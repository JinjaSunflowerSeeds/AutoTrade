import logging
import re
import sys

sys.path.append("./")
import os
import warnings

import pandas as pd
from utils.config_reader import (
    get_economy_data_conf,
    get_fundamentals_data_conf,
    get_merger_output_file,
    get_ticker_data_conf,
)
from utils.logging import LoggerUtility

warnings.filterwarnings("ignore")

"""
cmd: 
    python3 data/merger.py > merger.txt 2>&1 
    
    
Every dir gets combined into combined.csv
We read only the combined.csv from required dirs and merge them
For correlated stocks we rename the colums by appending the stock name to them to avoid merge collisions 
Times will be in UTC
Column names will be lower cased
"""

logger = LoggerUtility.setup_logger(__file__)


class Consolidator:
    def __init__(self):
        self.stock = None
        self.interval = None
        self.tmp_df = pd.DataFrame()
        self.df = pd.DataFrame()
        self.files = []

    def consolidate_dir(self, dir):
        # a utility to combine all csv files in a directory into one
        combined_df = pd.DataFrame()
        for file in os.listdir(dir):
            if file.endswith(".csv") :#and not file.startswith("combined"):
                file_path = os.path.join(dir, file)
                logger.info(file_path)
                df = pd.read_csv(file_path, index_col=None)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        if not combined_df.empty:
            output_file = os.path.join(dir, "combined.csv")
            # self.files.append(output_file)
            logger.info(f" -> Combined file={combined_df.shape}")
            logger.info(f"{combined_df.columns.tolist()}")
            combined_df.drop_duplicates("date", inplace=True)
            combined_df.sort_values(by="date", inplace=True)
            combined_df.to_csv(output_file, index=False)
            logger.info(
                f" ->Exported combined CSV: {output_file} =>{combined_df.shape}"
            )

    def get_dirs(self, dir):
        dirs = []
        for path, _, _ in os.walk(dir):
            if path == dir:
                continue
            if not os.path.isdir(path):
                dirs.append(dir)
            else:
                dirs += self.get_dirs(path)
        if len(dirs) == 0:
            dirs.append(dir)
        return set(dirs)


class Merger(Consolidator):
    def read_data(self, filepath):
        self.tmp_df = pd.read_csv(filepath)
        self.tmp_df = self.tmp_df[
            [i for i in self.tmp_df.columns.tolist() if not re.search("nnamed", i)]
        ]
        if self.stock not in filepath and "ohlcv" in filepath:
            self.tmp_df.rename(
                columns={
                    col: col + f'_{filepath.split("/")[-3]}'
                    for col in self.tmp_df
                    if col not in ["Date", "date"]
                },
                inplace=True,
            )
        logger.info(f"{filepath}, {self.tmp_df.shape}")

    def correct_date(self):
        # making all date to minute level and ensuring they are datetime otherwise merge by closest wont work
        if "date" in self.tmp_df.columns.tolist():
            self.tmp_df["date"] = pd.to_datetime(
                pd.to_datetime(self.tmp_df["date"], utc=True).dt.strftime(
                    "%Y-%m-%d, %H:%M:%S"
                )
            )
        elif "Earnings Date" in self.tmp_df.columns.tolist():
            self.tmp_df["date"] = pd.to_datetime(
                pd.to_datetime(self.tmp_df["Earnings Date"], utc=True).dt.strftime(
                    "%Y-%m-%d, %H:%M:%S"
                )
            )
        elif "Date" in self.tmp_df.columns.tolist():
            self.tmp_df["date"] = pd.to_datetime(
                pd.to_datetime(self.tmp_df["Date"], utc=True).dt.strftime(
                    "%Y-%m-%d, %H:%M:%S"
                )
            )
        elif "Unnamed: 0" in self.tmp_df.columns.tolist():
            self.tmp_df["date"] = pd.to_datetime(
                pd.to_datetime(self.tmp_df["Unnamed: 0"], utc=True).dt.strftime(
                    "%Y-%m-%d, %H:%M:%S"
                )
            )
        else:
            assert False, "date col not found"

    def drop_columns(self, columns_to_drop=["Date", "Earnings Date", "Unnamed"]):
        for c in columns_to_drop:
            if c in self.df.columns.tolist():
                self.df.drop(self.df.filter(regex=c).columns, axis=1, inplace=True)

    def impute(self):
        for c in ["Stock Splits", "Dividends"]:
            if c in self.df.columns.tolist():
                self.df[c] = self.df[c].fillna(0)
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)
        self.df.dropna(axis="columns", inplace=True)

    def merger(self):
        if not len(self.df):
            self.df = self.tmp_df
        else:
            # self.df= self.df.merge(self.tmp_df, how='left', on='date')
            self.df.sort_values(by="date", inplace=True)
            self.tmp_df.sort_values(by="date", inplace=True)
            # logger.info(self.tmp_df.columns.tolist())
            # logger.info(self.df.columns.tolist())
            self.df = pd.merge_asof(
                self.df, self.tmp_df, on="date", direction="nearest"
            )
        logger.info("merged df shape={}".format(self.df.shape))

    def get_data_files(self):
        # only read the required files
        # the stock and its correlations
        (
            self.stock,
            self.interval,
            lookback,
            indices,
            correlated_stocks,
            data_output_base_dir,
        ) = get_ticker_data_conf()
        self.files = []
        # this has to comes first otherwise the merge will messed it up (we left join on first df and its importnt to be the main df)
        for s in [self.stock] + correlated_stocks + indices:
            d = data_output_base_dir.format(s, self.interval)
            for f in os.listdir(d):
                if "combined" in f and ".csv" in f:
                    self.files.append(d + "/" + f)
        t = len(self.files)
        logger.info("   Loaded {} ohlcv files=> {}".format(t, self.files))
        # economy data
        data_output_base_dir = get_economy_data_conf()[1]
        for f in os.listdir(data_output_base_dir[:-3]):
            if ".DS_Store" in f:
                continue
            d = data_output_base_dir.format(f)
            for ff in os.listdir(d):
                if "combined" in ff and ".csv" in ff:
                    self.files.append(d + "/" + ff)
        logger.info(
            "   Loaded {} economy files=> {}".format(
                len(self.files) - t, self.files[t:]
            )
        )
        t = len(self.files)
        # fundamentals data
        data_output_base_dir = get_fundamentals_data_conf()[2]
        for f in os.listdir(data_output_base_dir[:-3]):
            if ".DS_Store" in f:
                continue
            d = data_output_base_dir.format(f)
            for ff in os.listdir(d):
                if 'DS_Store' in ff:
                    continue
                for fff in os.listdir(d + "/" + ff):
                    if "combined" in fff and ".csv" in fff:
                        self.files.append(d + "/" + ff + "/" + fff)
        logger.info(
            "   Loaded {} fundamental files=> {}".format(
                len(self.files) - t, self.files[t:]
            )
        )

    def export_to_csv(self):
        dir , interval = get_merger_output_file()
        logger.info("Exporting merged file to {}".format(dir))
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir+='/{}.csv'.format(interval)
        self.df.loc[:, ~self.df.columns.str.contains("^unnamed")].to_csv(dir)

    def driver(self):
        (
            self.stock,
            self.interval,
            lookback,
            indices,
            correlated_stocks,
            data_output_base_dir,
        ) = get_ticker_data_conf()
        dirs = []
        for f in ["economy", "fundamentals", "ohlcv"]:
            dirs += self.get_dirs("./files/{}".format(f))
        logger.info("dir to combine csv files in {dirs}")
        for dir in dirs:
            self.consolidate_dir(dir)
        logging.info("Files to combined")
        self.get_data_files()
        for f in self.files:
            if ".csv" not in f:
                continue
            assert "combined" in f, f
            self.read_data(f)
            self.correct_date()
            self.drop_columns()
            self.merger()
        self.impute()
        self.df.columns = self.df.columns.str.lower()
        self.export_to_csv()
        


if __name__ == "__main__":
    m = Merger()
    m.driver()
    logger.info("Done with merger!")
