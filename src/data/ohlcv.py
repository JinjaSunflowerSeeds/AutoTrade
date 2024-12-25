import sys

sys.path.append("./")
import os
from datetime import datetime

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from joblib import delayed, Parallel
from utils.config_reader import get_ticker_data_conf
from utils.logging import LoggerUtility

yf.pdr_override()


"""
cmd: 
    python3 data/ohlcv.py > ohlcv.txt 2>&1 

We simply download and export the data to csv files for the max range avaialble. 
Each stock in the list will be downloaded and exported to a separate dir/csv file. 

For each stock we export the following:
    - date(index), open,high,low,close, volume

Outout: 
    Filw will be exported to a new dir each time with the following format: e.g., ./files/ohlcv/INTC/1m/2024_12_19_0930_to_2024_12_24_1259.csv
    We then consolidate the historical data into a single file for each stock. ./files/ohlcv/INTC/1m/combined.csv
"""


class OHLCVBase:
    def __init__(self, ticker, main_stock, interval, filepath, lookback) -> None:
        self.ticker = ticker
        self.main_stock = main_stock
        self.interval = interval
        self.filepath: str = filepath.format(ticker, self.interval)
        # max range
        self.start_date = None  # self.get_start_time(interval)
        self.end_date = None  #  (datetime.now() + relativedelta(days=2),)
        self.df = pd.DataFrame()

        self.logger = LoggerUtility.setup_logger(__file__)

        self.logger.info(
            f"\nStock: {stock}, \nInterval: {interval}, \nLookback: {lookback}, \nIndices: {indices}, \nCorrelated Stocks: {correlated_stocks} \nData Output Base Dir: {self.filepath}"
        )

    def get_start_time(self, interval):
        if interval == "1m":
            return datetime.now() - relativedelta(days=7)
        return datetime.now() - relativedelta(years=5)

    def export_data(self):
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        filepath = self.filepath + "/{}_to_{}.csv".format(
            self.df.index.min().strftime("%Y_%m_%d_%H%M"),
            self.df.index.max().strftime("%Y_%m_%d_%H%M"),
        )
        self.logger.info(" Exporting csv to {}".format(filepath))
        self.logger.info(self.df)
        self.df.to_csv(filepath)

    def log_stats(self):
        self.logger.info(" Stats: ")
        self.logger.info("  -> Number of rows: {}".format(len(self.df)))
        self.logger.info("  -> Number of columns: {}".format(len(self.df.columns)))
        self.logger.info("  -> Column names: {}".format(self.df.columns))
        self.logger.info(f"  -> dtype:\n {self.df.dtypes}")
        self.logger.info(f"  -> Info:\n {self.df.info()}")
        self.logger.info(f"  -> nulls:\n { self.df.isnull().sum()}")

    def set_data_type(self):
        self.logger.info(" Setting data types...")
        self.df.sort_values(by="Date")
        self.df = self.df.astype(
            {"Open": float, "High": float, "Low": float, "Close": float, "Volume": int}
        )

    def sanity_check(self):
        self.logger.info(len(self.df))
        assert len(self.df) > 0, "No data found."
        assert len(self.df.columns) == 5, "Wrong number of columns"
        assert len(self.df.index.unique()) == len(
            self.df
        ), "repeated dates for {}".format(t)
        assert len(self.df) > 0, "Empty dataframe for {}".format(t)
        assert len(self.df.index.unique()) == len(
            self.df.index.unique()
        ), "missing days for {}".format(t)

    def consolidate_data(self):
        self.logger.info("Consolidating data...")
        for subdir, _, _ in os.walk(self.filepath):
            self.logger.info(f"Combining directory: {subdir}")
            combined_df = pd.DataFrame()
            for file in os.listdir(subdir):
                if file.endswith(".csv") and not file.startswith("combined"):
                    file_path = os.path.join(subdir, file)
                    df = pd.read_csv(file_path, index_col=None)
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
            if not combined_df.empty:
                output_file = os.path.join(subdir, "combined.csv")
                self.logger.info(f" -> Combined file={combined_df.shape}")
                combined_df.drop_duplicates("Date", inplace=True)
                combined_df.to_csv(output_file, index=False)
                self.logger.info(
                    f" ->Exported combined CSV: {output_file} =>{combined_df.shape}"
                )


class OHLCV(OHLCVBase):
    def download_data(self):
        self.logger.info(f" Getting data {self.ticker}...")
        tmp = yf.download(
            self.ticker,
            self.start_date,
            self.end_date,
            auto_adjust=True,
            keepna=True,
            interval=self.interval,
            progress=False,
        )
        if "Date" not in tmp.columns:
            tmp.index.names = ["Date"]
        self.df = pd.concat([self.df, tmp], axis=0)
        self.logger.info(
            "  -> {} downloaded: {}, {}, {}".format(
                self.ticker, tmp.shape, self.df.index.min(), self.df.index.max()
            )
        )

    def driver(self, logger_file_date):
        self.logger = LoggerUtility.setup_logger(
            name=__file__, file_date=logger_file_date
        )  # need to reset the logger so the worker inherits it
        try:
            self.logger.info("Data engine starting...")
            self.download_data()
            self.log_stats()
            self.set_data_type()
            self.sanity_check()
            self.export_data()
            self.consolidate_data()
            self.logger.info("Success downloads!")
        except Exception as e:
            self.logger.error(e)
            exit(1)


if __name__ == "__main__":
    stock, interval, lookback, indices, correlated_stocks, data_output_base_dir = (
        get_ticker_data_conf()
    )
    exec_date = datetime.now().strftime("%Y-%m-%d %H-%M")
    results = Parallel(n_jobs=-1)(
        delayed(
            OHLCV(
                ticker=s,
                main_stock=stock,
                interval=interval,
                lookback=lookback,
                filepath=data_output_base_dir,
            ).driver
        )(exec_date)
        for s in [stock] + indices + correlated_stocks
    )
    print(results)
    print("OHLCV done\n", "=" * 200)
