import sys

sys.path.append("./")
import os
from datetime import datetime

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from joblib import delayed, Parallel
import datetime
from utils.config_reader import get_ticker_data_conf

yf.pdr_override()

"""
date, open,high,low,close volume for main stock
for other correlated stuff we just export the close value and name it as the stock itself

filw will be exported to a new dir each time with the following format: e.g., 
 ./files/technical/NVDA/2024_12_24_17_13/NVDA_ohlcv_1m.csv
"""

class OHLCVBase:
    def __init__(self, ticker,main_stock, interval, filepath, lookback) -> None:
        self.ticker = ticker
        self.main_stock = main_stock
        self.interval = interval
        self.filepath = filepath.format(self.ticker, self.interval)
        # max range
        self.start_date = None  # self.get_start_time(interval)
        self.end_date = None  #  (datetime.now() + relativedelta(days=2),)
        self.df = pd.DataFrame()

    def get_start_time(self, interval):
        if interval == "1m":
            return datetime.now() - relativedelta(days=7)
        return datetime.now() - relativedelta(years=5)

    def export_data(self):
        print(" Exporting csv to {}".format(self.filepath))
        if self.ticker != self.main_stock:
            # self.df[self.ticker] = self.df.Close
            self.df.columns=[c+'_{}'.format(self.ticker) for c in self.df.columns]
            print(self.df)
            # self.df = self.df[self.ticker]
        self.df.to_csv(self.filepath)

    def print_stats(self):
        print(" Stats: ")
        print("  -> Number of rows: {}".format(len(self.df)))
        print("  -> Number of columns: {}".format(len(self.df.columns)))
        print("  -> Column names: {}".format(self.df.columns))
        print("  -> dtype:\n", self.df.dtypes)
        print("  -> Info:\n", self.df.info())
        print("  -> nulls:\n", self.df.isnull().sum())

    def set_data_type(self):
        print(" Setting data types...")
        self.df.sort_values(by="Date")
        self.df = self.df.astype(
            {"Open": float, "High": float, "Low": float, "Close": float, "Volume": int}
        )

    def sanity_check(self):
        print(len(self.df))
        assert len(self.df) > 0, "No data found."
        assert len(self.df.columns) == 5, "Wrong number of columns"
        assert len(self.df.index.unique()) == len(
            self.df
        ), "repeated dates for {}".format(t)
        assert len(self.df) > 0, "Empty dataframe for {}".format(t)
        assert len(self.df.index.unique()) == len(
            self.df.index.unique()
        ), "missing days for {}".format(t)

class OHLCV(OHLCVBase):
    def download_data(self):
        print(f" Getting data {self.ticker}...")
        # get the new data
        try:
            tmp = yf.download(
                self.ticker,
                self.start_date ,
                self.end_date,
                auto_adjust=True,
                keepna=True,
                interval=self.interval,
                progress=False,
            )
            if "Date" not in tmp.columns:
                tmp.index.names = ["Date"]
            self.df = pd.concat([self.df, tmp], axis=0)
            print("  -> {} downloaded: {}".format(self.ticker, tmp.shape))
        except Exception as e:
            print(e)
            exit(1)

    def driver(self, stock):
        print("Data engine starting...")
        self.download_data()
        self.print_stats()
        self.set_data_type()
        # self.sanity_check()
        self.export_data()
        print("Success downloads!\n", "*" * 100)


if __name__ == "__main__":

    stock, interval, lookback, indices, correlated_stocks, data_output_base_dir = (
        get_ticker_data_conf()
    )
    print(f"Stock: {stock}, \nInterval: {interval}, \nLookback: {lookback}, \nIndices: {indices}, \nCorrelated Stocks: {correlated_stocks}, \nData Output Base Dir: {data_output_base_dir}\n","+" * 100)
    if not os.path.exists(data_output_base_dir):
        os.makedirs(data_output_base_dir)

    OHLCV(
                ticker=stock,
                main_stock=stock,
                interval=interval,
                lookback=lookback,
                filepath=data_output_base_dir + "/{}_ohlcv_{}.csv",
            ).driver(stock)


    results = Parallel(n_jobs=-1)(
        delayed(
            OHLCV(
                ticker=s,
                main_stock=stock,
                interval=interval,
                lookback=lookback,
                filepath=data_output_base_dir + "/{}_ohlcv_{}.csv",
            ).driver
        )(s)
        for s in [stock] + indices + correlated_stocks
    )
    print("OHLCV done\n", "=" * 200)
    print("All done\n", "=" * 200)
