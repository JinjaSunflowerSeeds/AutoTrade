import os
import sys

sys.path.append("./")
import csv
import json
import requests
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import quandl
from fake_useragent import UserAgent
from full_fred.fred import Fred
from utils.config_reader import get_economy_data_conf
from utils.logging import LoggerUtility

logger = LoggerUtility.setup_logger(__file__)

class QuandlAPI:
    def __init__(self, api_key="esXmxpmMy8SwBFGvX9Nk"):
        self.api_key = api_key
        quandl.ApiConfig.api_key = self.api_key
        quandl.ApiConfig.verify_ssl = False
        self.headers = {"User-Agent": UserAgent().random}

    def log_test_request(self):
        url = f"https://data.nasdaq.com/api/v3/datasets/FRED/DTB3.csv?api_key={self.api_key}"
        response = requests.get(url, headers=self.headers)
        logger.info(response.json())

class CNN:
    BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"

    def __init__(self, start_date, filepath):
        self.start_date = str(start_date)[:10]
        self.filepath = filepath
        self.headers = {"User-Agent": UserAgent().random}
        self.data = pd.DataFrame()

    def export_file(self, name="fear_n_greed"):
        dir_path = self.filepath.format(name)
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{self.data.date.min()}_to_{self.data.date.max()}.csv"
        file_path = os.path.join(dir_path, filename)
        self.data.to_csv(file_path, index=False)

    def fetch_data(self):
        response = requests.get(self.BASE_URL, headers=self.headers)
        data = response.json()
        records = [
            [datetime.fromtimestamp(int(item["x"]) / 1000).strftime("%Y-%m-%d"), float(item["y"]), item["rating"]]
            for item in data["fear_and_greed_historical"]["data"]
        ]
        self.data = pd.DataFrame(records, columns=["date", "score", "rating"])
        logger.info(f"Fetched Fear & Greed data: {self.data.shape}\n{self.data}")

class FederalReserve:
    def __init__(self, start_date, end_date, filepath, api_key):
        self.fred = Fred(api_key)
        self.start_date = start_date
        self.end_date = end_date
        self.filepath = filepath
        self.data = {}
        logger.info(f"Initialized FederalReserve with filepath={filepath}, start_date={start_date}, end_date={end_date}")

    def fetch_series(self, series_id, rename_column):
        df = self.fred.get_series_df(series_id)[["date", "value"]]
        df = df.rename(columns={"value": rename_column})
        df["date"] = pd.to_datetime(df.date)
        logger.info(f"Fetched {rename_column} data: {df.shape}\n{df}")
        return df

    def export_file(self, data, name):
        dir_path = self.filepath.format(name)
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{data.date.min().strftime('%Y_%m_%d')}_to_{data.date.max().strftime('%Y_%m_%d')}.csv"
        file_path = os.path.join(dir_path, filename)
        data.to_csv(file_path, index=False)

    def export_all(self):
        for name, data in self.data.items():
            self.export_file(data, name)
        logger.info("Exported all Federal Reserve data.")

    def fetch_data(self):
        self.data = {
            "gdp": self.fetch_series("GDPPOT", "gdp"),
            "cpi": self.fetch_series("CPIAUCSL", "cpi"),
            "mortgage": self.fetch_series("MORTGAGE30US", "mortgage"),
            "treasury": self.fetch_series("T10Y2Y", "treasury"),
            "unemployment": self.fetch_series("UNRATE", "unemployment")
        }

if __name__ == "__main__":
    fred_api_key, data_output_base_dir, lookback_years = get_economy_data_conf()

    cnn = CNN(
        start_date=datetime.now() - relativedelta(years=max(3, lookback_years)),
        filepath=data_output_base_dir
    )
    cnn.fetch_data()
    cnn.export_file()

    federal_reserve = FederalReserve(
        start_date=datetime.now() - relativedelta(years=lookback_years),
        end_date=datetime.now(),
        filepath=data_output_base_dir,
        api_key=fred_api_key
    )
    federal_reserve.fetch_data()
    federal_reserve.export_all()
