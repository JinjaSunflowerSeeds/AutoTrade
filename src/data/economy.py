<<<<<<< HEAD
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
=======

import sys
sys.path.append('./')
import requests, csv, json, urllib
import pandas as pd
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fake_useragent import UserAgent
import quandl
from full_fred.fred import Fred
from utils.config_reader import get_economy_data_conf


class Quandl:
    def __init__(self):
        api_key='esXmxpmMy8SwBFGvX9Nk'
        quandl.ApiConfig.api_key = api_key
        quandl.ApiConfig.verify_ssl = False
        ua = UserAgent()
        self.headers = {
        'User-Agent': ua.random,
        }
        url="https://data.nasdaq.com/api/v3/datasets/FRED/DTB3.csv?api_key={}".format(api_key)
        print(requests.get(url , headers = self.headers).json())

        # df = quandl.get("FRED/DTB3", trim_start="2023-01-01", authtoken=api_key)
        # df=quandl.get('FRED/GDP', trim_start="2001-01-01")# start_date="2011-12-31", end_date="2023-12-01")
        # print(df)

class CNN:#market greed and fear indicator
    def __init__(self, start_date, filepath):
        self.BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
        self.START_DATE = str(start_date)[:10]
        self.filepath=filepath
        ua = UserAgent()

        self.headers = {
        'User-Agent': ua.random,
        }
        self.CNN_index= None

    def export_file(self):
        self.CNN_index.to_csv(self.filepath, index=False)

    def get_fear_n_greed(self):
        r = requests.get(self.BASE_URL , headers = self.headers)
        data = r.json()
        print(data.keys())
        l=[]
        for d in ((data['fear_and_greed_historical']['data'])):
            t = int(d['x'])
            t = datetime.fromtimestamp(t / 1000).strftime('%Y-%m-%d')
            s = float(d['y'])
            r= d['rating']
            l.append([t,s,r])
              
        self.CNN_index= pd.DataFrame(l, columns=['date', 'score', 'rating'])
        print(self.CNN_index.shape)
        print(self.CNN_index)

class FederalReserve:
    def __init__(self, start_ds, end_ds, filepath, key):
        # self.api_key='6fe7ae22575d4e763b079ff23965a811'
        self.fred = Fred(key)
        self.start_ds=start_ds
        self.end_ds=end_ds
        self.filepath=filepath
        #creating daily df so later the left join on weekly/monthly ochl data will be easier
        self.gdp=  pd.DataFrame(pd.date_range(start=start_ds.strftime("%Y-%m-%d"), end=end_ds.strftime("%Y-%m-%d")), columns=['date'])
        self.cpi= pd.DataFrame(pd.date_range(start=start_ds.strftime("%Y-%m-%d"), end=end_ds.strftime("%Y-%m-%d")), columns=['date'])
        self.mortgage= pd.DataFrame(pd.date_range(start=start_ds.strftime("%Y-%m-%d"), end=end_ds.strftime("%Y-%m-%d")), columns=['date'])
        self.treasury= pd.DataFrame(pd.date_range(start=start_ds.strftime("%Y-%m-%d"), end=end_ds.strftime("%Y-%m-%d")), columns=['date'])
        self.unemployment= pd.DataFrame(pd.date_range(start=start_ds.strftime("%Y-%m-%d"), end=end_ds.strftime("%Y-%m-%d")), columns=['date'])

    def export_data(self):
        print("Exporting economy data to {}".format(self.filepath))
        self.gdp.to_csv(self.filepath.format('gdp'), index=False)
        self.cpi.to_csv(self.filepath.format('cpi'), index=False)
        self.mortgage.to_csv(self.filepath.format('mortgage'), index=False)
        self.treasury.to_csv(self.filepath.format('treasury'), index=False)
        self.unemployment.to_csv(self.filepath.format('unemployment'), index=False)

    def get_gdp(self):
        gdp= self.fred.get_series_df('GDPPOT')[['date','value']]
        gdp=gdp.rename({'value':'gdp'}, axis='columns')
        gdp['date']= pd.to_datetime(gdp.date)
        gdp= gdp[(gdp.date>=self.start_ds)&(gdp.date<=self.end_ds)]
        self.gdp= self.gdp.merge(gdp, on='date', how='left')
        self.gdp['gdp']= self.gdp['gdp'].ffill()
        print(self.gdp)

    def get_cpi(self):
        cpi= self.fred.get_series_df('CPIAUCSL')[['date','value']]
        cpi= cpi.rename({'value':'cpi'}, axis='columns')
        cpi['date']= pd.to_datetime(cpi.date)
        cpi = cpi[(cpi.date>=self.start_ds)&(cpi.date<=self.end_ds)]
        
        self.cpi= self.cpi.merge(cpi, on='date', how='left')
        self.cpi['cpi']= self.cpi['cpi'].ffill()
        print(self.cpi)

    def get_mortgage(self):
        mortgage= self.fred.get_series_df('MORTGAGE30US')[['date','value']]
        mortgage= mortgage.rename({'value':'mortgage'}, axis='columns')
        mortgage['date']= pd.to_datetime(mortgage.date)
        mortgage = mortgage[(mortgage.date>=self.start_ds)&(mortgage.date<=self.end_ds)]
        
        self.mortgage= self.mortgage.merge(mortgage, on='date', how='left')
        self.mortgage['mortgage']= self.mortgage['mortgage'].ffill()
        print(self.mortgage)

    def get_10yr_treasury(self):
        treasury= self.fred.get_series_df('T10Y2Y')[['date','value']]
        treasury= treasury.rename({'value':'treasury'}, axis='columns')
        treasury['date']= pd.to_datetime(treasury.date)
        treasury = treasury[(treasury.date>=self.start_ds)&(treasury.date<=self.end_ds)]
        self.treasury= self.treasury.merge(treasury, on='date', how='left')
        self.treasury['treasury']= self.treasury['treasury'].ffill()
        print(self.treasury)

    def get_unemployment(self):
        unemployment= self.fred.get_series_df('UNRATE')[['date','value']]
        unemployment= unemployment.rename({'value':'unemployment'}, axis='columns')
        unemployment['date']= pd.to_datetime(unemployment.date)
        unemployment = unemployment[(unemployment.date>=self.start_ds)&(unemployment.date<=self.end_ds)]
        
        self.unemployment= self.unemployment.merge(unemployment, on='date', how='left')
        self.unemployment['unemployment']= self.unemployment['unemployment'].ffill()
        print(self.unemployment)

if __name__=='__main__':
    fred_api_key, data_output_base_dir, lookback= get_economy_data_conf()
    cnn= CNN(start_date= datetime.now() - relativedelta(years=int(max(3,lookback))),#fails for more than 3yr
                filepath=data_output_base_dir+'fear_n_greed.csv')
    cnn.get_fear_n_greed()
    cnn.export_file()
    #
    fr= FederalReserve(end_ds= datetime.now(), start_ds= datetime.now()- relativedelta(years=lookback)
                       ,filepath=data_output_base_dir+'{}.csv', key=fred_api_key)
    fr.get_gdp()
    fr.get_cpi()
    fr.get_mortgage()
    fr.get_10yr_treasury()
    fr.get_unemployment()
    fr.export_data()
>>>>>>> 5013666 (first commit)
