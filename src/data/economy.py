
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
