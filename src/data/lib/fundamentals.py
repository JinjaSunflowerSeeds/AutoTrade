
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
yf.pdr_override()


    # -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:49:09 2021

@author: Administrator
"""


#this website is called macrotrends
#this script is designed to scrape its financial statements
#yahoo finance only contains the recent 5 year
#macrotrends can trace back to 2005 if applicable
import re
import json
import pandas as pd
import requests
import os
# os.chdir('k:/')



class HistoricalPE:
    def __init__(self, ticker='AAPL', company='apple', filepath=None):
        # 'AAPL/apple'
        self.url='https://www.macrotrends.net/stocks/charts/{}/{}/financial-statements?freq=Q'.format(ticker,company)
        self.ticker=ticker
        self.filepath=filepath.format(ticker, 'financial_statements')

    #simply scrape
    def scrape(self,**kwargs):
        session=requests.Session()
        session.headers.update(
                {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'})
        response=session.get(self.url,**kwargs)
        return response

    def etl(self, response):#create dataframe
        #regex to find the data
        num=re.findall('(?<=div\>\"\,)[0-9\.\"\:\-\, ]*',response.text)
        text=re.findall('(?<=s\: \')\S+(?=\'\, freq)',response.text)
        #convert text to dict via json
        dicts=[json.loads('{'+i+'}') for i in num]
        #create dataframe
        df=pd.DataFrame()
        for ind,val in enumerate(text):
            df[val]=dicts[ind].values()
        df.index=dicts[ind].keys()
        names={
                'cost-goods-sold':'cost',
                'gross-profit':'g-profit',
                'research-development-expenses':'rnd',
                'selling-general-administrative-expenses':'sgae',
                'total-non-operating-income-expense':'tnoie',
                'total-provision-income-taxes':'tpit',
                'income-from-continuous-operations':'ifco',
                'eps-basic-net-earnings-per-share':'eps',
                'eps-earnings-per-share-diluted':'eps-diluted'
       }
        df.rename(columns=names, inplace=True)
        return df

    def main(self):
        print('Scraping data from macrotrends...')
        print("  ->{}".format(self.url))
        response=self.scrape()
        df=self.etl(response)
        print(" Exporting to {}".format(self.filepath))
        df.to_csv(self.filepath)
        print("Done: {}".format(df.shape))



class History:
    def __init__(self, ticker,
                 start_date,
                 end_date, filepath):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.filepath= filepath

        self.df = pd.DataFrame()

        self.dividends=pd.DataFrame()
        self.splits=pd.DataFrame()
        self.share_count=pd.DataFrame()
        self.quarterly_income_stmt=pd.DataFrame()
        self.balance_sheet=pd.DataFrame()
        self.cashflow=pd.DataFrame()
        self.earnings_dates=pd.DataFrame()
        self.options=pd.DataFrame()
        self.major_holders=pd.DataFrame()
        self.institutional_holders=pd.DataFrame()
        self.mutualfund_holders=pd.DataFrame()



    def get_historical_data(self):
        print(" Getting historical data...")
        hist= yf.Ticker(self.ticker)
        self.dividends = pd.DataFrame(hist.actions[hist.actions['Stock Splits']==0]['Dividends'], columns=['Dividends'])
        self.splits= pd.DataFrame(hist.actions[hist.actions['Stock Splits']>0]['Stock Splits'], columns=['Stock Splits'])
        self.share_count=pd.DataFrame(hist.get_shares_full(start=self.start_date, end=self.end_date), columns=['shares'])
        self.quarterly_income_stmt =hist.quarterly_income_stmt.T
        self.balance_sheet =hist.balance_sheet.T
        self.cashflow = hist.cashflow.T
        self.major_holders = pd.DataFrame(hist.major_holders.values, columns=[ 'Share%','Name'])
        self.institutional_holders =hist.institutional_holders
        self.mutualfund_holders =hist.mutualfund_holders
        self.earnings_dates = hist.get_earnings_dates(limit=40000)
        self.options =pd.DataFrame(hist.options, columns=['expiration'] ).sort_values('expiration')

    def print_info(self):
        print(self.dividends.head(1))
        print("*"*100)
        print(self.splits.head(1))
        print("*"*100)
        print(self.share_count.head(1))
        print("*"*100)
        print(self.quarterly_income_stmt.head(1))
        print("*"*100)
        print(self.balance_sheet.head(1))
        print("*"*100)
        print(self.cashflow.head(1))
        print("*"*100)
        print(self.earnings_dates.head(1))
        print("*"*100)
        print(self.options.head(1))
        print("*"*100)
        print(self.major_holders.head(1))
        print("*"*100)
        print(self.institutional_holders.head(1))
        print("*"*100)
        print(self.mutualfund_holders.head(1))
        print("*"*100)


    def export_data(self):
        print(" Exporting the data to {}...".format(self.filepath))
        # df=pd.concat([self.dividends_n_splits, self.share_count, self.bquarterly_in])
        self.earnings_dates.to_csv(self.filepath.format(self.ticker, "earnings_dates"))
        self.dividends.to_csv(self.filepath.format(self.ticker, "dividends"))

        self.splits.to_csv(self.filepath.format(self.ticker, "splits"))
        self.share_count.to_csv(self.filepath.format(self.ticker, "share_count"))
        self.quarterly_income_stmt.to_csv(self.filepath.format(self.ticker, "quarterly_income_stmt"))
        self.balance_sheet.to_csv(self.filepath.format(self.ticker, "balance_sheet"))
        self.cashflow.to_csv(self.filepath.format(self.ticker, "cashflow"))
        self.options.to_csv(self.filepath.format(self.ticker, "options"))
        self.major_holders.to_csv(self.filepath.format(self.ticker, "major_holders"))
        self.institutional_holders.to_csv(self.filepath.format(self.ticker, "institutional_holders"))
        self.mutualfund_holders.to_csv(self.filepath.format(self.ticker, "mutualfund_holders"))

if __name__ == "__main__":
    _filepath, lookback= get_merger_output_file()
    h = History(ticker=_ticker,
                 start_date= datetime.now() - relativedelta(years=lookback),
                 end_date= datetime.now(),filepath=_filepath)
    h.get_historical_data()
    h.print_info()
    h.export_data()
    print("Yohai","\n", "*"*100)
    # macrotrends
    h=HistoricalPE(ticker=_ticker, company='apple',filepath=_filepath)
    h.main()
