import sys
sys.path.append('./')
import pandas as pd
import os
from utils.config_reader import get_ticker_data_conf, get_economy_data_conf, get_merger_output_file, get_fundamentals_data_conf
import warnings
warnings.filterwarnings("ignore")


class Merger:
    def __init__(self):
        self.stock= None
        self.interval= None
        self.tmp_df=pd.DataFrame()
        self.df=pd.DataFrame()

    def read_data(self, filepath):
        self.tmp_df= pd.read_csv(filepath)
        print(filepath, self.tmp_df.shape)

    def correct_date(self):
        # making all date to minute level and ensuring they are datetime otherwise merge by closest wont work
        if 'date' in self.tmp_df.columns.tolist():
            self.tmp_df['date']= pd.to_datetime(pd.to_datetime(self.tmp_df['date'] ,utc=True).dt.strftime('%Y-%m-%d, %H:%M:%S'))
        elif 'Earnings Date' in self.tmp_df.columns.tolist():
            self.tmp_df['date']= pd.to_datetime(pd.to_datetime(self.tmp_df['Earnings Date'] ,utc=True).dt.strftime('%Y-%m-%d, %H:%M:%S'))
        elif 'Date' in self.tmp_df.columns.tolist():
            self.tmp_df['date']=pd.to_datetime(pd.to_datetime(self.tmp_df['Date'] ,utc=True).dt.strftime('%Y-%m-%d, %H:%M:%S'))
        elif 'Unnamed: 0' in self.tmp_df.columns.tolist():
            self.tmp_df['date']=pd.to_datetime(pd.to_datetime(self.tmp_df['Unnamed: 0'] ,utc=True).dt.strftime('%Y-%m-%d, %H:%M:%S'))
        else:
            assert False, 'date col not found'

    def drop_columns(self, columns_to_drop= ['Date', 'Earnings Date', 'Unnamed']):
        for c in columns_to_drop:
            if c in self.df.columns.tolist():
                self.df.drop(self.df.filter(regex=c).columns, axis=1, inplace=True)

    def impute(self):
        for c in ['Stock Splits', 'Dividends']:
            if c in self.df.columns.tolist():
                self.df[c]= self.df[c].fillna(0)
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)
        self.df.dropna(axis='columns', inplace=True)

    def merger(self):
        if not len(self.df) :
            self.df= self.tmp_df
        else: 
            # self.df= self.df.merge(self.tmp_df, how='left', on='date')
            self.df.sort_values(by='date', inplace=True)
            self.tmp_df.sort_values(by='date', inplace=True)
            print(self.tmp_df.columns.tolist())
            print(self.df.columns.tolist())
            self.df= pd.merge_asof(self.df, self.tmp_df, on='date', direction='nearest')
        print("merged df shape={}".format(self.df.shape))


    def get_data_files(self):
        # the stock and its correlations
        self.stock, self.interval, lookback, indices, correlated_stocks, data_output_base_dir= get_ticker_data_conf()
        files=[]
        # this has to comes first otherwise the merge will messed it up (we left join on first df and its importnt to be the main df)
        # TODO fixme such that main stock is alsway the first df and not its correlated ones
        for f in os.listdir(data_output_base_dir):
            if self.interval in f:
                files.append(data_output_base_dir +"/" + f)

        print("   Loaded {} ohclv files".format(len(files)))
        # economy data
        data_output_base_dir= get_economy_data_conf()[1]
        for f in os.listdir(data_output_base_dir):
            files.append(data_output_base_dir + f)
        print("   Loaded {} economy files".format(len(files))) 
        # fundamentals data
        data_output_base_dir= get_fundamentals_data_conf()[0]
        for f in os.listdir(data_output_base_dir):
            files.append(data_output_base_dir + f)
        print("   Loaded {} fundamentals files".format(len(files)))
        return files

    def driver(self):
        for f in self.get_data_files():
            if '.DS_Store' in f or '.csv' not in f:
                continue
            m.read_data(f)
            m.correct_date()
            m.drop_columns()
            m.merger()
        m.impute()
        self.df.columns = self.df.columns.str.lower()
        print("Exporting merged file to {}".format(get_merger_output_file().format(self.stock, self.interval) ) )
        self.df.loc[:, ~self.df.columns.str.contains('^unnamed')].to_csv(get_merger_output_file().format(self.stock, self.interval) )

if __name__ == '__main__':
    m= Merger()
    m.driver()
