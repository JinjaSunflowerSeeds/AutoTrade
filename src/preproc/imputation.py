import numpy as np
import pandas as pd
# from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

# predict the price (intrinsic value) based on the fundementals. Maybe you can make it
# XXL, XL, L, M, S, XS and XSS categories based on the retrun range predicted for the next quarter or year

# Our target variable in this research is stock’s quarterly
# relative returns with respect to the Dow Jones Industrial Average
# (DJIA).

# you can score multiple stocks and then invest in the top ones and refresh in the next period

# use sharpe to get risk adjusted return

class PreProc:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df=pd.DataFrame()

    def read_file(self):
        print(" Reading file {} ...".format(self.filepath))
        self.df = pd.read_csv(self.filepath)
        print("  -> df={}".format(self.df.shape))

    def set_data(self, data):
        self.df=data
    def get_data(self):
        return self.df

    def interpolate_missing_values(self):
        # commonly used for ohlcv data
        columns= self.df.columns
        print(" Interpolating missing values for {}...".format(columns))
        assert columns is not None, "not col was given for transformation"
        if(self.df.isna().sum().sum()):
            print("  ->N/A's count=\n{}".format(self.df.isna().sum()))
        self.df.sort_values(by='date', inplace=True)
        self.df[columns]=self.df[columns].interpolate(method='linear')


    def export_data(self):
        print(" Exporting the data to {}...".format(self.filepath))
        self.df.to_csv(self.filepath)

    def driver(self, data):
        self.set_data(data)
        self.interpolate_missing_values()
        print("*"*100)
        return self.get_data()

if __name__ == '__main__':
    pp = PreProc(filepath='./files/ohlcv_1d.csv')
    pp.read_file()
    pp.interpolate_missing_values()
    pp.export_data()
