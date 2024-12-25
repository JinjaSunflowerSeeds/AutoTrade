import pandas as pd
import numpy as np

class FeatureEngineering:
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

    def export_data(self, filepath=None):
        print(" Exporting the data to {}...".format(filepath))
        self.df.to_csv(filepath)

    def annotate_df(self):
        print("Annotating dataframe...")
        # print(self.df.columns.tolist())
        self.df['date'] = pd.to_datetime(self.df.index)
        self.df.reset_index(drop=True, inplace=True)
        self.df['quarter']= self.df['date'].dt.quarter
        self.df['month']= self.df['date'].dt.month
        self.df['week']= self.df['date'].dt.isocalendar().week
        self.df['day_of_year']= self.df['date'].dt.day_of_year
        self.df['hour']= self.df['date'].dt.hour

    def add_up_cnt(self):
        print("Adding number of gains in past x days...")
        self.df['up_cnt_7d'] = (self.df['log-ret']
                          .rolling(window=7,min_periods=0)
                          .agg(lambda x: (x > 0).sum()))/7.0
        self.df['up_cnt_30d'] = (self.df['log-ret']
                          .rolling(window=30,min_periods=0)
                          .agg(lambda x: (x > 0).sum()))/30.0

    def add_price_rng(self):
        print("Adding price range...")
        self.df['high_price_7d'] = self.df['close'].rolling(window=7,min_periods=0).max()-self.df['close']
        self.df['low_price_7d'] = self.df['close'].rolling(window=7,min_periods=0).min()-self.df['close']
        self.df['high_price_30d'] = self.df['close'].rolling(window=30,min_periods=0).max()-self.df['close']
        self.df['low_price_30d'] = self.df['close'].rolling(window=30,min_periods=0).min()-self.df['close']

    def add_price_seq(self):
        print("Adding when was the latest time price was higher or lower...")
        # time since last higher price: tslHp
        # time since lower price: tslLp
        h=[0]*len(self.df)
        l=[0]*len(self.df)
        self.df.sort_values(by='date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        for i in range(1, len(self.df)):
            p=self.df.loc[i].close
            previ_df =self.df.loc[:i-1]
            prev_high = previ_df[previ_df.close>=p].index.max()
            prev_low = previ_df[previ_df.close<=p].index.max()
            h[i] = i-prev_high if not pd.isna(prev_high) else i
            l[i] = i-prev_low if not pd.isna(prev_low) else i 
        self.df['tslHp']=h
        self.df['tslLp']=l

    def add_sin_cos(self):
        print("Adding sin and cos features...")
        l= [('hour', 365.25*24), ('day_of_year', 365.25), ('week', 365.25/7), ('month', 365.25/12), ('quarter', 365.25/4)]
        for (name, period) in l:
            self.df['sin_{}'.format(name)] = np.sin(2 * np.pi*self.df[name] / period )
            self.df['cos_{}'.format(name)] = np.cos(2 * np.pi*self.df[name] / period )
        # self.df['sin'] = np.sin(2 * np.pi / 365.25 *self.df.day_of_year)
        # self.df['cos'] = np.cos(2 * np.pi / 365.25 *self.df.day_of_year)      
        
    def add_next_open(self):
        # WARN adding here because we predicting close higher than open (so in this format its not leakage)
        print("  -> adding next_open to the dataset")
        self.df['next_open'] = self.df.open.shift(-1) 
        self.df['next_open'].fillna(self.df.open, inplace=True)
        
    def add_regression_slope(self):
        for i in [3, 7, 30]:
            self.df['slope_{}d'.format(i)] = self.df['close'].rolling(window=i, min_periods=2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            self.df['poly_{}d'.format(i)] = self.df['close'].rolling(window=i, min_periods=2).apply(lambda x: np.polyfit(range(len(x)), x, 2)[0])

    # def daily_variation(self):
        
    def driver(self, data):
        self.set_data(data)
        self.annotate_df()
        self.add_sin_cos()
        self.add_up_cnt()
        self.add_price_rng()
        self.add_price_seq()
        self.add_next_open()
        self.add_regression_slope()
        self.export_data('./files/training/fe_1d.csv')
        print("-"*100)
        return self.get_data()

if __name__ == '__main__':
    fe = FeatureEngineering(filepath='./files/ta_1d.csv')
    fe.read_file()
    fe.annotate_df()
    fe.add_sin_cos()
    fe.add_up_cnt()
    fe.add_price_rng()
    fe.export_data('./files/training/fe_1d.csv')
