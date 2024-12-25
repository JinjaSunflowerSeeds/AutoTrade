import pandas as pd
import copy


from preproc.imputation import PreProc
from preproc.technical_indicators import TA
from preproc.trading_signals import TS
from preproc.feature_eng import FeatureEngineering
from model.xgb.xgb import XGB
from utils.config_reader import get_merger_output_file
from lib.color_logger import MyLogger

class Driver(MyLogger):
    def __init__(self):
        super().__init__()
        self.df= None
        self._pp= PreProc(filepath=None)
        self._ta= TA(filepath=None)
        self._fe= FeatureEngineering(filepath=None)
        self._ts= TS()
        self._xgb= XGB()

    def read_file(self, filepath):
        self.log.info(" Reading file {} ...".format(filepath))
        self.df = pd.read_csv(filepath, index_col=False)
        self.df= self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        self.log.info("  -> df={}".format(self.df.shape))
        self.log.info("  -> date range from {} to {}".format(self.df['date'].min(), self.df['date'].max()))
        self.log.info(self.df.columns.tolist())

    def driver(self):
        for i in range(len(self.df) - self._xgb.start_from, len(self.df)+1):
            train= copy.deepcopy(self.df.iloc[:i])
            train=self._pp.driver(train)# impute
            train=self._ta.driver(train)# technical indocators
            train=self._ts.driver(train)# trading signals
            train=self._fe.driver(train)# feature eng
            self._xgb.driver(train)# training
            self._xgb.export_data()
        self._xgb.print_overall_important_features()


if __name__ == '__main__':
    d= Driver()
    d.read_file(filepath=get_merger_output_file())
    d.driver()
