# fit a final xgboost model on the housing dataset and make a prediction
import sys
from re import L

sys.path.append("./model/xgb")
sys.path.append("./model/labeling")


import os

import numpy as np
import pandas as pd
from config.train_conf import FeatureList, TrainConf
from lib.color_logger import MyLogger

# from labeling.labeler import Labeler
from model.labeling.labeler import Labeler

from xgb_params_v2 import XgbUtils
from xgboost import XGBClassifier, XGBRegressor


class XGB(TrainConf, FeatureList, XgbUtils, MyLogger, Labeler):
    def __init__(self):
        super().__init__()
        FeatureList.__init__(self)
        MyLogger.__init__(self)
        Labeler.__init__(self)

        self.df = pd.DataFrame()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.preds = []
        self.output_cols = ["date", "pred", "prob"]
        self.feature_importance_scores = [0] * len(self.features)

    def add_label(self, manual_labels_dir):
        # self.four_out_of_five()
        # self.short_label_profit_stop_loss()
        # self.long_label_profit_stop_loss()
        self.rsi_strategy()
        # self.add_manual_labels(manual_labels_dir)
        # if self.do_binary:
        #     self.add_binary_label()
        # else:
        #     self.add_regression_label(label_col="log-ret")

        # self.log.info("Train={}, Test={}".format(self.train.shape, self.test.shape))
        
    def split_data(self,split_idx):
        self.train = self.df.iloc[:split_idx]
        self.test = self.df.iloc[split_idx:]#.tail(split_idx)
        self.train_labels = self.train.label
        self.test_labels = self.test.label
        # self.train, self.test= self.train[self.features], self.test[self.features]#creating a copy of train/test cuz pca messed the column name up
        self.log.info("Train={}, Test={}".format(self.train.shape, self.test.shape))
        self.log.info(" ->Train 1s={}, Test 1s={}".format(self.train[self.train.label==1].shape, self.test[self.test.label==1].shape))

    def driver(self, data, manual_labels_dir, split_idx):
        self.set_features()
        self.set_data(data)
        self.add_label(manual_labels_dir)
        self.model = self.get_xbg_model(self.do_binary, self.df)
        self.split_data(split_idx)
        self.scale_feautres()
        if self.do_pca:
            self.train, self.test = self.pca_transform(self.train, self.test)

        self.fit(self.train, self.train_labels)
        self.predict(self.train, self.test, self.train_labels)
        self.feature_importance(self.features)
