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

from xgb_params import XgbUtils
from xgboost import XGBClassifier, XGBRegressor


class XGB(TrainConf, FeatureList, XgbUtils, MyLogger, Labeler):
    def __init__(self):
        super().__init__()
        FeatureList.__init__(self)
        MyLogger.__init__(self)
        Labeler.__init__(self)

        self.model = self.get_xbg_model(self.do_binary)
        self.df = pd.DataFrame()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.preds = []
        self.output_cols = ["date", "pred", "prob"]
        self.feature_importance_scores = [0] * len(self.features)

    def add_label(self, manual_labels_dir):
        self.short_label_profit_stop_loss()
        # self.long_label_profit_stop_loss()
        # self.add_manual_labels(manual_labels_dir)
        # if self.do_binary:
        #     self.add_binary_label()
        # else:
        #     self.add_regression_label(label_col="log-ret")

        # self.log.info("Train={}, Test={}".format(self.train.shape, self.test.shape))
        
    def split_data(self):
        self.train = self.df.iloc[:-1]
        self.test = self.df.tail(1)
        self.log.info("Train={}, Test={}".format(self.train.shape, self.test.shape))
        return self.train[self.features], self.test[self.features]#creating a copy of train/test cuz pca messed the column name up

    def driver(self, data, manual_labels_dir):
        self.set_data(data)
        self.add_label(manual_labels_dir)
        train, test= self.split_data()
        self.scale_feautres()
        if self.do_pca:
            train, test = self.pca_transform(train, test)
        labels = self.train.label

        self.fit(train, labels)
        self.predict(train, test, labels)
        self.feature_importance(train.columns.tolist())
