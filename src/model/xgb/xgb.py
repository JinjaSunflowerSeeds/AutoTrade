# fit a final xgboost model on the housing dataset and make a prediction
import sys
sys.path.append('./model/xgb')

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from xgb_params import (
    XgbUtils
)
from config.train_conf import (
    FeatureList,
    TrainConf
)
from lib.color_logger import MyLogger

class XGB(TrainConf, FeatureList,  XgbUtils, MyLogger):
    def __init__(self):
        super().__init__()
        FeatureList.__init__(self)
        MyLogger.__init__(self)

        self.model = self.get_xbg_model(self.do_binary)
        self.df = None
        self.train = None
        self.test = None
        self.preds = []
        self.output_cols = ["date", "pred", "prob"]
        self.feature_importance_scores=[0]*len(self.features)

    def add_label(self):
        if self.do_binary:
            self.add_binary_label()
        else:
            self.add_regression_label(label_col="log-ret")

    def add_regression_label(self, label_col="close"):
        self.df["label"] = self.df[label_col].shift(-1)
        self.train = self.df.iloc[:-1]
        self.test = self.df.tail(1)
        self.log.info("Train={}, Test={}".format(self.train.shape, self.test.shape))

    def add_binary_label(self, label_col="log-ret"):
        self.log.info(
            " Adding binary label with threshold={} and direction={}".format(
                self.threshold, self.direction
            )
        )
        # given open, will it close higher
        # 100*(c_1- c_0)/c_0 -> 100*(c_1 - o_1) / o_1
        # shift, means is tomorrow closes higher than its open
        self.df['label']= 100.0*(self.df.close - self.df.open)/self.df.open
        self.df["label"] = self.df["label"].shift(-1)
        # self.df["label"] = self.df[label_col].shift(-1)
        if self.direction == "buy":
            self.df["label"] = self.df.label > self.threshold
        elif self.direction == "sell":
            self.df["label"] = self.df.label < -self.threshold
        else:
            assert False, "wrong direction was given"
        self.df["label"] = self.df["label"].astype("int")
        self.train = self.df.iloc[:-1]
        self.test = self.df.tail(1)
        self.log.info(
            "Train={}, 1's:{:0.2f}%, Test={}".format(
                self.train.shape,
                100.0*len(self.train[self.train.label == 1])/len(self.train),
                self.test.shape,
            )
        )

    def driver(self, data):
        self.set_data(data)
        self.add_label()
        self.scale_feautres()
        train, test, = (
            self.train[self.features],
            self.test[self.features],
        )
        if self.do_pca:
            train, test = self.pca_transform(train, test)
        labels = self.train.label

        self.fit(train, labels)
        self.predict(train, test, labels)
        self.feature_importance(train.columns.tolist())
