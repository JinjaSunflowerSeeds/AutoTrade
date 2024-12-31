
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
# import machine learning libraries
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error as MSE,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from strategy.trading_strategy import (
    open_close_strategy_gain
)


class XgbUtils:
    def __init__(self):
        self.hyper_space={
            'max_depth': hp.quniform("max_depth", 3, 18, 1),
            'gamma': hp.uniform ('gamma', 1,9),
            'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
            'reg_lambda' : hp.uniform('reg_lambda', 0,1),
            'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
            'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
            'n_estimators': [180, 250, 500, 1000, 10000,],
            'seed': 0
            }

    def get_xbg_model(self, do_binary):
        if do_binary:
            return XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,       # Smaller learning rate for better convergence
    subsample=0.7,            # Subsample to reduce overfitting
    colsample_bytree=0.7,     # Focus on relevant features
    scale_pos_weight=90/10.0,  # Handle imbalance
    reg_lambda=3,             # L2 regularization
    reg_alpha=1,              # L1 regularization
    random_state=1989,
    n_jobs=-1,
    eval_metric="auc"         # Use AUC-ROC as the evaluation metric
)
            return XGBClassifier(
                # colsample_bytree= 1.0, eta= 0.01, max_depth= 5,
                # n_estimators= 500,  subsample= 0.8,
                #  reg_lambda= 3,

                n_estimators=500,
                max_depth=5,
                # eta=0.1,
                # subsample=0.7,
                colsample_bytree=0.01,

                # colsample_bynode=0.28,

                # reg_alpha=1,#l1
                # reg_lambda=50,  # l2


                n_jobs=-1,
                random_state=1989,
            )
        else:
            return XGBRegressor(
                n_estimators=1000,
                max_depth=7,
                eta=0.1,
                subsample=0.7,
                colsample_bytree=0.8,
                colsample_bynode=0.28,
                reg_lambda=3,  # l2
                n_jobs=-1,
                random_state=1989,
            )

    def make_date_future_n_add_label_for_export(self, df_=None):
        # label is already from future
        p = deepcopy(self.df[["date", "label"]].reset_index(drop=True, inplace=False))
        p["label"] = p.label.shift(1)
        df_ = df_.merge(p, on="date")
        # df_['date'] = pd.to_datetime(df_['date'])
        # self.df['date'] = pd.to_datetime(self.df['date'])
        # shift the dates one day forward to make it the day forcast is made for ( pd.offsets.BusinessDay(n=1) is not accurate as it does not consider market holidays)
        # changing input (date prediction was made, pred) ->to-> (date pred was made for, pred)
        # get last record, make its date next one, for other ones shift the prediction up, then join back
        l = df_.tail(1)
        t = pd.DataFrame(
            [
                (
                    l.date.values[0] + pd.offsets.BusinessDay(n=1),
                    l.pred.values[0],
                    l.prob.values[0],
                )
            ],
            columns=self.output_cols,
        )
        df_[["pred", "prob"]] = df_[["pred", "prob"]].shift(1)
        df_ = pd.concat([df_.iloc[1:], t], axis=0)
        #add close and open
        df_ =df_.merge(self.df[['date', 'open', 'close', 'low', 'high']], on='date', how='left')
        return df_

    def export_data(self, filepath_prefix="./files/xgb/"):
        pred_df = self.make_date_future_n_add_label_for_export(self.get_predictions())
        self.metrics(pred_df.iloc[:-1])  # last one is NaN
        f = filepath_prefix + "{}_binary_output.csv".format(self.direction)
        if not self.do_binary:
            f= filepath_prefix + "regression_output.csv"
        self.log.info(" Exporting to " + f)
        pred_df.to_csv(f)
        print(pred_df)

    def print_overall_important_features(self, top_cnt=-1):
        for i in sorted(zip(self.feature_importance_scores, self.features), reverse=True)[
            :top_cnt
        ]:
            self.log.info(i)

    def feature_importance(self, features, top_cnt=10):
        for i in range(len(self.model.feature_importances_)):
            self.feature_importance_scores[i] += self.model.feature_importances_[i]
        for i in sorted(zip(self.model.feature_importances_, features), reverse=True)[
            :top_cnt
        ]:
            self.log.info(i)

    def scale_feautres(self):
        self.log.info("  Scaling data...")
        # self.log.info("   ->features:{}".format(self.features))
        # mm = preprocessing.StandardScaler().fit(self.train[self.features])
        print(self.train[self.features][self.train[self.features].isnull().any(axis=1)])
        mm = preprocessing.MinMaxScaler().fit(self.train[self.features])
        self.train[self.features] = mm.transform(self.train[self.features])
        self.test[self.features] = mm.transform(self.test[self.features])

    def metrics(self, pred_df):
        try:
            if False :
                self.log.warning("   ->accuracy={:.3f}".format(100 * accuracy_score(pred_df.label, pred_df.pred>0.5)))
                self.log.warning(
                    "   ->precision_50={:.3f}, #trades={:.0f}".format(100 * precision_score(pred_df.label, pred_df.pred>0.5, average='weighted'), pred_df.pred.sum())
                )
                self.log.warning(
                    "   ->precision_75={:.3f}, #trades={:.0f}".format(100 * precision_score(pred_df.label, [i>0.75 for i in pred_df.prob], average='weighted'),
                                                    np.sum([i>0.75 for i in pred_df.prob]))
                )
                self.log.warning("   ->auc={:.3f}".format(100 * roc_auc_score(pred_df.label, pred_df.prob, multi_class='ovr')))
                self.log.warning(
                    "   ->f1={:.3f}".format(
                    100 * f1_score(pred_df.label, pred_df.pred, average="weighted")
                    )
                )
                self.log.warning("   ->recall={:.3f}".format(100 * recall_score(pred_df.label, pred_df.pred)))
                self.log.warning("   ->$gain={}".format(open_close_strategy_gain(pred_df, self.direction)))
            elif self.do_binary:
                self.log.warning("   ->accuracy={:.3f}".format(100 * accuracy_score(pred_df.label, pred_df.pred>0.5)))
                self.log.warning(
                    "   ->precision_50={:.3f}, #trades={:.0f}".format(100 * precision_score(pred_df.label, pred_df.pred>0.5, average='weighted'), pred_df.pred.sum())
                )
                self.log.warning(
                    "   ->precision_75={:.3f}, #trades={:.0f}".format(100 * precision_score(pred_df.label, [i>0.75 for i in pred_df.prob]),
                                                    np.sum([i>0.75 for i in pred_df.prob]))
                )
                self.log.warning("   ->auc={:.3f}".format(100 * roc_auc_score(pred_df.label, pred_df.prob)))
                self.log.warning(
                    "   ->f1={:.3f}".format(
                    100 * f1_score(pred_df.label, pred_df.pred, average="weighted")
                    )
                )
                self.log.warning("   ->recall={:.3f}".format(100 * recall_score(pred_df.label, pred_df.pred)))
                self.log.warning("   ->$gain={}".format(open_close_strategy_gain(pred_df, self.direction)))
            else:
                self.log.warning("   ->rmse={}".format(np.sqrt(MSE(pred_df.label, pred_df.pred))))
                self.log.warning("   ->r2={}".format(r2_score(pred_df.label, pred_df.pred, multioutput="variance_weighted")))
                self.log.warning("   ->corr={}".format(np.corrcoef(pred_df.label, pred_df.pred)[0][1]))
        except Exception as e:
            self.log.warning("   ->Error:{}".format(e))

    def predict(self, train, test, train_label):
        self.log.info("  Predictng...")
        train['pred'] = self.model.predict(train)
        self.log.info(test)
        yhat = self.model.predict(test)
        test_prob, train_prob = 0, None
        if self.do_binary:
            test_prob = self.model.predict_proba(test)[0][1]
            train['prob'] = [i[1] for i in self.model.predict_proba(train[self.features])]

        self.log.info("  ->Train")
        train['label']=train_label
        self.metrics(train)
        assert len(self.test) == 1
        self.preds.append([self.test.date.values[0], yhat[0], test_prob])

    def set_data(self, data):
        self.log.info(
            " Unused features:{}".format(
            [i for i in data.columns.tolist() if i not in self.features])
        )
        self.df = data

    def get_predictions(self):
        return pd.DataFrame(self.preds, columns=self.output_cols)

    def pca_transform(self, train, test):
        pca = PCA(n_components=self.pca_n_components, svd_solver="full")
        pca.fit(train.fillna(0))
        self.log.info(
            "pca.explained_variance_ratio_",
            len(pca.explained_variance_ratio_),
            pca.explained_variance_ratio_,
        )
        train = pd.DataFrame(
            pca.transform(train.fillna(0)),
            columns=[
                "pca_" + str(i) for i in range(len(pca.explained_variance_ratio_))
            ],
        )
        test = pd.DataFrame(
            pca.transform(test.fillna(0)),
            columns=[
                "pca_" + str(i) for i in range(len(pca.explained_variance_ratio_))
            ],
        )
        return train, test


    def hyer_tune(self, train_df, labels):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(train_df, labels,
                                                            test_size=0.2,
                                                            random_state=42)

        # Define the XGBoost model
        xgb_model = XGBClassifier()

        # Define the hyperparameters and their possible values for tuning
        param_grid  = {
                        # 'learning_rate': [0.01, 0.1, 0.2],
                        'n_estimators': [ 100, 500, 1000],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.3, 0.8, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        # 'min_child_weight': [1, 3, 5],
                        # 'gamma': [0, 0.1, 0.2, 0.3],
                        'eta': [0.01, 0.1,0.5],
                        'reg_alpha': [0, 1, 2 ,3],
                        'reg_lambda':[0, 1, 2 ,3]
                        }

        # Perform Grid Search Cross-Validation
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                                   scoring='precision', cv=3, n_jobs=-1, verbose=3)
        grid_search.fit(X_train, y_train)

        # self.log.info the best hyperparameters
        self.log.info("Best Hyperparameters:", grid_search.best_params_)

        # Evaluate the model on the test set
        accuracy = grid_search.score(X_test, y_test)
        self.log.info("Accuracy on Test Set:", accuracy)
        exit(1)

    def fit(self, df, labels):
        self.log.info("  Training...")
        # do hyper by using 80% for traiing and 20% for testing, the best will be based on the latter
        # XgbUtils().hyer_tune(df, labels)
        # print(df.columns.tolist())
        self.log.warning(f"nulls before train {df.shape}=> {df[df.isnull().any(axis=1)]}")
        x=df.columns[df.isnull().any()]
        if len(x)>0:
            print(x)
            # assert False and "nan columns" 
        self.model.fit(df, labels)
