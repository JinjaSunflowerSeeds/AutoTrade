from copy import deepcopy

import numpy as np
import pandas as pd
import shap

# import machine learning libraries
import xgboost as xgb
from hyperopt import fmin, hp, STATUS_OK, tpe, Trials
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error as MSE,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from strategy.trading_strategy import open_close_strategy_gain
from xgboost import XGBClassifier


class ModelSettings:
    def __init__(self):
        self.hyper_space = {
            "max_depth": hp.quniform("max_depth", 3, 18, 1),
            "gamma": hp.uniform("gamma", 1, 9),
            "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
            "n_estimators": [
                180,
                250,
                500,
                1000,
                10000,
            ],
            "seed": 0,
        }

    def get_xbg_model(self, do_binary, df):
        if do_binary:
            # return XGBClassifier(
            #     learning_rate=0.007,
            #     n_estimators=100,
            #     max_depth=3,
            #     min_child_weight=5,
            #     gamma=0.4,
            #     subsample=0.55,
            #     colsample_bytree=0.85,
            #     reg_alpha=0.005,
            #     objective="binary:logistic",
            #     nthread=4,
            #     scale_pos_weight=1,
            #     seed=27,
            #     eval_metric="aucpr",
            # )
            return XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,  # Smaller learning rate for better convergence
                subsample=0.7,  # Subsample to reduce overfitting
                colsample_bytree=0.7,  # Focus on relevant features
                # scale_pos_weight=1.0*df[df.label==0].shape[0] / df[df.label==1].shape[0],  # Handle imbalance
                reg_lambda=3,  # L2 regularization
                reg_alpha=11,  # L1 regularization
                random_state=1989,
                n_jobs=-1,
                eval_metric="aucpr",
                objective="binary:logistic",
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
                eval_metric="aucpr",
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


class Evaluation:
    def __init__(self):
        pass

    def binary_eval(self, pred_df):
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        # Calculate accuracy, precision, and number of trades for each threshold
        # accuracy_vals = [np.round(100 * accuracy_score(pred_df.label, pred_df.prob > t), 2) for t in thresholds]
        precision_vals = [np.round(100 * precision_score(pred_df[pred_df.prob>t].label, pred_df[ pred_df.prob>t].prob > t, average="weighted"), 2) for t in thresholds]
        n_trades = [np.sum(pred_df.prob > t) for t in thresholds]
        # Calculate overall metrics
        # roc_auc = 100 * roc_auc_score(pred_df.label, pred_df.prob)
        precision, recall, _ = precision_recall_curve(pred_df.label, pred_df.prob)
        pr_auc = 100 * auc(recall, precision)
        # f1 = 100 * f1_score(pred_df.label, pred_df.pred, average="weighted")
        recall_val = 100 * recall_score(pred_df.label, pred_df.pred)
        # Log the results
        self.log.warning(f"   -> Thresholds: {thresholds}")
        # self.log.warning(f"   -> Accuracy: {accuracy_vals}")
        self.log.warning(f"   -> (Precision, #Trades): {[i for i in zip(precision_vals,n_trades)]}")
        # self.log.warning(f"   -> ROC AUC: {roc_auc:.3f}")
        self.log.warning(f"   -> PR AUC: {pr_auc:.3f}")
        # self.log.warning(f"   -> F1 Score: {f1:.3f}")
        self.log.warning(f"   -> Recall: {recall_val:.3f}")
        # self.log.warning(f"   -> $ Gain: {open_close_strategy_gain(pred_df, self.direction)}")
        # plt.plot(precision, recall, label="PR Curve")
        pred_df.to_csv("./files/xgb/tmp_pred.csv")


    def regression_eval(self, pred_df):
        self.log.warning(
            "   ->rmse={}".format(np.sqrt(MSE(pred_df.label, pred_df.pred)))
        )
        self.log.warning(
            "   ->r2={}".format(
                r2_score(pred_df.label, pred_df.pred, multioutput="variance_weighted")
            )
        )
        self.log.warning(
            "   ->corr={}".format(np.corrcoef(pred_df.label, pred_df.pred)[0][1])
        )

    def metrics(self, pred_df):
        try:
            if self.do_binary:
                self.binary_eval(pred_df)
            else:
                self.regression_eval(pred_df)
        except Exception as e:
            self.log.warning("   ->Error:{}".format(e))


class XgbUtils(ModelSettings, Evaluation):
    def __init__(self):
        ModelSettings.__init__(self)
        Evaluation.__init__(self)

    def make_date_future_n_add_label_for_export(self, df_=pd.DataFrame()):
        df_ = df_.merge(
            self.df[["date", "label", "open", "close", "low", "high"]],
            on="date",
            how="left",
        )
        return df_

    def export_data(self, filepath_prefix="./files/xgb/"):
        pred_df = self.make_date_future_n_add_label_for_export(self.get_predictions())
        self.metrics(pred_df.iloc[:-1])  # last one is NaN
        f = filepath_prefix + "{}_binary_output.csv".format(self.direction)
        if not self.do_binary:
            f = filepath_prefix + "regression_output.csv"
        self.log.info(" Exporting to " + f)
        pred_df.to_csv(f)
        print(pred_df)
        print(pred_df[["prob", "close"]].corr().prob)

    def print_overall_important_features(self, top_cnt=-1):
        for i in sorted(
            zip(self.feature_importance_scores, self.features), reverse=True
        )[:top_cnt]:
            self.log.info(i)

    def feature_importance(self, features, top_cnt=10):
        for i in range(len(self.model.feature_importances_)):
            self.feature_importance_scores[i] += self.model.feature_importances_[i]
        for i in sorted(zip(self.model.feature_importances_, features), reverse=True)[
            :top_cnt
        ]:
            self.log.info(i)
        #
        # test
        # X_test_sample = self.train[self.features]  # Use the test set for SHAP value calculation
        # explainer = shap.Explainer(self.model, X_test_sample)
        # shap_values = explainer(X_test_sample)
        # shap_importance = np.abs(shap_values.values).mean(axis=0)
        # feature_names = X_test_sample.columns# Create a DataFrame for SHAP importance
        # shap_importance_df = pd.DataFrame({
        #                                     'Feature': feature_names,
        #                                     'Importance': shap_importance
        #                                 })
        # top_10_features = shap_importance_df.sort_values(by='Importance', ascending=False).head(top_cnt)
        # self.log.info("Train Shap importance:")
        # print(top_10_features) #
        # test
        X_test_sample = self.test[
            self.features
        ]  # Use the test set for SHAP value calculation
        explainer = shap.Explainer(self.model, X_test_sample)
        shap_values = explainer(X_test_sample)
        shap_importance = np.abs(shap_values.values).mean(axis=0)
        feature_names = X_test_sample.columns  # Create a DataFrame for SHAP importance
        shap_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": shap_importance}
        )
        top_10_features = shap_importance_df.sort_values(
            by="Importance", ascending=False
        ).head(10)
        self.log.info("Test Shap importance:")
        print(top_10_features)  #

    def scale_feautres(self):
        self.log.info("  Scaling data...")
        mm = preprocessing.MinMaxScaler().fit(self.train[self.features])
        self.train[self.features] = mm.transform(self.train[self.features])
        self.test[self.features] = mm.transform(self.test[self.features])

    def predict(self, train, test, train_label):
        self.log.info("  Predictng...")
        train["pred"] = self.model.predict(train[self.features])
        test["pred"] = self.model.predict(test[self.features])
        # self.log.info(test)
        yhat = self.model.predict(test[self.features])
        test_prob, train_prob = 0, None
        if self.do_binary:
            test["prob"] = [
                i[1] for i in self.model.predict_proba(test[self.features])
            ]  # self.model.predict_proba(test[self.features])[0][1]
            train["prob"] = [
                i[1] for i in self.model.predict_proba(train[self.features])
            ]

        self.log.info("  ->Train")
        train["label"] = self.train_labels
        self.log.info(
            f"   ->>prob={train['prob'].mean()}, 1={train[train.label==1]['prob'].mean()}, 0={train[train.label==0]['prob'].mean()}"
        )
        self.metrics(train)
        # assert len(self.test) == 1
        self.log.info("  ->Test")
        test["label"] = self.test_labels
        self.metrics(test)
        self.log.info(
            f"   ->>prob={test['prob'].mean()}, 1={test[test.label==1]['prob'].mean()}, 0={test[test.label==0]['prob'].mean()}"
        )
        self.preds.append([self.test.date.values[0], yhat[0], test.prob.values[0]])

    def set_features(self):
        self.features= [i for i in set(self.get_features() +self.get_lag_features())]
        self.feature_importance_scores = [0] * len(self.features)
        
    def set_data(self, data):
        
        self.log.info(
            " Unused features:{}".format(
                len([i for i in data.columns.tolist() if i not in self.features])
            )
        )
        self.df = data


    def get_predictions(self):
        return pd.DataFrame(self.preds, columns=self.output_cols)

    def pca_transform(self, train, test):
        pca = PCA(n_components=self.pca_n_components, svd_solver="full")
        pca.fit(train[self.features].fillna(0))
        self.log.info(
            "pca.explained_variance_ratio_",
            len(pca.explained_variance_ratio_),
            pca.explained_variance_ratio_,
        )
        train = pd.DataFrame(
            pca.transform(train[self.features].fillna(0)),
            columns=[
                "pca_" + str(i) for i in range(len(pca.explained_variance_ratio_))
            ],
        )
        test = pd.DataFrame(
            pca.transform(test[self.features].fillna(0)),
            columns=[
                "pca_" + str(i) for i in range(len(pca.explained_variance_ratio_))
            ],
        )
        return train, test

    def hyer_tune(self, train_df, labels):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            train_df, labels, test_size=0.2, random_state=42
        )

        # Define the XGBoost model
        xgb_model = XGBClassifier()

        # Define the hyperparameters and their possible values for tuning
        param_grid = {
            # 'learning_rate': [0.01, 0.1, 0.2],
            "n_estimators": [100, 500, 1000],
            "max_depth": [3, 5, 7],
            "subsample": [0.3, 0.8, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            # 'min_child_weight': [1, 3, 5],
            # 'gamma': [0, 0.1, 0.2, 0.3],
            "eta": [0.01, 0.1, 0.5],
            "reg_alpha": [0, 1, 2, 3],
            "reg_lambda": [0, 1, 2, 3],
        }

        # Perform Grid Search Cross-Validation
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring="precision",
            cv=3,
            n_jobs=-1,
            verbose=3,
        )
        grid_search.fit(X_train, y_train)

        # self.log.info the best hyperparameters
        self.log.info("Best Hyperparameters:", grid_search.best_params_)

        # Evaluate the model on the test set
        accuracy = grid_search.score(X_test, y_test)
        self.log.info("Accuracy on Test Set:", accuracy)
        exit(1)

    def fit(self, df, labels):
        self.log.info("  Training...")
        df = df[self.features]
        # do hyper by using 80% for traiing and 20% for testing, the best will be based on the latter
        # XgbUtils().hyer_tune(df, labels)
        # print(df.columns.tolist())
        
        if len(df.columns[df.isnull().any()]) > 0:
            self.log.error(f"null columns of df={df.columns[df.isnull().any()]}")
            self.log.warning(f"nulls before train {df.shape}=> null shape={df[df.isnull().any(axis=1)].shape}")
            # df.fillna(0, inplace=True)
            # assert False and "nan columns"
        self.model.fit(df, labels, eval_set=[(df, labels) ], verbose=False)
        evals_result = self.model.evals_result()
        training_loss = evals_result['validation_0']  # Training loss over epochs
        self.log.info(f" Training loss(prauc) {training_loss['aucpr'][-1]}")

         

    def chart_preds(self):
        self.log.info("  Charting...")
        pred_df = self.make_date_future_n_add_label_for_export(self.get_predictions())
        plt.plot(
            pred_df["date"],
            (pred_df["close"] - pred_df.close.min())
            / (pred_df.close.max() - pred_df.close.min()),
            label="label",
        )
        plt.plot(
            pred_df["date"],
            (pred_df["prob"] - pred_df.prob.min())
            / (pred_df.prob.max() - pred_df.prob.min()),
            label="prob",
        )
        plt.grid()
        plt.legend()
        plt.show()
