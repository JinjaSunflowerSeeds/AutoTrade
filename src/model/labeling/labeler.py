import os

import pandas as pd


class Labeler:
    def add_manual_labels(self, manual_labels_dir):
        self.log.info("Adding manual labels from {}".format(manual_labels_dir))
        self.df["label"] = 0
        self.df["date"] = pd.to_datetime(self.df.index)
        self.df.index.name = "index"

        for file in os.listdir(manual_labels_dir):
            self.log.info("Reading manual labels from {}".format(manual_labels_dir+'/'+ file))
            labeled_data = pd.read_csv(
                os.path.join(manual_labels_dir, file), index_col=0
            )
            labeled_data["date"] = pd.to_datetime(labeled_data.index)
            labeled_data.index.name = "index"
            labeled_data.reset_index(inplace=True)
            buys = labeled_data[labeled_data.label == 'Buy']["date"]
            sells = labeled_data[labeled_data.label == 'Sell']["date"]
            self.df.loc[self.df["date"].isin(buys), "label"] = 1
            self.df.loc[self.df["date"].isin(sells), "label"] = 1#2
        self.log.info(
            f"Buy label=1: {100.0 * (self.df['label'] == 1).mean():.2f}%, sell label 2s={100.0 * (self.df['label'] == 2).mean():.2f}% of rows"
        )

    def add_regression_label(self, label_col="close"):
        self.df["label"] = self.df[label_col].shift(-1)
        self.log.info("Regression label added")

    def add_binary_label(self, label_col="log-ret"):
        self.log.info(
            " Adding binary label with threshold={} and direction={}".format(
                self.threshold, self.direction
            )
        )
        # given open, will it close higher
        # 100*(c_1- c_0)/c_0 -> 100*(c_1 - o_1) / o_1
        # shift, means is tomorrow closes higher than its open
        self.df["label"] = 100.0 * (self.df.close - self.df.open) / self.df.open
        self.df["label"] = self.df["label"].shift(-1)
        # self.df["label"] = self.df[label_col].shift(-1)
        if self.direction == "buy":
            self.df["label"] = self.df.label > self.threshold
        elif self.direction == "sell":
            self.df["label"] = self.df.label < -self.threshold
        else:
            assert False, "wrong direction was given"
        self.df["label"] = self.df["label"].astype("int")
        self.log.info(
            "df={}, 1's:{:0.2f}%, Test={}".format(
                self.df.shape,
                100.0 * len(self.df[self.df.label == 1]) / len(self.df),
            )
        )
