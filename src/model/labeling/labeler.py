import os

import pandas as pd


class Labeler:
    def add_manual_labels(self, manual_labels_dir):
        self.log.info("Adding manual labels from {}".format(manual_labels_dir))
        self.df["label"] = 0
        self.df["date"] = pd.to_datetime(self.df.index)
        self.df.index.name = "index"

        for file in os.listdir(manual_labels_dir):
            self.log.info(
                "Reading manual labels from {}".format(manual_labels_dir + "/" + file)
            )
            labeled_data = pd.read_csv(
                os.path.join(manual_labels_dir, file), index_col=0
            )
            labeled_data["date"] = pd.to_datetime(labeled_data.index)
            labeled_data.index.name = "index"
            labeled_data.reset_index(inplace=True)
            buys = labeled_data[labeled_data.label == "Buy"]["date"]
            sells = labeled_data[labeled_data.label == "Sell"]["date"]
            self.df.loc[self.df["date"].isin(buys), "label"] = 1
            self.df.loc[self.df["date"].isin(sells), "label"] = 1  # 2
        self.log.info(
            f"Buy label=1: {100.0 * (self.df['label'] == 1).mean():.2f}%, sell label 2s={100.0 * (self.df['label'] == 2).mean():.2f}% of rows"
        )

    def short_label_profit_stop_loss(
        self, profit_target=0.002, stop_loss=0.0005, steps_ahead=30
    ):
        """
        Label each data point based on whether the price increases by 0.1% (profit_target) before hitting a
        0.05% stop loss (stop_loss) in the next 30 steps.

        :param profit_target: The target profit as a fraction (0.1% = 0.001)
        :param stop_loss: The stop loss as a fraction (0.05% = 0.0005)
        :param steps_ahead: Number of steps to look ahead in the future for profit/stop loss.
        """
        print(
            f"profit_target={profit_target}, stop_loss={stop_loss}, steps_ahead={steps_ahead} => reward/risk={profit_target/stop_loss}"
        )
        self.df["label"] = 0
        self.df["sell"] = 0
        sells_idx = []
        t = 0
        while t < len(self.df):
            purchase_price = self.df.loc[t, "close"]
            stop_loss_price = purchase_price * (1 + stop_loss)
            profit_target_price = purchase_price * (1 - profit_target)
            label_as_sell = False
            future_price = purchase_price
            for j in range(1, steps_ahead + 1):
                if t + j >= len(self.df):
                    break
                future_price = self.df.loc[t + j, "close"]
                if future_price > stop_loss_price:
                    break
                elif future_price <= profit_target_price:
                    label_as_sell = True
                    sells_idx.append(j)
                    break
            if label_as_sell:
                self.df.loc[t, "label"] = 1
                self.df.loc[t + sells_idx[-1], "sells"] = 1
                t += sells_idx[-1]
            t += 1

        print(
            f"Buy label=1: {100.0 * (self.df['label'] == 1).mean():.2f}% trades={self.df[self.df.label==1].shape[0]} profit={(self.df[self.df.label==1]).close.sum()-self.df[self.df.sells==1].close.sum():.2f}"
        )

    def long_label_profit_stop_loss(
        self, profit_target=0.002, stop_loss=0.0005, steps_ahead=30
    ):
        """
        Label each data point based on whether the price increases by 0.1% (profit_target) before hitting a
        0.05% stop loss (stop_loss) in the next 30 steps.

        :param profit_target: The target profit as a fraction (0.1% = 0.001)
        :param stop_loss: The stop loss as a fraction (0.05% = 0.0005)
        :param steps_ahead: Number of steps to look ahead in the future for profit/stop loss.
        """
        print(
            f"profit_target={profit_target}, stop_loss={stop_loss}, steps_ahead={steps_ahead} => reward/risk={profit_target/stop_loss}"
        )
        self.df["label"] = 0
        self.df["sells"] = 0
        t = 0
        while t < len(self.df):
            purchase_price = self.df.loc[t, "close"]
            stop_loss_price = purchase_price * (1 - stop_loss)
            profit_target_price = purchase_price * (1 + profit_target)
            label_as_buy = False
            future_price = purchase_price
            for j in range(1, steps_ahead + 1):
                if t + j >= len(self.df):
                    break
                future_price = self.df.loc[t + j, "close"]
                if future_price <= stop_loss_price:
                    break
                elif future_price >= profit_target_price:
                    label_as_buy = True
                    break
            if label_as_buy:
                self.df.loc[t, "label"] = 1
                self.df.loc[t + j, "sells"] = 1
                t += j
            t += 1

        print(
            f"Buy label=1: {100.0 * (self.df['label'] == 1).mean():.2f}% trades={self.df[self.df.label==1].shape[0]} profit={(self.df[self.df.label==1]).close.sum()-self.df[self.df.sells==1].close.sum():.2f}"
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
