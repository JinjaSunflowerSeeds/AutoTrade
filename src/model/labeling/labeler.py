import os,sys
sys.path.append('./model/labeling')

import pandas as pd
import numpy as np


from rsi_based import RsiLabeling

class Labeler(RsiLabeling):
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


    def bull_flag(self, length=10, flagpole_length=5):
        """Identifies bull flag patterns in a given dataframe."""

        bull_flags = []
        i=1
        while i < self.df.shape[0]: #in range(1,len(self.df)):
            if self.df['close'].iloc[i] > self.df['open'].iloc[i] and self.df.iloc[i].volume >  self.df.iloc[:i].volume.mean() and self.df.iloc[i].volume > self.df.iloc[i-1].volume:
                if self.df['close'].iloc[i:i+5].is_monotonic_increasing and self.df.iloc[i].close > self.df.iloc[i].vwap:
                    print(self.df.iloc[i].volume, self.df.iloc[:i].volume.mean())
                    bull_flags.append(i) 
                    i+=3
            i+=1
                    
                    # Consolidation criteria: small range
        # for i in range(flagpole_length, len(self.df) - length):
        #     # Detect flagpole: a strong upward movement
        #     flagpole = self.df['close'].iloc[i-flagpole_length:i]
        #     if flagpole.is_monotonic_increasing:
        #         # Check for consolidation (flag)
        #         flag = self.df['close'].iloc[i:i+length]
        #         flag_high = flag.max()
        #         flag_low = flag.min()
                
        #         # Consolidation criteria: small range and near the top of the flagpole
        #         if (flag_high - flag_low) < 0.3 * (flagpole.max() - flagpole.min()) and \
        #         (flag.mean() > 0.7 * flagpole.max()):
        #             # Look for breakout above the consolidation range
        #             breakout = self.df['close'].iloc[i+length:i+length+3]
        #             if breakout.max() > flag_high:
        #                 bull_flags.append((i-flagpole_length, i+length))
        # Identify bull flag pattern
        self.df['label']=0
        self.df.loc[self.df.index.isin(bull_flags), 'label'] = 1
        self.df['buy']= self.df.label
        self.df['sell']=0
        print(
            f"Buy label=1: {100.0 * (self.df['label'] == 1).mean():.2f}% trades={self.df[self.df.label==1].shape[0]}"
        )



    def vwap_strategy(self):
        return "if 3 throughs to the line, the third one determines the direction (remember the hedge funds want to buy close to vwap)"
    def four_out_of_five(self, vol_t=1.0):
        # expect the first hit to be confirmation and as we go probability if increse or hold, its confirmation
        # TODO the volume threshold should be the typical volume for the stock and 1.2* of it .. NOT the day's average
        self.df["buy"] = 0
        self.df["sell"] = 0
        self.df["red"] = self.df.close < self.df.open
        profit_target = 0.001  # in %
        stop_loss = profit_target / 3
        t=0
        while t <= len(self.df) - 5:
            prev_vol = self.df.iloc[max(0,t-60*8):t].volume.mean()
            next_df = self.df.iloc[t + 1 : t + 6]
            if (
                self.df.iloc[t].volume > vol_t * prev_vol
                and next_df.volume.median() > vol_t * prev_vol
            ):
                if (
                    # next_df.red.sum() >= 4 # 4 reds out of 5
                    # and not self.df.iloc[t].red 
                    True
                    and next_df.close.min() < self.df.iloc[t].close* (1 - profit_target)#makes the profit 
                    and next_df.close.max() < self.df.iloc[t].close *(1+ stop_loss) #doesnt trigger stop loss
                    and self.df.iloc[t].rsi_14>=70 #rsi oversold
                ):
                    self.df.loc[t, "sell"] = 1
                    t+=5
                if (
                    # next_df.red.sum() <= 1
                    True
                    and next_df.close.max() > self.df.iloc[t].close * (1+ profit_target)
                    and next_df.close.min() > self.df.iloc[t].close * (1 - stop_loss)
                    and self.df.iloc[t].rsi_14<=30
                    # and self.df.iloc[t].red
                ):
                    self.df.loc[t, "buy"] = 1
                    t+=5
            t+=1
        self.df["label"] = self.df.sell
        # self.df["label"] = self.df.buy
        self.df.loc[self.df.sell+self.df.buy>=1, 'label']=1
        print(
            f"label=1: {100.0 * (self.df['label'] == 1).mean():.2f}% (buy={100.0 * (self.df['buy'] == 1).mean():.2f}% sell={100.0 * (self.df['sell'] == 1).mean():.2f}%) trades={self.df[self.df.label==1].shape[0]} profit={(self.df[self.df.label==1]).close.sum()-self.df[self.df.sell==1].close.sum():.2f}"
        )

    def short_label_profit_stop_loss(
        self, profit_target=0.003, stop_loss=0.0005, steps_ahead=30
    ):
        
        self.df["label"] = 0
        self.df["sell"] = 0
        self.df["buy"] = 0
        t = 0
        while t < len(self.df):
            purchase_price = self.df.loc[t, "close"]
            stop_loss_price = purchase_price * (1 + stop_loss)
            profit_target_price = purchase_price * (1 - profit_target)
            label_as_buy = False
            future_price = purchase_price
            for j in range(1, steps_ahead + 1):
                if t + j >= len(self.df) or self.df.iloc[t].rsi_14<50 :
                    break
                future_price = self.df.loc[t + j, "close"]
                if future_price > stop_loss_price:
                    break
                elif future_price < profit_target_price:
                    label_as_buy = True
                    # break
            if label_as_buy:
                self.df.loc[t, "label"] = 1
                self.df.loc[t, "sell"] = 1
                # self.df.loc[t + j, "buy"] = 1
                t += j
            t += 1
        # self.df.loc[self.df['label'] == 1, 'sell']=1

        print(
            f"Buy label=1: {100.0 * (self.df['label'] == 1).mean():.2f}% trades={self.df[self.df.label==1].shape[0]} profit={(self.df[self.df.label==1]).close.sum()-self.df[self.df.sell==1].close.sum():.2f}"
        )


    def long_label_profit_stop_loss(
        self, profit_target=0.005, stop_loss=0.001, steps_ahead=30
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
        self.df["buy"] = 0
        # self.short_label_profit_stop_loss()
        # return
        t = 0
        while t < len(self.df):
            purchase_price = self.df.loc[t, "close"]
            stop_loss_price = purchase_price * (1 - stop_loss)
            profit_target_price = purchase_price * (1 + profit_target)
            label_as_buy = False
            future_price = purchase_price
            for j in range(1, steps_ahead + 1):
                if t + j >= len(self.df) or self.df.iloc[t].rsi_14>=50 :
                    break
                future_price = self.df.loc[t + j, "close"]
                if future_price <= stop_loss_price:
                    break
                elif future_price >= profit_target_price:
                    label_as_buy = True
                    # break
            if label_as_buy:
                self.df.loc[t, "label"] = 1
                self.df.loc[t, "buy"] = 1
                # self.df.loc[t + j, "sell"] = 1
                t += j
            t += 1 

        print(
            f"Buy label=1: {100.0 * (self.df['label'] == 1).mean():.2f}% trades={self.df[self.df.label==1].shape[0]} profit={(self.df[self.df.label==1]).close.sum()-self.df[self.df.sell==1].close.sum():.2f}"
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
            "df={}, 1's:{:0.2f}%".format(
                self.df.shape,
                100.0 * len(self.df[self.df.label == 1]) / len(self.df),
            )
        )
