from datetime import datetime
from math import log

import numpy as np
import pandas as pd
import pytz


class BacktestPlatform:
    def __init__(self, data):
        # Initial budget and parameters
        self.total_budget = 10000  # Total budget
        self.allocation_per_trade = 1  # 10% of total budget
        self.trade_allocations = []
        self.completed_trades = []
        self.data = data

    def record_trade(
        self,
        entr_idx,
        exit_idx,
        buy_price,
        curr_price,
        pct_change,
        type,
        available_budget,
        profit,
    ):
        self.completed_trades.append(
            {
                "entry_index": self.data.iloc[entr_idx].date,
                "exit_index": self.data.iloc[exit_idx].date,
                "entry_price": buy_price,
                "exit_price": curr_price,
                "pct_change": f"{100*pct_change:0.2f}",
                "type": type,
                "profit_loss": profit,
                "remaining_budget": available_budget,
            }
        )

    def simple_3wall_backtest(self, profit_target=0.5, stop_loss=0.25, n_periods=3):
        # hittting the target, stop or the wall(max bars or end of day)
        available_budget = self.total_budget
        self.completed_trades = []
        self.data["date"] = pd.to_datetime(self.data["date"])
        print(f"#signals={self.data[self.data.signal != 0].shape}")
        for index, row in self.data.iterrows():
            if row["signal"] == 0:
                continue
            # # waiting for confirmation, which is when the signal stops firing
            if self.data.iloc[index+1].signal != 0:
                continue
            else:
                index +=1
            buy_price = self.data.iloc[index].close # row["close"]
            max_allocation = available_budget * self.allocation_per_trade

            for i in range(1, n_periods + 1):
                if index + i >= len(self.data):
                    break
                curr_price = self.data.loc[index + i, "close"]
                delta = row["signal"] * (curr_price - buy_price)
                pct_change = delta / buy_price
                profit = max_allocation * pct_change
                t_type = None
                if (
                    i == n_periods
                    or self.data.iloc[index].date.day
                    < self.data.iloc[index + i + 1].date.day
                ):  # no hold overnight
                    t_type = "wall"
                elif 100 * pct_change >= profit_target:
                    t_type = "profit"
                elif 100 * pct_change <= -stop_loss:
                    t_type = "loss"
                if t_type is not None:
                    available_budget += profit
                    self.record_trade(
                        index,
                        index + i,
                        buy_price,
                        curr_price,
                        pct_change,
                        t_type,
                        available_budget,
                        profit,
                    )
                    break
        b_df = pd.DataFrame(self.completed_trades)
        print(b_df)
        print("_" * 100)
        print(f"profit_target= {profit_target},stop_loss= {stop_loss},n_periods= {n_periods}")
        print(
            f"#trade={b_df.shape[0]}, profit={b_df.profit_loss.sum():.2f}, win_rate={np.sum(b_df.profit_loss>0)/b_df.shape[0]:.2f}, max_win={b_df.profit_loss.max():.2f}, max_loss={b_df.profit_loss.min():.2f}"
        )
        print("_" * 100)
        print(self.data.date.min(), self.data.date.max())


class TradingPlatform:
    def __init__(self, data):
        self.data = data
        self.strategies = []
        self.results = {}
        print(self.data.shape)

    def add_strategy(self, name, conditions, actions):
        self.strategies.append(
            {"name": name, "conditions": conditions, "actions": actions}
        )

    def evaluate_strategies(self):
        singals = [0] * len(self.data)
        for strategy in self.strategies:
            self.data[strategy["name"]] = 0
            for index in range(len(self.data)):
                if all(
                    cond(self.data.iloc[index], self.data)
                    for cond in strategy["conditions"]
                ):
                    for action in strategy["actions"]:
                        singals[index] = action(self.data, index)
        self.data["signal"] = singals


class RSIDataSetup:
    def __init__(
        self, data_file, ema_period=14, rsi_period=14, rsi_overbought_threshold=80
    ) -> None:
        self.rsi_period = rsi_period
        self.data = pd.read_csv(data_file)
        print(f"converting from utc to east shape: {self.data.shape}")
        self.data["date"] =  pd.to_datetime(self.data['date']).dt.tz_localize(pytz.timezone('UTC')).dt.tz_convert(pytz.timezone('US/Eastern'))
        self.data.sort_values(by="date", inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.data["ema"] = self.data["close"].ewm(span=ema_period, adjust=False).mean()
        # self.data['rsi'] = self.calculate_rsi()
        self.data["overbought_rsi_con"] = self.data["rsi_14"] > rsi_overbought_threshold


def get_day(row):
    return str(row["date"].date())
    return str(datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d"))


# Define conditions and actions
def condition_high_of_day(row, data):
    df_ = data[(data.date >= get_day(row)) & (data.date <= row.date)]
    return len(df_) > 0 and  df_.tail(3).close.max() >= df_.close.max()
    df_ = data[(data.date >= get_day(row)) & (data.date < row.date)]
    # df_ = data[(data.date >= get_day(row)) & (data.date < row.date)]
    # return len(df_) > 0 and row["close"] > df_.close.max()


def intra_day(row, data):  # trade before end of day
    date_obj = row["date"] #datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
    end_of_day =data[ data["date"].dt.date == date_obj.date()]["date"].max()
      
    return date_obj < end_of_day


def price_below_ema(row, data):
    return row["close"] < row["ema"]


def condition_rsi_overbought(row, data):
    return row["overbought_rsi_con"]  # > threshold

def rsi_cross_below(row, data):
    df_ = data[(data.date >= get_day(row)) & (data.date < row.date)]
    return df_.shape[0]>0 and not row["overbought_rsi_con"] and df_.tail(1).overbought_rsi_con.iloc[0]

def condition_volume_above_avg(row, data):
    df_ = data[(data.date >= get_day(row)) & (data.date < row.date)]
    return row["volume"] > df_.volume.mean()


def action_short(data, index):
    return -1


if __name__ == "__main__":
    """
        this is for intraday only
    """
    data_file = "/Users/ragheb/myprojects/stock/src/files/training/fe_1d.csv"
    data = RSIDataSetup(
        data_file=data_file, ema_period=3, rsi_period=14, rsi_overbought_threshold=70
    ).data

    # Initialize platform
    platform = TradingPlatform(data)
    platform.add_strategy(
        name="High of Day Short",
        conditions=[
            condition_high_of_day,
            condition_rsi_overbought,
            # rsi_cross_below,
            # condition_volume_above_avg,
            intra_day,
            #  price_below_ema
        ],
        actions=[action_short],
    )

    # Evaluate and backtest
    platform.evaluate_strategies()
    print(
        platform.data[platform.data.signal != 0][["date", "signal", "close", "rsi_14"]]
    )

    backtest = BacktestPlatform(platform.data)
    for profit_target, stop_loss, n_periods in [
        # [0.1, 50, 1],
        # [0.1, 50, 2],
        [0.1, 50, 2],
        [0.1, 50, 3],
        [0.2, 50, 3],
        [0.5, 50, 5],
        # [2, 1.5, 5],
        # [0.5, 2, 10],
        # [1, 0.5, 20],
        # [5, 5, 60],
    ]:
        backtest.simple_3wall_backtest(profit_target, stop_loss, n_periods)


#  filters based on overall market trend (stock is more likely to move in the direction of the market for example if the day or n periods prior is bulleish)
# enter when confirmation is there
# price below ema on 2min chart
#  3taps alwasy fllow through
#  distribution of log returns by time of day
#  double top or 3 bar reeverse .. 
# shorter period no stop loss
#  after a losing day, is it a wining day
# if stock ralies, at the end of day goes down as shorts covering
#  search and learn about the 80% value area 
#  if after earning is pop, will it contunue for the next few month
#  after dropping or poping x%, it will revert always
#  after x red, if starts greem, it will end green
    #  probability of candel starting red and finishing red->48%
    # opening green and finishing green is 53%
    # if it opens red, 97% chance the low will be on average 24 cent lower than open
#  30 minute opening range breakout
