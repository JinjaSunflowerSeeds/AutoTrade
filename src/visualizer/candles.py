import os
import sys
from logging import Logger

sys.path.append("./")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseEvent
from model.labeling.labeler import Labeler
from utils.config_reader import get_manual_output_path, get_merger_final_output_file

"""
cmd: 
    python3 visualizer/candles.py
    
Each buy/sell must match a sell/buy action. See this func on how buy/sell are distinguished and labeled (default is buy if odd number of clicks have happened)
    -> see set_labels()
"""


class InteractiveCandlestickChartBase:
    def get_pnl(self, df):
        profit = df[df.label == "Buy"].close.sum() - df[df.label == "Sell"].close.sum()
        ppt = [
            f"{(i[0]-i[1])[0]:.2f}"
            for i in zip(
                df[df.label == "Buy"].close.values, df[df.label == "Sell"].close.values
            )
        ]
        return profit[0], ppt

    def save_labels_to_file(self):
        self.set_labels()
        df_ = pd.DataFrame(self.labels)
        df_.to_csv(self.outputfile, index=False)
        profit, ppt = self.get_pnl(df_)
        print(
            f"Labels saved to  {self.outputfile}. profit={profit:.2f}, pre trade={ppt})"
        )

    def get_df_info(self, x_coord, x_dates):
        step = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) / len(x_dates)
        x = x_coord - self.ax.get_xlim()[0]
        idx = int(np.ceil(x / step) - 1)
        clicked_date = x_dates[idx]
        price = self.df[self.df.index == clicked_date][
            ["open", "high", "low", "close"]
        ].close.values
        print(f"Clicked on: {clicked_date} (Idx: {idx}), (price is {price})")
        return clicked_date, idx, price

    def on_click(self, event: MouseEvent):
        if event.inaxes != self.ax:
            print("Clicked outside of the candlestick chart.")
            return
        # Convert the click's x-coordinate to the closest index
        x_coord = event.xdata
        if x_coord is None:
            print("Invalid click detected.")
            return
        # Find the closest date in the data
        x_dates = list(self.df.index)
        x_index = min(range(len(x_dates)), key=lambda i: abs(i - x_coord))
        if x_index < 0 or x_index >= len(self.df):
            print("Clicked out of range.")
            return
        date, idx, price = self.get_df_info(x_coord, x_dates)
        self.labels.append({"date": date, "label": self.label, "close": price})
        self.save_labels_to_file()
        #
        line = self.ax.axvline(x=event.xdata, color="black", linestyle="--")
        plt.draw()

    def show(self):
        plt.tight_layout()
        plt.show()

    def scale_rsi(self, rsi_value, col='volume'):
        rsi_min, rsi_max = self.df.rsi_14.min(), self.df.rsi_14.max()
        volume_min, volume_max = self.df[col].min(), self.df[col].max()
        return volume_min + (rsi_value - rsi_min) / (rsi_max - rsi_min) * (
                volume_max - volume_min
            )
    def plot_candlestick(self):
        mpf.plot(
            self.df,
            type="candle",
            ax=self.ax,
            volume=self.ax_vol,
            tight_layout=True,
            addplot=self.add_plot(),
            style='yahoo',
             show_nontrading=False
            # xaxis_date=False
        )
        self.ax.set_title(f"{self.df.index.min()}  <to>  {self.df.index.max()}")
        
    def get_predictions(self,model_prediction_file):
        preds= pd.read_csv(model_prediction_file)[['date', 'prob', 'label']]
        preds['date'] = pd.to_datetime(preds.date)
        # preds.set_index("date", inplace=True)
        print("pred df=", preds.shape)
        self.df = self.df.merge(
            preds[["date", "prob", "label"]], how="left", on="date"
        )
        print("merged df=", self.df.shape)
        


class InteractiveCandlestickChart(InteractiveCandlestickChartBase, Labeler):
    def __init__(
        self,
        filepath,
        outputfile,
        fe_file,
        model_prediction_file,
        start_date,
        end_date,
        strategy="long",
        tail=10000,
    ):
        super().__init__()
        Labeler.__init__(self)
        InteractiveCandlestickChartBase.__init__(self)
        #
        self.label = strategy
        # Load and prepare the data
        print(filepath, outputfile, fe_file, start_date, end_date, strategy, tail)
        self.set_data(filepath, fe_file, model_prediction_file, start_date, end_date, strategy, tail)
        #
        self.outputfile = outputfile.format(
            label=self.label,
            start_date=self.df.index[0].strftime("%Y_%m_%d_%H_%M"),
            end_date=self.df.index[-1].strftime("%Y_%m_%d_%H_%M"),
        )
        if not os.path.exists(self.outputfile):
            os.makedirs(os.path.dirname(self.outputfile), exist_ok=True)
        #
        self.fig = None
        self.ax = None
        self.ax_vol = None
        #
        self.labels = []
        # Storage for labels
        self.labels = []
        self.draw_plot()

    def draw_plot(self):
        # Create the plot (with two subplots
        self.fig, (self.ax, self.ax_vol) = plt.subplots(
            2, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
        )
        self.plot_candlestick()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def set_data(
        self, filepath, fe_filepath,model_prediction_file, start_date, end_date, strategy="long", tail=10000
    ):
        self.df = pd.read_csv(filepath)  # .tail(60)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.add_candle_pattern_labels(fe_filepath)
        self.get_predictions(model_prediction_file)
        # labeling 
        # self.four_out_of_five() 
        # self.long_label_profit_stop_loss()
        # self.bull_flag()
        self.rsi_strategy()
        print("_" * 100)
        print(self.df.date.min(), self.df.date.max())
        # these two work for intraday only
        x = len(self.df[(self.df.date >= start_date) & (self.df.date < end_date)])
        print(self.df.date.min() , end_date)
        prev_df = self.df[self.df.date < start_date].tail(x)
        self.df['prev_close'] = prev_df.close.values[-1] if prev_df.shape[0] else np.nan
        self.df["prev_high"] = prev_df.high.max() if prev_df.shape[0] else np.nan
        self.df["prev_low"] = prev_df.low.min() if prev_df.shape[0] else np.nan
        self.df["prev_vwap"] = (prev_df['close'] * prev_df['volume']).sum() / prev_df['volume'].sum() if prev_df.shape[0] else np.nan
        # filter to the day of interest only
        self.df = self.df[
            (self.df.date >= start_date) & (self.df.date < end_date)
        ].tail(tail)
        self.df.reset_index(inplace=True)
        # self.df= self.df.iloc[0:int(len(self.df)/2)]
        # ignore first 1 minute
        # self.df = self.df.iloc[1:]
        self.df.set_index("date", inplace=True)

    def add_candle_pattern_labels(self, filepath):
        patterns = pd.read_csv(filepath)
        patterns["date"] = pd.to_datetime(patterns["date"])
        x = self.df.shape[0]
        self.df = self.df.merge(
            patterns[["date", "vwap", "rsi_14", "macd"]], how="left", on="date"
        )
        assert self.df.shape[0] == x
        print(self.df.shape) 

    def set_labels(self):
        # this works if you click on the buy chart and then on the sell chart. Even for SHORTs.
        t = 1
        while t < len(self.labels):
            if self.labels[t - 1]["close"] < self.labels[t]["close"]:  # long
                self.labels[t - 1]["label"] = "Buy"
                self.labels[t]["label"] = "Sell"
            else:  # short
                self.labels[t - 1]["label"] = "Sell"
                self.labels[t]["label"] = "Buy"
            t += 2

    def add_plot(self):
        add_plots = []
        # Add buy and sell markers
        markers = {
            "buy": {"column": "buy", "offset": -0.04, "marker": "^", "color": "green"},
            "sell": {"column": "sell", "offset": -0.02, "marker": "x", "color": "red"},
        }

        for label, params in markers.items():
            marker_data = [
                i.low + params["offset"]
                if getattr(i, params["column"]) == 1
                else np.nan
                for i in self.df.itertuples()
            ]
            add_plots.append(
                mpf.make_addplot(
                    marker_data,
                    scatter=True,
                    markersize=10,
                    marker=params["marker"],
                    color=params["color"],
                    ylabel=label,
                    ax=self.ax,
                )
            )

        # model predicted probability
        l=[]
        for i,row in self.df.iterrows():
            if row.prob> 0.5:
                l.append(row.high-0.02)
            else:
                l.append(np.nan)
        add_plots.append(
            mpf.make_addplot(
                l,
                type="line",
                    scatter=True,
                    markersize=8,
                color="black",
                ylabel="probability",
                ax=self.ax,
            )
        ) 
        
        # Add line plots 
        line_plots = [
            {"data": self.df.vwap, "color": "darkblue", "label": "vwap" },
            # {
            #     "data": self.df.rsi_14.map(self.scale_rsi),
            #     "label": "rsi_14",
            #     "color": "darkblue",
            # },
            # {
            #     "data": self.df.prev_high,
            #     "color": "red",
            #     "label": "prev_high",
            #     "linestyle": "--",
            # },
            # {
            #     "data": self.df.prev_low,
            #     "color": "red",
            #     "label": "prev_low",
            #     "linestyle": "--",
            # },
            {
                "data": self.df.prev_vwap,
                "color": "black",
                "label": "prev_vwap",
                "linestyle": "--",
            },
            {
                "data": self.df.prev_close,
                "color": "black",
                "label": "prev_close",
                # "linestyle": "--",
            },
        ]
        for plot in line_plots:
            add_plots.append(
                mpf.make_addplot(
                    plot["data"],
                    type="line", 
                    color=plot["color"],
                    label=plot["label"],
                    linestyle=plot.get("linestyle", "-"),
                    markersize=10,
                    ax=self.ax,
                )
            )

        # RSI plots


        self.df["rsi_lb"], self.df["rsi_ub"] = 30, 70
        rsi_plots = [
            {
                "data": self.df.rsi_14.map(self.scale_rsi),
                "label": "rsi_14",
                "color": "darkblue",
            },
            {
                "data": self.df["rsi_lb"].map(self.scale_rsi),
                "label": "rsi_lb",
                "color": "red",
                "linestyle": "--",
            },
            {
                "data": self.df["rsi_ub"].map(self.scale_rsi),
                "label": "rsi_ub",
                "color": "red",
                "linestyle": "--",
            },
        ]
        for plot in rsi_plots:
            add_plots.append(
                mpf.make_addplot(
                    plot["data"],
                    type="line",
                    scatter=True,
                    color=plot["color"],
                    label=plot["label"],
                    linestyle=plot.get("linestyle", "-"),
                    markersize=10,
                    ax=self.ax_vol,
                )
            )


        # Mean volume
        self.df["mean_vol"] = self.df.volume.mean()
        add_plots.append(
            mpf.make_addplot(
                self.df.mean_vol,
                type="line",
                color="black",
                ylabel="mean vol",
                ax=self.ax_vol,
            )
        )

        return add_plots


if __name__ == "__main__":
    label_output_path = get_manual_output_path()
    filepath = get_merger_final_output_file()
    model_prediction_file= "/Users/ragheb/myprojects/stock/src/files/xgb/buy_binary_output.csv"
    model_prediction_file= "/Users/ragheb/myprojects/stock/src/files/xgb/tmp_pred.csv"
    if not os.path.exists(label_output_path):
        os.makedirs(label_output_path)
    for i in [4]:#[18, 19, 20, 23, 24, 26, 27, 30, 31]:
        chart = InteractiveCandlestickChart(
            filepath,
            label_output_path + "/{label}_{start_date}_to_{end_date}.csv",
            "/Users/ragheb/myprojects/stock/src/files/training/fe_1d.csv",
            model_prediction_file, 
            f"12/{i}/2024",
            f"12/{i+1}/2024", 
            strategy="long",
            tail=60000,
        )
        chart.show()
