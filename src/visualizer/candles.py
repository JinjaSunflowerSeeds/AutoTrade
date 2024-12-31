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


class InteractiveCandlestickChart(Labeler):
    def __init__(
        self,
        filepath,
        outputfile,
        fe_file,
        start_date,
        end_date,
        strategy="long",
        tail=10000,
    ):
        super().__init__()
        Labeler.__init__(self)
        self.label = strategy
        # Load and prepare the data
        print(filepath, outputfile,fe_file,start_date, end_date, strategy ,    tail )
        self.set_data(filepath, fe_file, start_date, end_date, strategy, tail)
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

    def set_data(
        self, filepath, fe_filepath, start_date, end_date, strategy="long", tail=10000
    ):
        self.df = pd.read_csv(filepath)  # .tail(60)
        self.df["date"] = pd.to_datetime(self.df["date"])
        if strategy == "long":
            self.long_label_profit_stop_loss()
        else:
            self.short_label_profit_stop_loss()
        self.df = self.df[
            (self.df.date >= start_date) & (self.df.date < end_date)
        ].tail(tail)
        self.df.reset_index(inplace=True)
        self.add_candle_pattern_labels(fe_filepath)
        # self.df= self.df.iloc[0:int(len(self.df)/2)]
        # ignore first 1 minute
        self.df = self.df.iloc[1:]
        self.df.set_index("date", inplace=True)

    def add_candle_pattern_labels(self, filepath):
        patterns = pd.read_csv(filepath)
        patterns["date"] = pd.to_datetime(patterns["date"])
        self.df = self.df.merge(
            patterns[["date", "vwap"]], how="left", on="date"
        )
        print(self.df.shape)

    def draw_plot(self):
        # Create the plot (with two subplots
        self.fig, (self.ax, self.ax_vol) = plt.subplots(
            2, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
        )
        self.plot_candlestick()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def add_plot(self):
        add_plots = []
        l, ll, lll = [], [], []
        for _, i in self.df.iterrows(): 
            if i.buy == 1:
                ll.append(i.low - 0.04)
            else:
                ll.append(np.nan)
            if i.sell == 1:
                lll.append(i.low - 0.02)
            else:
                lll.append(np.nan)

        add_plots.append(
            mpf.make_addplot(
                ll,
                scatter=True,
                markersize=10,
                marker="^",
                color="green",
                ylabel="buy",
                ax=self.ax,
            )
        )

        add_plots.append(
            mpf.make_addplot(
                lll,
                scatter=True,
                markersize=10,
                marker="x",
                color="red",
                ylabel="sell",
                ax=self.ax,
            )
        )
        add_plots.append(
            mpf.make_addplot(
                self.df.vwap,
                type="line",
                marker=".",
                color="darkblue",
                markersize=10,
                label="vwap",
                ax=self.ax,
            )
        )

        self.df["c"] = self.df.iloc[
            1:
        ].volume.mean()  # .rolling(window=7, min_periods=1).mean()
        add_plots.append(
            mpf.make_addplot(
                self.df.c,
                type="line",
                marker="-",
                color="black",
                markersize=10,
                ylabel="mena vol",
                ax=self.ax_vol,
            )
        )
        return add_plots

    def plot_candlestick(self):
        mc = mpf.make_marketcolors(up="g", down="r", inherit=True)
        style = mpf.make_mpf_style(marketcolors=mc)
        mpf.plot(
            self.df,
            type="candle",
            style=style,
            ax=self.ax,
            volume=self.ax_vol,
            show_nontrading=True,
            tight_layout=True,
            addplot=self.add_plot(),
        )
        self.ax.set_title("Interactive Candlestick Chart for Manual Labeling")

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
        # for i in range(len(self.labels)):
        #     if i % 2 != 0:
        #         self.labels[i]["label"] = "Buy"
        #     else:
        #         self.labels[i]["label"] = "Sell"

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

    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Usage example
    label_output_path = get_manual_output_path()
    # filepath = "/Users/ragheb/myprojects/stock/src/files/training/merger/NVDA/fe_1d.csv"
    filepath = get_merger_final_output_file()
    if not os.path.exists(label_output_path):
        os.makedirs(label_output_path)
    outputfile = label_output_path + "/{label}_{start_date}_to_{end_date}.csv"
    print(outputfile)
    print(filepath)
    chart = InteractiveCandlestickChart(
        filepath,
        outputfile,
        "/Users/ragheb/myprojects/stock/src/files/training/fe_1d.csv",
        "12/20/2024",
        "12/21/2024",
        strategy="short",
        tail=10000,
    )
    chart.show()
