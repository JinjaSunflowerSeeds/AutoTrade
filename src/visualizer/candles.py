import os
import sys
from logging import Logger

sys.path.append("./")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseEvent
from utils.config_reader import get_manual_output_path, get_merger_final_output_file

"""
cmd: 
    python3 visualizer/candles.py
    
You need to click on the buy chart and then on the sell chart. Even for SHORTs.
    -> see set_labels()
"""

class InteractiveCandlestickChart:
    def __init__(self, filepath, outputfile, label):
        self.label = label
        # Load and prepare the data
        self.df = pd.read_csv(filepath)#.tail(60)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["vwma_14"] = (
            self.df["close"]
            .mul(self.df["volume"])
            .rolling(window=14, min_periods=1)
            .sum()
            / self.df["volume"].rolling(window=14, min_periods=1).sum()
        )
        # self.df["vwma_14"]= (self.df['close'] * self.df['volume']).cumsum() / self.df['volume'].cumsum()
        s = "12/26/2024"
        e = "12/27/2024"
        self.df = self.df[(self.df.date >= s) & (self.df.date < e)]
        # ignore first 1 minute
        self.df = self.df.iloc[1:]
        self.df.set_index("date", inplace=True)
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

    def add_plot(self):
        add_plots = []
        # l = []
        # for _, i in self.df.iterrows():
        #     if i.label == 1:
        #         l.append(i.low - 0.01)
        #     else:
        #         l.append(np.nan)
        # add_plots.append(
        #     mpf.make_addplot(
        #         l,
        #         scatter=True,
        #         markersize=10,
        #         marker="^",
        #         color="black",
        #         ylabel="Buy Signal",
        #     )
        # )
        add_plots.append(
            mpf.make_addplot(
                self.df.vwma_14,
                type="line",
                marker=".",
                color="darkblue",
                markersize=10,
                label="vwma_14",
                ax=self.ax,
            )
        )
        # add_plots.append(
        #     mpf.make_addplot(
        #         100*self.df.rsi_14,type='line', marker='.', color="green", markersize=100, ylabel="RSI", ax=self.ax_vol
        #     )
        # )
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
        self.labels.append({"date": date, "close": price})
        self.save_labels_to_file()
        #
        line = self.ax.axvline(x=event.xdata, color="black", linestyle="--")
        plt.draw()

    def set_labels(self):
        # this works if you click on the buy chart and then on the sell chart. Even for SHORTs.
        for i in range(len(self.labels)):
            if i % 2 != 0:
                self.labels[i]["label"] = "Buy"
            else:
                self.labels[i]["label"] = "Sell"
                
    def get_pnl(self, df):
        profit= df[df.label=='Buy'].close.sum()-df[df.label=='Sell'].close.sum()
        ppt= [f"{(i[0]-i[1])[0]:.2f}" for i in zip(df[df.label=='Buy'].close.values , df[df.label=='Sell'].close.values)]
        return profit[0], ppt

    def save_labels_to_file(self):
        self.set_labels()
        df_ = pd.DataFrame(self.labels)
        df_.to_csv(self.outputfile, index=False)
        profit, ppt= self.get_pnl(df_)
        print(
            f"Labels saved to  {self.outputfile}. profit={profit:.2f}, pre trade={ppt})"
        )

    def show(self):
        # self.fig.subplots_adjust(hspace=0.1)
        self.fig.tight_layout()
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
    chart = InteractiveCandlestickChart(filepath, outputfile, "Buy")
    chart.show()
