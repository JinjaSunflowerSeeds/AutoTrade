import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseEvent


class InteractiveCandlestickChart:
    def __init__(self, filepath, outputfile, ticker, label):
        self.label = label
        # Load and prepare the data
        self.df = pd.read_csv(filepath).tail(50)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df.set_index("date", inplace=True)
        #
        self.outputfile = outputfile.format(
            stock=ticker,
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
        )
        self.ax.set_title("Interactive Candlestick Chart for Manual Labeling")

    def get_df_info(self, x_coord, x_dates):
        step = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) / len(x_dates)
        x = x_coord - self.ax.get_xlim()[0]
        idx = int(np.ceil(x / step) - 1)
        clicked_date = x_dates[idx]
        low_price = self.df[self.df.index == clicked_date]["low"].values[0]
        print(f"Clicked on: {clicked_date} (Idx: {idx}), (low price is {low_price})")
        return clicked_date, idx, low_price

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
        date, idx, low_price = self.get_df_info(x_coord, x_dates)
        self.labels.append({"date": date, "label": self.label})
        self.save_labels_to_file()
        #
        line = self.ax.axvline(x=event.xdata, color="black", linestyle="--")
        plt.draw()

    def save_labels_to_file(self):
        labels_df = pd.DataFrame(self.labels)
        labels_df.to_csv(self.outputfile, index=False)
        print(f"Labels saved to  {self.outputfile}.")

    def show(self):
        self.fig.tight_layout()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Usage example
    ticker = "NVDA"
    filepath = "/Users/ragheb/myprojects/stock/src/files/training/fe_1d.csv"
    outputfile = "/Users/ragheb/myprojects/stock/src/files/training/manual_labels/{stock}/{label}_{start_date}_to_{end_date}.csv"
    chart = InteractiveCandlestickChart(filepath, outputfile, ticker, "Buy")
    chart.show()
