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

class ModelPredictionChart:
    def __init__(self, filepath, model_prediction_file):
        # Load and prepare the data
        self.get_data(filepath)
        self.get_predictions(model_prediction_file)
        print(self.df.shape, self.preds.shape)
        self.df = self.df.merge(self.preds, on='date')
        # for i in ['open', 'high', 'low', 'close', 'prob']:
        #     self.df[i] = (self.df[i]-self.df[i].min())/(self.df[i].max()-self.df[i].min())
        print(self.df.shape)
        print(self.df)
        #
        self.fig = None
        self.ax = None
        self.ax_vol = None
        #
        self.labels = []
        # Storage for labels
        self.labels = []
        self.draw_plot()
    
    def get_predictions(self,model_prediction_file):
        self.preds= pd.read_csv(model_prediction_file)[['date', 'prob', 'label']]
        self.preds['date'] = pd.to_datetime(self.preds.date)
        self.preds.set_index("date", inplace=True)
        print(self.preds)
        
    def get_data(self,filepath):
        self.df = pd.read_csv(filepath)#.tail(60)
        self.df["date"] = pd.to_datetime(self.df["date"])
        s = "12/30/2024"
        e = "12/31/2024"
        self.df = self.df[(self.df.date >= s) & (self.df.date < e)]
        self.df.set_index("date", inplace=True)

    def draw_plot(self):
        # Create the plot (with two subplots
        self.fig, (self.ax, self.ax_vol) = plt.subplots(
            2, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
        )
        self.plot_candlestick()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def add_plot(self):
        add_plots = []
        l, ll = [], []
        for _, i in self.df.iterrows():
            if i.label == 1:
                l.append(i.low - 0.01)
            else:
                l.append(np.nan)
            if i.prob>self.df['prob'].quantile(0.9):
                ll.append(i.low - 0.01)
            else:
                ll.append(np.nan)
        add_plots.append(
            mpf.make_addplot(
                l,
                scatter=True,
                markersize=10,
                marker="^",
                color="black",
                ylabel="Buy Signal",
                ax=self.ax,
            )
        )
        add_plots.append(
            mpf.make_addplot(
                ll,
                scatter=True,
                markersize=10,
                marker="X",
                color="blue",
                ylabel="ml action signal",
                ax=self.ax,
            )
        )
        # add_plots.append(
        #     mpf.make_addplot(
        #         self.df.prob,
        #         type="scatter",
        #         marker=".",
        #         color="darkblue",
        #         markersize=10,
        #         label="prob",
        #         ax=self.ax,
        #     )
        # )  
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
        #
        line = self.ax.axvline(x=event.xdata, color="black", linestyle="--")
        plt.draw()
 
   
    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Usage example
    label_output_path = get_manual_output_path()
    # filepath = "/Users/ragheb/myprojects/stock/src/files/training/merger/NVDA/fe_1d.csv"
    filepath = get_merger_final_output_file()
    model_prediction_file= "/Users/ragheb/myprojects/stock/src/files/xgb/buy_binary_output.csv"
    if not os.path.exists(label_output_path):
        os.makedirs(label_output_path)
    print(filepath)
    print(model_prediction_file)
    chart = ModelPredictionChart(filepath, model_prediction_file)
    chart.show()
