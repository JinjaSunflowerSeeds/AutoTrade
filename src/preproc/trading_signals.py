# in here we code the common analysis traders use as signals got buy and sell using TA
# such as rsi divergence

# https://tradeciety.com/the-13-best-candlestick-signals
import sys
sys.path.append('./preproc/trading_signals')
# sys.path.append('./lib/visualization')
from combination_signals import (
    macd_rsi_signal,
    rsi_bollinger,
    rsi_bollinger_vwap
)
from single_signal import(
    macd_crosses_over,
    rsi_territory,
    mfi_signal,
    moving_average,
    vwap_signal,
    ichimoku_signal,
    kdj_signal,
)
from divergence import (
    add_regression_slope
)
from lib.visualization.optimas import(
    find_local_optima_DO_NOT_USE_FOR_TRAININGs
)

class TS:
    def __init__(self):
        pass

    def driver(self, df):
        print("Adding trading signals...")
        c= df.columns.tolist()
        df= add_regression_slope(df,'close')
        df= add_regression_slope(df,'kdjk')

        df= macd_crosses_over(df)
        df= rsi_territory(df)
        df= macd_rsi_signal(df)
        df= mfi_signal(df)
        df=moving_average(df)
        df=rsi_bollinger(df)
        df=vwap_signal(df)
        df=rsi_bollinger_vwap(df)
        df=ichimoku_signal(df)
        df=kdj_signal(df)

        # find_local_optima_DO_NOT_USE_FOR_TRAININGs(df,'close')

        print(" Added {}".format([i for i in df.columns.tolist() if i not in c]))
        return df
