

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def plot_optimas(df,col):

  x=df.tail(500)
  plt.scatter(x.index, x[col],  marker='.')
  plt.scatter(x[x.optima_DO_NOT_USE_FOR_TRAINING==1].index, x[x.optima_DO_NOT_USE_FOR_TRAINING==1][col],  marker='o', color='g')
  plt.scatter(x[x.optima_DO_NOT_USE_FOR_TRAINING==-1].index, x[x.optima_DO_NOT_USE_FOR_TRAINING==-1][col],  marker='*', color='r')
  plt.show()


def find_local_optima_DO_NOT_USE_FOR_TRAININGs(df, col, N=5):

  # x= df.loc[(df.cnt>=start)&(df.cnt<=end)&(df.optima_DO_NOT_USE_FOR_TRAINING==1)]
  # y= df.loc[(df.cnt>=start)&(df.cnt<=end)&(df.optima_DO_NOT_USE_FOR_TRAINING==-1)]
  # if len(y)<2 or len(x)<2:
  #   return 0
  # # LL and LH
  # if  x.tail(1).close.values[0] < x.tail(2).head(1).close.values[0]  and\
  #     y.tail(1).close.values[0]< y.tail(2).head(1).close.values[0] :
  #   return -1
  # # HH and HL
  # if x.tail(1).close.values[0] > x.tail(2).head(1).close.values[0] and\
  #     y.tail(1).close.values[0] > y.tail(2).head(1).close.values[0] :
  #   return 1
  # return 0
  df['optima_DO_NOT_USE_FOR_TRAINING']=0
  df.loc[df.iloc[argrelextrema(df[col].values, np.greater, order=N)[0]].index, 'optima_DO_NOT_USE_FOR_TRAINING']=1
  df.loc[df.iloc[argrelextrema(df[col].values, np.less, order=N)[0]].index, 'optima_DO_NOT_USE_FOR_TRAINING']=-1

  print("  ->optima_DO_NOT_USE_FOR_TRAINING={}:{}".format(len(df[df.optima_DO_NOT_USE_FOR_TRAINING==1]), len(df[df.optima_DO_NOT_USE_FOR_TRAINING==-1])))
  plot_optimas(df,col)
  exit(1)

  # return df
