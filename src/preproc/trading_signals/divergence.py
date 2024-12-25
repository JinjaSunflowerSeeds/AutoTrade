import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge



def get_trend(df, idx, N):
  try:
    x = df.iloc[max(0, idx.cnt - N): idx.cnt]
    slope, _ = np.polyfit(x.cnt, x.close, 1)
    return slope
  except Exception as e:
    print(e)
  return 0

def add_regression_slope(df, col, N=5):
  df.sort_values('date', inplace=True)
  df['cnt']= [i for i in range(len(df))]

  new_col ='{}_slope'.format(col)
  new_col_dir ='{}_slope_dir'.format(col)


  df[new_col]=df.apply(lambda row: get_trend(df, row, N), axis=1)
  df[new_col_dir] = df.apply(lambda row:  1 if row[new_col] > 0 else -1 if row[new_col] < 0 else 0, axis=1)

  print("  ->{}={}:{}".format(new_col_dir, len(df[df[new_col_dir]==1]), len(df[df[new_col_dir]==-1])))
  df.drop(['cnt'], inplace=True, axis='columns')
  return df
