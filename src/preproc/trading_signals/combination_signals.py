

def macd_rsi_signal(df):
    df['macd_rsi']=0
    # df.loc[(df['macd'] > df['macds']) & (df['rsi_14'] < 30), 'macd_rsi'] = 1
    # df.loc[(df['macd'] < df['macds']) & (df['rsi_14'] > 70),'macd_rsi'] = -1
    df.loc[(df['macd'] > df['macds']) & (df['rsi_14'] < 30) & (df.shift()['rsi_14'] > 30), 'macd_rsi'] = 1
    df.loc[(df['macd'] < df['macds']) & (df['rsi_14'] > 70)& (df.shift()['rsi_14'] < 70),'macd_rsi'] = -1
    print("  ->macd_rsi={}:{}".format(len(df[df.macd_rsi==1]), len(df[df.macd_rsi==-1])))
    return df


def rsi_bollinger(df):
    df['rsi_boll_signal']=0
    df.loc[(df['rsi_14'] < 30)&(df['close']<=df['boll_lb_20']), 'rsi_boll_signal']=1
    df.loc[(df['rsi_14'] >=70)&(df['close']>=df['boll_ub_20']), 'rsi_boll_signal']=-1
    print("  ->rsi_boll_signal={}:{}".format(len(df[df.rsi_boll_signal==1]), len(df[df.rsi_boll_signal==-1])))
    return df

def rsi_bollinger_vwap(df):
    df['rsi_boll_vwap_signal']=0
    df.loc[(df['rsi_14'] < 30)&(df['close']<=df['boll_lb_20']) &(df.open>=df.vwma_14)&(df.close>=df.vwma_14), 'rsi_boll_vwap_signal']=1
    df.loc[(df['rsi_14'] >=70)&(df['close']>=df['boll_ub_20'])&((df.open<=df.vwma_14)|(df.close<=df.vwma_14)), 'rsi_boll_vwap_signal']=-1
    print("  ->rsi_boll_vwap_signal={}:{}".format(len(df[df.rsi_boll_vwap_signal==1]), len(df[df.rsi_boll_vwap_signal==-1])))
    return df

def Keltner_channel(df):
    # df['keltner_channel_middle']=ema
    df['keltner_channel_ub']=df.close.ewm(com=0.5).mean() + 2*df.atr
    df['keltner_channel_ub']=df.close.ewm(com=0.5).mean() - 2*df.atr

    return df
