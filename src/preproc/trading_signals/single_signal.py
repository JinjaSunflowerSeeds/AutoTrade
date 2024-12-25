# crossing singals make more sense otherwise it will be too many same but usless labels

def macd_crosses_over(df):
    df['macd_crossover']=0
    df.loc[(df['macd'].shift() < df['macds'].shift()) & (df['macd'] > df['macds']), 'macd_crossover']=1
    df.loc[(df['macd'].shift() > df['macds'].shift()) & (df['macd'] < df['macds']), 'macd_crossover']=-1
    print("  ->macd_crossover={}:{}".format(len(df[df.macd_crossover==1]), len(df[df.macd_crossover==-1])))
    return df

def rsi_territory(df):
    assert df['rsi_14'].max()<=100 and df['rsi_14'].max()>1 and df['rsi_14'].min()>=0
    # entered the region
    df['rsi_singal']=0
    df.loc[(df.rsi_14<20)&(df.shift().rsi_14>20), 'rsi_singal']=1
    df.loc[(df.rsi_14>80)&(df.shift().rsi_14<80), 'rsi_singal']=-1
    #exited the region
    df['rsi_singal_v2']=0
    df.loc[(df.rsi_14>20)&(df.shift().rsi_14<20), 'rsi_singal_v2']=1
    df.loc[(df.rsi_14<80)&(df.shift().rsi_14>80), 'rsi_singal_v2']=-1

    print("  ->rsi_singal={}:{}".format(len(df[df.rsi_singal==1]), len(df[df.rsi_singal==-1])))
    print("  ->rsi_singal_v2={}:{}".format(len(df[df.rsi_singal_v2==1]), len(df[df.rsi_singal_v2==-1])))
    return df

def mfi_signal(df):
    assert df['mfi_14'].max()<=1 and df['mfi_14'].min()>=0
    df['mfi_signal']=0
    df.loc[(df['mfi_14'] < 0.2) & (df['mfi_14'].shift() >= 0.2), 'mfi_signal']=1
    df.loc[(df['mfi_14'] > 0.8) & (df['mfi_14'].shift() <= 0.8),'mfi_signal']=-1
    print("  ->mfi_signal={}:{}".format(len(df[df.mfi_signal==1]), len(df[df.mfi_signal==-1])))

    return df

def moving_average(df):
    df['ma_signal']= 0
    df['ma_50']=df['close'].rolling(window=50).mean()
    df['ma_200']=df['close'].rolling(window=200).mean()

    df.loc[(df.ma_50>df.ma_200)&(df.shift().ma_50<df.shift().ma_200), 'ma_signal']= 1
    df.loc[(df.ma_50<df.ma_200)&(df.shift().ma_50>df.shift().ma_200), 'ma_signal']= -1
    df.drop(columns=['ma_50','ma_200'],inplace=True)
    print("  ->ma_signal={}:{}".format(len(df[df.ma_signal==1]), len(df[df.ma_signal==-1])))
    return df

def vwap_signal(df):
    df['vwap_signal']=0
    df.loc[((df.open>=df.vwma_14)&(df.close>=df.vwma_14))& ~((df.shift().open>=df.shift().vwma_14)&(df.shift().close>=df.shift().vwma_14)), 'vwap_signal']=1
    df.loc[(df.open<=df.vwma_14)|(df.close<=df.vwma_14)& ~((df.shift().open<=df.shift().vwma_14)|(df.shift().close<=df.shift().vwma_14)), 'vwap_signal']=-1
    print("  ->vwap_signal={}:{}".format(len(df[df.vwap_signal==1]), len(df[df.vwap_signal==-1])))
    return df

def ichimoku_signal(df):
    df['Tenkan-sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['Kijun-sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    df['spanA'] = ((df['Tenkan-sen'] + df['Kijun-sen']) / 2).shift(26)
    df['spanB'] = (df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2
    df['Chikou Span'] = df['close'].shift(-26)

    # Generate signals
    df['ichimoku_signal'] = 0  # 0: No Signal, 1: Buy Signal, -1: Sell Signal

    # Buy Signal: Tenkan-sen > Kijun-sen and Close > spanA
    df.loc[(df['close'] > df['spanA'])&(df['close'] > df['spanB'])& ~((df.shift()['close'] > df.shift()['spanA'])&(df.shift()['close'] > df.shift()['spanB'])), 'ichimoku_signal'] = 1

    # Sell Signal: Tenkan-sen < Kijun-sen and Close < spanA
    df.loc[(df['close'] < df['spanA'])&(df['close'] < df['spanB'])& ~((df.shift()['close'] < df.shift()['spanA'])&(df.shift()['close'] < df.shift()['spanB'])), 'ichimoku_signal'] = -1

    print("  ->ichimoku_signal={}:{}".format(len(df[df.ichimoku_signal==1]), len(df[df.ichimoku_signal==-1])))

    return df

def kdj_signal(df):
    # Stochastic Oscillator
    assert df['kdjk'].max()<=100 and df['kdjk'].max()>1 and df['kdjk'].min()>=0
    # entered the region
    df['kdjk_singal']=0
    df.loc[(df.kdjk<20)&(df.shift().kdjk>20), 'kdjk_singal']=1
    df.loc[(df.kdjk>80)&(df.shift().kdjk<80), 'kdjk_singal']=-1
    #exited the region
    df['kdjk_singal_v2']=0
    df.loc[(df.kdjk>20)&(df.shift().kdjk<20), 'kdjk_singal_v2']=1
    df.loc[(df.kdjk<80)&(df.shift().kdjk>80), 'kdjk_singal_v2']=-1
    # Stochastic crossover
    # When an increasing %K line crosses above the %D line in an oversold region, it is generating a buy signal.
    # When a decreasing %K line crosses below the %D line in an overbought region, this is a sell signal.
    # These signals tend to be more reliable in a range-bound market. They are less reliable in a trending market.
    df['kdjk_singal_v3']=0
    df.loc[(df.kdjk>df.kdjd)&(df.shift().kdjk<df.shift().kdjd)&(df.kdjk<20), 'kdjk_singal_v3']=1
    df.loc[(df.kdjk<df.kdjd)&(df.shift().kdjk>df.shift().kdjd)&(df.kdjk>80), 'kdjk_singal_v3']=-1

    print("  ->kdjk_singal={}:{}".format(len(df[df.kdjk_singal==1]), len(df[df.kdjk_singal==-1])))
    print("  ->kdjk_singal_v2={}:{}".format(len(df[df.kdjk_singal_v2==1]), len(df[df.kdjk_singal_v2==-1])))
    print("  ->kdjk_singal_v3={}:{}".format(len(df[df.kdjk_singal_v3==1]), len(df[df.kdjk_singal_v3==-1])))
    return df
