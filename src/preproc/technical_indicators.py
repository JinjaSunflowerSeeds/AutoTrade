import sys
sys.path.append('./')
sys.path.append('./lib')

import pandas as pd
import numpy as np
import pandas_ta as ta
from stockstats import wrap
from lib.candlestick import candlestick
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from lib.color_logger import MyLogger
import copy

from utils.config_reader import  get_merger_final_output_file

class CandlestickChart: 
    def candle_stick(self):
        self.log.info(" Adding candle stick patterns ...")
        self.df = candlestick.inverted_hammer(self.df, target='inverted_hammer')
        self.log.info(f"  -> inverted_hammer={self.df[self.df['inverted_hammer']!=0].shape}")
        self.df = candlestick.bearish_engulfing(self.df, target="bearish_engulfing")
        self.log.info(f"  -> bearish_engulfing={self.df[self.df['bearish_engulfing']!=0].shape}")
        self.df = candlestick.dark_cloud_cover(self.df, target="dark_cloud_cover")
        self.log.info(f"  -> dark_cloud_cover={self.df[self.df['dark_cloud_cover']!=0].shape}")
        self.df = candlestick.evening_star_doji(self.df, target="evening_star_doji")
        self.log.info(f"  -> evening_star_doji={self.df[self.df['evening_star_doji']!=0].shape}")
        self.df = candlestick.morning_star(self.df, target="morning_star")
        self.log.info(f"  -> morning_star={self.df[self.df['morning_star']!=0].shape}")
        self.df = candlestick.shooting_star(self.df, target="shooting_star")
        self.log.info(f"  -> shooting_star={self.df[self.df['shooting_star']!=0].shape}")
        self.df = candlestick.bearish_harami(self.df, target="bearish_harami")
        self.log.info(f"  -> bearish_harami={self.df[self.df['bearish_harami']!=0].shape}")
        self.df = candlestick.doji(self.df, target="doji")
        self.log.info(f"  -> doji={self.df[self.df['doji']!=0].shape}")
        self.df = candlestick.gravestone_doji(self.df, target="gravestone_doji")
        self.log.info(f"  -> gravestone_doji={self.df[self.df['gravestone_doji']!=0].shape}")
        self.df = candlestick.morning_star_doji(self.df, target="morning_star_doji")
        self.log.info(f"  -> morning_star_doji={self.df[self.df['morning_star_doji']!=0].shape}")
        self.df = candlestick.star(self.df, target="star")
        self.log.info(f"  -> star={self.df[self.df['star']!=0].shape}")
        self.df = candlestick.bullish_engulfing(self.df, target="bullish_engulfing")
        self.log.info(f"  -> bullish_engulfing={self.df[self.df['bullish_engulfing']!=0].shape}")
        self.df = candlestick.doji_star(self.df, target="doji_star")
        self.log.info(f"  -> doji_star={self.df[self.df['doji_star']!=0].shape}")
        self.df = candlestick.hammer(self.df, target="hammer")
        self.log.info(f"  -> hammer={self.df[self.df['hammer']!=0].shape}")
        self.df = candlestick.piercing_pattern(self.df, target="piercing_pattern")
        self.log.info(f"  -> piercing_pattern={self.df[self.df['piercing_pattern']!=0].shape}")
        self.df = candlestick.bullish_harami(self.df, target="bullish_harami")
        self.log.info(f"  -> bullish_harami={self.df[self.df['bullish_harami']!=0].shape}")
        self.df = candlestick.dragonfly_doji(self.df, target="dragonfly_doji")
        self.log.info(f"  -> dragonfly_doji={self.df[self.df['dragonfly_doji']!=0].shape}")
        self.df = candlestick.hanging_man(self.df, target="hanging_man")
        self.log.info(f"  -> hanging_man={self.df[self.df['hanging_man']!=0].shape}")
        self.df = candlestick.rain_drop(self.df, target="rain_drop")
        self.log.info(f"  -> rain_drop={self.df[self.df['rain_drop']!=0].shape}")
        self.df = candlestick.evening_star(self.df, target="evening_star")
        self.log.info(f"  -> evening_star={self.df[self.df['evening_star']!=0].shape}")
        self.df = candlestick.rain_drop_doji(self.df, target="rain_drop_doji")
        self.log.info(f"  -> rain_drop_doji={self.df[self.df['rain_drop_doji']!=0].shape}")

        # self.df = candlestick.tweezer_bottom(self.df, target="tweezer_bottom")
        # self.df = candlestick.tweezer_top(self.df, target="tweezer_top")
        # self.df = candlestick.bearish_marubozu(self.df, target="bearish_mar")
        # self.df = candlestick.bullish_marubozu(self.df, target="bullish_mar")
        # self.df = candlestick.bearish_spinning_top(self.df, target="bearish_spinning_top")
        # self.df = candlestick.bullish_spinning_top(self.df, target="bullish_spinning_top")
        # self.df = candlestick.bearish_abandoned_baby(self.df, target="bearish_abandon")
        # self.df = candlestick.bullish_abandoned_baby(self.df, target="bullish_abandoned")
        # self.df = candlestick.bearish_triple_top(self.df, target="bearish_triple_top")
        # self.df = candlestick.bullish_triple_top(self.df, target="bullish_triple_top")
        # self.df = candlestick.bearish_head_and_shoulders(self.df, target="bearish_head_and_shoulders")
        # self.df = candlestick.bullish_head_and_shoulders(self.df, target="bullish_head_and_shoulders")
        # self.df = candlestick.bearish_inverted_hammer(self.df, target="bearish_inverted_hammer")
        # self.df = candlestick.bullish_inverted_hammer(self.df, target="bullish_inverted_hammer")

        self.df[['inverted_hammer',"bearish_engulfing","dark_cloud_cover","evening_star_doji","morning_star",
                 "shooting_star","bearish_harami","doji","gravestone_doji","morning_star_doji","star",
                 "bullish_engulfing","doji_star","hammer","piercing_pattern","bullish_harami",
                 "dragonfly_doji","hanging_man","rain_drop","evening_star","rain_drop_doji"]]= self.df[['inverted_hammer',"bearish_engulfing","dark_cloud_cover","evening_star_doji","morning_star",
                 "shooting_star","bearish_harami","doji","gravestone_doji","morning_star_doji","star",
                 "bullish_engulfing","doji_star","hammer","piercing_pattern","bullish_harami",
                 "dragonfly_doji","hanging_man","rain_drop","evening_star","rain_drop_doji"]].astype('float')

    def cs_patterns(self):
        self.log.info(" Adding candle stick patterns ...")
        patterns = self.df.ta.cdl_pattern(name="all")
        patterns.columns = [p.replace('CDL_','cs_').lower() for p in patterns.columns]
        self.log.info(f"  -> {patterns.columns.tolist()}")
        self.df = pd.concat([self.df, patterns], axis=1)

class TA(CandlestickChart, MyLogger):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.df=pd.DataFrame()

    def read_file(self):
        self.log.info(" Reading file {} ...".format(self.filepath))
        self.df = pd.read_csv(self.filepath)
        self.log.info("  -> df={}".format(self.df.shape))

    def set_data(self, data):
        self.df=data

    def get_data(self):
        return self.df

    def export_data(self, filepath=None):
        self.log.info(" Exporting the data to {}...".format(filepath))
        self.df.to_csv(filepath)

    def print_stats(self):
        self.log.info(" Stats: ")
        self.log.info(self.df)
        self.log.info(self.df.shape)
        self.log.info(self.df.isna().sum())
        self.log.info(self.df.corr())
        cor=[]
        for c in self.df.columns:
            if c!='ticker':
                try:
                    cor.append([self.df[c].corr(self.df['close'].shift(-1)), c])
                except Exception as e:
                    self.log.info(c, e)
        for i in sorted(cor, key=lambda x: x[0]):
            self.log.info(i)

        # self.log.info(self.df.corr(self.df.shift(-1).close).sort_values())

    def eight_trigram(self):
        self.log.info(" Adding the 8 trigrams...")
        self.df[["BullishHorn", "BearHorn", "BullishHigh", "BearHigh", "BullishLow", "BearLow", "BullishHarami" , "BearHarami"]] = 0
        self.df.loc[
            (self.df.high>self.df.shift(1).high) &
            (self.df.low<self.df.shift(1).low) &
            (self.df.close>self.df.shift(1).close),'BullishHorn'] = 1
        self.df.loc[
            (self.df.high>self.df.shift(1).high) &
            (self.df.low<self.df.shift(1).low) &
            (self.df.close<self.df.shift(1).close),'BearHorn'] = 1
        self.df.loc[
            (self.df.high>self.df.shift(1).high) &
            (self.df.low>self.df.shift(1).low) &
            (self.df.close>self.df.shift(1).close),'BullishHigh'] = 1
        self.df.loc[
            (self.df.high>self.df.shift(1).high) &
            (self.df.low>self.df.shift(1).low) &
            (self.df.close<self.df.shift(1).close),'BearHigh'] = 1
        self.df.loc[
            (self.df.high<self.df.shift(1).high) &
            (self.df.low<self.df.shift(1).low) &
            (self.df.close>self.df.shift(1).close),'BullishLow'] = 1
        self.df.loc[
            (self.df.high<self.df.shift(1).high) &
            (self.df.low<self.df.shift(1).low) &
            (self.df.close<self.df.shift(1).close),'BearLow'] = 1
        self.df.loc[
            (self.df.high<self.df.shift(1).high) &
            (self.df.low>self.df.shift(1).low) &
            (self.df.close>self.df.shift(1).close),'BullishHarami'] = 1
        self.df.loc[
            (self.df.high<self.df.shift(1).high) &
            (self.df.low>self.df.shift(1).low) &
            (self.df.close<self.df.shift(1).close),'BearHarami'] = 1

    def gen_ta_aux(self):
        # Add ta features filling NaN values
        self.df = add_all_ta_features(
            self.df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        self.log.info(" ->add aux ta")
        self.log.info(self.df.columns.tolist())
        exit(1)

    def gen_ta(self):
        self.log.info(" Adding technical indicators...")
        # self.gen_ta_aux()
        # https://pypi.org/project/stockstats/
        ta = wrap(self.df)
        # middle
        self.df['middle']= ta['middle']
        # log return of close ( ln( close / last close))
        self.df['log-ret']= 100*ta['log-ret']
        # count positive retruns in the past 10 windows
        ta['up'] = ta['log-ret'] > 0
        self.df['up_10_c']=ta['up_10_c']
        # rsi 6, 14 and 26
        self.df['rsi_6'] = ta['rsi_6']
        self.df['rsi_14'] = ta['rsi_14']
        self.df['rsi_26'] = ta['rsi_26']
        self.df['stochrsi_6'] = ta['stochrsi_6']
        self.df['stochrsi_14'] = ta['stochrsi_14']
        self.df['stochrsi_26'] = ta['stochrsi_26']
        # WT - Wave Trend (Retrieve the LazyBear's Wave Trend )
        self.df['wt1']= ta['wt1']
        self.df['wt2'] = ta['wt2']
        # SMMA - Smoothed Moving Average
        self.df['close_7_smma'] = ta['close_7_smma']
        self.df['close_14_smma'] = ta['close_14_smma']
        self.df['close_21_smma']=ta['close_21_smma']
        # ROC - Rate of Change
        self.df['close_10_roc']=ta['close_10_roc']
        self.df['close_21_roc']=ta['close_21_roc']
        # MAD - Mean Absolute Deviation
        self.df['close_5_mad']=ta['close_5_mad']
        self.df['close_10_mad']=ta['close_10_mad']
        self.df['close_25_mad']=ta['close_25_mad']
        # TRIX - Triple Exponential Average
        self.df['close_12_trix'] = ta['close_12_trix']
        self.df['close_12_tema'] = ta['close_12_tema']
        # VR - Volume Variation Index
        self.df['vr_26']=ta['vr_26']
        # WR - Williams Overbought/Oversold Index
        self.df['wr_14']=ta['wr_14']
        # CCI - Commodity Channel Index
        self.df['cci_14'] = ta['cci_14']
        # TR - True Range of Trading
        self.df['atr_14'] = ta['atr_14']
        # Supertrend
        self.df['supertrend'] = ta['supertrend']
        self.df['supertrend_ub'] = ta['supertrend_ub']
        self.df['supertrend_lb'] = ta['supertrend_lb']
        # DMA - Difference of Moving Average
        self.df['dma'] = ta['dma']
        # DMI - Directional Movement Index
        self.df['pdi'] = ta['pdi']
        self.df['ndi'] = ta['ndi']
        self.df['dx'] = ta['dx']
        self.df['adx'] = ta['adx']
        self.df['adxr'] = ta['adxr']
        # KDJ Indicator (Stochastic Oscillator)
        self.df['kdjk'] = ta['kdjk']
        self.df['kdjd'] = ta['kdjd']
        self.df['kdjj'] = ta['kdjj']
        # CR - Energy Index
        self.df['cr'] = ta['cr']
        self.df['cr-ma1'] = ta['cr-ma1']
        self.df['cr-ma2'] = ta['cr-ma2']
        self.df['cr-ma3'] = ta['cr-ma3']
        # Bollinger Bands
        self.df['boll_20'] = ta['boll_20']
        self.df['boll_ub_20'] = ta['boll_ub_20']
        self.df['boll_lb_20'] = ta['boll_lb_20']
        # MACD - Moving Average Convergence Divergence
        self.df['macd'] = ta['macd']
        self.df['macds'] = ta['macds']#is the signal line.
        self.df['macdh'] = ta['macdh']#is he histogram line.
        # PPO - Percentage Price Oscillator
        self.df['ppo'] = ta['ppo']
        self.df['ppos'] = ta['ppos']
        self.df['ppoh'] = ta['ppoh']
        # Volume Weighted Moving Average
        self.df['vwma_14'] = ta['vwma_14']
        # Simple Moving Average
        self.df['close_14_sma'] = ta['close_14_sma']
        self.df['close_14_mstd'] = ta['close_14_mstd']
        # CHOP - Choppiness Index
        self.df['chop_14'] = ta['chop_14']
        # MFI - Money Flow Index
        self.df['mfi_14'] = ta['mfi_14']
        # ERI - Elder-Ray Index
        self.df['eribull']=ta['eribull'] #retrieves the 13 periods bull power = taperiods bull power
        self.df['eribear']=ta['eribear'] #retrieves the 13 periods bear power = taperiods bear power
        self.df['eribull_5']=ta['eribull_5'] #retrieves the 5 periods bull power = taperiods bull power
        self.df['eribear_5']=ta['eribear_5'] #retrieves = ta'] #retrieves
        # KER - Kaufman's efficiency ratio
        self.df['close_10_ker'] = ta['close_10_ker']
        # KAMA - Kaufman's Adaptive Moving Average
        self.df['close_10,2,30_kama'] = ta['close_10,2,30_kama']
        # Aroon Oscillator
        self.df['aroon_14'] = ta['aroon_14']
        # Awesome Oscillator
        self.df['ao'] = ta['ao']
        # Balance of Power
        self.df['bop'] = ta['bop']
        # Chande Momentum Oscillator
        self.df['cmo'] = ta['cmo']
        # oppock Curve
        self.df['coppock'] = ta['coppock']
        # Ichimoku Cloud
        self.df['ichimoku'] = ta['ichimoku']
        # Linear Regression Moving Average
        self.df['close_10_lrma'] = ta['close_10_lrma']
        # Correlation Trend Indicator
        self.df['cti'] = ta['cti']
        # the Gaussian Fisher Transform Price Reversals indicator
        self.df['ftr'] = ta['ftr']
        # Relative Vigor Index (RVGI)
        self.df['rvgi'] = ta['rvgi']
        self.df['rvgis'] = ta['rvgis']
        self.df['rvgi_5'] = ta['rvgi_5']
        self.df['rvgis_5'] = ta['rvgis_5']
        # Inertia Indicator
        self.df['inertia'] = ta['inertia']
        # Know Sure Thing (kst)
        self.df['kst'] = ta['kst']
        # Pretty Good Oscillator (PGO)
        self.df['pgo'] = ta['pgo']
        # Psychological Line (PSL)
        self.df['psl'] = ta['psl']
        # Percentage Volume Oscillator(PVO)
        self.df['pvo'] = ta['pvo']
        self.df['pvos'] = ta['pvos']
        self.df['pvoh'] = ta['pvoh']
        # Quantitative Qualitative Estimation(QQE)
        self.df['qqe'] = ta['qqe']
        self.df['qqel'] = ta['qqel']#retrieves the QQE long
        self.df['qqes'] = ta['qqes']#retrieves the QQE short

    def driver(self, data):
        self.set_data(data)
        self.cs_patterns()
        self.gen_ta()
        self.eight_trigram()
        self.log.info("="*100)
        return self.get_data()

if __name__ == '__main__':
    ta = TA(filepath=get_merger_final_output_file())
    ta.read_file()
    ta.gen_ta()
    ta.eight_trigram()
    ta.cs_patterns()
    ta.print_stats()
    ta.export_data('./files/ta_1d.csv')
