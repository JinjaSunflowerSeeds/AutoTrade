import sys
import pandas as pd

from talib import  CDLDOJI, CDLENGULFING, CDLEVENINGSTAR, CDLHAMMER

sys.path.append("./")
from utils.config_reader import get_merger_final_output_file
from utils.logging import LoggerUtility

logger = LoggerUtility.setup_logger(__name__)

class PriceActionFeatures:
    def __init__(self, filepath: str):
        """
        Initialize with a DataFrame containing OHLCV data.
        :param df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
        """
        self.features = pd.DataFrame()
        self.df = pd.read_csv(filepath)
        self.df["date"] = pd.to_datetime(self.df["date"])

    def add_candlestick_patterns(self):
        """
        Add candlestick pattern features using TA-Lib.
        """

        patterns = {
            "Hammer": CDLHAMMER,
            "Doji": CDLDOJI,
            "EveningStar": CDLEVENINGSTAR,
            "Engulfing": CDLENGULFING,
        }
        for name, func in patterns.items():
            self.df[f"pattern_{name}"] = func(
                self.df["open"], self.df["high"], self.df["low"], self.df["close"]
            )
        self.df[['open', 'high', 'low', 'close', 'volume','pattern_Hammer','pattern_Doji','pattern_EveningStar','pattern_Engulfing']].to_csv("candlestick_patterns.csv")
        print(self.df)

    def add_volume_features(self):
        """
        Add volume-related features.
        """
        self.features["volume_change"] = self.df["volume"].pct_change()
        self.features["volume_moving_avg"] = self.df["volume"].rolling(10).mean()

    def add_support_resistance(self):
        """
        Add support and resistance levels based on local minima and maxima.
        """
        self.features["support"] = self.df["low"].rolling(window=20).min()
        self.features["resistance"] = self.df["high"].rolling(window=20).max()

    def add_price_change(self):
        """
        Add features based on price changes.
        """
        self.features["price_change_pct"] = self.df["close"].pct_change()
        # compared to past 10 periods
        self.features["high_low_diff"] = self.df["high"] - self.df["low"]

    def add_bollinger_bands(self, window=20):
        """
        Add Bollinger Bands as features.
        """
        # we need to identify actions that are anomaly
        sma = self.df["close"].rolling(window).mean()
        std = self.df["close"].rolling(window).std()
        self.features["bollinger_upper"] = sma + (2 * std)
        self.features["bollinger_lower"] = sma - (2 * std)

    def generate_features(self):
        """
        Generate all features and return as a DataFrame.
        """
        self.add_candlestick_patterns()
        self.add_volume_features()
        self.add_support_resistance()
        self.add_price_change()
        self.add_bollinger_bands()
        # Drop NaN rows caused by rolling operations
        self.features.dropna(inplace=True)
        return self.features


# Example usage:
if __name__ == "__main__":
    filepath = get_merger_final_output_file()
    # Generate features
    feature_extractor = PriceActionFeatures(filepath)
    features = feature_extractor.generate_features()
    print(features)
