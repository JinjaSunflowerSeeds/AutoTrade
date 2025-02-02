from candlestick.patterns.candlestick_finder import CandlestickFinder


class BearishEngulfing(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]
<<<<<<< HEAD
        prev_prev_candle = self.data.iloc[idx + 2 * self.multi_coeff]
=======
>>>>>>> 5013666 (first commit)

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]
<<<<<<< HEAD

        # return (
        #     (prev_close > prev_open)  # Previous candle was bullish
        #     & (open > prev_close)  # Current open is higher than previous close
        #     & (close < prev_open)  # Current close is lower than previous open
        # )
        return (
            open >= prev_close > prev_open
            and open > close
            and prev_open >= close
            and open - close > prev_close - prev_open
        )

=======
        
        return (open >= prev_close > prev_open and
                open > close and
                prev_open >= close and 
                open - close > prev_close - prev_open)
        
>>>>>>> 5013666 (first commit)
        # return (prev_close > prev_open and
        #         0.3 > abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.1 and
        #         close < open and
        #         abs(close - open) / (high - low) >= 0.7 and
        #         prev_high < open and
        #         prev_low > close)
