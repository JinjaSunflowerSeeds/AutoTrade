class RsiLabeling:
 
         
    def rsi_strategy(self, lookahead=5, lb=40, ub=60, gain=0.5):
        # if vol is p30 and more, and rsi in range and lookahead delta is favorable
        self.df[["label", "buy", "sell"]] = 0
        self.df['SMA_200'] = self.df['close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['close'].rolling(window=9).mean()
        for t in range(len(self.df)):
            if self.df.iloc[t].volume < self.df.iloc[:t].volume.quantile(0.3):
                continue
            
            rsi_14=self.df.iloc[t].rsi_14
            close_series = self.df.iloc[t : t + lookahead].close.values
            
            sma_50 = self.df.at[t, 'SMA_50']
            sma_200 = self.df.at[t, 'SMA_200']
            
            b_delta = 100 * (close_series.max() - close_series[0]) / close_series[0]
            s_delta = 100 * (close_series.min() - close_series[0]) / close_series[0]
            
            if rsi_14 < lb and b_delta > gain:  # close_series.is_monotonic_increasing:
                if close_series[0] > sma_200 and close_series[0] > sma_50:
                    self.df.loc[t, ["label", "buy"]] = 1
                # t += lookahead  # Skip to t+5 in next iteration
            elif (
                rsi_14 > ub and s_delta < -gain
            ):  # close_series.is_monotonic_decreasing:
                if close_series[0] < sma_200 and close_series[0] < sma_50:
                    self.df.loc[t, ["label", "sell"]] = 1
                # t += lookahead
        print(
            f"Buy label=1: {100.0 * self.df.label.mean():.2f}% trades={(self.df['label'] == 1).sum()}"
        )


# has touched 32 times the vwap and always bounced, on the third trend will be the same
#  fin the thresholds based on the probability distribution (after what number with 2std prob it has gone up) 
# if hits high of the day and rsi oversold, it will pull back
