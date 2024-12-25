
import numpy as np
import scipy.stats as st 


def calc_confidence_interval(data, alpha=0.95):
    """
    Calculate confidence interval for a given data
    """
    a = 1.0 * alpha
    n = len(data)
    m, se = np.mean(data), st.sem(data)
    h = se * st.t.ppf((1 + a) / 2., n-1)
    return h

def open_close_strategy_gain(pred_df, direction, t=0.5):
    x= pred_df[pred_df.prob>t].drop_duplicates(subset=['date'])
    gain= np.sum(x.close - x.open)
    max_gain= np.sum(x.high - x.open)
    max_loss= np.sum(x.low - x.open)
    
    h= calc_confidence_interval(x.close - x.open) 

    if direction=='buy':
        return "{:.3f} [range=({:.3f}; {:.3f}), ci={:.3f}]".format(gain, max_loss, max_gain, h)
    return "{:.3f} [range=({:.3f}; {:.3f}), ci={:.3f}]".format(-gain,-max_loss, -max_gain, h)
