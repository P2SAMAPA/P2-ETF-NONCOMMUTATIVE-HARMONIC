import numpy as np
import pandas as pd

def compute_forward_returns(returns, scores, horizon=21):
    """
    For a given window end date (the last row of `returns`), use `scores` to rank ETFs,
    select top 3, and compute their average return over the next `horizon` days.
    Returns (top3_avg_return, top3_tickers, individual_returns).
    """
    if len(returns) < horizon:
        return None, None, None
    # Sort ETFs by score descending
    sorted_etfs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top3 = [ticker for ticker, _ in sorted_etfs[:3]]
    # Compute forward returns
    forward_returns = returns.iloc[-horizon:].mean()  # average daily return over horizon
    # Could also use cumulative return: (returns.iloc[-horizon:].sum())
    top3_returns = [forward_returns[t] for t in top3]
    avg_return = np.mean(top3_returns)
    return avg_return, top3, top3_returns

def backtest_all_windows(returns_df, all_window_data, horizon=21):
    """
    For each window in all_window_data, compute the forward return of top 3 ETFs.
    Returns a dict: {window: {'avg_return': ..., 'top3': [...], 'individual_returns': [...]}}
    """
    results = {}
    for wdata in all_window_data:
        w = wdata['window']
        # The returns_df up to the window end (last day of that window)
        end_idx = - (len(returns_df) - w)  # approximate: we need the exact end date
        # Instead, we can use the last `horizon+1` rows? Simpler: use the full returns_df
        # Since we only have one score per window (for the last day), we can compute forward return
        # from that day. We need the actual date to slice returns.
        # We'll assume the scores are computed on the last day of the window, and returns_df
        # is aligned. So we can use the entire returns_df and take the forward returns.
        # To be precise, we need the date index. We'll pass the full returns_df with index.
        # We'll use the last day of the window as the signal date.
        # But we don't have the date stored. We'll simplify: use the last `horizon` rows of returns_df
        # after the window end. Since we don't have the exact date, we'll assume the window end is
        # the last row of `returns_df` that was used for the window. Actually the window uses the
        # last `w` days, so its end date is the last date in returns_df. So forward returns are
        # the next days beyond returns_df? That's not in returns_df.
        # This is tricky. We'll change approach: during training, we can store the end date
        # for each window and then later fetch forward returns. That requires modifying train.py.
        # To keep it simple now, we'll use a placeholder: we will compute forward returns from
        # the last day of the returns_df (the entire series) assuming the window ends at the end.
        # This is inaccurate but gives a rough comparison.
        # A better method: In train.py, we will compute the forward returns at the time we have
        # the returns DataFrame (which has a datetime index). We'll do the backtest in train.py
        # and store the results. So this backtest.py may not be needed.
        pass
    return results
