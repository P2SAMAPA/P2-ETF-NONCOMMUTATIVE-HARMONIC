import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
import config
import data_manager as dm
from noncommutative_harmonic import noncommutative_harmonic_scores

def normalize_scores(score_dict):
    scores = np.array(list(score_dict.values()))
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {k: 0.0 for k in score_dict}
    norm = (scores - min_s) / (max_s - min_s)
    return {ticker: float(norm[i]) for i, ticker in enumerate(score_dict.keys())}

def compute_forward_return(returns_df, signal_date, top3_tickers, horizon=21):
    """
    Given the full returns DataFrame (index=datetime), a signal date, and list of top3 tickers,
    compute the average cumulative return over the next `horizon` trading days.
    Returns float (average cumulative return).
    """
    # Find the position of signal_date in the index
    if signal_date not in returns_df.index:
        # If exact date not found, take the next available date after signal_date
        idx = returns_df.index.searchsorted(signal_date)
        if idx >= len(returns_df):
            return np.nan
        signal_date = returns_df.index[idx]
    signal_idx = returns_df.index.get_loc(signal_date)
    end_idx = min(signal_idx + horizon, len(returns_df) - 1)
    if end_idx <= signal_idx:
        return np.nan
    # Cumulative returns from signal_date+1 to end_idx
    future_returns = returns_df.iloc[signal_idx+1:end_idx+1]
    if future_returns.empty:
        return np.nan
    cum_returns = (1 + future_returns).prod() - 1  # per ETF
    top3_returns = [cum_returns[t] for t in top3_tickers if t in cum_returns.index]
    if not top3_returns:
        return np.nan
    return np.mean(top3_returns)

def run_for_window(returns, window_days, returns_full):
    """
    Compute scores and forward returns for one window.
    """
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    signal_date = ret_window.index[-1]  # last day of window
    try:
        raw_scores = noncommutative_harmonic_scores(ret_window)
    except Exception as e:
        print(f"    Error: {e}")
        return None
    norm_scores = normalize_scores(raw_scores)
    sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    top3_tickers = [t for t, _ in sorted_norm[:config.TOP_N]]
    top3_norm_scores = [s for _, s in sorted_norm[:config.TOP_N]]
    top3_raw_scores = [raw_scores[t] for t in top3_tickers]

    # Forward return (backtest)
    forward_ret = compute_forward_return(returns_full, signal_date, top3_tickers, horizon=21)

    top_etfs = [{"ticker": t, "harmonic_score_norm": s_norm, "raw_score": s_raw}
                for t, s_norm, s_raw in zip(top3_tickers, top3_norm_scores, top3_raw_scores)]
    return {
        "window": window_days,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "top_etfs": top_etfs,
        "all_scores_raw": raw_scores,
        "all_scores_norm": norm_scores,
        "forward_return_21d": forward_ret if not np.isnan(forward_ret) else None
    }

def main():
    print("Loading master data...")
    dm.load_master_data()
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print("  No data -> skipping")
            continue
        all_window_results = []
        best_forward_ret = -np.inf
        best_window = None
        best_data = None
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, w, returns)  # pass full returns for forward calc
            if out:
                all_window_results.append(out)
                if out["forward_return_21d"] is not None and out["forward_return_21d"] > best_forward_ret:
                    best_forward_ret = out["forward_return_21d"]
                    best_window = w
                    best_data = out
            else:
                print(f"    Failed for window {w}")
        # Also keep best by raw score magnitude (optional)
        # We'll store both
        results["universes"][uni_name] = {
            "best_window_by_forward_return": best_window,
            "best_window_data": best_data,
            "all_windows": all_window_results
        }
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/noncommutative_harmonic_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
