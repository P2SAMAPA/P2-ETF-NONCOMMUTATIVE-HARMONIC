import os
import json
from datetime import datetime
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

def compute_forward_return(returns_df, signal_idx, top3_tickers, horizon=21):
    """
    Compute average cumulative return of top3 ETFs over next `horizon` days.
    returns_df: DataFrame with datetime index
    signal_idx: integer index of the signal day (last day of the window)
    """
    if signal_idx + horizon >= len(returns_df):
        return None
    future_returns = returns_df.iloc[signal_idx+1:signal_idx+horizon+1]
    cum_ret = (1 + future_returns).prod() - 1  # cumulative return per ETF
    top3_cum = [cum_ret[t] for t in top3_tickers if t in cum_ret.index]
    if not top3_cum:
        return None
    return float(np.mean(top3_cum))

def run_for_window(returns, window_days, full_returns_df):
    """Process one rolling window: compute scores and forward return."""
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    signal_date = ret_window.index[-1]
    signal_idx = full_returns_df.index.get_loc(signal_date)

    try:
        raw_scores = noncommutative_harmonic_scores(ret_window)
    except Exception as e:
        print(f"    Error: {e}")
        return None

    norm_scores = normalize_scores(raw_scores)
    sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    top3_tickers = [t for t, _ in sorted_norm[:config.TOP_N]]
    top3_norm = [s for _, s in sorted_norm[:config.TOP_N]]
    top3_raw = [raw_scores[t] for t in top3_tickers]

    # Forward return (21 days)
    forward_ret = compute_forward_return(full_returns_df, signal_idx, top3_tickers, horizon=21)

    top_etfs = [{"ticker": t, "harmonic_score_norm": s_norm, "raw_score": s_raw}
                for t, s_norm, s_raw in zip(top3_tickers, top3_norm, top3_raw)]

    return {
        "window": window_days,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "top_etfs": top_etfs,
        "all_scores_raw": raw_scores,
        "all_scores_norm": norm_scores,
        "forward_return_21d": forward_ret
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
            out = run_for_window(returns, w, returns)  # pass full returns for index
            if out:
                all_window_results.append(out)
                if out["forward_return_21d"] is not None and out["forward_return_21d"] > best_forward_ret:
                    best_forward_ret = out["forward_return_21d"]
                    best_window = w
                    best_data = out
            else:
                print(f"    Failed for window {w}")
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
