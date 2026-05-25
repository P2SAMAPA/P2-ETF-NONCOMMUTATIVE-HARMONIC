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

def rolling_walkforward_backtest(returns_df, window_days, top_n=3):
    """
    For each possible day t (where we have window_days of history and at least one next day),
    compute scores on the window ending at t, pick top_n ETFs, and record their next day return.
    Returns: average next-day return of top_n ETFs across all signals.
    """
    n = len(returns_df)
    all_next_returns = []
    # We need at least window_days + 1 rows
    for t in range(window_days, n - 1):
        window = returns_df.iloc[t - window_days : t]
        next_day = returns_df.iloc[t]
        try:
            raw_scores = noncommutative_harmonic_scores(window)
        except Exception as e:
            continue
        norm_scores = normalize_scores(raw_scores)
        sorted_etfs = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [ticker for ticker, _ in sorted_etfs[:top_n]]
        next_returns = [next_day[t] for t in top_etfs if t in next_day.index]
        if next_returns:
            all_next_returns.extend(next_returns)
    if not all_next_returns:
        return None
    return float(np.mean(all_next_returns))

def run_for_window(returns, window_days):
    if len(returns) < window_days + 1:
        return None
    try:
        # For the "best window" by score (original method), we compute scores on the last window
        # and produce a dictionary of scores for display.
        ret_window = returns.iloc[-window_days:]
        raw_scores = noncommutative_harmonic_scores(ret_window)
        norm_scores = normalize_scores(raw_scores)
        sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [{"ticker": t, "harmonic_score_norm": s, "raw_score": raw_scores[t]} for t, s in sorted_norm[:config.TOP_N]]
    except Exception as e:
        print(f"    Error computing scores: {e}")
        return None

    # Backtest: average next-day return of top 3 over the entire history
    avg_return = rolling_walkforward_backtest(returns, window_days, top_n=config.TOP_N)

    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores_raw": raw_scores,
        "all_scores_norm": norm_scores,
        "backtest_avg_next_return": avg_return
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
        best_avg_ret = -np.inf
        best_window = None
        best_data = None
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, w)
            if out:
                all_window_results.append(out)
                if out["backtest_avg_next_return"] is not None and out["backtest_avg_next_return"] > best_avg_ret:
                    best_avg_ret = out["backtest_avg_next_return"]
                    best_window = w
                    best_data = out
            else:
                print(f"    Failed for window {w}")
        results["universes"][uni_name] = {
            "best_window_by_backtest": best_window,
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
