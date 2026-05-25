import streamlit as st
import pandas as pd
import numpy as np
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Noncommutative Harmonic Analysis", layout="wide")

st.markdown("""
<style>
.hero-card {
    background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 0.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}
.hero-card h3 {
    font-size: 2rem;
    margin: 0;
    font-weight: bold;
}
.hero-card p {
    font-size: 1.2rem;
    margin: 0.5rem 0 0;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center;">🔊 Noncommutative Harmonic Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Fourier transform on the permutation group S₃ | Matrix norm as per‑ETF signal</p>', unsafe_allow_html=True)

st.sidebar.markdown("## 🧩 Noncommutative FT")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows evaluated:** {', '.join(map(str, config.WINDOWS))} days")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'noncommutative_harmonic_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']

def display_universe(universe_name, uni_data, window_data, window_label):
    top3 = window_data["top_etfs"]
    norm_scores = window_data["all_scores_norm"]
    raw_scores = window_data["all_scores_raw"]
    st.markdown(f'<h2 style="font-size: 1.8rem; margin-top: 1rem;">{universe_name.replace("_", " ").title()} <span style="font-size: 0.9rem; background: #e0e0e0; padding: 0.2rem 0.8rem; border-radius: 20px;">{window_label}</span></h2>', unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, etf in enumerate(top3):
        with cols[idx]:
            st.markdown(f"""
            <div class="hero-card">
                <h3>{etf['ticker']}</h3>
                <p>Fourier norm: {etf['harmonic_score_norm']:.3f}</p>
                <p style="font-size:0.9rem;">raw: {etf['raw_score']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
    with st.expander(f"Full ranking for {universe_name}"):
        df_full = pd.DataFrame(list(norm_scores.items()), columns=["Ticker", "Normalized Fourier Norm"])
        df_full["Raw Score"] = df_full["Ticker"].apply(lambda t: raw_scores[t])
        df_full = df_full.sort_values("Normalized Fourier Norm", ascending=False)
        st.dataframe(df_full, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["📊 Best Window (Auto)", "🔍 Choose Window (Manual)", "📈 Backtest (Per‑ETF)"])

with tab1:
    st.header("🔊 Top ETFs by Noncommutative Fourier Norm (Auto Best Window)")
    with st.expander("📖 Interpretation", expanded=False):
        st.markdown("""
        - **Noncommutative harmonic analysis** generalises Fourier transform to functions on non‑abelian groups.
        - We map each ETF's return series to a function on the permutation group S₃ (6 elements) via quantiles, weighted by return magnitude.
        - The Fourier coefficient at the 2‑dimensional irreducible representation is a 2×2 complex matrix.
        - The **Frobenius norm** of this matrix measures non‑commutative structure.
        - Higher norm = stronger non‑abelian signal – potentially more complex, exploitable patterns.
        - The best window is automatically selected as the one with the highest **average backtest return** (average across all ETFs' per‑ETF averages).
        """)
    for universe_name, uni_data in data["universes"].items():
        if not uni_data or not uni_data.get("all_windows"):
            st.warning(f"No window data for {universe_name}")
            continue
        best_data = uni_data.get("best_window_data")
        if best_data is None and uni_data["all_windows"]:
            best_data = uni_data["all_windows"][-1]  # fallback to longest window
            win_label = f"window {best_data['window']}d (fallback)"
        elif best_data:
            win_label = f"best window {best_data['window']}d"
        else:
            st.warning(f"No data for {universe_name}")
            continue
        display_universe(universe_name, uni_data, best_data, win_label)

with tab2:
    st.header("🔍 Manual Window Selection")
    st.markdown("Choose a rolling window to inspect the Fourier norms per ETF.")
    for universe_name, uni_data in data["universes"].items():
        if not uni_data or not uni_data.get("all_windows"):
            st.warning(f"No window data for {universe_name}")
            continue
        available_windows = [wd["window"] for wd in uni_data["all_windows"]]
        sel_win = st.selectbox(f"Window for {universe_name.replace('_', ' ').title()}", available_windows, key=f"manual_{universe_name}")
        win_data = next((wd for wd in uni_data["all_windows"] if wd["window"] == sel_win), None)
        if win_data:
            display_universe(universe_name, uni_data, win_data, f"window {sel_win}d")
        else:
            st.warning("No data for selected window.")

with tab3:
    st.header("📈 Walk‑Forward Backtest (Per‑ETF Average Next‑Day Return)")
    st.markdown("""
    **Method:**  
    For each window length, we simulate a daily walk‑forward:
    - On each day, compute the Fourier norm for all ETFs (trailing window).
    - Select the top 3 ETFs by Fourier norm.
    - Record the **next day's return** for each selected ETF individually.
    - For each ETF, we compute the **average** of those next‑day returns across all days it was selected.
    
    **The table below shows, for each universe and window, the top 3 ETFs by that average backtest return.**  
    (This tells you which specific ETFs have historically performed well when the signal selected them.)
    """)
    
    for universe_name, uni_data in data["universes"].items():
        if not uni_data or not uni_data.get("all_windows"):
            continue
        
        st.subheader(universe_name.replace("_", " ").title())
        
        rows = []
        for wd in uni_data["all_windows"]:
            w = wd["window"]
            backtest_dict = wd.get("backtest_per_etf_avg_return", {})
            if not backtest_dict:
                continue
            # Sort by backtest average (descending) and take top 3
            sorted_by_backtest = sorted(backtest_dict.items(), key=lambda x: x[1], reverse=True)[:config.TOP_N]
            for ticker, avg_ret in sorted_by_backtest:
                rows.append({
                    "Window (days)": w,
                    "Ticker": ticker,
                    "Avg next‑day return (%)": f"{avg_ret*100:.4f}%"
                })
        if rows:
            df_backtest = pd.DataFrame(rows)
            st.dataframe(df_backtest, use_container_width=True)
        else:
            st.info("No backtest data available for this universe.")
        
        # Optional: show full detail for a selected window
        with st.expander(f"See full Fourier‑based rankings for a specific window (original signal)"):
            available_windows = [wd["window"] for wd in uni_data["all_windows"]]
            sel_win = st.selectbox(f"Select window for {universe_name.replace('_', ' ').title()}", available_windows, key=f"backtest_detail_{universe_name}")
            win_data = next((wd for wd in uni_data["all_windows"] if wd["window"] == sel_win), None)
            if win_data:
                st.markdown("**Top 3 by Fourier norm (original signal)**")
                top3_fourier = win_data["top_etfs"]
                cols = st.columns(3)
                for idx, etf in enumerate(top3_fourier):
                    with cols[idx]:
                        st.markdown(f"""
                        <div style="background: #f0f2f6; padding: 0.5rem; border-radius: 0.5rem; text-align: center;">
                            <b>{etf['ticker']}</b><br>
                            Fourier norm: {etf['harmonic_score_norm']:.3f}<br>
                            raw: {etf['raw_score']:.4f}
                        </div>
                        """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Noncommutative Harmonic Analysis | Fourier transform on S₃ for ETF returns")
