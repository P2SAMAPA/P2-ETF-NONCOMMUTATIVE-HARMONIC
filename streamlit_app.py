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

tab1, tab2, tab3 = st.tabs(["📊 Best Window (Auto)", "🔍 Choose Window (Manual)", "📈 Backtest + ETFs"])

with tab1:
    st.header("🔊 Top ETFs by Noncommutative Fourier Norm (Auto Best Window)")
    with st.expander("📖 Interpretation", expanded=False):
        st.markdown("""
        - **Noncommutative harmonic analysis** generalises Fourier transform to functions on non‑abelian groups.
        - We map each ETF's return series to a function on the permutation group S₃ (6 elements) via quantiles, weighted by return magnitude.
        - The Fourier coefficient at the 2‑dimensional irreducible representation is a 2×2 complex matrix.
        - The **Frobenius norm** of this matrix measures non‑commutative structure.
        - Higher norm = stronger non‑abelian signal – potentially more complex, exploitable patterns.
        - The best window is automatically selected as the one with the highest **average next‑day return** from the walk‑forward backtest.
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
    st.header("📈 Walk‑Forward Backtest + ETF Selection")
    st.markdown("""
    For each rolling window length, we simulate a **daily walk‑forward**:
    - On each day, compute the Fourier norm using the trailing `window` days.
    - Rank ETFs and select the top 3.
    - Record the **next day's return** of those ETFs.
    - The backtest result is the **average of these next‑day returns** across all days.
    """)

    # Check if any backtest data exists
    has_backtest = False
    for uni_data in data["universes"].values():
        if uni_data and uni_data.get("all_windows"):
            for wd in uni_data["all_windows"]:
                if "backtest_avg_next_return" in wd and wd["backtest_avg_next_return"] is not None:
                    has_backtest = True
                    break
        if has_backtest:
            break

    if not has_backtest:
        st.info("⚠️ **Backtest data not available.**\n\n"
                "Please re‑run `train.py` with the latest version that includes walk‑forward backtest.")
    else:
        for universe_name, uni_data in data["universes"].items():
            if not uni_data or not uni_data.get("all_windows"):
                continue
            st.subheader(universe_name.replace("_", " ").title())
            
            # Select window for this universe
            available_windows = [wd["window"] for wd in uni_data["all_windows"]]
            sel_win = st.selectbox(f"Select window for {universe_name.replace('_', ' ').title()}", available_windows, key=f"backtest_win_{universe_name}")
            win_data = next((wd for wd in uni_data["all_windows"] if wd["window"] == sel_win), None)
            
            if win_data:
                # Display backtest metric
                avg_ret = win_data.get("backtest_avg_next_return")
                if avg_ret is not None:
                    st.metric("Average next-day return (backtest)", f"{avg_ret*100:.4f}%")
                else:
                    st.warning("Backtest data not available for this window.")
                
                # Display top 3 ETFs for this window (same as in Tab 2)
                top3 = win_data["top_etfs"]
                norm_scores = win_data["all_scores_norm"]
                raw_scores = win_data["all_scores_raw"]
                
                st.markdown("### Top 3 ETFs for this window")
                cols = st.columns(3)
                for idx, etf in enumerate(top3):
                    with cols[idx]:
                        st.markdown(f"""
                        <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                            <h3>{etf['ticker']}</h3>
                            <p>Fourier norm: {etf['harmonic_score_norm']:.3f}</p>
                            <p style="font-size:0.8rem;">raw: {etf['raw_score']:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with st.expander(f"Full ranking for {universe_name} (window {sel_win}d)"):
                    df_full = pd.DataFrame(list(norm_scores.items()), columns=["Ticker", "Normalized Fourier Norm"])
                    df_full["Raw Score"] = df_full["Ticker"].apply(lambda t: raw_scores[t])
                    df_full = df_full.sort_values("Normalized Fourier Norm", ascending=False)
                    st.dataframe(df_full, use_container_width=True)
            
            # Show table of all windows' backtest returns for reference
            rows = []
            for wd in uni_data["all_windows"]:
                avg_ret = wd.get("backtest_avg_next_return")
                rows.append({
                    "Window (days)": wd["window"],
                    "Avg next-day return (%)": f"{avg_ret*100:.4f}%" if avg_ret is not None else "N/A"
                })
            df_bt = pd.DataFrame(rows)
            st.markdown("### Backtest results for all windows")
            st.dataframe(df_bt, use_container_width=True)
            
            best_win = uni_data.get("best_window_by_backtest")
            if best_win is not None:
                best_avg = next((wd["backtest_avg_next_return"] for wd in uni_data["all_windows"] if wd["window"] == best_win), None)
                if best_avg is not None:
                    st.success(f"**Best window:** {best_win} days → Avg next-day return = {best_avg*100:.4f}%")
            st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.caption("Noncommutative Harmonic Analysis | Fourier transform on S₃ for ETF returns")
