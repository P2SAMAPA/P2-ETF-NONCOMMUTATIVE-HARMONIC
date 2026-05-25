import streamlit as st
import pandas as pd
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
    """Generic display function for a given window's data."""
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

# Create three tabs
tab1, tab2, tab3 = st.tabs(["📊 Best Window (Auto)", "🔍 Choose Window (Manual)", "📈 Backtest"])

with tab1:
    st.header("🔊 Top ETFs by Noncommutative Fourier Norm (Auto Best Window)")
    with st.expander("📖 Interpretation", expanded=False):
        st.markdown("""
        - **Noncommutative harmonic analysis** generalises Fourier transform to functions on non‑abelian groups.
        - We map each ETF's return series to a function on the permutation group S₃ (6 elements) via quantisation.
        - The Fourier coefficient at the 2‑dimensional irreducible representation is a 2×2 complex matrix.
        - The **Frobenius norm** of this matrix measures how much the distribution of returns aligns with non‑commutative structure.
        - Higher norm = stronger non‑abelian signal – potentially more complex, exploitable patterns.
        - The best window is automatically selected as the one with the highest forward 21‑day return (if available), otherwise the longest window.
        """)
    for universe_name, uni_data in data["universes"].items():
        if not uni_data or not uni_data.get("all_windows"):
            st.warning(f"No window data for {universe_name}")
            continue
        # Try to get best_window_data, else fallback to first window
        best_data = uni_data.get("best_window_data")
        if best_data is None and uni_data["all_windows"]:
            best_data = uni_data["all_windows"][0]  # fallback to first window
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
    st.header("📈 Backtest: Forward 21‑Day Return of Top 3 ETFs")
    st.markdown("""
    For each rolling window length, we rank ETFs by the raw Fourier norm, select the top 3,
    and compute their **average cumulative return** over the next 21 trading days.
    The table below shows the best window per universe (by highest forward return).
    """)
    # Check if any universe has backtest data
    has_backtest_data = False
    for universe_name, uni_data in data["universes"].items():
        if uni_data and uni_data.get("all_windows"):
            for wd in uni_data["all_windows"]:
                if "forward_return_21d" in wd and wd["forward_return_21d"] is not None:
                    has_backtest_data = True
                    break
        if has_backtest_data:
            break

    if not has_backtest_data:
        st.info("No backtest data found. Please re‑run `train.py` with the latest version that includes forward return computation, then upload the new results.")
    else:
        for universe_name, uni_data in data["universes"].items():
            if not uni_data or not uni_data.get("all_windows"):
                st.warning(f"No window data for {universe_name}")
                continue
            # Find best by forward return
            best_win_data = None
            best_fwd = -np.inf
            for wd in uni_data["all_windows"]:
                fwd = wd.get("forward_return_21d")
                if fwd is not None and fwd > best_fwd:
                    best_fwd = fwd
                    best_win_data = wd
            if best_win_data is None:
                st.warning(f"No backtest data for {universe_name}")
                continue
            best_win = best_win_data["window"]
            best_fwd_str = f"{best_fwd:.2%}" if best_fwd != -np.inf else "N/A"
            st.markdown(f"### {universe_name.replace('_', ' ').title()}")
            st.markdown(f"**Best window:** {best_win} days → **Avg 21d return of top 3 ETFs:** {best_fwd_str}")
            # Show all windows table
            rows = []
            for wd in uni_data["all_windows"]:
                fwd = wd.get("forward_return_21d")
                if fwd is None:
                    continue
                rows.append({
                    "Window (days)": wd["window"],
                    "Signal Date": wd.get("signal_date", "N/A"),
                    "21d Forward Return": f"{fwd:.2%}" if fwd is not None else "N/A"
                })
            if rows:
                df_backtest = pd.DataFrame(rows)
                st.dataframe(df_backtest, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Noncommutative Harmonic Analysis | Fourier transform on S₃ for ETF returns")
