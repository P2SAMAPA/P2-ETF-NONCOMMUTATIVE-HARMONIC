# Noncommutative Harmonic Analysis Engine for ETFs

Applies non‑commutative Fourier analysis on the permutation group S₃ to ETF returns. The per‑ETF score is the Frobenius norm of the Fourier coefficient at the 2‑dimensional irreducible representation – a measure of non‑abelian structure.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Returns quantised into 6 bins → functions on S₃
- Fourier transform at the standard 2‑dim irrep of S₃
- Score = Frobenius norm of the resulting complex matrix
- Best window automatically selected (largest raw norm)
- Two‑tab Streamlit dashboard (auto best + manual window selection)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-noncommutative-harmonic-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Run training: `python train.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. GitHub Actions runs daily.

## Interpretation

- Non‑commutative harmonic analysis generalises classical Fourier analysis.
- A high Fourier norm indicates that the ETF’s return pattern is “non‑abelian” – it cannot be reduced to a sum of commuting modes.
- This is a novel signal derived from abstract harmonic analysis, distinct from all classical methods.

## Requirements

See `requirements.txt`.
