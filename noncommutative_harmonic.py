import numpy as np

# Irreducible representation matrices for S_3 (2-dimensional)
rho = [
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),   # identity
    np.array([[-0.5, 0.8660254], [0.8660254, 0.5]], dtype=complex),   # (12)
    np.array([[-0.5, -0.8660254], [-0.8660254, 0.5]], dtype=complex),  # (13)
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),   # (23)
    np.array([[-0.5, -0.8660254], [0.8660254, -0.5]], dtype=complex),  # (123)
    np.array([[-0.5, 0.8660254], [-0.8660254, -0.5]], dtype=complex)   # (132)
]

def quantise_returns_to_group_indices(returns, n_bins=6):
    """Map each return to a group element 0..5 via quantiles."""
    returns_clean = returns.dropna().values
    if len(returns_clean) == 0:
        return np.array([])
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    thresholds = np.quantile(returns_clean, quantiles)
    indices = np.digitize(returns_clean, thresholds)
    indices = np.clip(indices, 0, n_bins - 1)
    return indices.astype(int)

def noncommutative_fourier_coefficient(returns):
    """
    Fourier coefficient at the 2D irrep of S_3, weighted by return values.
    Returns a 2x2 complex matrix.
    """
    seq = quantise_returns_to_group_indices(returns)
    if len(seq) == 0:
        return np.zeros((2, 2), dtype=complex)
    returns_clean = returns.dropna().values
    if len(returns_clean) != len(seq):
        return np.zeros((2, 2), dtype=complex)
    total = np.zeros((2, 2), dtype=complex)
    for r_val, g_idx in zip(returns_clean, seq):
        total += r_val * rho[g_idx]
    norm = np.sum(np.abs(returns_clean))
    if norm > 0:
        total /= norm
    return total

def harmonic_score(returns):
    """Frobenius norm of the Fourier coefficient."""
    mat = noncommutative_fourier_coefficient(returns)
    return float(np.linalg.norm(mat, ord='fro'))

def noncommutative_harmonic_scores(returns_df):
    """Compute scores for all ETFs."""
    scores = {}
    for ticker in returns_df.columns:
        scores[ticker] = harmonic_score(returns_df[ticker])
    return scores
