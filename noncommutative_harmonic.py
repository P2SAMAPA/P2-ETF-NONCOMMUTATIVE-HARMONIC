import numpy as np

# Define the non‑abelian group S_3 (6 elements)
# We assign each element an index 0..5.
# Irreducible representation: 2‑dimensional (standard representation).
# We'll use the representation matrices ρ(g) for g in S_3.
# For simplicity, we hardcode them as 2x2 complex matrices (unitary).

# The six group elements: identity, transpositions (3), 3‑cycles (2)
# Represent S_3 as permutations of {1,2,3}
# ρ(identity) = [[1,0],[0,1]]
# ρ(12) = [[-1/2, sqrt(3)/2], [sqrt(3)/2, 1/2]]
# ρ(13) = [[-1/2, -sqrt(3)/2], [-sqrt(3)/2, 1/2]]
# ρ(23) = [[1,0],[0,-1]]
# ρ(123) = [[-1/2, -sqrt(3)/2], [sqrt(3)/2, -1/2]]
# ρ(132) = [[-1/2, sqrt(3)/2], [-sqrt(3)/2, -1/2]]

rho = [
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),   # id
    np.array([[-0.5, 0.8660254], [0.8660254, 0.5]], dtype=complex),   # (12)
    np.array([[-0.5, -0.8660254], [-0.8660254, 0.5]], dtype=complex),  # (13)
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),   # (23)
    np.array([[-0.5, -0.8660254], [0.8660254, -0.5]], dtype=complex),  # (123)
    np.array([[-0.5, 0.8660254], [-0.8660254, -0.5]], dtype=complex)   # (132)
]

def quantise_returns_to_group_elements(returns, n_bins=6):
    """
    Map each return value to a group element index 0..5 based on quantiles.
    """
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        return []
    # Use quantiles to assign bins (balanced)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    thresholds = np.quantile(returns_clean, quantiles)
    # For each return, find bin
    bins = np.digitize(returns_clean, thresholds)  # values 0..n_bins-1
    # Ensure within range (0..5)
    bins = np.clip(bins, 0, n_bins - 1)
    return bins.astype(int)

def noncommutative_fourier_coefficient(sequence, rho_matrices):
    """
    Compute the Fourier transform at the given representation:
    f̂(ρ) = Σ_g f(g) ρ(g), where f(g) is the frequency of element g.
    Returns a 2x2 complex matrix.
    """
    # Count occurrences of each group element
    counts = np.bincount(sequence, minlength=len(rho_matrices))
    # total number of elements
    n = len(sequence)
    if n == 0:
        return np.zeros((2,2), dtype=complex)
    # Normalise by n to get probability
    prob = counts / n
    # Sum prob[g] * ρ(g)
    result = np.zeros((2,2), dtype=complex)
    for g, p in enumerate(prob):
        if p > 0:
            result += p * rho_matrices[g]
    return result

def harmonic_score(returns):
    """
    Compute per‑ETF score = Frobenius norm of the noncommutative Fourier coefficient.
    """
    seq = quantise_returns_to_group_elements(returns)
    if len(seq) == 0:
        return 0.0
    fourier_mat = noncommutative_fourier_coefficient(seq, rho)
    # Frobenius norm = sqrt( sum |entry|^2 )
    score = np.linalg.norm(fourier_mat, ord='fro')
    return float(score)

def noncommutative_harmonic_scores(returns):
    """
    Apply harmonic score to each ETF in the returns DataFrame.
    """
    scores = {}
    for ticker in returns.columns:
        s = harmonic_score(returns[ticker])
        scores[ticker] = s
    return scores
