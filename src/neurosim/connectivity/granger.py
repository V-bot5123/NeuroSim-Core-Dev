"""
neurosim.granger
================
Statistical edge validation for directed Effective Connectivity.

The Edge Significance Problem
------------------------------
Estimating the Effective Connectivity (EC) matrix via MVAR regression
recovers a directed A-matrix, but does not answer a critical question:
which directed edges are statistically meaningful and which are fitting
artefacts of regularisation?

Two failure modes arise in practice:

1. **Spurious Functional Correlation:** High Pearson correlation between
   two regions (|FC[i,j]| > 0.5) can arise from shared upstream drive
   rather than direct causal projection. Treating FC as effective
   connectivity assigns control energy to non-causal pathways — the
   "Teleportation Error" described in neurosim.connectivity.

2. **Hidden Directed Influence:** A causal projection from region j to i
   may produce low pairwise correlation if the signal is weak relative to
   noise, yet the directed F-test can still detect it via residual
   variance reduction in the MVAR model.

NeuroSim Solution: Granger F-test + FDR Validation
----------------------------------------------------
We implement the pairwise Granger causality F-test (Granger, 1969;
Seth et al., 2015) to assign statistical confidence to each directed
edge in the MVAR-estimated EC matrix:

    H0: the lags of region j do not reduce prediction error for region i.
    H1: including j's lags significantly reduces the residual SS.

The test statistic is:

    F = ((RSS_restricted - RSS_full) / p) / (RSS_full / (T - K - 1))

where p is the lag order, K = N * p is the total number of predictors,
and T is the number of timepoints. Under H0, F ~ F(p, T-K-1).

At clinical parcellation resolutions (N = 200–1000 regions), a single
run of granger_causality_matrix() performs N*(N-1) simultaneous F-tests
(up to ~999,000 at N=1000). Without multiple-testing correction, the
expected number of false positives at alpha=0.05 is 0.05 * N*(N-1).
Benjamini-Hochberg FDR control (Benjamini & Hochberg, 1995) is applied
by default to bound the false discovery rate at alpha across all tests.

References
----------
Granger, C. W. J. (1969). Investigating causal relations by econometric
    models and cross-spectral methods. Econometrica, 37(3), 424–438.
Seth, A. K., Barrett, A. B., & Barnett, L. (2015). Granger causality
    analysis in neuroscience and neuroimaging. Journal of Neuroscience,
    35(8), 3293–3297.
Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
    rate: a practical and powerful approach to multiple testing.
    Journal of the Royal Statistical Society B, 57(1), 289–300.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import f as f_dist


# Granger causality F-test matrix

def granger_causality_matrix(
    timeseries: NDArray,
    order: int = 1,
    alpha: float = 0.05,
    fdr_correction: bool = True,
) -> Dict:
    """Test directed Granger causality between all region pairs.

    For each ordered pair (j→i), fits a full MVAR model including all
    regions' lags, then a restricted model excluding region j's lags.
    The F-statistic tests whether j's history significantly reduces
    prediction error for region i beyond what all other regions explain.

    At clinical parcellation resolutions (N ≥ 200), the N*(N-1) simultaneous
    F-tests require multiple-testing correction. Benjamini-Hochberg FDR
    control is applied by default via ``fdr_correction=True``.

    Parameters
    ----------
    timeseries     : (N, T) ndarray
        BOLD time series. N regions × T timepoints.
    order          : int
        MVAR lag order p (default 1 TR).
    alpha          : float
        Significance level. Interpreted as FDR rate when
        ``fdr_correction=True``, as per-test alpha otherwise (default 0.05).
    fdr_correction : bool
        If True, apply Benjamini-Hochberg FDR correction across all
        N*(N-1) simultaneous tests before thresholding (default True).
        Set to False only for exploratory analysis on small N.

    Returns
    -------
    dict with keys:

    - ``F_matrix``       : (N, N) ndarray — F-statistics. F[i, j] tests j→i.
    - ``p_matrix``       : (N, N) ndarray — raw p-values. p[i, j] tests j→i.
    - ``significant``    : (N, N) bool ndarray — edges surviving the
                           threshold (FDR-corrected if fdr_correction=True).
    - ``n_causal_edges`` : int — number of significant directed edges.
    - ``alpha``          : float — significance level used.
    - ``order``          : int — lag order used.
    - ``df1``            : int — numerator degrees of freedom (= order).
    - ``df2``            : int — denominator degrees of freedom.
    - ``fdr_correction`` : bool — whether BH FDR correction was applied.

    Raises
    ------
    ValueError
        If timeseries is not 2D, has insufficient timepoints for the
        requested order, or contains non-finite values.
    """
    _validate_timeseries(timeseries, order)

    N, T = timeseries.shape
    X, Y = _build_lagged_matrix(timeseries, order)

    T_eff = X.shape[0]
    K = X.shape[1]
    df1 = order
    df2 = T_eff - K - 1

    if df2 <= 0:
        raise ValueError(
            f"Insufficient degrees of freedom for Granger F-test. "
            f"Need T - N*order - 1 > 0. Got df2={df2}. "
            f"Reduce order or acquire more TRs."
        )

    F_matrix = np.zeros((N, N))
    p_matrix = np.ones((N, N))

    # Pre-compute full-model RSS for each target node
    rss_full = np.array([_ols_rss(X, Y[:, i])[1] for i in range(N)])

    for i in range(N):
        y_i = Y[:, i]

        for j in range(N):
            if i == j:
                continue

            # Restricted model: remove all lags belonging to region j
            restricted_cols = [c for c in range(K) if (c % N) != j]
            _, rss_restr = _ols_rss(X[:, restricted_cols], y_i)

            delta = max(rss_restr - rss_full[i], 0.0)

            if rss_full[i] < 1e-16:
                warnings.warn(
                    f"Node {i} has near-zero full-model residuals. "
                    f"F-test for edge ({j}→{i}) may be unreliable.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            F = (delta / df1) / (rss_full[i] / df2)
            F_matrix[i, j] = F
            p_matrix[i, j] = float(1.0 - f_dist.cdf(F, df1, df2))

    # Multiple-testing correction across all N*(N-1) off-diagonal tests
    if fdr_correction:
        off_diag = ~np.eye(N, dtype=bool)
        rejected = _benjamini_hochberg(p_matrix[off_diag], alpha)
        significant = np.zeros((N, N), dtype=bool)
        significant[off_diag] = rejected
    else:
        significant = p_matrix < alpha
        np.fill_diagonal(significant, False)

    return {
        "F_matrix": F_matrix,
        "p_matrix": p_matrix,
        "significant": significant,
        "n_causal_edges": int(significant.sum()),
        "alpha": alpha,
        "order": order,
        "df1": df1,
        "df2": df2,
        "fdr_correction": fdr_correction,
    }


# Spurious FC vs hidden causality detector

def causality_vs_correlation_summary(
    timeseries: NDArray,
    order: int = 1,
    alpha: float = 0.05,
    fc_spurious_threshold: float = 0.5,
    fc_hidden_threshold: float = 0.3,
) -> Dict:
    """Identify where Functional Connectivity and Granger causality disagree.

    Two diagnostically important discordance patterns are flagged:

    - **Spurious FC:** |FC[i,j]| > fc_spurious_threshold but the Granger
      F-test is non-significant. Suggests shared upstream drive rather
      than a direct causal projection.

    - **Hidden causality:** Granger F-test is significant but
      |FC[i,j]| < fc_hidden_threshold. Suggests a weak directed signal
      that pairwise correlation fails to detect.

    Parameters
    ----------
    timeseries            : (N, T) ndarray — BOLD time series.
    order                 : int   — MVAR lag order (default 1).
    alpha                 : float — Granger significance level (default 0.05).
    fc_spurious_threshold : float — FC magnitude above which a non-causal
                            edge is flagged as spurious (default 0.5).
    fc_hidden_threshold   : float — FC magnitude below which a significant
                            Granger edge is flagged as hidden (default 0.3).

    Returns
    -------
    dict with keys:

    - ``fc_matrix``             : (N, N) ndarray — Pearson FC matrix.
    - ``granger_result``        : dict — Full output of granger_causality_matrix().
    - ``spurious_fc_pairs``     : list of (i, j) — High-FC, non-causal pairs.
    - ``hidden_causality_pairs``: list of (i, j) — Low-FC, significant-Granger pairs.
    - ``n_spurious``            : int — Count of spurious FC pairs.
    - ``n_hidden``              : int — Count of hidden causal edges.
    """
    _validate_timeseries(timeseries, order)

    N = timeseries.shape[0]
    FC = np.corrcoef(timeseries)
    granger = granger_causality_matrix(timeseries, order=order, alpha=alpha)
    significant = granger["significant"]

    spurious: List[Tuple[int, int]] = []
    hidden: List[Tuple[int, int]] = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            fc_ij = abs(FC[i, j])
            if fc_ij > fc_spurious_threshold and not significant[i, j]:
                spurious.append((i, j))
            if significant[i, j] and fc_ij < fc_hidden_threshold:
                hidden.append((i, j))

    return {
        "fc_matrix": FC,
        "granger_result": granger,
        "spurious_fc_pairs": spurious,
        "hidden_causality_pairs": hidden,
        "n_spurious": len(spurious),
        "n_hidden": len(hidden),
    }


# Private helpers

def _benjamini_hochberg(p_values: NDArray, alpha: float) -> NDArray:
    """Apply Benjamini-Hochberg FDR correction to a flat array of p-values.

    Controls the expected proportion of false discoveries among all
    rejected hypotheses at level alpha. Returns a boolean array of the
    same length: True where H0 is rejected after FDR control.

    Benjamini & Hochberg (1995), JRSS-B, 57(1), 289–300.
    """
    m = len(p_values)
    if m == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha

    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)

    k = int(np.where(below)[0].max())
    rejected = np.zeros(m, dtype=bool)
    rejected[order[: k + 1]] = True
    return rejected


def _build_lagged_matrix(
    timeseries: NDArray,
    order: int,
) -> Tuple[NDArray, NDArray]:
    """Construct MVAR design matrix X and response matrix Y.

    Returns X of shape (T-order, N*order) and Y of shape (T-order, N).
    """
    N, T = timeseries.shape
    n_samples = T - order

    X = np.zeros((n_samples, N * order))
    for lag in range(1, order + 1):
        col_start = (lag - 1) * N
        col_end = lag * N
        X[:, col_start:col_end] = timeseries[:, order - lag: T - lag].T

    Y = timeseries[:, order:].T
    return X, Y


def _ols_rss(X: NDArray, y: NDArray) -> Tuple[NDArray, float]:
    """Fit OLS via least squares and return (coefficients, residual SS).

    Uses lstsq rather than normal equations for numerical stability
    when X is near-collinear (common at high parcellation resolutions).
    """
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    return beta, float(np.dot(residuals, residuals))


def _validate_timeseries(timeseries: NDArray, order: int) -> None:
    """Raise informative errors for malformed inputs."""
    if not isinstance(timeseries, np.ndarray):
        raise ValueError("timeseries must be a numpy array.")
    if timeseries.ndim != 2:
        raise ValueError(
            f"timeseries must be a 2D array of shape (N_nodes, T_timepoints). "
            f"Got shape: {timeseries.shape}."
        )
    if order < 1:
        raise ValueError(f"order must be >= 1. Got order={order}.")
    N, T = timeseries.shape
    min_T = N * order + 2
    if T <= min_T:
        raise ValueError(
            f"Insufficient time points for Granger causality F-test. "
            f"Need T > N*order + 1 = {min_T}. Got T={T}, N={N}, order={order}. "
            f"Consider reducing parcellation resolution or acquiring more TRs."
        )
    if not np.isfinite(timeseries).all():
        raise ValueError("timeseries contains NaN or Inf values.")
