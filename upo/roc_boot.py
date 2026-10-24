import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def _bootstrap_indices(
    y: np.ndarray,
    n_boot: int,
    seed: Optional[int] = 42,
    stratified: bool = True,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    idxs = []
    for _ in range(n_boot):
        if stratified and pos.size > 0 and neg.size > 0:
            bpos = rng.choice(pos, size=pos.size, replace=True)
            bneg = rng.choice(neg, size=neg.size, replace=True)
            boot = np.concatenate([bpos, bneg])
        else:
            boot = rng.choice(len(y), size=len(y), replace=True)
        rng.shuffle(boot)
        idxs.append(boot)
    return idxs


def _auc_and_bands_boot(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    boot_idxs: List[np.ndarray],
    fpr_grid: np.ndarray,
    ci: float = 0.95,
) -> Tuple[float, Tuple[float, float], np.ndarray, np.ndarray, np.ndarray]:
    # Full-sample AUC and ROC
    auc_full = roc_auc_score(y_true, p_pred)
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    # Ensure end-points and deduplicate FPR taking max TPR at each FPR
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    ufpr, idx_start = np.unique(fpr, return_index=True)
    idx_end = np.r_[idx_start[1:], fpr.size]
    tpr_env = np.empty_like(ufpr, dtype=float)
    for i in range(ufpr.size):
        a, b = idx_start[i], idx_end[i]
        tpr_env[i] = np.max(tpr[a:b]) if b > a else tpr[a]
    tpr_full = np.interp(fpr_grid, ufpr, tpr_env)

    # Bootstrap on shared indices
    auc_boot, tpr_boot = [], []
    for bidx in boot_idxs:
        yb = (
            y_true.iloc[bidx] if isinstance(y_true, pd.Series) else y_true[bidx]
        )
        pb = (
            p_pred.iloc[bidx] if isinstance(p_pred, pd.Series) else p_pred[bidx]
        )
        if len(np.unique(yb)) < 2:
            continue
        auc_boot.append(roc_auc_score(yb, pb))
        f_b, t_b, _ = roc_curve(yb, pb)
        f_b = np.concatenate([[0.0], f_b, [1.0]])
        t_b = np.concatenate([[0.0], t_b, [1.0]])
        ord_b = np.argsort(f_b)
        f_b = f_b[ord_b]
        t_b = t_b[ord_b]
        uf_b, is_b = np.unique(f_b, return_index=True)
        ie_b = np.r_[is_b[1:], f_b.size]
        t_env_b = np.empty_like(uf_b, dtype=float)
        for i in range(uf_b.size):
            a, b = is_b[i], ie_b[i]
            t_env_b[i] = np.max(t_b[a:b]) if b > a else t_b[a]
        tpr_boot.append(np.interp(fpr_grid, uf_b, t_env_b))

    alpha = (1 - ci) / 2
    auc_lo, auc_hi = (
        np.quantile(auc_boot, [alpha, 1 - alpha])
        if len(auc_boot)
        else (np.nan, np.nan)
    )
    if len(tpr_boot):
        T = np.vstack(tpr_boot)
        tpr_lo = np.quantile(T, alpha, axis=0)
        tpr_hi = np.quantile(T, 1 - alpha, axis=0)
    else:
        tpr_lo = tpr_hi = np.full_like(fpr_grid, np.nan)
    # Return full-sample curve for plotting, not bootstrap mean
    return auc_full, (auc_lo, auc_hi), tpr_full, tpr_lo, tpr_hi


def plot_roc_with_bands(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: Optional[int] = 42,
    stratified: bool = True,
    ci: float = 0.95,
    fpr_grid: Optional[np.ndarray] = None,
    fill_alpha: float = 0.2,
    ax: Optional[plt.Axes] = None,
):
    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 201)
    if ax is None:
        fig, ax = plt.subplots()

    # One shared set of bootstrap indices for all models
    boot_idxs = _bootstrap_indices(
        y_true, n_boot=n_boot, seed=seed, stratified=stratified
    )

    info = {}
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    cmap = mpl.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, len(preds)))
    for c, (name, p) in zip(colors, preds.items()):
        auc_full, (lo, hi), tpr_curve, tpr_lo, tpr_hi = _auc_and_bands_boot(
            y_true, p, boot_idxs, fpr_grid, ci
        )
        ax.plot(
            fpr_grid,
            tpr_curve,
            linewidth=2,
            color=c,
            label=f"{name}: AUC={auc_full:.3f} [{lo:.3f}, {hi:.3f}]",
        )
        ax.fill_between(
            fpr_grid, tpr_lo, tpr_hi, alpha=fill_alpha, linewidth=0, color=c
        )

        info[name] = {
            "auc": auc_full,
            "ci_low": lo,
            "ci_high": hi,
            "fpr_grid": fpr_grid,
            "tpr_lo": tpr_lo,
            "tpr_hi": tpr_hi,
            "tpr_curve": tpr_curve,
        }

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC with shared-bootstrap {int(ci*100)}% bands + AUC CIs")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    return ax, info


def _ap_and_bands_boot(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    boot_idxs: List[np.ndarray],
    recall_grid: np.ndarray,
    ci: float = 0.95,
) -> Tuple[float, Tuple[float, float], np.ndarray, np.ndarray, np.ndarray]:
    """Compute AP and precision bands across a fixed recall grid using bootstrap.

    Returns
    -------
    ap_full : float
        Average precision on full sample.
    (ap_lo, ap_hi) : Tuple[float, float]
        Percentile bootstrap CI bounds for AP.
    prec_mean, prec_lo, prec_hi : np.ndarray
        Mean and CI envelopes of precision evaluated on `recall_grid`.
    """

    # Helpers to prepare PR curves for interpolation
    def _dedup_recall_take_max(
        recall: np.ndarray, precision: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Sort by recall ascending, then take max precision per recall value
        order = np.argsort(recall)
        r_sorted = recall[order]
        p_sorted = precision[order]
        uniq, idx_start = np.unique(r_sorted, return_index=True)
        idx_end = np.r_[idx_start[1:], r_sorted.size]
        max_prec = np.empty_like(uniq, dtype=float)
        for i in range(uniq.size):
            a, b = idx_start[i], idx_end[i]
            if b <= a:
                max_prec[i] = p_sorted[a]
            else:
                max_prec[i] = np.max(p_sorted[a:b])
        return uniq, max_prec

    def _prepare_pr_arrays(
        y: np.ndarray, p: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pr, rc, _ = precision_recall_curve(y, p)
        # Deduplicate and sort ascending by recall
        rc, pr = _dedup_recall_take_max(rc, pr)
        # Ensure coverage of [0, 1]
        if rc[0] > 0:
            rc = np.concatenate([[0.0], rc])
            pr = np.concatenate([[1.0], pr])
        if rc[-1] < 1:
            rc = np.concatenate([rc, [1.0]])
            pr = np.concatenate([pr, [pr[-1]]])
        # Enforce precision envelope (non-increasing with recall)
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        return rc, pr

    # Full-sample AP and PR curve (use this curve for the line)
    ap_full = average_precision_score(y_true, p_pred)
    rec_full, prec_full = _prepare_pr_arrays(y_true, p_pred)
    prec_full_interp = np.interp(recall_grid, rec_full, prec_full)

    # Bootstrap on shared indices
    ap_boot, prec_boot = [], []
    for bidx in boot_idxs:
        yb = (
            y_true.iloc[bidx] if isinstance(y_true, pd.Series) else y_true[bidx]
        )
        pb = (
            p_pred.iloc[bidx] if isinstance(p_pred, pd.Series) else p_pred[bidx]
        )
        if len(np.unique(yb)) < 2:
            continue
        ap_boot.append(average_precision_score(yb, pb))
        r_b, p_b = _prepare_pr_arrays(yb, pb)
        prec_boot.append(np.interp(recall_grid, r_b, p_b))

    alpha = (1 - ci) / 2
    ap_lo, ap_hi = (
        np.quantile(ap_boot, [alpha, 1 - alpha])
        if len(ap_boot)
        else (np.nan, np.nan)
    )
    if len(prec_boot):
        P = np.vstack(prec_boot)
        prec_lo = np.quantile(P, alpha, axis=0)
        prec_hi = np.quantile(P, 1 - alpha, axis=0)
        prec_mean = P.mean(axis=0)
    else:
        prec_lo = prec_hi = prec_mean = np.full_like(recall_grid, np.nan)
    # Return full-sample curve for plotting, not the bootstrap mean
    return ap_full, (ap_lo, ap_hi), prec_full_interp, prec_lo, prec_hi


def plot_prc_with_bands(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: Optional[int] = 42,
    stratified: bool = True,
    ci: float = 0.95,
    recall_grid: Optional[np.ndarray] = None,
    fill_alpha: float = 0.2,
    ax: Optional[plt.Axes] = None,
    y_pad: float = 0.02,
):
    """Plot Precision-Recall curves with bootstrap precision bands and AP CIs.

    Parameters mirror `plot_roc_with_bands`, but use recall on X and precision on Y.
    Bootstrapping uses a single shared set of resampled indices across models
    for paired uncertainty estimates.
    The y-axis lower bound is tightened to just below the minimum of the
    plotted PR curves and the chance (prevalence) line, with padding `y_pad`.
    """
    if recall_grid is None:
        recall_grid = np.linspace(0, 1, 201)
    if ax is None:
        fig, ax = plt.subplots()

    # One shared set of bootstrap indices for all models
    boot_idxs = _bootstrap_indices(
        y_true, n_boot=n_boot, seed=seed, stratified=stratified
    )

    # No-skill baseline at prevalence
    y_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    prevalence = np.mean(y_arr)
    ax.hlines(prevalence, 0, 1, linestyle="--", linewidth=1, label="Chance")

    info = {}
    y_min_observed = 1.0
    cmap = mpl.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, len(preds)))
    for c, (name, p) in zip(colors, preds.items()):
        ap_full, (lo, hi), prec_curve, prec_lo, prec_hi = _ap_and_bands_boot(
            y_true, p, boot_idxs, recall_grid, ci
        )
        ax.plot(
            recall_grid,
            prec_curve,
            linewidth=2,
            color=c,
            label=f"{name}: AP={ap_full:.3f} [{lo:.3f}, {hi:.3f}]",
        )
        ax.fill_between(
            recall_grid,
            prec_lo,
            prec_hi,
            alpha=fill_alpha,
            linewidth=0,
            color=c,
        )

        info[name] = {
            "ap": ap_full,
            "ci_low": lo,
            "ci_high": hi,
            "recall_grid": recall_grid,
            "prec_lo": prec_lo,
            "prec_hi": prec_hi,
            "prec_curve": prec_curve,
            "prevalence": prevalence,
        }

        # Track minimum y value from bands and center curve (robust to all-NaN bands)
        lo_min = np.nanmin(prec_lo) if np.any(np.isfinite(prec_lo)) else np.inf
        curve_min = (
            np.nanmin(prec_curve) if np.any(np.isfinite(prec_curve)) else np.inf
        )
        y_min_observed = min(y_min_observed, lo_min, curve_min)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PRC with shared-bootstrap {int(ci*100)}% bands + AP CIs")
    ax.set_xlim(0, 1)
    # Tighten lower y-limit to min(curves, prevalence), padded
    y_lower = max(0.0, min(prevalence, y_min_observed) - float(y_pad))
    if y_lower >= 1:
        y_lower = 0.0
    ax.set_ylim(y_lower, 1)
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    return ax, info
