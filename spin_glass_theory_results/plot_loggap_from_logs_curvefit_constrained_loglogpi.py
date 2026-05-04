#!/usr/bin/env python3
"""
Extract log(1-Pi_k) vs k from *.logs.json files emitted by spin-glass-theory_multigpu.py
and plot log(-log(Pi_k)) vs log(k) for each h value.

Usage:
  python plot_loggap_from_logs_curvefit_constrained_loglogpi.py \
      --logs ./spin-glass-theory-data-N24-disorder1024-m1-curvefit.png.logs.json \
      --out loggap_vs_logk.png

You can pass multiple --logs; curves for the same h across files will be merged.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


H_HEADER = re.compile(r"^\s*h\s*=\s*([0-9.eE+-]+)")
K_LINE = re.compile(r"^\s*k=\s*(\d+):\s*log\(1-Pi_k\)\s*=\s*([-\d.eE+]+)\s*[±+/-]\s*([-\d.eE+]+)")
PARAMS_LINE = re.compile(r"Parameters:\s*N=([-\d.eE+]+),\s*beta=([-\d.eE+]+),\s*j0=([-\d.eE+]+),\s*m=([-\d.eE+]+)")
TEACHER_M_LINE = re.compile(r"Teacher m_l .*:\s*([-\d.eE+]+)")
LOG_CM_LINE = re.compile(r"Value of log_Cm:\s*([-\d.eE+]+)")
LAMBDA_MEAN_LINE = re.compile(r"Lambda mean:\s*([-\d.eE+]+)")


def model_log_corrected(x: np.ndarray, a: float, b: float, l: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return -a * np.log(x) - 0.5 * (l ** 2) * x + b


def model_log_theory_from_logk(logk: np.ndarray, nu_hat: float, lambda_hat: float, log_cm: float) -> np.ndarray:
    logk = np.asarray(logk, dtype=np.float64)
    return -nu_hat * logk - 0.5 * (lambda_hat ** 2) * np.exp(logk) + log_cm


def _logneglog_pi_from_loggap(log_gap: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """Convert log(1-Pi_k) to log(-log(Pi_k)) with numeric safety."""
    lg = np.asarray(log_gap, dtype=np.longdouble)
    max_log_gap = np.log1p(-eps)
    lg = np.minimum(lg, max_log_gap)
    # log(Pi_k) = log(1 - exp(log_gap)) computed stably via log1p
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        log_pi = np.log1p(-np.exp(lg))
    # Avoid log(0) or log(1) effects at machine precision
    log_pi = np.clip(log_pi, np.log(eps), np.log1p(-eps))
    return np.log(-log_pi).astype(np.float64)


def model_logneglog_pi_from_logk(logk: np.ndarray, nu_hat: float, lambda_hat: float, log_cm: float) -> np.ndarray:
    log_gap = model_log_theory_from_logk(logk, nu_hat, lambda_hat, log_cm)
    return _logneglog_pi_from_loggap(log_gap)


def fit_monotone_params(
    h_values: List[float],
    data_by_h: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[float, Tuple[float, float, float]] | None:
    """Joint fit with nu_hat(h) constrained to be non-decreasing."""
    n = len(h_values)
    if n == 0:
        return {}

    nu_guess = np.zeros(n, dtype=np.float64)
    logcm_guess = np.zeros(n, dtype=np.float64)
    lambda_guess = np.full(n, 1.0, dtype=np.float64)

    for i, h in enumerate(h_values):
        logk, y, _ = data_by_h[h]
        slope, intercept = np.polyfit(logk, y, deg=1)
        nu_guess[i] = max(0.0, -slope)
        logcm_guess[i] = intercept

    nu_guess = np.maximum.accumulate(nu_guess)
    x0 = np.concatenate([nu_guess, lambda_guess, logcm_guess])

    bounds = [(None, None)] * n + [(0.0, None)] * n + [(None, None)] * n
    constraints = [{"type": "ineq", "fun": lambda x, i=i: x[i + 1] - x[i]} for i in range(n - 1)]
    constraints += [{"type": "ineq", "fun": lambda x, i=i: x[n + i + 1] - x[n + i]} for i in range(n - 1)]
    if h_values[0] == 0.0:
        bounds[n] = (0.0, 0.0)

    def objective(x: np.ndarray) -> float:
        nu = x[:n]
        lam = x[n:2 * n]
        logcm = x[2 * n:]
        total = 0.0
        for i, h in enumerate(h_values):
            logk, y, std = data_by_h[h]
            sigma = np.where(std > 0, std, 1.0)
            pred = model_logneglog_pi_from_logk(logk, nu[i], lam[i], logcm[i])
            resid = (y - pred) / sigma
            total += float(np.sum(resid ** 2))
        return total

    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        return None

    nu = result.x[:n]
    lam = result.x[n:2 * n]
    logcm = result.x[2 * n:]
    return {h: (float(nu[i]), float(lam[i]), float(logcm[i])) for i, h in enumerate(h_values)}


def parse_log_file(path: Path) -> Tuple[Dict[float, List[Tuple[int, float, float]]], Dict[str, object]]:
    """Return mapping h -> list of (k, mean, std) and parsed theory metadata."""
    with path.open() as f:
        lines = json.load(f)

    data: Dict[float, List[Tuple[int, float, float]]] = defaultdict(list)
    meta: Dict[str, object] = {
        "m_unsafe": None,
        "teacher_m": None,
        "log_Cm": None,
        "nu_theory": None,
        "lambda_mean_by_h": {},
        "fit_params_by_h": {},
    }
    current_h = None

    for raw in lines:
        if not isinstance(raw, str):
            continue
        # Global parameters
        m_params = PARAMS_LINE.search(raw)
        if m_params and meta["m_unsafe"] is None:
            try:
                meta["m_unsafe"] = float(m_params.group(4))
            except ValueError:
                pass

        m_teacher = TEACHER_M_LINE.search(raw)
        if m_teacher and meta["teacher_m"] is None:
            try:
                meta["teacher_m"] = float(m_teacher.group(1))
            except ValueError:
                pass

        m_logcm = LOG_CM_LINE.search(raw)
        if m_logcm and meta["log_Cm"] is None:
            try:
                meta["log_Cm"] = float(m_logcm.group(1))
            except ValueError:
                pass

        # Detect h block header
        m_h = H_HEADER.match(raw)
        if m_h:
            try:
                current_h = float(m_h.group(1))
            except ValueError:
                current_h = None
            continue

        m_lambda = LAMBDA_MEAN_LINE.search(raw)
        if m_lambda and current_h is not None:
            try:
                meta["lambda_mean_by_h"][current_h] = float(m_lambda.group(1))
            except ValueError:
                pass

        # Parse k lines within current h block
        m_k = K_LINE.match(raw)
        if m_k and current_h is not None:
            k = int(m_k.group(1))
            mean = float(m_k.group(2))
            std = float(m_k.group(3))
            data[current_h].append((k, mean, std))

    # Sort entries per h by k
    for h in list(data.keys()):
        data[h] = sorted(data[h], key=lambda t: t[0])
    if meta["m_unsafe"] is not None and meta["teacher_m"] is not None:
        meta["nu_theory"] = float(meta["m_unsafe"]) * (1.0 - float(meta["teacher_m"]))
    return data, meta


def merge_datasets(
    datasets: List[Dict[float, List[Tuple[int, float, float]]]]
) -> Dict[float, List[Tuple[int, float, float]]]:
    """Merge datasets from multiple files; prefer later entries for duplicate k at same h."""
    merged: Dict[float, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    for ds in datasets:
        for h, entries in ds.items():
            for k, mean, std in entries:
                merged[h][k] = (mean, std)

    merged_sorted: Dict[float, List[Tuple[int, float, float]]] = {}
    for h, kmap in merged.items():
        merged_sorted[h] = [(k, *kmap[k]) for k in sorted(kmap.keys())]
    return merged_sorted


def merge_meta(metas: List[Dict[str, object]]) -> Dict[str, object]:
    merged = {
        "nu_theory": None,
        "log_Cm": None,
        "lambda_mean_by_h": {},
        "fit_params_by_h": {},
    }
    for meta in metas:
        if merged["nu_theory"] is None:
            merged["nu_theory"] = meta.get("nu_theory")
        if merged["log_Cm"] is None:
            merged["log_Cm"] = meta.get("log_Cm")
        for h, val in meta.get("lambda_mean_by_h", {}).items():
            merged["lambda_mean_by_h"][h] = val
        for h, params in meta.get("fit_params_by_h", {}).items():
            merged["fit_params_by_h"][h] = params
    return merged


def plot_loggap(
    data: Dict[float, List[Tuple[int, float, float]]],
    out_path: Path,
    title: str | None = None,
    theory_meta: Dict[str, object] | None = None
):
    plt.figure(figsize=(10, 8))
    # plt.figure(figsize=(16, 8))
    fontsize=20

    h_values = sorted(data.keys())
    print('h_values:', h_values)
    if h_values:
        h_norm = plt.Normalize(vmin=min(h_values), vmax=max(h_values)+0.1)
        h_cmap = plt.cm.Reds_r
    else:
        h_norm = None
        h_cmap = None

    data_by_h: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for h in h_values:
        entries = data[h]
        if not entries:
            continue
        k_vals = np.array([k for k, _, _ in entries], dtype=np.float64)
        means = np.array([m for _, m, _ in entries], dtype=np.float64)
        stds = np.array([s for _, _, s in entries], dtype=np.float64)
        logk = np.log(k_vals)
        y_vals = _logneglog_pi_from_loggap(means)
        y_lower = _logneglog_pi_from_loggap(means - stds)
        y_upper = _logneglog_pi_from_loggap(means + stds)
        finite = np.isfinite(y_vals) & np.isfinite(y_lower) & np.isfinite(y_upper)
        if not np.all(finite):
            logk = logk[finite]
            y_vals = y_vals[finite]
            y_lower = y_lower[finite]
            y_upper = y_upper[finite]
        if logk.size < 2:
            continue
        data_by_h[h] = (logk, y_vals, 0.5 * (y_upper - y_lower))

    fit_h_values = sorted(data_by_h.keys())
    monotone_params = fit_monotone_params(fit_h_values, data_by_h)
    print('monotone_params:', monotone_params)

    for h in h_values:
        entries = data[h]
        if not entries:
            continue
        k_vals = np.array([k for k, _, _ in entries], dtype=np.float64)
        means = np.array([m for _, m, _ in entries], dtype=np.float64)
        stds = np.array([s for _, _, s in entries], dtype=np.float64)

        logk = np.log(k_vals)
        y_vals = _logneglog_pi_from_loggap(means)
        y_lower = _logneglog_pi_from_loggap(means - stds)
        y_upper = _logneglog_pi_from_loggap(means + stds)
        finite = np.isfinite(y_vals) & np.isfinite(y_lower) & np.isfinite(y_upper)
        if not np.all(finite):
            logk = logk[finite]
            y_vals = y_vals[finite]
            y_lower = y_lower[finite]
            y_upper = y_upper[finite]
            if logk.size == 0:
                continue
        color = h_cmap(h_norm(h)) if h_cmap is not None else None
        line, = plt.plot(
            logk,
            y_vals,
            marker="o",
            markersize=4,
            linestyle="-",
            linewidth=2.0,
            label=f"h={h:g}",
            color=color,
        )
        color = line.get_color()
        plt.fill_between(
            logk,
            y_lower,
            y_upper,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

        if theory_meta:
            nu_theory = theory_meta.get("nu_theory")
            log_Cm = theory_meta.get("log_Cm")
            lambda_mean = theory_meta.get("lambda_mean_by_h", {}).get(h)
            if monotone_params and h in monotone_params:
                nu_hat, lambda_hat, log_cm_hat = monotone_params[h]
                mu_hat = 0.5*lambda_hat**2
                print('nu_hat:', nu_hat)
                print('lambda_hat:', lambda_hat)
                print('log_cm_hat:', log_cm_hat)
                fit_line = model_logneglog_pi_from_logk(logk, nu_hat, lambda_hat, log_cm_hat)
                plt.plot(
                    logk,
                    fit_line,
                    linestyle="--",
                    alpha=0.5,
                    color=color,
                    linewidth=2.5,
                    label=rf"Fit: $\hat{{\nu}}={nu_hat:.2f}$, $\hat{{\mu}}={mu_hat:.6f}$",
                )

    plt.xlabel('$\\log ~ k$', fontsize=fontsize)
    plt.ylabel('$\\log(-\\log(\\Pi_k))$', fontsize=fontsize)
    
    plt.legend(
    loc='lower left',
    bbox_to_anchor=(0.02, 0.02), 
    fontsize=18,
    frameon=True,              # Show/hide box (True/False)
    framealpha=0.9,            # Box transparency (0-1, lower = more transparent)
    edgecolor='black',         # Box border color
    fancybox=True,             # Rounded corners
    borderpad=1.0,             # Padding inside box (in font units)
    labelspacing=0.5,          # Vertical space between legend entries
    handlelength=2.0,          # Length of legend lines/markers
    handletextpad=0.5,         # Space between marker and text
    borderaxespad=0.3,         # Pad between axes and legend box
    columnspacing=2.0,         # Horizontal space between columns (if multi-column)
    ncol=1,                    # Number of columns
    )
    plt.ylim(-60, 2)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    if title:
        plt.title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Plot log(-log(Pi_k)) vs log k from logs.json files.")
    ap.add_argument("--logs", nargs="+", required=True, help="Paths to *.logs.json files (one or many).")
    ap.add_argument("--out", required=True, help="Output image path (png/pdf/svg).")
    ap.add_argument("--title", default=None, help="Optional plot title.")
    args = ap.parse_args()

    parsed = [parse_log_file(Path(p)) for p in args.logs]
    datasets = [item[0] for item in parsed]
    metas = [item[1] for item in parsed]
    merged = merge_datasets(datasets)
    merged_meta = merge_meta(metas)

    if not merged:
        raise SystemExit("No data parsed from provided logs.")

    plot_loggap(merged, Path(args.out), title=args.title, theory_meta=merged_meta)


if __name__ == "__main__":
    main()
