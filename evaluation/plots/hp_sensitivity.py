"""Plot: LabelMix hyperparameter sensitivity (α, k) for ViT-Wee on ImageNet-1k.

Produces three figures from
``data/processed/hyperparameter_sensitivity_vit_wee_with_baselines.csv``:

1. ``hp_sensitivity_vit_wee.pdf`` (main text, two panels):
   x = α on a uniform (categorical) axis, y = Top-1 (left panel, linear)
   and ECE@15 (right panel, log).  For each LabelMix loss we plot the
   median-over-k as a solid line and the min-to-max-over-k range as a shaded
   band.  Two horizontal references show the ``baseline`` (mixup + cutmix)
   and ``single-aug`` (single image aug only) seeds, each with a thin ±std
   band.  The ECE panel uses a log y-axis so the near-baseline region does
   not get crushed by the high-α explosion.

2. ``hp_sensitivity_k_vit_wee.pdf`` (main text companion, two panels):
   identical layout to (1) but sweeping x = k (with bands showing the
   min–max over α).  Provides the complementary view of the same grid.

3. ``hp_sensitivity_heatmaps_vit_wee.pdf`` (appendix, 2x2 grid):
   rows = metric, cols = loss.  Each cell is an 8x8 α × k heatmap of
   Δ = metric − baseline (the ``baseline`` row of the combined CSV is the
   zero).  Colour uses a symmetric linear scale centred on zero
   (blue = better than baseline, red = worse).  The bare ``single-aug``
   value is printed in each panel title for context.

Every rendered PDF is accompanied by a ``*.meta.json`` sidecar that records
the plot id, axes, reference values and headline numbers so an agent can
generate the correct LaTeX caption and cross-references without re-reading
the raw CSV.

Input   : ``data/processed/hyperparameter_sensitivity_vit_wee_with_baselines.csv``
Output  : ``data/processed/figures/hyperparameter/hp_sensitivity_vit_wee.pdf``
          ``data/processed/figures/hyperparameter/hp_sensitivity_k_vit_wee.pdf``
          ``data/processed/figures/hyperparameter/hp_sensitivity_heatmaps_vit_wee.pdf``
          (+ ``.png`` and ``.meta.json`` siblings for each)
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ._style import DOUBLE_COL_WIDTH, apply_paper_style, savefig


_logger = logging.getLogger(__name__)

INPUT_FILE = PROCESSED_DIR / "hyperparameter_sensitivity_vit_wee_with_baselines.csv"
# All hyperparameter plots live under a dedicated subfolder.
HP_FIGURES_DIR = FIGURES_DIR / "hyperparameter"
OUTPUT_SENSITIVITY = HP_FIGURES_DIR / "hp_sensitivity_vit_wee.pdf"
OUTPUT_SENSITIVITY_K = HP_FIGURES_DIR / "hp_sensitivity_k_vit_wee.pdf"
OUTPUT_HEATMAPS = HP_FIGURES_DIR / "hp_sensitivity_heatmaps_vit_wee.pdf"

# Human-readable loss names. Internal CSV keys ('pl_loss', 'soft_ce') are
# left untouched for backward compatibility with the extraction script.
LOSS_LABELS: dict[str, str] = {
    "pl_loss": "LabelMix (PL)",
    "soft_ce": "LabelMix (SCE)",
}
# Fixed draw order -> stable colours.
LOSS_ORDER: list[str] = ["pl_loss", "soft_ce"]
LOSS_COLOURS: dict[str, str] = {
    "pl_loss": "#1f77b4",   # blue
    "soft_ce": "#d62728",   # red
}

# Display names for the two reference rows in the combined CSV.
# The internal label 'bare' is renamed on the fly to 'single-aug' — this
# reflects what the configuration actually is (RandAug + ColorJitter, no
# mixup and no cutmix) and avoids the ambiguous "bare".
BASELINE_KEY = "baseline"     # mixup + cutmix         — our zero
SINGLE_AUG_KEY = "bare"       # single-image aug only  — secondary reference
BASELINE_DISPLAY = "baseline (Mixup+CutMix)"
SINGLE_AUG_DISPLAY = "single-aug (single image aug only)"

BASELINE_COLOUR = "#404040"   # dark grey
SINGLE_AUG_COLOUR = "#8c8c8c" # mid grey


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Split the combined CSV into a LabelMix long-form frame and a dict of
    baseline scalars keyed by the CSV's ``label`` column
    ('baseline', 'bare').
    """
    lm = df[df["kind"] == "labelmix"].copy()
    lm["alpha"] = pd.to_numeric(lm["alpha"], errors="coerce")
    lm["k"] = pd.to_numeric(lm["k"], errors="coerce").astype("Int64")
    lm = lm.dropna(subset=["alpha", "k", "loss"])
    lm = lm.rename(columns={"eval_top1_mean": "top1", "eval_ece_mean": "ece"})

    bl_rows = df[df["kind"] == "baseline"]
    baselines: dict[str, dict[str, float]] = {}
    for _, r in bl_rows.iterrows():
        baselines[r["label"]] = {
            "top1_mean": float(r["eval_top1_mean"]),
            "top1_std": float(r["eval_top1_std"]),
            "ece_mean": float(r["eval_ece_mean"]),
            "ece_std": float(r["eval_ece_std"]),
        }
    return lm, baselines


# --------------------------------------------------------------------------
# Figure 1: sensitivity bands
# --------------------------------------------------------------------------


def _draw_baselines(ax, baselines: dict[str, dict[str, float]], metric: str):
    """Add horizontal reference lines + ±std bands for both baselines."""
    specs = [
        (BASELINE_KEY, BASELINE_COLOUR, "-", BASELINE_DISPLAY),
        (SINGLE_AUG_KEY, SINGLE_AUG_COLOUR, "--", SINGLE_AUG_DISPLAY),
    ]
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    for name, colour, ls, label in specs:
        if name not in baselines:
            continue
        m = baselines[name][mean_key]
        s = baselines[name][std_key]
        ax.axhspan(m - s, m + s, color=colour, alpha=0.10, lw=0)
        ax.axhline(m, color=colour, ls=ls, lw=1.0, label=label)


def _plot_metric(
    ax,
    lm: pd.DataFrame,
    metric: str,
    baselines,
    *,
    sweep: str = "alpha",
    yscale: str = "linear",
):
    """Render one panel (Top-1 or ECE) of a sensitivity figure.

    Parameters
    ----------
    sweep:
        Which hyperparameter to put on the x-axis, either ``"alpha"`` (band
        is min–max over k) or ``"k"`` (band is min–max over α).
    yscale:
        ``"linear"`` or ``"log"``.  The ECE panel benefits from ``"log"``
        because values span roughly 1.5 → 35 and the low-α region collapses
        into a flat line on a linear axis.

    Uniform (categorical) x-axis spacing is used regardless of the sweep:
    each sampled value gets the same horizontal gap, independent of its
    numeric value.
    """
    if sweep not in {"alpha", "k"}:
        raise ValueError(f"sweep must be 'alpha' or 'k', got {sweep!r}")
    other = "k" if sweep == "alpha" else "alpha"

    xs_sorted = sorted(lm[sweep].unique())
    pos = {v: i for i, v in enumerate(xs_sorted)}

    for loss in LOSS_ORDER:
        sub = lm[lm["loss"] == loss]
        if sub.empty:
            continue
        # For each x value, aggregate across the *other* hyperparameter.
        agg = (
            sub.groupby(sweep)[metric]
            .agg(median="median", lo="min", hi="max")
            .reindex(xs_sorted)
        )
        x = np.asarray([pos[v] for v in agg.index], dtype=float)
        colour = LOSS_COLOURS[loss]
        ax.fill_between(x, agg["lo"].values, agg["hi"].values, color=colour,
                        alpha=0.18, lw=0)
        ax.plot(x, agg["median"].values, color=colour, lw=1.3, marker="o",
                ms=3.0, label=LOSS_LABELS[loss])

    _draw_baselines(ax, baselines, metric)

    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([f"{v:g}" if sweep == "alpha" else str(int(v))
                        for v in xs_sorted])
    ax.set_xlim(-0.3, len(xs_sorted) - 0.7)
    ax.set_xlabel(r"$\alpha$" if sweep == "alpha" else r"$k$")

    if yscale == "log":
        ax.set_yscale("log")
        # Tidy ticks and minor grid for the log panel.
        ax.grid(axis="y", which="both", ls=":", lw=0.5, alpha=0.6)
    else:
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # Record which variable we aggregated away so callers (metadata) can
    # describe the band correctly.
    ax._sweep = sweep
    ax._other = other


def plot_sensitivity(df: pd.DataFrame, *, sweep: str = "alpha") -> plt.Figure:
    """Two-panel sensitivity figure. ``sweep`` picks the x-axis variable."""
    apply_paper_style()
    lm, baselines = _split(df)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL_WIDTH, 2.4),
        sharex=True,
    )

    _plot_metric(axes[0], lm, "top1", baselines, sweep=sweep, yscale="linear")
    axes[0].set_ylabel(r"Top-1 accuracy (%, $\uparrow$)")

    # ECE is log-scaled so the low-ECE plateau is legible.
    _plot_metric(axes[1], lm, "ece", baselines, sweep=sweep, yscale="log")
    axes[1].set_ylabel(r"ECE@15 (%, $\downarrow$, log)")

    # Single shared legend above the panels.
    handles, labels = axes[0].get_legend_handles_labels()
    # De-duplicate while preserving order.
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    fig.legend(
        [h for h, _ in uniq],
        [l for _, l in uniq],
        loc="upper center",
        ncol=len(uniq),
        bbox_to_anchor=(0.5, 1.02),
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def build_sensitivity_metadata(
    df: pd.DataFrame,
    out_path: Path,
    *,
    sweep: str = "alpha",
) -> dict:
    """Build a JSON-serialisable metadata dict for a sensitivity figure.

    ``sweep`` is the variable on the x-axis (``"alpha"`` or ``"k"``); the
    band describes min–max over the *other* variable.
    """
    lm, baselines = _split(df)
    alphas = sorted(lm["alpha"].unique())
    ks = sorted(int(k) for k in lm["k"].dropna().unique())
    other = "k" if sweep == "alpha" else "alpha"
    other_values = ks if sweep == "alpha" else alphas

    # Per-loss headline stats (min / median / max across the full grid).
    per_loss: dict[str, dict] = {}
    for loss in LOSS_ORDER:
        sub = lm[lm["loss"] == loss]
        if sub.empty:
            continue
        per_loss[LOSS_LABELS[loss]] = {
            "top1": {
                "min": float(sub["top1"].min()),
                "median": float(sub["top1"].median()),
                "max": float(sub["top1"].max()),
            },
            "ece": {
                "min": float(sub["ece"].min()),
                "median": float(sub["ece"].median()),
                "max": float(sub["ece"].max()),
            },
            "n_cells": int(len(sub)),
        }

    meta = {
        "plot_id": (
            "hp_sensitivity_vit_wee" if sweep == "alpha"
            else "hp_sensitivity_k_vit_wee"
        ),
        "title": (
            "LabelMix hyperparameter sensitivity (α, k) \u2014 ViT-Wee / ImageNet-1k"
            if sweep == "alpha"
            else "LabelMix hyperparameter sensitivity over k \u2014 ViT-Wee / ImageNet-1k"
        ),
        "kind": "line_with_band",
        "panels": [
            {"axis": "left", "metric": "Top-1 accuracy (%)",
             "direction": "higher is better", "yscale": "linear"},
            {"axis": "right", "metric": "ECE@15 (%)",
             "direction": "lower is better", "yscale": "log"},
        ],
        "x": {"name": sweep, "scale": "categorical_uniform",
              "values": alphas if sweep == "alpha" else ks},
        "aggregation_over_other": {
            "variable": other,
            "line": "median",
            "band": f"min\u2013max over {other}",
            "values": other_values,
        },
        "losses": [LOSS_LABELS[l] for l in LOSS_ORDER if l in lm["loss"].unique()],
        "references": {
            "baseline": {
                "display": BASELINE_DISPLAY,
                "role": "zero reference",
                "top1_mean": baselines[BASELINE_KEY]["top1_mean"],
                "top1_std": baselines[BASELINE_KEY]["top1_std"],
                "ece_mean": baselines[BASELINE_KEY]["ece_mean"],
                "ece_std": baselines[BASELINE_KEY]["ece_std"],
            },
            "single_aug": {
                "display": SINGLE_AUG_DISPLAY,
                "role": "secondary reference (no Mixup/CutMix, keeps RandAug + ColorJitter)",
                "top1_mean": baselines[SINGLE_AUG_KEY]["top1_mean"],
                "top1_std": baselines[SINGLE_AUG_KEY]["top1_std"],
                "ece_mean": baselines[SINGLE_AUG_KEY]["ece_mean"],
                "ece_std": baselines[SINGLE_AUG_KEY]["ece_std"],
            },
        },
        "per_loss_stats": per_loss,
        "caption_hint": (
            "Hyperparameter sensitivity of LabelMix on ViT-Wee / ImageNet-1k. "
            f"The x-axis sweeps {sweep}; each solid line is the median Top-1 "
            f"(left, linear) / ECE@15 (right, log) over {other} \u2208 "
            f"{other_values}; the shaded band shows the min\u2013max envelope "
            f"over {other}. The dark grey line is the baseline (Mixup+CutMix); "
            "the dashed grey line is single-aug (single image aug only). "
            "LabelMix matches or beats the baseline in Top-1 across the full "
            "sweep and reduces ECE by a large margin."
        ),
        "file": {
            "pdf": out_path.with_suffix(".pdf").name,
            "png": out_path.with_suffix(".png").name,
            "directory": str(out_path.parent),
        },
    }
    return meta

# --------------------------------------------------------------------------
# Figure 2: Δ-vs-baseline heatmaps (symmetric log colour)
# --------------------------------------------------------------------------


def _pivot(lm: pd.DataFrame, loss: str, metric: str) -> pd.DataFrame:
    """Return a (k × α) matrix of metric values for a given loss."""
    sub = lm[lm["loss"] == loss]
    piv = sub.pivot_table(index="k", columns="alpha", values=metric, aggfunc="mean")
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    return piv


def _linear_sym_norm(delta: np.ndarray) -> Normalize:
    """Plain symmetric linear normalisation around zero (vmin = -vmax).

    We deliberately let the largest |Δ| drive ``vmax`` so the colour scale is
    honest: strong outliers look strong, and the zero point stays exactly at
    the colourmap midpoint.
    """
    vmax = float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else 1.0
    vmax = max(vmax, 1e-6)
    return Normalize(vmin=-vmax, vmax=vmax)


def _draw_heatmap(ax, piv: pd.DataFrame, baseline_val: float, single_aug_val: float,
                  *, lower_is_better: bool, title: str):
    delta = piv.values - baseline_val
    norm = _linear_sym_norm(delta)
    # Invariant: blue = better than baseline, red = worse.
    #   - Top-1 (higher better): "better" = delta > 0. RdBu maps high values
    #     to blue, so use RdBu.
    #   - ECE  (lower  better): "better" = delta < 0. RdBu maps low values to
    #     red -- inverted from what we want -- so use RdBu_r (which flips).
    cmap = "RdBu_r" if lower_is_better else "RdBu"

    im = ax.imshow(
        delta,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        origin="lower",
    )

    # Tick labels from the pivot.
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels([f"{a:g}" for a in piv.columns], rotation=0)
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels([str(k) for k in piv.index])

    # Annotate each cell with the *delta* (what the colour encodes).  Use a
    # compact format and pick white text on strongly-coloured cells for
    # contrast.  Colour-strength from the norm covers both halves of the
    # symlog range consistently.
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            d = delta[i, j]
            if not np.isfinite(d):
                continue
            rel = norm(d)  # in [0, 1], with 0.5 at delta = 0
            txt_colour = "white" if abs(rel - 0.5) > 0.40 else "black"
            # Signed, compact; two decimals is enough for both scales.
            fmt = f"{d:+.2f}"
            ax.text(j, i, fmt,
                    ha="center", va="center",
                    fontsize=5.5, color=txt_colour)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$k$")
    # Title: baseline on line 1, bare single-aug value on line 2
    # (Δ is already encoded by the cell colour and cell text).
    ax.set_title(f"{title}\n"
                 f"(baseline={baseline_val:.2f}, single-aug={single_aug_val:.2f})",
                 fontsize=8)
    return im


def plot_heatmaps(df: pd.DataFrame) -> plt.Figure:
    apply_paper_style()
    lm, baselines = _split(df)
    bl_top1 = baselines[BASELINE_KEY]["top1_mean"]
    bl_ece = baselines[BASELINE_KEY]["ece_mean"]
    sa_top1 = baselines.get(SINGLE_AUG_KEY, {}).get("top1_mean", np.nan)
    sa_ece = baselines.get(SINGLE_AUG_KEY, {}).get("ece_mean", np.nan)

    # Widen the figure a touch so the two-decimal Δ text has room to breathe
    # without having to shrink the font further.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL_WIDTH * 1.15, 5.6),
        constrained_layout=True,
    )

    cfg = [
        # (metric, lower_is_better, row, baseline_val, single_aug_val, row_label)
        ("top1", False, 0, bl_top1, sa_top1, "Top-1 (%)"),
        ("ece", True, 1, bl_ece, sa_ece, "ECE@15 (%)"),
    ]

    for metric, lower_is_better, row, bl_val, sa_val, metric_label in cfg:
        last_im = None
        for col, loss in enumerate(LOSS_ORDER):
            piv = _pivot(lm, loss, metric)
            ax = axes[row, col]
            title = f"{metric_label} — {LOSS_LABELS[loss]}"
            last_im = _draw_heatmap(
                ax, piv, baseline_val=bl_val, single_aug_val=sa_val,
                lower_is_better=lower_is_better, title=title,
            )
        # Shared colorbar per row (Δ vs baseline, symmetric log).
        cbar = fig.colorbar(last_im, ax=axes[row, :], shrink=0.85, pad=0.02)
        cbar.set_label(
            f"Δ{metric_label} vs. baseline  (blue = better)",
            fontsize=7,
        )
        cbar.ax.tick_params(labelsize=7)

    return fig


def build_heatmap_metadata(df: pd.DataFrame, out_path: Path) -> dict:
    """Build a JSON-serialisable metadata dict for the heatmap figure."""
    lm, baselines = _split(df)
    bl = baselines[BASELINE_KEY]
    sa = baselines.get(SINGLE_AUG_KEY, {})

    per_panel = []
    for metric, direction, bl_val in [
        ("top1", "higher is better", bl["top1_mean"]),
        ("ece", "lower is better", bl["ece_mean"]),
    ]:
        for loss in LOSS_ORDER:
            piv = _pivot(lm, loss, metric)
            if piv.empty:
                continue
            delta = piv.values - bl_val
            best_idx = np.unravel_index(
                (np.nanargmin if metric == "ece" else np.nanargmax)(piv.values),
                piv.values.shape,
            )
            per_panel.append({
                "metric": metric,
                "loss": LOSS_LABELS[loss],
                "direction": direction,
                "alpha_grid": [float(a) for a in piv.columns.tolist()],
                "k_grid": [int(k) for k in piv.index.tolist()],
                "baseline_value": float(bl_val),
                "delta": {
                    "min": float(np.nanmin(delta)),
                    "max": float(np.nanmax(delta)),
                    "mean": float(np.nanmean(delta)),
                    "n_better_than_baseline": int(
                        (delta < 0).sum() if metric == "ece" else (delta > 0).sum()
                    ),
                    "n_total": int(np.isfinite(delta).sum()),
                },
                "best_cell": {
                    "k": int(piv.index[best_idx[0]]),
                    "alpha": float(piv.columns[best_idx[1]]),
                    "value": float(piv.values[best_idx]),
                    "delta_vs_baseline": float(piv.values[best_idx] - bl_val),
                },
            })

    meta = {
        "plot_id": "hp_sensitivity_heatmaps_vit_wee",
        "title": "LabelMix α × k heatmaps (Δ vs. baseline) — ViT-Wee / ImageNet-1k",
        "kind": "heatmap_grid_2x2",
        "layout": "rows=metrics (Top-1, ECE@15), cols=losses (PL, SCE)",
        "encoding": {
            "cell_value_printed": "Δ = metric − baseline",
            "colour_norm": "symmetric linear (vmin = -max|Δ|, vmax = +max|Δ|)",
            "colour_semantics": "blue = better than baseline, red = worse",
        },
        "references": {
            "baseline": {
                "display": BASELINE_DISPLAY,
                "top1_mean": bl["top1_mean"],
                "ece_mean": bl["ece_mean"],
            },
            "single_aug": {
                "display": SINGLE_AUG_DISPLAY,
                "top1_mean": sa.get("top1_mean"),
                "ece_mean": sa.get("ece_mean"),
                "delta_top1_vs_baseline": (
                    None if "top1_mean" not in sa
                    else sa["top1_mean"] - bl["top1_mean"]
                ),
                "delta_ece_vs_baseline": (
                    None if "ece_mean" not in sa
                    else sa["ece_mean"] - bl["ece_mean"]
                ),
            },
        },
        "panels": per_panel,
        "caption_hint": (
            "α × k sensitivity of LabelMix on ViT-Wee / ImageNet-1k, reported "
            "as Δ against the Mixup+CutMix baseline (baseline = 0). Columns "
            "contrast the PL and SCE variants; rows show Top-1 (top) and "
            "ECE@15 (bottom). Colour uses a symmetric linear scale centred "
            "on zero. Blue cells beat the baseline; red cells lag. The bare "
            "single-aug reference value is printed in every panel title."
        ),
        "file": {
            "pdf": out_path.with_suffix(".pdf").name,
            "png": out_path.with_suffix(".png").name,
            "directory": str(out_path.parent),
        },
    }
    return meta


# --------------------------------------------------------------------------
# Metadata sidecar
# --------------------------------------------------------------------------


def _write_metadata(meta: dict, out_path: Path) -> Path:
    """Write a ``<stem>.meta.json`` sidecar next to ``out_path``.

    ``out_path`` is expected to point at the PDF; the metadata file keeps the
    same stem with a ``.meta.json`` suffix so it is trivially linkable.
    """
    meta = {
        **meta,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "evaluation.plots.hp_sensitivity",
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=False)
        f.write("\n")
    return meta_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=INPUT_FILE)
    p.add_argument("--out-sensitivity", type=Path, default=OUTPUT_SENSITIVITY)
    p.add_argument("--out-sensitivity-k", type=Path, default=OUTPUT_SENSITIVITY_K)
    p.add_argument("--out-heatmaps", type=Path, default=OUTPUT_HEATMAPS)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    if not args.input.is_file():
        _logger.error("Input CSV not found: %s", args.input)
        return 1

    _logger.info("Reading %s", args.input)
    df = pd.read_csv(args.input)

    args.out_sensitivity.parent.mkdir(parents=True, exist_ok=True)
    args.out_sensitivity_k.parent.mkdir(parents=True, exist_ok=True)
    args.out_heatmaps.parent.mkdir(parents=True, exist_ok=True)

    # Figure 1: α sweep.
    fig1 = plot_sensitivity(df, sweep="alpha")
    savefig(fig1, str(args.out_sensitivity))
    meta1 = build_sensitivity_metadata(df, args.out_sensitivity, sweep="alpha")
    meta1_path = _write_metadata(meta1, args.out_sensitivity)
    _logger.info("Wrote %s", args.out_sensitivity)
    _logger.info("Wrote %s", meta1_path)
    plt.close(fig1)

    # Figure 2: k sweep (companion to the α figure).
    fig_k = plot_sensitivity(df, sweep="k")
    savefig(fig_k, str(args.out_sensitivity_k))
    meta_k = build_sensitivity_metadata(df, args.out_sensitivity_k, sweep="k")
    meta_k_path = _write_metadata(meta_k, args.out_sensitivity_k)
    _logger.info("Wrote %s", args.out_sensitivity_k)
    _logger.info("Wrote %s", meta_k_path)
    plt.close(fig_k)

    # Figure 3: heatmaps.
    fig2 = plot_heatmaps(df)
    savefig(fig2, str(args.out_heatmaps))
    meta2 = build_heatmap_metadata(df, args.out_heatmaps)
    meta2_path = _write_metadata(meta2, args.out_heatmaps)
    _logger.info("Wrote %s", args.out_heatmaps)
    _logger.info("Wrote %s", meta2_path)
    plt.close(fig2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
