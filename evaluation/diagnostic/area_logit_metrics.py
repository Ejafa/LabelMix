"""Area-vs-logit diagnostic metrics.

This module consumes ``logits.pt`` artifacts produced by
:mod:`evaluation.diagnostic.infer` and the composed manifest written by
:mod:`evaluation.diagnostic.generate`, and computes three summary metrics
per model:

    M1. Area-logit Spearman correlation (mean over samples).
    M2. Present-class pairwise ranking accuracy (micro-average over eligible
        pairs, threshold ``min_area_diff``).
    M3. Largest-area top-logit accuracy (mean over samples).

All three are reported per-``k`` and aggregated over ``k ∈ K_ALLOWED``
(default: ``{3, 4, 5, 6}``).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_logger = logging.getLogger("evaluation.diagnostic.area_logit_metrics")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: k values considered for all metrics. Samples with k outside this set are
#: ignored end-to-end (both per-k rows and the aggregated ``"all"`` row).
K_ALLOWED: Tuple[int, ...] = (3, 4, 5, 6)

#: Minimum area difference for Metric 2 to include a pair.
MIN_AREA_DIFF: float = 0.05


# ---------------------------------------------------------------------------
# Low-level Spearman (tie-free fallback with no scipy dependency)
# ---------------------------------------------------------------------------


def _has_ties(x: np.ndarray) -> bool:
    return bool(np.unique(x).size != x.size)


def _spearman_no_ties(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho assuming no ties in either vector.

    Uses the simplified rank-difference formula
    ``rho = 1 - 6 * sum(d^2) / (n (n^2 - 1))`` which is exact when there are
    no ties; we guarantee that by skipping tied samples at the caller.

    Returns NaN when ``n < 2``.
    """
    n = int(a.shape[0])
    if n < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    d = ra - rb
    return float(1.0 - 6.0 * float((d * d).sum()) / (n * (n * n - 1)))


# ---------------------------------------------------------------------------
# Per-sample metric computation
# ---------------------------------------------------------------------------


@dataclass
class _SampleContrib:
    """What a single sample contributes to the three running tallies."""

    # Metric 1
    spearman: Optional[float]            # None if undefined (ties / n<2)
    # Metric 2
    pair_correct: int
    pair_total: int
    # Metric 3
    top_correct: Optional[int]           # None if largest-area tied


def _sample_contrib(
    present_classes: Sequence[int],
    class_area_ratios: Mapping[str, float],
    logits_row: np.ndarray,  # (C,) float
    *,
    min_area_diff: float,
) -> _SampleContrib:
    """Compute (m1, m2, m3) contributions for a single composed sample."""
    present = [int(c) for c in present_classes]
    n = len(present)

    areas = np.asarray(
        [float(class_area_ratios[str(c)]) for c in present], dtype=np.float64
    )
    lg = np.asarray([float(logits_row[c]) for c in present], dtype=np.float64)

    # ---- Metric 1: Spearman (skip on ties or n<2) ---------------------------
    if n < 2 or _has_ties(areas) or _has_ties(lg):
        spearman: Optional[float] = None
    else:
        spearman = _spearman_no_ties(areas, lg)
        if not math.isfinite(spearman):
            spearman = None

    # ---- Metric 2: pairwise accuracy w/ area-diff threshold ----------------
    pair_correct = 0
    pair_total = 0
    if n >= 2:
        # Only enumerate i<j; for each eligible pair count one comparison.
        for i in range(n):
            ai, li = areas[i], lg[i]
            for j in range(i + 1, n):
                aj, lj = areas[j], lg[j]
                if abs(ai - aj) < min_area_diff:
                    continue
                pair_total += 1
                if ai > aj:
                    if li > lj:
                        pair_correct += 1
                else:  # aj > ai  (strict because |diff| >= threshold > 0)
                    if lj > li:
                        pair_correct += 1

    # ---- Metric 3: largest-area vs argmax-logit over present ---------------
    if n == 0:
        top_correct: Optional[int] = None
    else:
        area_max = float(areas.max())
        tied = int((areas == area_max).sum())
        if tied > 1:
            top_correct = None  # skip per spec
        else:
            argmax_area = int(np.argmax(areas))
            argmax_logit = int(np.argmax(lg))
            top_correct = int(argmax_area == argmax_logit)

    return _SampleContrib(
        spearman=spearman,
        pair_correct=pair_correct,
        pair_total=pair_total,
        top_correct=top_correct,
    )


# ---------------------------------------------------------------------------
# Public API: one model -> per-k and aggregated metric rows
# ---------------------------------------------------------------------------


@dataclass
class ModelMetricsRow:
    model: str
    k: str                  # either "3"/"4"/"5"/"6" or "all"
    n_samples: int          # composed samples considered (post-k-filter)
    spearman_mean: float
    spearman_n: int         # samples with a defined Spearman
    pair_acc: float
    pair_n: int             # eligible pairs
    top_acc: float
    top_n: int              # samples with non-tied largest area


def compute_metrics_for_model(
    logits_blob: Mapping[str, object],
    *,
    k_allowed: Sequence[int] = K_ALLOWED,
    min_area_diff: float = MIN_AREA_DIFF,
    model_name: Optional[str] = None,
) -> List[ModelMetricsRow]:
    """Compute per-k and aggregated ('all') metric rows for one model.

    ``logits_blob`` is the in-memory dict saved by
    :func:`evaluation.diagnostic.infer.export_logits_for_run`.
    """
    # Resolve tensors / lists with defensive typing ---------------------------
    logits_t = logits_blob["logits"]
    if isinstance(logits_t, torch.Tensor):
        logits_np = logits_t.detach().float().cpu().numpy()
    else:
        logits_np = np.asarray(logits_t, dtype=np.float64)

    ks_t = logits_blob["k"]
    if isinstance(ks_t, torch.Tensor):
        ks = ks_t.detach().cpu().numpy().astype(np.int64)
    else:
        ks = np.asarray(ks_t, dtype=np.int64)

    sample_ids: List[str] = list(logits_blob["sample_ids"])  # type: ignore[arg-type]
    present_classes: List[List[int]] = list(
        logits_blob["present_classes"]  # type: ignore[assignment]
    )
    class_area_ratios: List[Dict[str, float]] = list(
        logits_blob["class_area_ratios"]  # type: ignore[assignment]
    )
    name = model_name or str(logits_blob.get("model_name", "model"))

    n_total = logits_np.shape[0]
    if not (len(sample_ids) == len(present_classes) == len(class_area_ratios) == ks.shape[0] == n_total):
        raise ValueError(
            "logits blob is internally inconsistent: "
            f"logits={n_total}, k={ks.shape[0]}, ids={len(sample_ids)}, "
            f"present={len(present_classes)}, areas={len(class_area_ratios)}"
        )

    k_set = set(int(k) for k in k_allowed)

    # Group sample indices by k (k-filter applied here so out-of-scope samples
    # never contribute to the 'all' row either).
    per_k_indices: Dict[int, List[int]] = {k: [] for k in sorted(k_set)}
    for i, k in enumerate(ks.tolist()):
        k = int(k)
        if k in k_set:
            per_k_indices[k].append(i)

    # Accumulators ------------------------------------------------------------
    def _empty_bucket() -> Dict[str, float]:
        return {
            "n_samples": 0,
            "spearman_sum": 0.0,
            "spearman_n": 0,
            "pair_correct": 0,
            "pair_total": 0,
            "top_correct": 0,
            "top_n": 0,
        }

    buckets: Dict[str, Dict[str, float]] = {
        str(k): _empty_bucket() for k in sorted(k_set)
    }
    buckets["all"] = _empty_bucket()

    for k, idx_list in per_k_indices.items():
        for i in idx_list:
            contrib = _sample_contrib(
                present_classes[i],
                class_area_ratios[i],
                logits_np[i],
                min_area_diff=min_area_diff,
            )
            for key in (str(k), "all"):
                b = buckets[key]
                b["n_samples"] += 1
                if contrib.spearman is not None:
                    b["spearman_sum"] += float(contrib.spearman)
                    b["spearman_n"] += 1
                b["pair_correct"] += int(contrib.pair_correct)
                b["pair_total"] += int(contrib.pair_total)
                if contrib.top_correct is not None:
                    b["top_correct"] += int(contrib.top_correct)
                    b["top_n"] += 1

    # Materialize rows --------------------------------------------------------
    def _finalize(k_label: str, b: Dict[str, float]) -> ModelMetricsRow:
        sp_n = int(b["spearman_n"])
        pr_n = int(b["pair_total"])
        tp_n = int(b["top_n"])
        return ModelMetricsRow(
            model=name,
            k=k_label,
            n_samples=int(b["n_samples"]),
            spearman_mean=float(b["spearman_sum"] / sp_n) if sp_n > 0 else float("nan"),
            spearman_n=sp_n,
            pair_acc=float(b["pair_correct"] / pr_n) if pr_n > 0 else float("nan"),
            pair_n=pr_n,
            top_acc=float(b["top_correct"] / tp_n) if tp_n > 0 else float("nan"),
            top_n=tp_n,
        )

    rows: List[ModelMetricsRow] = [
        _finalize(str(k), buckets[str(k)]) for k in sorted(k_set)
    ]
    rows.append(_finalize("all", buckets["all"]))
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


CSV_COLUMNS: Tuple[str, ...] = (
    "model",
    "k",
    "n_samples",
    "spearman_mean",
    "spearman_n",
    "pair_acc",
    "pair_n",
    "top_acc",
    "top_n",
)


def row_to_dict(row: ModelMetricsRow) -> Dict[str, object]:
    return {
        "model": row.model,
        "k": row.k,
        "n_samples": row.n_samples,
        "spearman_mean": row.spearman_mean,
        "spearman_n": row.spearman_n,
        "pair_acc": row.pair_acc,
        "pair_n": row.pair_n,
        "top_acc": row.top_acc,
        "top_n": row.top_n,
    }
