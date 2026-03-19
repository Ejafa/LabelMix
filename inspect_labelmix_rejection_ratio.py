#!/usr/bin/env python3
"""Estimate LabelMix rejection-sampling ratios across alpha/K sweeps."""
import argparse
import csv
import io
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from PIL import Image

from timm import utils
from timm.data import resolve_data_config
from timm.data.balanced_dataset import _boxes_are_valid_and_tile, _layout_to_pixel_boxes
from timm.data.labelmix_layout import squarify_core

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    plt = None

_logger = logging.getLogger("inspect_labelmix_rejection_ratio")

DEFAULT_REJECTION_RATIO_ALPHA_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
DEFAULT_REJECTION_RATIO_K_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate LabelMix rejection-sampling ratios across alpha and k."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )
    parser.add_argument("--output-dir", default="output_rejection_ratio", type=str)
    parser.add_argument("--input-size", nargs=3, default=None, type=int, metavar=("C", "H", "W"))
    parser.add_argument("--img-size", default=None, type=int)
    parser.add_argument("--in-chans", default=None, type=int)
    parser.add_argument("--chans", default=None, type=int)
    parser.add_argument("--mean", nargs="*", default=None, type=float)
    parser.add_argument("--std", nargs="*", default=None, type=float)
    parser.add_argument("--interpolation", default=None, type=str)
    parser.add_argument("--num-classes", default=None, type=int)
    parser.add_argument(
        "--rejection-ratio-samples",
        default=2000,
        type=int,
        help="Requested LabelMix samples per alpha/K point for rejection-ratio estimation.",
    )
    parser.add_argument(
        "--rejection-ratio-history-points",
        default=24,
        type=int,
        help="Number of cumulative checkpoints to record while estimating rejection ratio.",
    )
    parser.add_argument(
        "--rejection-ratio-alpha-values",
        nargs="*",
        default=None,
        type=float,
        help="Alpha values to sweep for rejection-ratio estimation.",
    )
    parser.add_argument(
        "--rejection-ratio-k-values",
        nargs="*",
        default=None,
        type=int,
        help="K values to sweep for rejection-ratio estimation.",
    )
    parser.add_argument("--labelmix-sampling", action="store_true", default=False)
    parser.add_argument("--labelmix-sampling-min-side-px", default=8, type=int)
    parser.add_argument("--labelmix-sampling-max-aspect", default=10, type=float)
    parser.add_argument("--labelmix-sampling-bins", default=16, type=int)
    parser.add_argument("--labelmix-sampling-pool-size", default=128, type=int)
    parser.add_argument("--labelmix-sampling-low-watermark", default=32, type=int)
    parser.add_argument("--labelmix-sampling-max-attempts", default=200, type=int)
    args, _ = parser.parse_known_args()

    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        defaults = {a.dest: parser.get_default(a.dest) for a in parser._actions}
        for k, v in cfg.items():
            if getattr(args, k, defaults.get(k)) == defaults.get(k):
                setattr(args, k, v)

    return args


def _ensure_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


def _resolve_sweep_values(values: Optional[Sequence[Any]], default_values: Sequence[Any], cast) -> List[Any]:
    if values is None or len(values) == 0:
        values = default_values
    return sorted({cast(v) for v in values})


def _build_history_checkpoints(total_samples: int, history_points: int) -> List[int]:
    total_samples = max(1, int(total_samples))
    history_points = max(1, int(history_points))
    if total_samples <= history_points:
        return list(range(1, total_samples + 1))

    checkpoints = {1, total_samples}
    if history_points > 2:
        log_total = math.log(float(total_samples))
        for i in range(history_points):
            p = i / float(history_points - 1)
            checkpoints.add(int(round(math.exp(log_total * p))))
    return sorted({max(1, min(total_samples, int(v))) for v in checkpoints})


def _resolve_output_hw(args: argparse.Namespace) -> Tuple[int, int]:
    data_config = resolve_data_config(vars(args), model=None)
    _, h, w = data_config["input_size"]
    return int(h), int(w)


def _layout_is_valid_for_sampling(
    weights_desc: torch.Tensor,
    H: int,
    W: int,
    sampling_min_side_px: int,
    sampling_max_aspect: float,
) -> bool:
    base_layout_desc = squarify_core(weights_desc, canvas_size=1.0)
    base_layout_asc = torch.flip(base_layout_desc, dims=[0])
    boxes = _layout_to_pixel_boxes(base_layout_asc, H=H, W=W, canvas_size=1.0, eps=1e-7)
    if not _boxes_are_valid_and_tile(boxes, H=H, W=W):
        return False

    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    widths = (x1 - x0).to(dtype=torch.float32)
    heights = (y1 - y0).to(dtype=torch.float32)
    min_side = torch.minimum(widths, heights)
    max_side = torch.maximum(widths, heights)

    if sampling_min_side_px > 0 and torch.any(min_side < float(sampling_min_side_px)):
        return False
    if sampling_max_aspect > 0.0:
        aspect = max_side / torch.clamp(min_side, min=1.0)
        if torch.any(aspect > float(sampling_max_aspect)):
            return False
    return True


def _estimate_rejection_ratio_for_point(
    alpha: float,
    mix_k: int,
    sample_requests: int,
    history_points: int,
    H: int,
    W: int,
    sampling_min_side_px: int,
    sampling_max_aspect: float,
    sampling_max_attempts: int,
) -> Dict[str, Any]:
    sample_requests = max(1, int(sample_requests))
    sampling_max_attempts = max(1, int(sampling_max_attempts))
    checkpoints = _build_history_checkpoints(sample_requests, history_points)
    checkpoint_set = set(checkpoints)

    rejected_draws = 0
    total_draws = 0
    fallback_count = 0
    requests_with_rejection = 0
    history: List[Dict[str, Any]] = []

    if alpha > 0.0:
        dirichlet = torch.distributions.Dirichlet(torch.full((int(mix_k),), float(alpha)))
    else:
        dirichlet = None

    for request_idx in range(1, sample_requests + 1):
        had_rejection = False
        accepted = False

        if dirichlet is None:
            total_draws += 1
            accepted = True
        else:
            for _ in range(sampling_max_attempts):
                total_draws += 1
                weights = dirichlet.sample().to(dtype=torch.float32)
                weights_desc, _ = torch.sort(weights, descending=True)
                if _layout_is_valid_for_sampling(
                    weights_desc,
                    H=H,
                    W=W,
                    sampling_min_side_px=sampling_min_side_px,
                    sampling_max_aspect=sampling_max_aspect,
                ):
                    accepted = True
                    break
                rejected_draws += 1
                had_rejection = True

        if had_rejection:
            requests_with_rejection += 1
        if not accepted:
            fallback_count += 1

        if request_idx in checkpoint_set:
            history.append(
                {
                    "sample_requests": int(request_idx),
                    "raw_draws": int(total_draws),
                    "rejected_draws": int(rejected_draws),
                    "requests_with_rejection": int(requests_with_rejection),
                    "fallback_count": int(fallback_count),
                    "rejection_ratio": float(rejected_draws / max(1, total_draws)),
                    "request_rejection_ratio": float(requests_with_rejection / float(request_idx)),
                    "fallback_ratio": float(fallback_count / float(request_idx)),
                    "mean_draws_per_request": float(total_draws / float(request_idx)),
                }
            )

    final = history[-1]
    return {
        "alpha": float(alpha),
        "k": int(mix_k),
        "sample_requests": int(sample_requests),
        "raw_draws": int(total_draws),
        "rejected_draws": int(rejected_draws),
        "requests_with_rejection": int(requests_with_rejection),
        "fallback_count": int(fallback_count),
        "rejection_ratio": float(final["rejection_ratio"]),
        "request_rejection_ratio": float(final["request_rejection_ratio"]),
        "fallback_ratio": float(final["fallback_ratio"]),
        "mean_draws_per_request": float(final["mean_draws_per_request"]),
        "history": history,
    }


def _build_metric_grid(
    results_by_key: Dict[Tuple[float, int], Dict[str, Any]],
    alpha_values: Sequence[float],
    k_values: Sequence[int],
    metric_key: str,
) -> torch.Tensor:
    rows: List[List[float]] = []
    for alpha in alpha_values:
        row: List[float] = []
        for k in k_values:
            row.append(float(results_by_key[(float(alpha), int(k))][metric_key]))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def _render_rejection_ratio_surface(
    alpha_values: Sequence[float],
    k_values: Sequence[int],
    z_values: torch.Tensor,
    title: str,
) -> Image.Image:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable")

    x = torch.tensor(k_values, dtype=torch.float32)
    y = torch.tensor(alpha_values, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        xx.numpy(),
        yy.numpy(),
        z_values.numpy(),
        cmap="viridis",
        linewidth=0.4,
        edgecolor="black",
        antialiased=True,
        alpha=0.95,
    )
    ax.scatter(
        xx.numpy(),
        yy.numpy(),
        z_values.numpy(),
        c=z_values.numpy(),
        cmap="viridis",
        edgecolors="black",
        s=25,
        depthshade=False,
    )
    ax.set_xlabel("k")
    ax.set_ylabel("alpha")
    ax.set_zlabel("rejection ratio")
    ax.set_title(title)
    ax.set_xticks(list(k_values))
    ax.set_yticks(list(alpha_values))
    ax.set_zlim(0.0, 1.0)
    # Rotate the camera so the alpha axis sits close to the viewer-facing plane.
    ax.view_init(elev=24, azim=0)
    fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.12, label="rejection ratio")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    frame = Image.open(buf).convert("RGB")
    frame.load()
    buf.close()
    return frame


def _save_rejection_ratio_surface_plot(
    path: str,
    alpha_values: Sequence[float],
    k_values: Sequence[int],
    z_values: torch.Tensor,
    title: str,
) -> None:
    frame = _render_rejection_ratio_surface(
        alpha_values=alpha_values,
        k_values=k_values,
        z_values=z_values,
        title=title,
    )
    frame.save(path)


def _save_rejection_ratio_history_plot(
    path: str,
    results_by_key: Dict[Tuple[float, int], Dict[str, Any]],
    alpha_values: Sequence[float],
    k_values: Sequence[int],
) -> None:
    if plt is None:
        return

    num_k = len(k_values)
    cols = min(4, max(1, num_k))
    rows = int(math.ceil(num_k / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows), sharex=True, sharey=True)
    if hasattr(axes, "flatten"):
        axes_flat = list(axes.flatten())
    else:
        axes_flat = [axes]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in torch.linspace(0.1, 0.95, steps=len(alpha_values)).tolist()]

    for ax, k in zip(axes_flat, k_values):
        for color, alpha in zip(colors, alpha_values):
            history = results_by_key[(float(alpha), int(k))]["history"]
            xs = [int(item["sample_requests"]) for item in history]
            ys = [float(item["rejection_ratio"]) for item in history]
            ax.plot(xs, ys, color=color, linewidth=1.8, label=f"alpha={alpha:g}")
        ax.set_title(f"k={k}")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[num_k:]:
        ax.axis("off")

    for idx, ax in enumerate(axes_flat[:num_k]):
        if idx // cols == rows - 1:
            ax.set_xlabel("sample requests")
        if idx % cols == 0:
            ax.set_ylabel("rejection ratio")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(alpha_values)), frameon=False)
    fig.suptitle("LabelMix rejection-ratio convergence (lines = alpha, panels = k)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_rejection_ratio_surface_gif(
    path: str,
    results_by_key: Dict[Tuple[float, int], Dict[str, Any]],
    alpha_values: Sequence[float],
    k_values: Sequence[int],
) -> None:
    if plt is None:
        return

    reference_history = results_by_key[(float(alpha_values[0]), int(k_values[0]))]["history"]
    frames: List[Image.Image] = []
    for history_idx, checkpoint in enumerate(reference_history):
        z_rows: List[List[float]] = []
        for alpha in alpha_values:
            row: List[float] = []
            for k in k_values:
                row.append(
                    float(
                        results_by_key[(float(alpha), int(k))]["history"][history_idx]["rejection_ratio"]
                    )
                )
            z_rows.append(row)
        title = (
            "LabelMix rejection ratio "
            f"(up to {int(checkpoint['sample_requests'])} sample requests)"
        )
        frames.append(
            _render_rejection_ratio_surface(
                alpha_values=alpha_values,
                k_values=k_values,
                z_values=torch.tensor(z_rows, dtype=torch.float32),
                title=title,
            )
        )

    if not frames:
        return

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=350,
        loop=0,
    )


def get_rejection_sampling_ratio_strenght(args: argparse.Namespace) -> Dict[str, Any]:
    alpha_values = _resolve_sweep_values(
        getattr(args, "rejection_ratio_alpha_values", None),
        DEFAULT_REJECTION_RATIO_ALPHA_VALUES,
        float,
    )
    k_values = _resolve_sweep_values(
        getattr(args, "rejection_ratio_k_values", None),
        DEFAULT_REJECTION_RATIO_K_VALUES,
        int,
    )
    if not alpha_values or not k_values:
        raise ValueError("Rejection-ratio sweep requires at least one alpha and one k value.")

    num_classes = getattr(args, "num_classes", None)
    if num_classes is not None:
        for k in k_values:
            if int(k) > int(num_classes):
                raise ValueError(
                    f"Rejection-ratio sweep requires k <= num_classes "
                    f"(got k={k}, num_classes={num_classes})."
                )

    H, W = _resolve_output_hw(args)
    sample_requests = max(1, int(getattr(args, "rejection_ratio_samples", 2000)))
    history_points = max(1, int(getattr(args, "rejection_ratio_history_points", 24)))
    sampling_min_side_px = int(getattr(args, "labelmix_sampling_min_side_px", 6))
    sampling_max_aspect = float(getattr(args, "labelmix_sampling_max_aspect", 10.0))
    sampling_max_attempts = int(getattr(args, "labelmix_sampling_max_attempts", 200))

    _logger.info(
        "Running rejection-ratio sweep on %dx%d outputs for %d alpha values x %d k values.",
        H,
        W,
        len(alpha_values),
        len(k_values),
    )
    _logger.info(
        "Sampling constraints: min_side_px=%d max_aspect=%.4f max_attempts=%d samples_per_point=%d",
        sampling_min_side_px,
        sampling_max_aspect,
        sampling_max_attempts,
        sample_requests,
    )

    results_by_key: Dict[Tuple[float, int], Dict[str, Any]] = {}
    ordered_results: List[Dict[str, Any]] = []
    total_points = len(alpha_values) * len(k_values)
    point_idx = 0
    for alpha in alpha_values:
        for k in k_values:
            point_idx += 1
            _logger.info("Estimating rejection ratio for alpha=%.4f, k=%d (%d/%d)", alpha, k, point_idx, total_points)
            point_result = _estimate_rejection_ratio_for_point(
                alpha=float(alpha),
                mix_k=int(k),
                sample_requests=sample_requests,
                history_points=history_points,
                H=H,
                W=W,
                sampling_min_side_px=sampling_min_side_px,
                sampling_max_aspect=sampling_max_aspect,
                sampling_max_attempts=sampling_max_attempts,
            )
            results_by_key[(float(alpha), int(k))] = point_result
            ordered_results.append(point_result)

    output_dir = str(getattr(args, "output_dir", "output_test") or "output_test")
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "mode": "get_rejection_sampling_ratio_strenght",
        "output_hw": {"height": int(H), "width": int(W)},
        "alpha_values": [float(v) for v in alpha_values],
        "k_values": [int(v) for v in k_values],
        "sampling_constraints": {
            "labelmix_sampling": bool(_ensure_bool(getattr(args, "labelmix_sampling", False))),
            "labelmix_sampling_min_side_px": int(sampling_min_side_px),
            "labelmix_sampling_max_aspect": float(sampling_max_aspect),
            "labelmix_sampling_max_attempts": int(sampling_max_attempts),
            "labelmix_sampling_bins": int(getattr(args, "labelmix_sampling_bins", 16)),
            "labelmix_sampling_pool_size": int(getattr(args, "labelmix_sampling_pool_size", 128)),
            "labelmix_sampling_low_watermark": int(getattr(args, "labelmix_sampling_low_watermark", 32)),
            "note": (
                "Bins/pool-size/low-watermark are recorded for completeness; "
                "the geometry rejection ratio is driven by alpha, k, output size, "
                "min_side_px, and max_aspect, while max_attempts only affects truncation "
                "and fallback-related metrics."
            ),
        },
        "sample_requests_per_point": int(sample_requests),
        "history_points": int(history_points),
        "results": ordered_results,
    }

    json_path = os.path.join(output_dir, "rejection_ratio_strength.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(output_dir, "rejection_ratio_strength.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alpha",
                "k",
                "sample_requests",
                "raw_draws",
                "rejected_draws",
                "requests_with_rejection",
                "fallback_count",
                "rejection_ratio",
                "request_rejection_ratio",
                "fallback_ratio",
                "mean_draws_per_request",
            ],
        )
        writer.writeheader()
        for row in ordered_results:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    if plt is not None:
        rejection_surface = _build_metric_grid(
            results_by_key=results_by_key,
            alpha_values=alpha_values,
            k_values=k_values,
            metric_key="rejection_ratio",
        )
        _save_rejection_ratio_surface_plot(
            path=os.path.join(output_dir, "rejection_ratio_strength_surface.png"),
            alpha_values=alpha_values,
            k_values=k_values,
            z_values=rejection_surface,
            title="LabelMix rejection ratio across alpha and k",
        )
        _save_rejection_ratio_history_plot(
            path=os.path.join(output_dir, "rejection_ratio_strength_history.png"),
            results_by_key=results_by_key,
            alpha_values=alpha_values,
            k_values=k_values,
        )
        _save_rejection_ratio_surface_gif(
            path=os.path.join(output_dir, "rejection_ratio_strength_surface.gif"),
            results_by_key=results_by_key,
            alpha_values=alpha_values,
            k_values=k_values,
        )
    else:
        _logger.warning("matplotlib is unavailable; skipping rejection-ratio plots and GIF.")

    return summary


def main() -> None:
    utils.setup_default_logging()
    args = _parse_args()
    _logger.info("args=%s", json.dumps(vars(args), indent=2, default=str))

    summary = get_rejection_sampling_ratio_strenght(args)
    for row in summary["results"]:
        print(
            json.dumps(
                {
                    "alpha": row["alpha"],
                    "k": row["k"],
                    "rejection_ratio": row["rejection_ratio"],
                    "request_rejection_ratio": row["request_rejection_ratio"],
                    "fallback_ratio": row["fallback_ratio"],
                    "mean_draws_per_request": row["mean_draws_per_request"],
                }
            )
        )


if __name__ == "__main__":
    main()
