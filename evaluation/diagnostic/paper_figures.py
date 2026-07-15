"""Paper-ready figure export for Gradient x Input attribution panels.

Given the per-model attribution run folders produced by
:mod:`evaluation.diagnostic.attribution` + :mod:`evaluation.diagnostic.visualize`,
this module produces, for each (sample_id, class_id) that is present in ALL
selected models, a small **figure package** suitable for inclusion in a paper
or for a downstream agent to render its own figure:

    paper_figures/
        index.json                                # all packages at a glance
        class_names.json                          # index -> lemma (ImageNet-1k)
        <sample_id>/
            class_<cid>/
                figure_spec.json                  # full machine-readable spec
                caption.txt                       # suggested caption
            input.png                         # shared composed input
            input_with_box.png                # input + red target box
            <model_label>_boxed.png           # overlay + red-box tile per model
            figure.pdf / figure.png           # rendered panel (matplotlib)

The PDF is vector, the PNG is high-dpi raster.  If matplotlib is unavailable we
still emit the ``figure_spec.json``, ``caption.txt``, and the individual tile
PNGs so any other tool (or an agent) can assemble the figure from the spec.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from evaluation.common import DIAGNOSTIC_DIR, FIGURES_DIR, setup_logging

_logger = logging.getLogger("evaluation.diagnostic.paper_figures")


# ---------------------------------------------------------------------------
# Model discovery (shared with compose_boxed_grid, with display labels added)
# ---------------------------------------------------------------------------

DEFAULT_ATTRIB_ROOT = DIAGNOSTIC_DIR / "attributions"
# Final, paper-ready figure packages live under the processed figures tree
# (``evaluation/data/processed/figures/diagnostic_attribution``) rather than
# inside the raw attribution root, so ``data/raw`` stays append-only and all
# rendered artifacts sit in ``data/processed/figures/`` per project layout.
DEFAULT_OUT_DIR = FIGURES_DIR / "diagnostic_attribution"

# --- "Default command" defaults ---------------------------------------------
# Running ``python -m evaluation.scripts.diag_paper_figures`` with no extra
# arguments should reproduce our canonical paper-figure export:
#
#     --per-k-samples 10 --overwrite -v      (k = 3,4,5,6, full rendering)
#
# so we lift those into module-level defaults instead of leaving them as
# CLI-only overrides.
DEFAULT_PER_K_SAMPLES: int = 10
DEFAULT_K_VALUES: Sequence[int] = (1, 3, 4, 5, 6)
DEFAULT_OVERWRITE: bool = True
DEFAULT_VERBOSE: bool = True

# Full set of models we track per (sample, class).  For every entry here we
# copy an individual ``<label>_boxed.png`` tile into the class folder and emit
# a ``tiles`` entry in ``figure_spec.json`` -- this is the full metadata
# payload.  Each entry is (short_label, display_label, slug_keywords).
DEFAULT_MODEL_ORDER: Sequence[Tuple[str, str, Sequence[str]]] = (
    ("noaug", "No augmentation", ("noaug_no_aug",)),
    ("bare", "Single image augmentation", ("bare_cutmix_",)),
    ("baseline", "Baseline (Cutmix + Mixup)", ("baseline__baseline",)),
    ("mosaic", "Mosaic", ("mosaic__mosaic",)),
    ("soft-ce", "LabelMix SCE", ("soft-ce",)),
    ("pl-loss", "LabelMix PL", ("pl-loss",)),
)

# Subset of labels that actually appear in the rendered PDF/PNG figure.  All
# other models are still copied as individual tile PNGs and captured in the
# spec as metadata, but the rendered panel only shows this narrower
# comparison.  Order defines left-to-right placement in the figure.
DEFAULT_RENDER_MODELS: Sequence[str] = (
    "baseline",
    "mosaic",
    "soft-ce",
    "pl-loss",
)


@dataclass(frozen=True)
class ModelEntry:
    label: str          # short label used for filenames
    display: str        # pretty name shown in captions / rendered figure
    run_dir: Path       # directory with k*_*** sample folders + index.json


def _match_slot(run_dir_name: str,
                order: Sequence[Tuple[str, str, Sequence[str]]]
                ) -> Optional[Tuple[str, str]]:
    lower = run_dir_name.lower()
    for label, display, keywords in order:
        if all(kw.lower() in lower for kw in keywords):
            return label, display
    return None


def discover_models(
    attrib_root: Path,
    order: Sequence[Tuple[str, str, Sequence[str]]] = DEFAULT_MODEL_ORDER,
) -> List[ModelEntry]:
    candidates = [p for p in attrib_root.iterdir()
                  if p.is_dir() and not p.name.startswith("_")]
    by_label: Dict[str, Tuple[str, Path]] = {}
    for p in candidates:
        hit = _match_slot(p.name, order)
        if hit is None:
            continue
        label, display = hit
        by_label.setdefault(label, (display, p))
    out: List[ModelEntry] = []
    for label, display, _ in order:
        if label in by_label:
            _, run_dir = by_label[label]
            out.append(ModelEntry(label=label, display=display, run_dir=run_dir))
    return out


# ---------------------------------------------------------------------------
# Per-sample helpers
# ---------------------------------------------------------------------------

_SAMPLE_RE = re.compile(r"k\d{2}_\d+")
_BOXED_RE = re.compile(r"^class_(\d+)_boxed\.png$")


# In-memory cache: run_dir.name -> {sample_id: 1-D softmax tensor (numpy)}
_PROB_CACHE: Dict[str, Optional[Dict[str, "object"]]] = {}


def _softmax_probs_for_model(model_run_dir: Path) -> Optional[Dict[str, "object"]]:
    """Load ``logits.pt`` for a model and return ``{sample_id: softmax(np.ndarray)}``.

    Looks up ``<attrib_root>/../logits/<same-slug>/logits.pt``.  Returns
    ``None`` (cached) when the file is missing or cannot be decoded.  The
    tensors are converted to float64 probabilities on CPU.
    """
    key = model_run_dir.name
    if key in _PROB_CACHE:
        return _PROB_CACHE[key]

    logits_path = model_run_dir.parent.parent / "logits" / key / "logits.pt"
    if not logits_path.is_file():
        _logger.warning("  [probs] missing logits.pt for %s at %s",
                        key, logits_path)
        _PROB_CACHE[key] = None
        return None
    try:
        import numpy as np
        import torch  # type: ignore
        payload = torch.load(str(logits_path), map_location="cpu",
                             weights_only=False)
        sample_ids = list(payload["sample_ids"])
        logits = payload["logits"]
        if hasattr(logits, "detach"):
            logits = logits.detach()
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        assert probs.shape[0] == len(sample_ids), (
            f"logits rows ({probs.shape[0]}) != sample_ids ({len(sample_ids)})")
        mapping: Dict[str, "object"] = {
            sid: probs[i] for i, sid in enumerate(sample_ids)
        }
        _PROB_CACHE[key] = mapping
        _logger.info("  [probs] loaded %d softmax rows for %s",
                     len(mapping), key)
        return mapping
    except Exception as exc:  # pragma: no cover
        _logger.warning("  [probs] failed to load %s: %s", logits_path, exc)
        _PROB_CACHE[key] = None
        return None


def _prob_for(model_run_dir: Path, sample_id: str, class_id: int
              ) -> Optional[float]:
    table = _softmax_probs_for_model(model_run_dir)
    if table is None:
        return None
    row = table.get(sample_id)
    if row is None:
        return None
    try:
        return float(row[int(class_id)])
    except Exception:
        return None


def _list_sample_dirs(run_dir: Path) -> List[Path]:
    return sorted(p for p in run_dir.iterdir()
                  if p.is_dir() and _SAMPLE_RE.fullmatch(p.name))


def _class_ids_in_sample(sample_dir: Path) -> List[int]:
    out: List[int] = []
    for child in sample_dir.iterdir():
        m = _BOXED_RE.match(child.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _load_sample_json(sample_dir: Path) -> Optional[dict]:
    p = sample_dir / "_sample.json"
    if not p.is_file():
        return None
    with p.open("r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# ImageNet-1k class names (via timm.ImageNetInfo)
# ---------------------------------------------------------------------------


def _load_in1k_class_names() -> Dict[int, str]:
    try:
        from timm.data import ImageNetInfo  # type: ignore
    except Exception:  # pragma: no cover
        return {}
    try:
        info = ImageNetInfo("imagenet-1k")
    except Exception:
        return {}
    out: Dict[int, str] = {}
    for idx in range(info.num_classes()):
        try:
            desc = info.index_to_description(idx, detailed=False)
        except Exception:
            desc = ""
        # Lemmas are comma-joined; keep the shortest first name.
        out[idx] = desc.split(",", 1)[0].strip() if desc else ""
    return out


# ---------------------------------------------------------------------------
# Input-with-box helper (clean input + red target rectangle)
# ---------------------------------------------------------------------------


def _make_input_with_box(
    input_png: Path,
    reference_boxed: Path,
    out_path: Path,
    overwrite: bool,
) -> bool:
    """Produce a clean-input tile that still shows the red target box.

    We take the base ``input.png`` and copy over *only* the red-box pixels from
    one of the per-model ``class_<c>_boxed.png`` (boxes are identical across
    models since they are derived from the shared patch mask).
    """
    if out_path.is_file() and not overwrite:
        return True
    try:
        base = Image.open(input_png).convert("RGB")
        ref = Image.open(reference_boxed).convert("RGB")
    except Exception:
        return False
    if base.size != ref.size:
        ref = ref.resize(base.size, Image.BILINEAR)

    import numpy as np
    base_np = np.array(base, dtype=np.int16)
    ref_np = np.array(ref, dtype=np.int16)
    r, g, b = ref_np[..., 0], ref_np[..., 1], ref_np[..., 2]
    mask = (r > 200) & (g < 60) & (b < 60)
    out_np = base_np.copy()
    out_np[mask] = ref_np[mask]
    out_np = out_np.clip(0, 255).astype("uint8")
    Image.fromarray(out_np, mode="RGB").save(out_path)
    return True


# ---------------------------------------------------------------------------
# Figure-package builder (per sample_id, class_id)
# ---------------------------------------------------------------------------


@dataclass
class PaperFiguresConfig:
    attrib_root: Path = DEFAULT_ATTRIB_ROOT
    out_dir: Path = DEFAULT_OUT_DIR
    overwrite: bool = False
    render: bool = True                 # also render PDF + PNG via matplotlib
    render_dpi: int = 300
    tile_size_inches: float = 1.06      # five panels in the 5.5-inch text block
    max_samples: Optional[int] = None   # cap total number of figures produced
    sample_ids: Optional[Sequence[str]] = None   # filter to these sample_ids
    k_values: Optional[Sequence[int]] = None     # filter to these k values
    per_k_samples: Optional[int] = None          # cap number of sample_ids per k


def _relp(p: Path, root: Path) -> str:
    return os.path.relpath(p, root)


def _copy_if_present(src: Path, dst: Path, overwrite: bool) -> bool:
    if not src.is_file():
        return False
    if dst.exists() and not overwrite:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def _short_caption(class_id: int,
                   class_name: str,
                   k: int,
                   area_ratio: Optional[float],
                   models: Sequence[ModelEntry]) -> str:
    name_part = f'"{class_name}"' if class_name else f"class {class_id}"
    area_part = f", area {area_ratio * 100:.1f}%" if area_ratio is not None else ""
    models_str = ", ".join(m.display for m in models)
    return (
        f"Gradient x Input attribution for {name_part} (id {class_id}) on a "
        f"k={k} composed input{area_part}. Left panel: composed input with "
        f"the target patch outlined in red. Remaining panels: attribution "
        f"overlays from {models_str}. Intensity is normalized per panel to "
        f"the 99th percentile of absolute attribution."
    )


def _build_spec(
    sample_id: str,
    class_id: int,
    k: int,
    area_ratio: Optional[float],
    class_name: str,
    models: Sequence[ModelEntry],
    render_models: Sequence[ModelEntry],
    pkg_dir: Path,
    out_dir: Path,
    have_render: bool,
) -> dict:
    render_label_set = {m.label for m in render_models}
    tiles: List[dict] = []
    for m in models:
        prob = _prob_for(m.run_dir, sample_id, class_id)
        tiles.append({
            "model_label": m.label,
            "display_label": m.display,
            "boxed": _relp(pkg_dir / f"{m.label}_boxed.png", out_dir),
            "run_dir": m.run_dir.name,
            "prob": prob,  # softmax probability for ``class_id`` (None if missing)
            "in_rendered_figure": m.label in render_label_set,
        })
    return {
        "sample_id": sample_id,
        "k": int(k),
        "class_id": int(class_id),
        "class_name": class_name,
        "area_ratio": area_ratio,
        "input_png": _relp(pkg_dir / "input.png", out_dir),
        "input_with_box_png": _relp(pkg_dir / "input_with_box.png", out_dir),
        "tiles": tiles,
        "layout": {
            "orientation": "horizontal",
            "include_input_panel": True,
            # ``panel_order`` / ``panel_titles`` describe the RENDERED figure;
            # full metadata for the other models still lives under ``tiles``.
            "panel_order": ["input"] + [m.label for m in render_models],
            "panel_titles": {
                "input": "Input (target boxed)",
                **{m.label: m.display for m in render_models},
            },
            "panel_source_for_models": "boxed",
            "all_models": [m.label for m in models],
            "rendered_models": [m.label for m in render_models],
        },
        "caption": _short_caption(
            class_id, class_name, k, area_ratio, render_models
        ),
        "rendered": {
            "pdf": _relp(pkg_dir / "figure.pdf", out_dir) if have_render else None,
            "png": _relp(pkg_dir / "figure.png", out_dir) if have_render else None,
        },
    }


# ---------------------------------------------------------------------------
# Matplotlib rendering (paper style)
# ---------------------------------------------------------------------------


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # pragma: no cover
        _logger.warning("matplotlib unavailable (%s); skipping PDF/PNG rendering.", exc)
        return None


def _render_panel_pdf(
    pkg_dir: Path,
    spec: dict,
    models: Sequence[ModelEntry],
    tile_size_inches: float,
    dpi: int,
) -> bool:
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    # Match every other generated paper figure instead of maintaining a
    # separate sans-serif / hard-coded-size style for attribution panels.
    from evaluation.plots._style import BASE_FONT_SIZE, apply_paper_style

    apply_paper_style()

    input_panel_path = (pkg_dir / "input_with_box.png"
                        if (pkg_dir / "input_with_box.png").is_file()
                        else pkg_dir / "input.png")
    # Build (title, path) per panel.  For the model panels we append a second
    # line with the softmax probability of ``class_id`` so the reader sees how
    # likely the model deems the whole image to be of that class.
    probs_by_label: Dict[str, Optional[float]] = {}
    for tile in spec.get("tiles", []):
        probs_by_label[tile.get("model_label", "")] = tile.get("prob")

    def _panel_title(m: ModelEntry) -> str:
        p = probs_by_label.get(m.label)
        if p is None:
            return m.display
        return f"{m.display}\np={p:.2f}"

    # The caption explains the red target box; the short title avoids clipping
    # at the left edge in the five-across native-width layout.
    panels: List[Tuple[str, Path]] = [("Input", input_panel_path)]
    for m in models:
        panels.append((_panel_title(m), pkg_dir / f"{m.label}_boxed.png"))

    n = len(panels)
    fig_w = tile_size_inches * n + 0.05 * (n - 1)
    # Preserve the original 8.2:2.1 canvas ratio while shrinking both
    # dimensions to the native 5.5-inch paper width.
    fig_h = fig_w * (2.1 / 8.2)
    fig, axes = plt.subplots(
        1, n, figsize=(fig_w, fig_h),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.04, "hspace": 0.0},
    )
    if n == 1:
        axes = [axes]

    for ax, (title, path) in zip(axes, panels):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # pragma: no cover
            _logger.warning("  render: cannot open %s: %s", path, exc)
            ax.set_facecolor("#dddddd")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=BASE_FONT_SIZE, pad=2)

    cls_part = spec.get("class_name") or f"class {spec['class_id']}"
    area = spec.get("area_ratio")
    if area is not None:
        supt = f"{cls_part}  (k={spec['k']}, area={area*100:.1f}%)"
    else:
        supt = f"{cls_part}  (k={spec['k']})"
    fig.suptitle(supt, fontsize=BASE_FONT_SIZE)

    fig.savefig(pkg_dir / "figure.pdf")
    fig.savefig(pkg_dir / "figure.png", dpi=dpi)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def build_paper_figures(cfg: PaperFiguresConfig) -> Dict[str, int]:
    models = discover_models(cfg.attrib_root)
    if len(models) < 2:
        raise RuntimeError(
            f"Need >= 2 model run-dirs under {cfg.attrib_root}; found "
            f"{[m.label for m in models]}")

    # Subset of models that actually appear in the rendered PDF/PNG panel.
    # Preserve the order defined by ``DEFAULT_RENDER_MODELS`` and drop labels
    # that were not discovered on disk.
    by_label = {m.label: m for m in models}
    render_models: List[ModelEntry] = [
        by_label[lbl] for lbl in DEFAULT_RENDER_MODELS if lbl in by_label
    ]
    if not render_models:
        raise RuntimeError(
            "None of the configured render models "
            f"{list(DEFAULT_RENDER_MODELS)} were discovered under "
            f"{cfg.attrib_root}; available: {[m.label for m in models]}")
    render_label_set = {m.label for m in render_models}
    _logger.info(
        "[paper_figures] all_models=%s  rendered=%s",
        [m.label for m in models],
        [m.label for m in render_models],
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    sample_sets: List[set] = []
    for m in models:
        sample_sets.append({p.name for p in _list_sample_dirs(m.run_dir)})
    common_samples = sorted(set.intersection(*sample_sets))
    if cfg.sample_ids:
        keep = set(cfg.sample_ids)
        common_samples = [s for s in common_samples if s in keep]

    # Apply per-k cap: keep at most ``cfg.per_k_samples`` sample_ids per k.
    # ``common_samples`` is already sorted -> we take the first N of each k
    # group, which is deterministic across reruns.
    if cfg.per_k_samples is not None and cfg.per_k_samples > 0:
        seen_per_k: Dict[int, int] = {}
        limited: List[str] = []
        for sid in common_samples:
            try:
                k = int(sid.split("_", 1)[0].lstrip("k"))
            except Exception:
                continue
            n = seen_per_k.get(k, 0)
            if n >= cfg.per_k_samples:
                continue
            seen_per_k[k] = n + 1
            limited.append(sid)
        _logger.info("[paper_figures] per-k cap=%d -> %d sample_ids selected "
                     "(was %d)", cfg.per_k_samples, len(limited),
                     len(common_samples))
        common_samples = limited

    class_names = _load_in1k_class_names()
    packages: List[dict] = []

    n_pkgs = 0
    n_rendered = 0
    n_skipped = 0
    for sid in common_samples:
        k_val = int(sid.split("_", 1)[0].lstrip("k"))
        if cfg.k_values is not None and k_val not in set(int(k) for k in cfg.k_values):
            continue

        cls_sets: List[set] = []
        for m in models:
            cls_sets.append(set(_class_ids_in_sample(m.run_dir / sid)))
        common_classes = sorted(set.intersection(*cls_sets))
        if not common_classes:
            continue

        sample_meta = _load_sample_json(models[0].run_dir / sid) or {}
        area_ratios = sample_meta.get("class_area_ratios") or {}

        for cid in common_classes:
            if cfg.max_samples is not None and n_pkgs >= cfg.max_samples:
                break

            pkg_dir = cfg.out_dir / sid / f"class_{cid}"
            pkg_dir.mkdir(parents=True, exist_ok=True)

            # Shared composed input (any model has the same one).
            any_input = None
            for m in models:
                src = m.run_dir / sid / "input.png"
                if src.is_file():
                    any_input = src
                    break
            if any_input is None:
                n_skipped += 1
                continue
            _copy_if_present(any_input, pkg_dir / "input.png", cfg.overwrite)

            # Per-model tiles (only the boxed overlay is needed for the paper
            # figure; the raw heatmap and the unboxed overlay are intentionally
            # skipped to keep the output package small).
            any_ok = False
            for m in models:
                ok = _copy_if_present(m.run_dir / sid / f"class_{cid}_boxed.png",
                                      pkg_dir / f"{m.label}_boxed.png",
                                      cfg.overwrite)
                any_ok = any_ok or ok
            if not any_ok:
                n_skipped += 1
                continue

            # Clean input + target-class red box (for the left panel).
            _make_input_with_box(
                pkg_dir / "input.png",
                pkg_dir / f"{models[0].label}_boxed.png",
                pkg_dir / "input_with_box.png",
                cfg.overwrite,
            )

            area_ratio = None
            if str(cid) in area_ratios:
                area_ratio = float(area_ratios[str(cid)])
            class_name = class_names.get(int(cid), "")

            spec = _build_spec(sid, cid, k_val, area_ratio, class_name,
                               models, render_models, pkg_dir, cfg.out_dir,
                               have_render=cfg.render)
            (pkg_dir / "caption.txt").write_text(spec["caption"] + "\n")

            if cfg.render:
                have_render = _render_panel_pdf(
                    pkg_dir=pkg_dir,
                    spec=spec,
                    models=render_models,
                    tile_size_inches=cfg.tile_size_inches,
                    dpi=cfg.render_dpi,
                )
                spec["rendered"]["pdf"] = (_relp(pkg_dir / "figure.pdf", cfg.out_dir)
                                           if have_render else None)
                spec["rendered"]["png"] = (_relp(pkg_dir / "figure.png", cfg.out_dir)
                                           if have_render else None)
                if have_render:
                    n_rendered += 1

            with (pkg_dir / "figure_spec.json").open("w") as f:
                json.dump(spec, f, indent=2)

            packages.append({
                "sample_id": sid,
                "k": k_val,
                "class_id": int(cid),
                "class_name": class_name,
                "spec": _relp(pkg_dir / "figure_spec.json", cfg.out_dir),
            })
            n_pkgs += 1

        if cfg.max_samples is not None and n_pkgs >= cfg.max_samples:
            break

    (cfg.out_dir / "class_names.json").write_text(
        json.dumps({str(k): v for k, v in class_names.items()}, indent=2))
    (cfg.out_dir / "index.json").write_text(json.dumps({
        "attrib_root": str(cfg.attrib_root),
        "models": [
            {
                "label": m.label,
                "display": m.display,
                "run_dir": m.run_dir.name,
                "in_rendered_figure": m.label in render_label_set,
            }
            for m in models
        ],
        "rendered_models": [m.label for m in render_models],
        "n_packages": n_pkgs,
        "n_rendered": n_rendered,
        "n_skipped": n_skipped,
        "packages": packages,
    }, indent=2))
    return {"packages": n_pkgs, "rendered": n_rendered, "skipped": n_skipped}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Export paper-ready figure packages (input + per-model "
                     "attribution tiles) with a machine-readable "
                     "figure_spec.json per (sample, class), plus an optional "
                     "rendered PDF + PNG panel."),
    )
    p.add_argument("--attrib-root", type=Path, default=DEFAULT_ATTRIB_ROOT,
                   help="Attribution root produced by diag_gradient_x_input.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help=("Output dir for the paper-ready figure packages "
                         f"(default: {DEFAULT_OUT_DIR})."))
    # --overwrite / --no-overwrite: default ON so the bare command regenerates
    # everything deterministically (matches our last manual invocation).
    ow = p.add_mutually_exclusive_group()
    ow.add_argument("--overwrite", dest="overwrite", action="store_true",
                    help="Regenerate every package, overwriting existing "
                         f"files (default: {DEFAULT_OVERWRITE}).")
    ow.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                    help="Skip packages whose output files already exist.")
    p.set_defaults(overwrite=DEFAULT_OVERWRITE)

    p.add_argument("--no-render", action="store_true",
                   help="Skip PDF/PNG rendering; only emit tile PNGs + spec.")
    p.add_argument("--render-dpi", type=int, default=300)
    p.add_argument("--tile-size-inches", type=float, default=1.06)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap the total number of figure packages produced.")
    p.add_argument("--per-k-samples", type=int, default=DEFAULT_PER_K_SAMPLES,
                   help=("Keep only the first N composed sample_ids per k. "
                         "All target classes of those samples are then "
                         f"rendered (default: {DEFAULT_PER_K_SAMPLES}; "
                         "pass 0 to disable the cap)."))
    p.add_argument("--sample-ids", nargs="*", default=None,
                   help="Only export these sample_ids (e.g. k03_00071).")
    p.add_argument("--k", "--k-values", dest="k_values", nargs="*", type=int,
                   default=list(DEFAULT_K_VALUES),
                   help=("Only export these k values (e.g. --k 3 4). "
                         f"Default: {list(DEFAULT_K_VALUES)}."))

    # -v / --quiet: verbose ON by default so the bare command prints progress.
    vb = p.add_mutually_exclusive_group()
    vb.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                    help="Log INFO-level progress (default: on).")
    vb.add_argument("-q", "--quiet", dest="verbose", action="store_false",
                    help="Only log warnings and errors.")
    p.set_defaults(verbose=DEFAULT_VERBOSE)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_cli(argv)
    setup_logging(level="INFO" if args.verbose else "WARNING")

    # ``--per-k-samples 0`` is the explicit opt-out for the cap.
    if args.per_k_samples is None or int(args.per_k_samples) <= 0:
        per_k = None
    else:
        per_k = int(args.per_k_samples)

    cfg = PaperFiguresConfig(
        attrib_root=args.attrib_root,
        out_dir=(args.out_dir or DEFAULT_OUT_DIR),
        overwrite=bool(args.overwrite),
        render=(not args.no_render),
        render_dpi=int(args.render_dpi),
        tile_size_inches=float(args.tile_size_inches),
        max_samples=(int(args.max_samples) if args.max_samples is not None else None),
        sample_ids=(tuple(args.sample_ids) if args.sample_ids else None),
        k_values=(tuple(args.k_values) if args.k_values else None),
        per_k_samples=per_k,
    )
    stats = build_paper_figures(cfg)
    print(f"[paper_figures] stats: {stats}  out_dir={cfg.out_dir}")


if __name__ == "__main__":
    main()
