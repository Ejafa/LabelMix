"""Compose compound images that tile a given `class_<id>_boxed.png`
across the 4 trained vit-betwixt models (baseline, mosaic, soft-ce, pl-loss).

For every (sample_id, class_id) present in all selected model run-dirs we
produce ONE compound PNG that stitches the model tiles horizontally, adds a
thin header with the model's short label and a top-level title line
``sample_id | class_id``.

Output layout (relative to ``attributions_root``):

    _compound_boxed/<sample_id>/class_<class_id>.png

The script is intentionally dependency-light: only PIL + stdlib.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ATTRIB_ROOT = Path(
    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/LabelMix"
    "/evaluation/data/raw/diagnostic/attributions"
)

# Preferred left-to-right order in the compound image.  Each entry is a
# (short_label, keyword_list) pair: a run-dir is assigned to the FIRST entry
# whose keywords are ALL present in the directory name.
#
# NOTE on disambiguation: the "bare" run-dir name contains the substring
# "cutmix_0" (cutmix disabled), and the "cutmix" run-dir name contains the
# substring "cutmix_mixup_switch_prob_1.0".  We therefore use the longest
# discriminating substrings (``bare_cutmix_`` / ``cutmix_mixup_switch_prob``)
# as slot keywords so no directory can match both slots.  The attribution
# pipeline slugifies run names (strips ``=`` etc.) so keywords must use the
# slugified form.
DEFAULT_MODEL_ORDER: Sequence[Tuple[str, Sequence[str]]] = (
    ("bare", ("bare_cutmix_",)),
    ("baseline", ("baseline__baseline",)),
    ("cutmix", ("cutmix_mixup_switch_prob",)),
    ("mosaic", ("mosaic__mosaic",)),
    ("soft-ce", ("soft-ce",)),
    ("pl-loss", ("pl-loss",)),
)

COMPOUND_SUBDIR = "_compound_boxed"

# Styling
HEADER_H = 28           # px strip above each tile with model label
TITLE_H = 36            # px strip above the whole compound with the title
TILE_GAP = 6            # px horizontal gap between tiles
PAD = 8                 # px outer padding
BG_COLOR = (255, 255, 255)
HEADER_BG = (32, 32, 32)
HEADER_FG = (240, 240, 240)
TITLE_BG = (18, 18, 18)
TITLE_FG = (250, 250, 250)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntry:
    label: str          # short label shown in the header (e.g. "baseline")
    run_dir: Path       # directory that holds the k*_***** sample folders


def _match_model(run_dir_name: str,
                 order: Sequence[Tuple[str, Sequence[str]]]) -> Optional[str]:
    lower = run_dir_name.lower()
    for label, keywords in order:
        if all(kw.lower() in lower for kw in keywords):
            return label
    return None


def discover_models(attrib_root: Path,
                    order: Sequence[Tuple[str, Sequence[str]]] = DEFAULT_MODEL_ORDER,
                    ) -> List[ModelEntry]:
    """Return one `ModelEntry` per slot in ``order`` that we can locate under
    ``attrib_root``.  Slots without a match are silently skipped.
    """
    candidates = [p for p in attrib_root.iterdir()
                  if p.is_dir() and not p.name.startswith("_")]
    by_label: Dict[str, Path] = {}
    for p in candidates:
        label = _match_model(p.name, order)
        if label is None:
            continue
        # Keep the first hit per label; this is deterministic because we sort.
        by_label.setdefault(label, p)

    models: List[ModelEntry] = []
    for label, _ in order:
        if label in by_label:
            models.append(ModelEntry(label=label, run_dir=by_label[label]))
    return models


def list_sample_dirs(run_dir: Path) -> List[Path]:
    return sorted(p for p in run_dir.iterdir()
                  if p.is_dir() and re.fullmatch(r"k\d{2}_\d+", p.name))


_BOXED_RE = re.compile(r"^class_(\d+)_boxed\.png$")


def list_class_ids(sample_dir: Path) -> List[int]:
    out: List[int] = []
    for child in sample_dir.iterdir():
        m = _BOXED_RE.match(child.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.ImageFont:
    # Try a few common paths; fall back to PIL default (small bitmap font).
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered(draw: ImageDraw.ImageDraw,
                   text: str,
                   box: Tuple[int, int, int, int],
                   fill: Tuple[int, int, int],
                   font: ImageFont.ImageFont) -> None:
    x0, y0, x1, y1 = box
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:  # very old PIL
        tw, th = draw.textsize(text, font=font)
    tx = x0 + ((x1 - x0) - tw) // 2
    ty = y0 + ((y1 - y0) - th) // 2
    draw.text((tx, ty), text, fill=fill, font=font)


def compose_compound(tiles: Sequence[Tuple[str, Path]],
                     title: str) -> Image.Image:
    """Return a single compound image stitching ``tiles`` horizontally.

    Each tile is a (label, path-to-png).  All tiles are resized to a common
    height (= max of their heights) while preserving aspect ratio so that
    side-by-side comparison stays fair.
    """
    if not tiles:
        raise ValueError("No tiles to compose")

    loaded = [(lbl, Image.open(p).convert("RGB")) for lbl, p in tiles]
    target_h = max(im.height for _, im in loaded)
    resized: List[Tuple[str, Image.Image]] = []
    for lbl, im in loaded:
        if im.height != target_h:
            new_w = round(im.width * (target_h / im.height))
            im = im.resize((new_w, target_h), Image.BILINEAR)
        resized.append((lbl, im))

    n = len(resized)
    tile_widths = [im.width for _, im in resized]
    total_w = sum(tile_widths) + TILE_GAP * (n - 1) + 2 * PAD
    total_h = TITLE_H + HEADER_H + target_h + 2 * PAD

    out = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(out)

    # Title strip
    draw.rectangle((0, 0, total_w, TITLE_H), fill=TITLE_BG)
    _draw_centered(draw, title, (0, 0, total_w, TITLE_H),
                   fill=TITLE_FG, font=_load_font(20))

    x = PAD
    header_font = _load_font(16)
    for lbl, im in resized:
        tile_w = im.width
        # Header strip above the tile
        draw.rectangle((x, TITLE_H, x + tile_w, TITLE_H + HEADER_H),
                       fill=HEADER_BG)
        _draw_centered(draw, lbl,
                       (x, TITLE_H, x + tile_w, TITLE_H + HEADER_H),
                       fill=HEADER_FG, font=header_font)
        # Paste the tile itself
        out.paste(im, (x, TITLE_H + HEADER_H + PAD))
        x += tile_w + TILE_GAP

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_compounds(attrib_root: Path = DEFAULT_ATTRIB_ROOT,
                    order: Sequence[Tuple[str, Sequence[str]]] = DEFAULT_MODEL_ORDER,
                    out_dir: Optional[Path] = None,
                    overwrite: bool = False,
                    ) -> Dict[str, int]:
    """Produce compound PNGs for every (sample, class) available in ALL
    discovered models.  Returns a small stats dict.
    """
    models = discover_models(attrib_root, order=order)
    if len(models) < 2:
        raise RuntimeError(
            f"Need at least 2 model run-dirs under {attrib_root}; found "
            f"{[m.label for m in models]}")

    out_dir = out_dir or (attrib_root / COMPOUND_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intersect sample_ids across all models.
    sample_sets: List[set] = []
    for m in models:
        sample_sets.append({p.name for p in list_sample_dirs(m.run_dir)})
    common_samples = sorted(set.intersection(*sample_sets))

    index: List[Dict[str, object]] = []
    stats = {"samples": 0, "compounds": 0, "skipped_missing_class": 0}

    for sid in common_samples:
        # Intersect class ids present across all models for this sample.
        class_sets: List[set] = []
        for m in models:
            class_sets.append(set(list_class_ids(m.run_dir / sid)))
        common_classes = sorted(set.intersection(*class_sets))
        missing_here = sum(len(s) for s in class_sets) - len(common_classes) * len(models)
        stats["skipped_missing_class"] += missing_here

        if not common_classes:
            continue
        stats["samples"] += 1

        sample_out = out_dir / sid
        sample_out.mkdir(parents=True, exist_ok=True)

        for cid in common_classes:
            out_path = sample_out / f"class_{cid}.png"
            if out_path.exists() and not overwrite:
                stats["compounds"] += 1
                continue
            tiles: List[Tuple[str, Path]] = []
            for m in models:
                tiles.append((m.label,
                              m.run_dir / sid / f"class_{cid}_boxed.png"))
            title = f"{sid}  |  class {cid}"
            img = compose_compound(tiles, title=title)
            img.save(out_path, format="PNG")
            stats["compounds"] += 1
            index.append({
                "sample_id": sid,
                "class_id": cid,
                "path": str(out_path.relative_to(attrib_root)),
                "models": [m.label for m in models],
            })

    # Write a small companion index for downstream scripts.
    with (out_dir / "index.json").open("w") as f:
        json.dump({
            "models": [{"label": m.label, "run_dir": m.run_dir.name} for m in models],
            "attrib_root": str(attrib_root),
            "stats": stats,
            "items": index,
        }, f, indent=2)

    return stats


def _parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compose per-(sample, class) compound PNGs that stitch the "
            "`class_<id>_boxed.png` tile across all 4 trained models."
        )
    )
    p.add_argument("--attrib-root", type=Path, default=DEFAULT_ATTRIB_ROOT,
                   help="Root directory containing per-model run folders.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help=f"Output directory (default: <attrib-root>/{COMPOUND_SUBDIR}).")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate compound PNGs even if they already exist.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_cli(argv)
    stats = build_compounds(
        attrib_root=args.attrib_root,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
    )
    print(f"[compose_boxed_grid] stats: {stats}")


if __name__ == "__main__":
    main()
