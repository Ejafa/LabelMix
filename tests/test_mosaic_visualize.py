"""Visual smoke-test for the Mosaic augmentation.

Generates sample mosaic images so a human can verify the augmentation is
implemented correctly.  Two input modes are supported:

1. ``--data-dir <path>``: load real JPEG / PNG images from a directory
   (recursively).  This is the recommended mode for visual verification
   because solid-color tiles can be indistinguishable from padding /
   scaling artifacts.
2. Synthetic mode (default, no ``--data-dir``): generate colored tiles
   with a diagonal gradient and a large printed label index so each of
   the four quadrants is easy to tell apart.

Usage
-----
::

    # Real images (recommended)
    python tests/test_mosaic_visualize.py --data-dir /path/to/images \\
        --num-samples 8 --image-size 224

    # Synthetic fallback
    python tests/test_mosaic_visualize.py --num-samples 8 \\
        --image-size 224 --prob 1.0 \\
        --center-ratio 0.5 1.5 --scale-range 0.8 1.2 \\
        --out-dir ./mosaic_samples

The script prints the (labels, weights) associated with each mosaic image
next to the filename so you can eyeball the area fractions.

Close-mosaic verification
-------------------------
Use ``--close-epochs`` and ``--total-epochs`` with ``--force-epoch`` to
see that mosaic is disabled during the cooldown window — images past
``total_epochs - close_epochs`` will be passed-through (single-source).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

# Ensure the repo root is on PYTHONPATH when running the file directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from timm.data.mosaic_dataset import MosaicDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic single-sample dataset
# ---------------------------------------------------------------------------


_PALETTE: List[Tuple[int, int, int]] = [
    (230,  25,  75),  # red
    ( 60, 180,  75),  # green
    (255, 225,  25),  # yellow
    (  0, 130, 200),  # blue
    (245, 130,  48),  # orange
    (145,  30, 180),  # purple
    ( 70, 240, 240),  # cyan
    (240,  50, 230),  # magenta
    (210, 245,  60),  # lime
    (250, 190, 212),  # pink
    (  0, 128, 128),  # teal
    (220, 190, 255),  # lavender
    (170, 110,  40),  # brown
    (255, 250, 200),  # cream
    (128,   0,   0),  # maroon
    (170, 255, 195),  # mint
]


def _make_synthetic_tile(size: int, label: int) -> torch.Tensor:
    """Return a CHW float [0,1] tile with a color gradient + printed label.

    The tile is far from uniform:
      * a diagonal brightness gradient makes orientation/scale easy to see,
      * the label index is rendered in the center with a contrasting color.
    """
    S = int(size)
    r, g, b = _PALETTE[label % len(_PALETTE)]
    base = torch.tensor([r, g, b], dtype=torch.float32).view(3, 1, 1) / 255.0

    # Diagonal gradient in [0.5, 1.2] clipped below to [0, 1].
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, S),
        torch.linspace(0.0, 1.0, S),
        indexing="ij",
    )
    grad = (0.55 + 0.55 * (xx + yy) / 2.0).clamp(0.0, 1.0)  # H x W
    img = (base * grad.unsqueeze(0)).clamp(0.0, 1.0)

    # Slim corner bar for orientation (top-left = dark).
    bar = max(1, S // 48)
    img[:, 0:bar, :] *= 0.35
    img[:, :, 0:bar] *= 0.35

    # Draw a compact label number tightly around the tile center.
    try:
        from PIL import Image, ImageDraw, ImageFont

        pil = Image.new("RGB", (S, S), (0, 0, 0))
        arr = (img.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        pil = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil)
        txt = str(int(label))
        font_size = max(14, S // 6)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx_off, ty_off = bbox[0], bbox[1]
        except AttributeError:
            tw, th = draw.textsize(txt, font=font)
            tx_off, ty_off = 0, 0
        # Center the text's bounding box exactly on (S/2, S/2).
        pos = ((S - tw) // 2 - tx_off, (S - th) // 2 - ty_off)
        outline = (0, 0, 0) if sum((r, g, b)) > 380 else (255, 255, 255)
        fill = (255, 255, 255) if outline == (0, 0, 0) else (0, 0, 0)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((pos[0] + dx, pos[1] + dy), txt, font=font, fill=outline)
        draw.text(pos, txt, font=font, fill=fill)

        import numpy as np
        arr2 = np.asarray(pil, dtype="float32") / 255.0
        img = torch.from_numpy(arr2).permute(2, 0, 1).contiguous().float()
    except Exception:
        pass

    return img


class _SyntheticImageDataset(IterableDataset):
    """Emits a finite stream of (CHW float [0,1], label) pairs.

    Each sample is a colored tile with a diagonal gradient and the label
    index printed large in the center, so mosaic quadrant placement is
    easy to verify by eye.  Tiles are rendered at ``tile_size`` rather
    than at the mosaic output size so ``MosaicDataset`` has spatial room
    to take *different* random crops of each source.
    """

    def __init__(self, image_size: int, num_samples: int, num_classes: int = 16,
                 tile_size: Optional[int] = None) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.tile_size = int(tile_size) if tile_size is not None else int(image_size)

    def __len__(self) -> int:  # type: ignore[override]
        return self.num_samples

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        S = self.tile_size
        for i in range(self.num_samples):
            lbl = i % self.num_classes
            yield _make_synthetic_tile(S, lbl), lbl


# ---------------------------------------------------------------------------
# Real-image dataset (loads JPEG/PNG recursively from a directory)
# ---------------------------------------------------------------------------


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_image_files(root: str, limit: Optional[int] = None) -> List[str]:
    files: List[str] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(_IMG_EXTS):
                files.append(os.path.join(dp, fn))
                if limit is not None and len(files) >= limit:
                    return files
    return files


class _RealImageDataset(IterableDataset):
    """Loads real images from ``data_dir`` and yields (CHW float [0,1], lbl).

    Labels are synthesised as the file index — they are only used so that
    the mosaic's (labels, weights) output can be matched against the
    printed tiles' on-disk filenames (see the ``--dump-tiles`` flag).

    Note on tile size: images are resized so their *shorter* side equals
    ``tile_size`` (default: ``image_size``), aspect-ratio preserving.
    Picking ``tile_size > image_size`` gives ``MosaicDataset`` real room
    to take *different* random crops of each source on every call.
    """

    def __init__(self, data_dir: str, image_size: int, num_samples: int,
                 tile_size: Optional[int] = None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.image_size = int(image_size)
        self.num_samples = int(num_samples)
        self.tile_size = int(tile_size) if tile_size is not None else int(image_size)
        self.files = _list_image_files(data_dir, limit=max(num_samples, 64))
        if not self.files:
            raise FileNotFoundError(
                f"No images (.jpg/.jpeg/.png/.bmp/.webp) found under {data_dir!r}"
            )

    def __len__(self) -> int:  # type: ignore[override]
        return self.num_samples

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        from PIL import Image
        import numpy as np
        target_short = self.tile_size
        for i in range(self.num_samples):
            path = self.files[i % len(self.files)]
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                short = min(w, h)
                if short != target_short:
                    scale = float(target_short) / float(short)
                    new_w = max(target_short, int(round(w * scale)))
                    new_h = max(target_short, int(round(h * scale)))
                    im = im.resize((new_w, new_h), Image.BILINEAR)
                arr = np.asarray(im, dtype="float32") / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
            yield img, i % 1000


# ---------------------------------------------------------------------------
# Saving utilities
# ---------------------------------------------------------------------------


def _save_chw_image(tensor: torch.Tensor, path: str) -> None:
    """Save a CHW float [0,1] tensor as a PNG."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to save mosaic samples.") from exc

    img = tensor.detach().cpu().clamp(0.0, 1.0)
    if img.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")
    img = (img * 255.0).round().to(torch.uint8)
    arr = img.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sample Mosaic augmentation images for visual verification.",
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing real JPEG/PNG images to use as "
                             "input tiles (recursive). If omitted, synthetic "
                             "tiles with printed label indices are used.")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of mosaic samples to generate (default: 8).")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Mosaic output size S (square, default: 224).")
    parser.add_argument("--prob", type=float, default=1.0,
                        help="Mosaic probability (default: 1.0).")
    parser.add_argument("--center-ratio", type=float, nargs=2, default=[0.75, 1.25],
                        metavar=("LO", "HI"),
                        help="Center-ratio range (default: 0.75 1.25).")
    parser.add_argument("--scale-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Post-mosaic affine scale range, applied to the "
                             "full 2S x 2S canvas around the sampled center "
                             "BEFORE the S x S crop (YOLOv5 / MMYOLO style). "
                             "Default: None => disabled.")
    parser.add_argument("--close-epochs", type=int, default=0,
                        help="Close-mosaic last N epochs (default: 0).")
    parser.add_argument("--total-epochs", type=int, default=None,
                        help="Total training epochs (required if --close-epochs > 0).")
    parser.add_argument("--force-epoch", type=int, default=0,
                        help="Override the epoch reported to the dataset (default: 0).")
    parser.add_argument("--out-dir", type=str, default="./mosaic_samples",
                        help="Directory to write sample PNGs (default: ./mosaic_samples).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0). Controls all mosaic "
                             "randomness (crops / centers / post-scales) via "
                             "MosaicDataset's seed argument.")
    parser.add_argument("--source-scale", type=float, default=1.5,
                        help="Tile size fed into Mosaic is source_scale * "
                             "image_size, so _random_crop_tile has room to "
                             "take visibly different crops.  Use 1.0 to "
                             "reproduce the old no-randomness behavior "
                             "(default: 1.5).")
    args = parser.parse_args()

    if args.close_epochs > 0 and args.total_epochs is None:
        parser.error("--close-epochs > 0 requires --total-epochs.")
    if args.source_scale <= 0:
        parser.error("--source-scale must be > 0.")

    # Seed *all* RNGs used downstream: torch (DataLoader worker_info.seed),
    # python `random` (MosaicDataset's internal reseed salts on this when
    # seed is passed), and numpy (just for determinism of any tile RNG).
    torch.manual_seed(args.seed)
    import random as _random
    _random.seed(args.seed)
    try:
        import numpy as _np
        _np.random.seed(args.seed % (2 ** 32 - 1))
    except Exception:
        pass

    tile_size = max(args.image_size, int(round(args.image_size * args.source_scale)))

    num_needed = args.num_samples * 4 + 4
    if args.data_dir:
        synth = _RealImageDataset(
            data_dir=args.data_dir,
            image_size=args.image_size,
            num_samples=num_needed,
            tile_size=tile_size,
        )
        source_kind = f"real images from {args.data_dir}"
    else:
        synth = _SyntheticImageDataset(
            image_size=args.image_size,
            num_samples=num_needed,
            num_classes=16,
            tile_size=tile_size,
        )
        source_kind = "synthetic (labeled gradient tiles)"

    mosaic = MosaicDataset(
        base_dataset=synth,
        output_size=(args.image_size, args.image_size),
        prob=args.prob,
        center_ratio_range=tuple(args.center_ratio),
        post_scale_range=tuple(args.scale_range) if args.scale_range else None,
        close_epochs=args.close_epochs,
        total_epochs=args.total_epochs,
        seed=args.seed,
    )
    mosaic.set_epoch(args.force_epoch)

    os.makedirs(args.out_dir, exist_ok=True)

    print("Mosaic configuration:")
    print(f"  source               = {source_kind}")
    print(f"  tile_size (in)       = {tile_size}  (source_scale={args.source_scale})")
    print(f"  output_size          = ({args.image_size}, {args.image_size})")
    print(f"  prob                 = {args.prob}")
    print(f"  center_ratio_range   = {tuple(args.center_ratio)}")
    print(f"  post_scale_range     = {tuple(args.scale_range) if args.scale_range else None}")
    print(f"  close_epochs         = {args.close_epochs}")
    print(f"  total_epochs         = {args.total_epochs}")
    print(f"  force_epoch          = {args.force_epoch}")
    print(f"  seed                 = {args.seed}")
    print(f"  mosaic_enabled       = {mosaic._mosaic_enabled_this_epoch()}")
    print()

    saved = 0
    iterator = iter(mosaic)
    while saved < args.num_samples:
        try:
            img, tgt = next(iterator)
        except StopIteration:
            break
        if not (isinstance(tgt, tuple) and len(tgt) == 2):
            raise RuntimeError(f"Unexpected target format: {type(tgt)}")
        labels, weights = tgt
        # Passthrough samples have weights = [1, 0, 0, 0]; still save for
        # visual confirmation of close-mosaic / prob skipping.
        passthrough = bool(((weights > 0).sum().item() == 1) and (weights.max() > 0.99))
        mode = "passthrough" if passthrough else "mosaic"
        fname = f"{mode}_{saved:03d}.png"
        path = os.path.join(args.out_dir, fname)
        _save_chw_image(img, path)
        label_list = labels.tolist()
        weight_list = [round(float(w), 3) for w in weights.tolist()]
        print(f"  [{saved:03d}] {fname}  labels={label_list}  weights={weight_list}")
        saved += 1

    print(f"\nWrote {saved} sample image(s) to: {args.out_dir}")
    if saved == 0:
        print("WARNING: no samples produced — base dataset exhausted.")


if __name__ == "__main__":
    main()
