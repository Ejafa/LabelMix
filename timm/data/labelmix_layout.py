import torch

# =============================================================================
# Squarified Treemap (TorchScript-friendly core) + D4 layout cache
# =============================================================================
#
# Design goals:
# - The core layout algorithm is TorchScript-friendly (runs well on CPU).
# - If input weights are on CUDA, we automatically run the control-flow layout
#   on CPU to avoid GPU sync stalls from `.item()` and Python-style loops, then
#   move the rectangles back to the original device.
# - Output rectangles are in (x, y, w, h) within [0, canvas_size].
#
# Notes:
# - Randomization is included (randperm + random fill direction + carve side).
#   For deterministic results, call `torch.manual_seed(seed)` before running.
# - `sorted_weights` are expected to be sorted descending by the caller.


# ----------------------------
# TorchScript helpers
# ----------------------------

@torch.jit.script
def _worst_score(row_sum: float, row_min: float, row_max: float, side: float) -> float:
    """
    Aspect-ratio quality metric (lower is better). Equivalent to the classic
    'worst' metric used by squarify:
        max( (side^2 * max_a) / sum^2,  sum^2 / (side^2 * min_a) )
    """
    if row_sum <= 0.0 or row_min <= 0.0 or row_max <= 0.0 or side <= 0.0:
        return 1e18
    s2 = row_sum * row_sum
    w2 = side * side
    r1 = (w2 * row_max) / s2
    r2 = s2 / (w2 * row_min)
    return r1 if r1 > r2 else r2


@torch.jit.script
def _safe_div(num: float, den: float, eps: float) -> float:
    """Safe division for float control-flow (avoids Python max() in TorchScript)."""
    return num / den if den > eps else 0.0


# ----------------------------
# TorchScript core (CPU)
# ----------------------------

@torch.jit.script
def _squarify_core_cpu(sorted_weights_cpu: torch.Tensor, canvas_size: float) -> torch.Tensor:
    """
    TorchScript squarified treemap on CPU.

    Args:
        sorted_weights_cpu: (K,) CPU tensor of nonnegative weights (float32 recommended).
        canvas_size: square canvas side length.

    Returns:
        (K,4) float32 CPU tensor of rectangles (x, y, w, h) in [0, canvas_size].
    """
    # Ensure float32 for stable geometry math
    w_in = sorted_weights_cpu.to(dtype=torch.float32)

    K = int(w_in.numel())
    out = torch.zeros((K, 4), dtype=torch.float32, device=w_in.device)

    if K == 0 or canvas_size <= 0.0:
        return out

    # Treemaps typically treat negative weights as 0 area.
    w_in = torch.clamp(w_in, min=0.0)

    # Convert weights -> areas that fill the square canvas.
    total_area = canvas_size * canvas_size
    weight_sum = float(torch.sum(w_in).item())
    if weight_sum > 0.0:
        areas = w_in * (total_area / weight_sum)
    else:
        areas = torch.zeros_like(w_in)

    # Current remaining canvas (top-left origin).
    x = 0.0
    y = 0.0
    W = canvas_size
    H = canvas_size

    eps = 1e-12
    start = 0

    # Each iteration places one "strip" (row or column), then shrinks the remaining canvas.
    while start < K and W > 0.0 and H > 0.0:
        vertical_stack = W < H                  # If tall, place a horizontal strip (stack vertically)
        side = W if vertical_stack else H       # Fixed side length for scoring the strip

        if side <= eps:
            break

        # Choose the best row length incrementally (O(K) overall):
        # maintain (sum, min, max) without repeatedly reducing slices.
        a0 = float(areas[start].item())
        row_sum = a0
        row_min = a0
        row_max = a0
        best = _worst_score(row_sum, row_min, row_max, side)

        row_len = 1
        i = start + 1
        while i < K:
            ai = float(areas[i].item())
            new_sum = row_sum + ai
            new_min = row_min if row_min < ai else ai
            new_max = row_max if row_max > ai else ai
            score = _worst_score(new_sum, new_min, new_max, side)

            # Keep extending while aspect ratio improves or stays the same.
            if score <= best:
                best = score
                row_sum = new_sum
                row_min = new_min
                row_max = new_max
                row_len += 1
                i += 1
            else:
                break

        end = start + row_len

        # Randomize placement order inside the strip (keeps output indexing stable).
        perm = torch.randperm(row_len, device=areas.device) if row_len > 1 else torch.zeros((1,), dtype=torch.int64, device=areas.device)

        # Randomize direction within strip and whether we carve from the far side.
        reverse = float(torch.rand((1,), device=areas.device).item()) < 0.5
        from_end = float(torch.rand((1,), device=areas.device).item()) < 0.5

        if vertical_stack:
            # Place a horizontal strip of height row_h, fill it with varying widths.
            row_h = _safe_div(row_sum, W, eps)
            if row_h > H:
                row_h = H  # clamp in case of tiny numerical drift

            row_y = (y + H - row_h) if from_end else y

            if reverse:
                curr_x = x + W
                for k in range(row_len):
                    idx = int(perm[k].item())
                    area_k = float(areas[start + idx].item())

                    rect_w = (curr_x - x) if (k == row_len - 1) else _safe_div(area_k, row_h, eps)
                    if rect_w < 0.0:
                        rect_w = 0.0

                    curr_x -= rect_w
                    out[start + idx, 0] = curr_x
                    out[start + idx, 1] = row_y
                    out[start + idx, 2] = rect_w
                    out[start + idx, 3] = row_h
            else:
                curr_x = x
                for k in range(row_len):
                    idx = int(perm[k].item())
                    area_k = float(areas[start + idx].item())

                    rect_w = ((x + W) - curr_x) if (k == row_len - 1) else _safe_div(area_k, row_h, eps)
                    if rect_w < 0.0:
                        rect_w = 0.0

                    out[start + idx, 0] = curr_x
                    out[start + idx, 1] = row_y
                    out[start + idx, 2] = rect_w
                    out[start + idx, 3] = row_h
                    curr_x += rect_w

            # Shrink the remaining canvas by removing the placed strip.
            H -= row_h
            if not from_end:
                y += row_h

        else:
            # Place a vertical strip of width row_w, fill it with varying heights.
            row_w = _safe_div(row_sum, H, eps)
            if row_w > W:
                row_w = W  # clamp in case of tiny numerical drift

            row_x = (x + W - row_w) if from_end else x

            if reverse:
                curr_y = y + H
                for k in range(row_len):
                    idx = int(perm[k].item())
                    area_k = float(areas[start + idx].item())

                    rect_h = (curr_y - y) if (k == row_len - 1) else _safe_div(area_k, row_w, eps)
                    if rect_h < 0.0:
                        rect_h = 0.0

                    curr_y -= rect_h
                    out[start + idx, 0] = row_x
                    out[start + idx, 1] = curr_y
                    out[start + idx, 2] = row_w
                    out[start + idx, 3] = rect_h
            else:
                curr_y = y
                for k in range(row_len):
                    idx = int(perm[k].item())
                    area_k = float(areas[start + idx].item())

                    rect_h = ((y + H) - curr_y) if (k == row_len - 1) else _safe_div(area_k, row_w, eps)
                    if rect_h < 0.0:
                        rect_h = 0.0

                    out[start + idx, 0] = row_x
                    out[start + idx, 1] = curr_y
                    out[start + idx, 2] = row_w
                    out[start + idx, 3] = rect_h
                    curr_y += rect_h

            # Shrink the remaining canvas by removing the placed strip.
            W -= row_w
            if not from_end:
                x += row_w

        start = end

    return out


# ----------------------------
# Public API: device-aware core
# ----------------------------

def squarify_core(sorted_weights: torch.Tensor, canvas_size: float) -> torch.Tensor:
    """
    Device-aware wrapper around the TorchScript CPU core.

    - If `sorted_weights` is CUDA, runs the sequential control-flow on CPU and
      returns rectangles on the original CUDA device.
    - If `sorted_weights` is CPU, runs entirely on CPU.

    Args:
        sorted_weights: (K,) tensor, typically sorted descending.
        canvas_size: size of square canvas.

    Returns:
        (K,4) float32 tensor of rectangles on the same device as `sorted_weights`.
    """
    if sorted_weights.is_cuda:
        base_cpu = _squarify_core_cpu(sorted_weights.detach().cpu(), float(canvas_size))
        return base_cpu.to(device=sorted_weights.device)
    return _squarify_core_cpu(sorted_weights, float(canvas_size))


# ----------------------------
# Layout cache: D4 symmetries
# ----------------------------

def generate_layout_cache(sorted_weights: torch.Tensor, canvas_size: float = 1.0, clamp: bool = False) -> torch.Tensor:
    """
    Generate the base squarified treemap, then precompute all 8 D4 symmetries.

    Output layout_cache has shape (8, K, 4), each slice is (K,4) as (x,y,w,h):
      0: identity
      1: rot90
      2: rot180
      3: rot270
      4: flipH        (mirror left-right)
      5: flipV        (mirror top-bottom)
      6: transpose    (reflect across y=x)
      7: anti-transpose (reflect across y=-x within [0,S])

    Args:
        sorted_weights: (K,) weights, typically sorted descending.
        canvas_size: size of square canvas (default 1.0).
        clamp: if True, clamps for tiny numerical drift and enforces x+w<=S, y+h<=S.

    Returns:
        (8, K, 4) tensor on the same device as `sorted_weights`.
    """
    base = squarify_core(sorted_weights, float(canvas_size))  # (K,4) on input device
    S = float(canvas_size)
    K = int(base.size(0))

    # Views into base (no copies)
    x = base[:, 0]
    y = base[:, 1]
    w = base[:, 2]
    h = base[:, 3]

    # Common reused terms
    xr = S - x - w
    yr = S - y - h

    out = torch.empty((8, K, 4), dtype=base.dtype, device=base.device)

    # 0: identity
    out[0, :, 0] = x;  out[0, :, 1] = y;  out[0, :, 2] = w;  out[0, :, 3] = h
    # 1: rot90
    out[1, :, 0] = y;  out[1, :, 1] = xr; out[1, :, 2] = h;  out[1, :, 3] = w
    # 2: rot180
    out[2, :, 0] = xr; out[2, :, 1] = yr; out[2, :, 2] = w;  out[2, :, 3] = h
    # 3: rot270
    out[3, :, 0] = yr; out[3, :, 1] = x;  out[3, :, 2] = h;  out[3, :, 3] = w
    # 4: flipH (mirror left-right)
    out[4, :, 0] = xr; out[4, :, 1] = y;  out[4, :, 2] = w;  out[4, :, 3] = h
    # 5: flipV (mirror top-bottom)
    out[5, :, 0] = x;  out[5, :, 1] = yr; out[5, :, 2] = w;  out[5, :, 3] = h
    # 6: transpose (reflect across y=x)
    out[6, :, 0] = y;  out[6, :, 1] = x;  out[6, :, 2] = h;  out[6, :, 3] = w
    # 7: anti-transpose (reflect across y=-x within [0,S])
    out[7, :, 0] = yr; out[7, :, 1] = xr; out[7, :, 2] = h;  out[7, :, 3] = w

    if clamp:
        # Clamp x/y into [0,S] and w/h to nonnegative.
        out[:, :, 0].clamp_(0.0, S)
        out[:, :, 1].clamp_(0.0, S)
        out[:, :, 2].clamp_min_(0.0)
        out[:, :, 3].clamp_min_(0.0)

        # Enforce containment: x+w <= S and y+h <= S (avoids "independent clamp" artifacts).
        max_w = (S - out[:, :, 0]).clamp_min(0.0)
        max_h = (S - out[:, :, 1]).clamp_min(0.0)
        out[:, :, 2].copy_(torch.minimum(out[:, :, 2], max_w))
        out[:, :, 3].copy_(torch.minimum(out[:, :, 3], max_h))

    return out
