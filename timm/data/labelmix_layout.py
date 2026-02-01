import torch

# =====================================================================
# TorchScript Squarified Treemap Engine
# =====================================================================


@torch.jit.script
def get_aspect_ratio_score(areas: torch.Tensor, width: float) -> float:
    if areas.numel() == 0 or width <= 0.0:
        return 1e9
    s = float(torch.sum(areas))
    if s <= 0.0:
        return 1e9
    min_a = float(torch.min(areas))
    max_a = float(torch.max(areas))
    if min_a <= 0.0:
        return 1e9

    w2 = width * width
    s2 = s * s
    r1 = (w2 * max_a) / s2
    r2 = s2 / (w2 * min_a)
    return r1 if r1 > r2 else r2


@torch.jit.script
def squarify_core(sorted_weights: torch.Tensor, canvas_size: float) -> torch.Tensor:
    """
    Squarified treemap in normalized [0, canvas_size] coordinates.

    Args:
        sorted_weights: (K,) weights (not necessarily summing to 1), typically sorted descending.
        canvas_size: float, size of the square canvas.

    Returns:
        out: (K,4) float32 rectangles in (x, y, w, h).
    """
    total_area = canvas_size * canvas_size
    weight_sum = float(torch.sum(sorted_weights))
    if weight_sum > 0.0:
        areas = sorted_weights * (total_area / weight_sum)
    else:
        areas = torch.zeros_like(sorted_weights)

    x, y = 0.0, 0.0
    w, h = canvas_size, canvas_size
    K = int(areas.size(0))
    out = torch.zeros((K, 4), dtype=torch.float32, device=sorted_weights.device)

    start = 0
    while start < K:
        vertical_stack = w < h
        side = w if vertical_stack else h

        row_len = 1
        best = get_aspect_ratio_score(areas[start : start + 1], float(side))
        i = start + 1
        while i < K:
            score = get_aspect_ratio_score(areas[start : start + row_len + 1], float(side))
            if score <= best:
                best = score
                row_len += 1
                i += 1
            else:
                break

        end = start + row_len
        row_areas = areas[start:end]
        row_sum = float(torch.sum(row_areas))

        if vertical_stack:
            row_h = row_sum / w if w > 0.0 else 0.0
            curr_x = x
            for k in range(row_len):
                if k == row_len - 1:
                    rect_w = (x + w) - curr_x
                else:
                    rect_w = float(row_areas[k]) / row_h if row_h > 0.0 else 0.0
                out[start + k, 0] = curr_x
                out[start + k, 1] = y
                out[start + k, 2] = rect_w
                out[start + k, 3] = row_h
                curr_x += rect_w
            y += row_h
            h -= row_h
        else:
            row_w = row_sum / h if h > 0.0 else 0.0
            curr_y = y
            for k in range(row_len):
                if k == row_len - 1:
                    rect_h = (y + h) - curr_y
                else:
                    rect_h = float(row_areas[k]) / row_w if row_w > 0.0 else 0.0
                out[start + k, 0] = x
                out[start + k, 1] = curr_y
                out[start + k, 2] = row_w
                out[start + k, 3] = rect_h
                curr_y += rect_h
            x += row_w
            w -= row_w

        start = end

    return out


# =====================================================================
# Layout Cache with D4 Symmetries (layout-only augmentation)
# =====================================================================


def generate_layout_cache(sorted_weights: torch.Tensor, canvas_size: float = 1.0) -> torch.Tensor:
    """
    Generate base squarified treemap and precompute all 8 D4 symmetries
    in normalized coordinates.

    Returns:
        layout_cache: (8, K, 4) float tensor, each (K,4) is (x,y,w,h)
          0: identity
          1: rot90
          2: rot180
          3: rot270
          4: flipH
          5: flipV
          6: transpose
          7: anti-transpose
    """
    base_layout = squarify_core(sorted_weights, float(canvas_size))
    x, y, w, h = base_layout.unbind(-1)
    S = float(canvas_size)

    layouts = [
        torch.stack([x, y, w, h], dim=-1),
        torch.stack([y, S - x - w, h, w], dim=-1),
        torch.stack([S - x - w, S - y - h, w, h], dim=-1),
        torch.stack([S - y - h, x, h, w], dim=-1),
        torch.stack([S - x - w, y, w, h], dim=-1),
        torch.stack([x, S - y - h, w, h], dim=-1),
        torch.stack([y, x, h, w], dim=-1),
        torch.stack([S - y - h, S - x - w, h, w], dim=-1),
    ]

    layout_cache = torch.stack(layouts, dim=0)
    layout_cache[..., 0:2] = layout_cache[..., 0:2].clamp(0.0, S)
    layout_cache[..., 2:4] = layout_cache[..., 2:4].clamp(0.0, S)

    return layout_cache
