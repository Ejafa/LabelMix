from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Set

# Per-model parameter counts (in millions). Update these values with your
# manually verified counts for accurate VRAM scheduling.
MODEL_PARAMS_MILLIONS: Dict[str, float] = {
    "mobilenetv4_conv_large": 32.59,
    "mobilenetv4_conv_medium": 9.72,
    "mobilenetv4_hybrid_large": 37.76,
    "mobilenetv4_hybrid_medium": 11.07,
    "resnetv2_101": 44.54,
    "resnetv2_50": 25.55,
    "vit_base_patch16_rope_reg1_gap_256": 86.43,
    "vit_little_patch16_reg4_gap_256": 22.52,
    "vit_medium_patch16_reg1_gap_256": 38.88,
    "vit_wee_patch16_reg1_gap_256": 13.42,
}
DEFAULT_MODEL_PARAMS_MILLIONS = 30.0
_WARNED_UNKNOWN_MODELS: Set[str] = set()

# Approximate base VRAM from parameter count:
# base_model_gb ~= offset + slope * params_million
MODEL_VRAM_BASE_OFFSET_GB = 0.9
MODEL_VRAM_BASE_PER_MPARAM_GB = 0.065

OPTIMIZER_VRAM_MULTIPLIER: Dict[str, float] = {
    "sgd": 1.00,
    "momentum": 1.00,
    "nesterov": 1.00,
    "lars": 1.00,
    "adam": 1.30,
    "adamw": 1.30,
    "lamb": 1.30,
    "adafactor": 1.15,
    "adagrad": 1.15,
    "rmsprop": 1.15,
    "__default__": 1.20,
}


def _get_flag_value(args_list: Sequence[str], flag: str) -> Optional[str]:
    for i in range(len(args_list) - 1, -1, -1):
        token = str(args_list[i])
        if token == flag:
            if i + 1 < len(args_list):
                return str(args_list[i + 1])
            return None
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _resolve_effective_flag(flag: str, layers: Sequence[Sequence[str]]) -> Optional[str]:
    for args_list in reversed(layers):
        value = _get_flag_value(args_list, flag)
        if value is not None:
            return value
    return None


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _model_params_millions(model_name: str) -> float:
    key = str(model_name or "").strip().lower()
    if key in MODEL_PARAMS_MILLIONS:
        return float(MODEL_PARAMS_MILLIONS[key])
    if key and key not in _WARNED_UNKNOWN_MODELS:
        _WARNED_UNKNOWN_MODELS.add(key)
        print(
            "Warning: model not found in MODEL_PARAMS_MILLIONS: "
            f"'{key}', using default={DEFAULT_MODEL_PARAMS_MILLIONS}M."
        )
    return float(DEFAULT_MODEL_PARAMS_MILLIONS)


def _optimizer_factor(opt_name: str) -> float:
    opt = str(opt_name or "").strip().lower()
    return float(OPTIMIZER_VRAM_MULTIPLIER.get(opt, OPTIMIZER_VRAM_MULTIPLIER["__default__"]))


def _mode_factor(mode_value: Any) -> float:
    _ = mode_value
    return 1.1


def estimate_job_vram_gb(
    ray_gpus_per_node: int,
    config_data: Dict[str, Any],
    base_train_common: Sequence[str],
    exp_extra: Sequence[str],
    runner_extra: Sequence[str],
) -> Dict[str, Any]:
    layers = [base_train_common, exp_extra, runner_extra]

    model = str(config_data.get("model") or "")
    optimizer = str(
        _resolve_effective_flag("--opt", layers)
        or config_data.get("opt")
        or "adamw"
    ).strip()
    batch_size = _to_int(
        _resolve_effective_flag("--batch-size", layers)
        or config_data.get("batch_size")
        or 32,
        default=128,
    )
    img_size = _to_int(
        _resolve_effective_flag("--img-size", layers)
        or config_data.get("img_size")
        or 256,
        default=256,
    )
    if img_size <= 0:
        img_size = 256

    mode = (
        _resolve_effective_flag("--balanced-mode", layers)
        or config_data.get("balanced_mode")
    )

    model_params_m = _model_params_millions(model)
    base_model_gb = MODEL_VRAM_BASE_OFFSET_GB + (MODEL_VRAM_BASE_PER_MPARAM_GB * model_params_m)
    image_scale = (float(img_size) / 224.0) ** 2
    activation_gb = 0.010 * float(batch_size) * image_scale
    per_gpu_vram_gb = (base_model_gb + activation_gb) * _optimizer_factor(optimizer) * _mode_factor(mode)
    per_gpu_vram_gb = max(0.5, per_gpu_vram_gb)

    total_vram_gb = per_gpu_vram_gb * float(max(1, int(ray_gpus_per_node)))
    return {
        "estimated_vram_gb": round(total_vram_gb, 3),
        "vram_factors": {
            "model": model or "unknown",
            "model_params_m": round(model_params_m, 3),
            "optimizer": optimizer or "unknown",
            "batch_size": batch_size,
            "img_size": img_size,
            "mode": mode,
            "base_model_gb": round(base_model_gb, 3),
            "per_gpu_vram_gb": round(per_gpu_vram_gb, 3),
            "gpus_per_exp": max(1, int(ray_gpus_per_node)),
        },
    }
