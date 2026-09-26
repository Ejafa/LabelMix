"""Export native LaTeX result tables from the processed classification/runtime CSVs.

Run from the repository root with ``python -m evaluation.scripts.export_paper_result_tables``.
No plots or images are used to render tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..common import PROCESSED_DIR

# Keep table labels consistent with figure labels, without importing matplotlib.
METHOD_LABELS = {
    "noaug": "No Aug", "bare": "Single-Image Aug", "baseline": "Mixup+CutMix",
    "mixup": "Mixup", "cutmix": "CutMix", "mosaic": "RICAP", "fmix": "FMix",
    "gridmix": "GridMix", "resizemix": "ResizeMix", "saliencymix": "SaliencyMix",
    "smoothmix": "SmoothMix", "tokenmix": "TokenMix", "tla": "TLA",
    "labelmix-sce": "TreemapMix-SCE", "labelmix-mixed": "TreemapMix-mixed",
    "labelmix-pl": "TreemapMix-PL",
}
VIT_MODELS = {
    "vit_wee_patch16_reg1_gap_256": "ViT-Wee",
    "vit_little_patch16_reg4_gap_256": "ViT-Little",
    "vit_medium_patch16_reg1_gap_256": "ViT-Medium",
    "vit_betwixt_patch16_reg4_gap_256": "ViT-Betwixt",
}
ARCH_MODELS = {
    "convnextv2_nano": "ConvNeXtV2-Nano",
    "deit_tiny_patch16_224": "DeiT-Tiny",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
}
FOUR_METHODS = ["baseline", "mosaic", "labelmix-sce", "labelmix-pl"]


def _cells(df, methods, models, metric, scale=1., decimals=2):
    """Return model columns of mean/std cells, highlighting the best mean."""
    indexed = df.set_index(["type", "model"], verify_integrity=True)
    result = {method: [] for method in methods}
    for model in models:
        rows = indexed.loc[[(method, model) for method in methods]]
        means = rows[f"{metric}_mean"]
        best = means.max() if metric == "top1_acc" else means.min()
        for method in methods:
            row = indexed.loc[(method, model)]
            value = f"{scale * row[f'{metric}_mean']:.{decimals}f} \\pm {scale * row[f'{metric}_std']:.{decimals}f}"
            if row[f"{metric}_mean"] == best:
                value = r"\mathbf{" + value + "}"
            result[method].append("$" + value + "$")
    return result


def _row(cells):
    return "    " + " & ".join(cells) + r" \\" + "\n"


def _table(caption, label, columns, body, placement="t"):
    return (
        f"\\begin{{table*}}[{placement}]\n  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n  \\centering\n  \\footnotesize\n"
        "  \\setlength{\\tabcolsep}{4pt}\n"
        f"  \\begin{{tabular}}{{@{{}}{columns}@{{}}}}\n    \\toprule\n"
        + body + "    \\bottomrule\n  \\end{tabular}\n\\end{table*}\n"
    )


def paired_table(df, methods, models, caption, label, placement="t"):
    """Accuracy/ECE pairs per backbone, as in the long-horizon table."""
    body = _row([""] + [f"\\multicolumn{{2}}{{c}}{{{name}}}" for name in models.values()])
    body += "    " + " ".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(models))) + "\n"
    body += _row(["Method"] + [r"Acc. $\uparrow$", r"ECE $\downarrow$"] * len(models))
    body += "    \\midrule\n"
    accuracy = _cells(df, methods, models, "top1_acc")
    ece = _cells(df, methods, models, "ece/n_bins=15", 100.)
    for method in methods:
        cells = [cell for pair in zip(accuracy[method], ece[method], strict=True) for cell in pair]
        body += _row([METHOD_LABELS[method], *cells])
    return _table(caption, label, "l" + "cc" * len(models), body, placement)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "paper" / "fig")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def read(stem):
        return pd.read_csv(args.processed_dir / f"{stem}.csv")

    def write(stem, text):
        (args.output_dir / f"{stem}.tex").write_text(text, encoding="utf-8")

    long = read("in1k_per_experiment_long")
    write("tab_in1k_long_horizon", paired_table(
        long, ["bare", "baseline", "labelmix-sce", "labelmix-pl"],
        dict(list(VIT_MODELS.items())[:2]),
        r"ImageNet-1K long-horizon results for ViT-Wee and ViT-Little under a 320-epoch schedule. Accuracy is reported in percent, and ECE in percentage points using 15 bins. Values are mean $\pm$ one sample standard deviation over seeds 42, 43, and 44. Bold marks the best mean for each model and metric.",
        "tab:in1k-long-horizon-results"))

    lt = read("long_tail_per_experiment_short")
    body = _row(["Method", *VIT_MODELS.values()])
    for heading, metric, scale in [(r"Top-1 accuracy (\%) $\uparrow$", "top1_acc", 1.), (r"ECE (percentage points) $\downarrow$", "ece/n_bins=15", 100.)]:
        body += "    \\midrule\n" + _row([f"\\multicolumn{{5}}{{c}}{{{heading}}}"]) + "    \\midrule\n"
        cells = _cells(lt, FOUR_METHODS, VIT_MODELS, metric, scale)
        for method in FOUR_METHODS:
            body += _row([METHOD_LABELS[method], *cells[method]])
    write("tab_long_tail_results", _table(
        r"ImageNet-LT accuracy and calibration for four backbones. Values are mean $\pm$ one sample standard deviation over seeds 42 and 43. ECE uses 15 bins. Bold marks the best mean for each model and metric.",
        "tab:long-tail-results", "lcccc", body))

    write("tab_architecture_results", paired_table(
        read("architectures_per_experiment_short"), FOUR_METHODS, ARCH_MODELS,
        r"ImageNet-1K results on three additional architectures under the 110-epoch protocol. Values are mean $\pm$ one sample standard deviation over seeds 42, 43, and 44. Accuracy is reported in percent, and ECE in percentage points using 15 bins. Bold marks the best mean for each model and metric.",
        "tab:architecture-results", "htbp"))

    runtime = read("augmentation_loader_benchmark").set_index("configuration", verify_integrity=True)
    reference = runtime.loc["single_image", "throughput_images_per_second_mean"]
    labels = {"no_augmentation": "No Aug", "single_image": "Single-Image Aug", "mixup_cutmix": "Mixup+CutMix", "treemapmix_sce": r"TreemapMix-SCE ($K=4$)", "treemapmix_pl": r"TreemapMix-PL ($K=6$)", "mosaic": "Mosaic (recorded)"}
    body = _row(["Method", r"\makecell{Throughput\\(images/s)}", r"\makecell{Throughput\\reduction}", r"\makecell{Peak RSS\\(GiB)}", r"\makecell{Peak USS\\(GiB)}"]) + "    \\midrule\n"
    for key, label in labels.items():
        row = runtime.loc[key]
        values = [label]
        for metric, scale, decimals in [("throughput_images_per_second", 1., 1), ("peak_rss_mb", 1/1024, 2), ("peak_uss_mb", 1/1024, 2)]:
            values.append(f"${scale*row[f'{metric}_mean']:.{decimals}f} \\pm {scale*row[f'{metric}_std']:.{decimals}f}$")
        reduction = 100 * (1 - row["throughput_images_per_second_mean"] / reference)
        values.insert(2, f"${reduction:+.1f}\\%$")
        body += _row(values)
    write("tab_augmentation_loader_runtime", _table(
        r"Augmentation-loader throughput and peak host memory: mean $\pm$ sample standard deviation over five runs, with batch size 128, eight workers, 20 warm-up batches, and 100 measured batches. Throughput reduction is $100\times(1-T/T_{\mathrm{single}})$ using mean throughput. RSS and USS are process-tree resident and unique memory. The recorded Mosaic label does not establish a separate RICAP timing result.",
        "tab:augmentation-loader-runtime", "lcccc", body))

    short = read("in1k_per_experiment_short")
    methods = list(METHOD_LABELS)
    for suffix, metric, heading, scale, decimals in [
        ("acc", "top1_acc", "Top-1 accuracy (percent)", 1., 2),
        ("ece", "ece/n_bins=15", "ECE (15 bins, percentage points)", 100., 2),
        ("nll", "nll", "negative log-likelihood", 1., 3),
        ("brier", "brier", "Brier score", 1., 3),
    ]:
        arrow = r"$\uparrow$" if metric == "top1_acc" else r"$\downarrow$"
        body = _row(["Method", *[name + " " + arrow for name in VIT_MODELS.values()]]) + "    \\midrule\n"
        cells = _cells(short, methods, VIT_MODELS, metric, scale, decimals)
        for method in methods:
            body += _row([METHOD_LABELS[method], *cells[method]])
        write(f"tab_in1k_full_results_{suffix}", _table(
            f"Full ImageNet-1K {heading} results under the matched 110-epoch protocol. "
            r"Values are mean $\pm$ one sample standard deviation over seeds 42, 43, and 44. Each method uses one run per seed. Bold marks the best mean within each model column.",
            f"tab:in1k-full-results-{suffix}", "lcccc", body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
