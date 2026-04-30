"""Split a ``{name: run_dir}`` mapping YAML into ``n_shards`` sub-mappings.

This lets us fan out the offline evaluator across several GPUs in parallel
(each shard + CSV becomes a separate process)::

    python -m evaluation.scripts.shard_mapping \\
        --mapping evaluation/data/raw/in1k_mapping.yaml \\
        --n-shards 4 \\
        --out-prefix evaluation/data/raw/in1k_mapping_shard
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping", required=True)
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--out-prefix", required=True,
                   help="Shards are written to ``<prefix>_00.yaml`` ... "
                        "``<prefix>_{n-1}.yaml``.")
    args = p.parse_args(argv)

    mapping = yaml.safe_load(Path(args.mapping).read_text())
    if not isinstance(mapping, dict):
        raise SystemExit("Mapping must be a dict")

    # Deterministic round-robin so a given run always lands in the same shard.
    items = sorted(mapping.items())
    shards: list[dict[str, str]] = [dict() for _ in range(args.n_shards)]
    for i, (k, v) in enumerate(items):
        shards[i % args.n_shards][k] = v

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for i, shard in enumerate(shards):
        out_path = prefix.parent / f"{prefix.name}_{i:02d}.yaml"
        out_path.write_text(yaml.safe_dump(shard, sort_keys=True))
        print(f"shard {i}: {len(shard)} entries -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
