"""One-shot helper: merge per-lrx job YAMLs and split into 3 round-robin
node files for the cifar100 vit-wee aug+lr sweep.

Run after generating ``/tmp/_aug_lr_sweep/lrx_*.yaml`` via
``experiments/generate_jobs.py`` (one invocation per lrx).
"""
import glob
import os
import yaml

SRC_GLOB = "/tmp/_aug_lr_sweep/lrx_*.yaml"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_NAME_FMT = "cifar100_vitwee_aug_lr_sweep_210e_bs256_1seed_node_{ni}_jobs.yaml"


def main() -> None:
    src = sorted(glob.glob(SRC_GLOB))
    all_jobs = []
    defaults = None
    for p in src:
        with open(p) as f:
            d = yaml.safe_load(f)
        if defaults is None:
            defaults = d["defaults"]
        all_jobs.extend(d["jobs"])

    print(f"Total jobs: {len(all_jobs)}")
    names = [j["name"] for j in all_jobs]
    if len(names) != len(set(names)):
        raise SystemExit("Duplicate job names detected!")

    # Annotate every job with the scheduler hints it expects.  The daemon
    # uses these to (a) co-schedule same-model jobs on a GPU and (b) guard
    # against OOM by checking ``memory_mib_per_gpu`` against free memory
    # before launching.  Values mirror those used by the prior
    # cifar100_vitwee_aug_sweep_210e_bs256_2seed node files.
    for j in all_jobs:
        j.setdefault("gpus", 2)
        j.setdefault("model_key", "vit-wee")
        j.setdefault("memory_mib_per_gpu", 48935)

    # The naive round-robin (i % 3) ends up segregating by RandAugment
    # magnitude because the source ordering is grouped (model, trial, lr,
    # magnitude, seed) and there are exactly 3 magnitudes -- every other
    # group of 3 lands on the same node.  To balance the per-node mix
    # across all axes (lrx, randaug magnitude, config flavor) we shuffle
    # with a fixed seed before splitting.
    import random

    rng = random.Random(0xC1FA12_AA17)
    shuffled = list(all_jobs)
    rng.shuffle(shuffled)

    nodes = [[], [], []]
    for i, j in enumerate(shuffled):
        nodes[i % 3].append(j)

    for ni, jobs in enumerate(nodes):
        out_path = os.path.join(BASE_DIR, OUT_NAME_FMT.format(ni=ni))
        with open(out_path, "w") as f:
            yaml.safe_dump(
                {"defaults": defaults, "jobs": jobs},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        print(f"Node {ni}: {len(jobs)} jobs -> {out_path}")


if __name__ == "__main__":
    main()
