#!/usr/bin/env python3
"""Zip the diagnostic_attribution/k05_00051 folder and upload it to a
Hugging Face Hub dataset repo.

Defaults target ``KonstantinGarbers/labelmix-assets`` (dataset). The repo is
created if it does not yet exist. Authentication is taken from the environment
variable ``HF_TOKEN`` (or the canonical ``HUGGING_FACE_HUB_TOKEN``).

Usage::

    export HF_TOKEN=hf_xxx
    python evaluation/scripts/upload_k05_00051_to_hf.py

Override any default via CLI flags (see ``--help``).
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = (
    REPO_ROOT
    / "evaluation/data/processed/figures/diagnostic_attribution/k05_00051"
)
DEFAULT_REPO_ID = "KonstantinGarbers/labelmix-assets"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_PATH_IN_REPO = "diagnostic_attribution/k05_00051.zip"
DEFAULT_COMMIT_MSG = "Add diagnostic attribution bundle k05_00051"


def _resolve_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Set HF_TOKEN (write scope) in "
            "the environment before running this script."
        )
    return token


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help="Folder to zip and upload.",
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        choices=["dataset", "model", "space"],
    )
    p.add_argument("--path-in-repo", default=DEFAULT_PATH_IN_REPO)
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist yet.",
    )
    p.add_argument(
        "--no-create",
        action="store_true",
        help="Skip create_repo; assume the repo already exists.",
    )
    p.add_argument("--commit-message", default=DEFAULT_COMMIT_MSG)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    token = _resolve_token()

    src: Path = args.src.resolve()
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    with tempfile.TemporaryDirectory() as tmp:
        zip_stem = Path(tmp) / src.name  # shutil appends .zip
        archive = Path(
            shutil.make_archive(
                base_name=str(zip_stem),
                format="zip",
                root_dir=str(src.parent),
                base_dir=src.name,
            )
        )
        size_mb = archive.stat().st_size / 1e6
        print(f"[zip] {archive.name}  ({size_mb:.2f} MB)")

        api = HfApi(token=token)
        if not args.no_create:
            create_repo(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                private=args.private,
                exist_ok=True,
                token=token,
            )
            print(
                f"[repo] ensured {args.repo_type}:{args.repo_id} "
                f"(private={args.private})"
            )

        api.upload_file(
            path_or_fileobj=str(archive),
            path_in_repo=args.path_in_repo,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
            token=token,
        )

        slug = (
            f"datasets/{args.repo_id}"
            if args.repo_type == "dataset"
            else args.repo_id
        )
        print(
            "[done] "
            f"https://huggingface.co/{slug}/blob/main/{args.path_in_repo}"
        )


if __name__ == "__main__":
    main()
