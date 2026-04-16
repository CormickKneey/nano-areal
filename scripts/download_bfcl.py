"""
Download BFCL-V4 data files directly from the gorilla GitHub repository.

Files are saved to ~/.cache/nano-areal/bfcl/ (or --data-dir override).
Only multi-turn categories are downloaded by default (sufficient for RL training).

Usage:
    uv run python scripts/download_bfcl.py
    uv run python scripts/download_bfcl.py --data-dir ./data/bfcl
    uv run python scripts/download_bfcl.py --all   # include single-turn categories
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

GORILLA_RAW = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main"
    "/berkeley-function-call-leaderboard/bfcl_eval/data"
)

# Multi-turn categories (used for RL training)
MULTI_TURN_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

# Additional single-turn categories (optional)
SINGLE_TURN_CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
]

# Sub-directories to download alongside main files
SUBDIRS = [
    "possible_answer",
    "multi_turn_func_doc",
]


def _fetch(url: str) -> bytes | None:
    """Download URL, return bytes or None on 404."""
    try:
        req = Request(url, headers={"User-Agent": "nano-areal/0.1"})
        with urlopen(req, timeout=30) as r:
            return r.read()
    except HTTPError as e:
        if e.code == 404:
            return None
        raise


def download_category(category: str, data_dir: Path, verbose: bool = True) -> int:
    """
    Download main file + sub-directory files for one category.
    Returns number of files downloaded.
    """
    fname = f"BFCL_v4_{category}.json"
    count = 0

    # Main file
    dest = data_dir / fname
    if dest.exists():
        if verbose:
            print(f"  skip  {fname}  (already exists)")
    else:
        url = f"{GORILLA_RAW}/{fname}"
        data = _fetch(url)
        if data:
            dest.write_bytes(data)
            n = len([l for l in data.decode().splitlines() if l.strip()])
            if verbose:
                print(f"  ok    {fname}  ({n} samples)")
            count += 1
        else:
            if verbose:
                print(f"  miss  {fname}  (not found at {url})")

    # Sub-directory files (possible_answer, multi_turn_func_doc)
    for subdir in SUBDIRS:
        sub_dest = data_dir / subdir / fname
        if sub_dest.exists():
            if verbose:
                print(f"  skip  {subdir}/{fname}")
            continue
        url = f"{GORILLA_RAW}/{subdir}/{fname}"
        data = _fetch(url)
        if data:
            sub_dest.parent.mkdir(parents=True, exist_ok=True)
            sub_dest.write_bytes(data)
            if verbose:
                print(f"  ok    {subdir}/{fname}")
            count += 1
        # Sub-dir files are optional; silence 404s

    return count


def default_data_dir() -> Path:
    return Path.home() / ".cache" / "nano-areal" / "bfcl"


def main():
    parser = argparse.ArgumentParser(description="Download BFCL-V4 data files")
    parser.add_argument(
        "--data-dir", default=None,
        help=f"Destination directory (default: {default_data_dir()})"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Also download single-turn categories"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    categories = MULTI_TURN_CATEGORIES[:]
    if args.all:
        categories += SINGLE_TURN_CATEGORIES

    print(f"Downloading BFCL-V4 to: {data_dir}\n")

    total = 0
    for cat in categories:
        print(f"[{cat}]")
        total += download_category(cat, data_dir)

    print(f"\nDone. {total} file(s) downloaded.")
    print(f"Set BFCL_DATA_DIR={data_dir} or pass --bfcl-data-dir to train.py")


if __name__ == "__main__":
    main()
