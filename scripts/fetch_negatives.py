#!/usr/bin/env python
"""Download the AMPlify non-AMP FASTA used as the negative training pool.

The file is distributed with AMPlify (GPL-3.0) and is therefore not bundled with
this repository. It is required by both training configurations
(`data.neg_fasta`). The downloaded file is checked against a recorded SHA-256.

Usage:
    python scripts/fetch_negatives.py [--out data/negatives] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

FILENAME = "AMPlify_non_AMP_train_imbalanced.fa"
URL = (
    "https://raw.githubusercontent.com/bcgsc/AMPlify/master/"
    "data/AMPlify_non_AMP_train_imbalanced.fa"
)
SHA256 = "b8d5e54644ff8351938103893d3c5b3bfd181e627371f6e7224220cb0269f921"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/negatives")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / FILENAME

    if target.exists() and not args.force:
        digest = sha256_file(target)
        if digest == SHA256:
            print(f"already present and verified: {target}")
            return 0

    print(f"downloading {args.url}")
    try:
        with urllib.request.urlopen(args.url) as response:
            payload = response.read()
    except Exception as exc:  # noqa: BLE001 - surface the reason, not a traceback
        print(
            f"download failed: {exc}\n"
            "Obtain AMPlify from https://github.com/bcgsc/AMPlify and place\n"
            f"{FILENAME} in {out_dir}, then re-run this script.",
            file=sys.stderr,
        )
        return 1

    target.write_bytes(payload)
    digest = sha256_file(target)
    (out_dir / "SHA256SUMS").write_text(f"{digest}  {FILENAME}\n")

    if digest != SHA256:
        print(
            f"checksum mismatch\nexpected {SHA256}\ngot      {digest}\n"
            "The upstream file may have changed; review before training.",
            file=sys.stderr,
        )
        return 2
    print(f"wrote {target} ({len(payload)} bytes, sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
