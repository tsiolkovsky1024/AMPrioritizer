"""Verify the released dataset against its recorded checksums."""
from __future__ import annotations

import json
from pathlib import Path

from .provenance import ROOT, sha256_file


DATASET_ROOT = ROOT / "data"
VERSION_DIRS = {"v2_censor_safe": DATASET_ROOT}
V2_REQUIRED = (
    "peptides.parquet", "mic_long.parquet", "mic_wide.parquet",
    "splits.parquet", "cluster_assignments.parquet", "top_species.json",
    "censoring_aggregation_audit.json",
)


def verify(version: str) -> dict:
    version_dir = VERSION_DIRS[version]
    manifest_path = version_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path.relative_to(ROOT)}")
    manifest = json.loads(manifest_path.read_text())
    failures = []
    for record in manifest.get("files", []):
        path = version_dir / record["path"]
        if not path.exists():
            failures.append(f"missing:{record['path']}")
        elif path.stat().st_size != record["bytes"]:
            failures.append(f"size:{record['path']}")
        elif sha256_file(path) != record["sha256"]:
            failures.append(f"sha256:{record['path']}")
    if version == "v2_censor_safe" and manifest.get("cluster_cross_split_leaks") != 0:
        failures.append("cluster split leakage")
    result = {"version": version, "ok": not failures, "failures": failures}
    if failures:
        raise SystemExit(json.dumps(result, indent=2))
    return result

