"""Integrity checks for the released dataset tables."""
from pathlib import Path

import pandas as pd
import pytest

from amp_prioritizer.data_versions import verify

DATA = Path(__file__).resolve().parents[1] / "data" / "current"


def test_manifest_checksums():
    assert verify("v2_censor_safe")["ok"]


@pytest.fixture(scope="module")
def tables():
    return {
        "peptides": pd.read_parquet(DATA / "peptides.parquet"),
        "mic": pd.read_parquet(DATA / "mic_long.parquet"),
        "splits": pd.read_parquet(DATA / "splits.parquet"),
        "clusters": pd.read_parquet(DATA / "cluster_assignments.parquet"),
    }


def test_row_counts(tables):
    assert len(tables["peptides"]) == 68_421
    assert len(tables["mic"]) == 106_728
    counts = tables["splits"]["split"].value_counts().to_dict()
    assert counts == {"train": 55_269, "val": 6_553, "test": 6_599}


def test_observation_types(tables):
    mic = tables["mic"]
    assert int(mic.censored_right.sum()) == 22_742
    assert int(mic.censored_left.sum()) == 470
    assert not (mic.censored_right & mic.censored_left).any()


def test_no_cluster_crosses_splits(tables):
    leaks = tables["clusters"].groupby("cluster_rep")["split"].nunique().gt(1).sum()
    assert leaks == 0


def test_every_mic_row_has_a_split(tables):
    keys = set(tables["splits"].dedup_key)
    assert set(tables["mic"].dedup_key) <= keys
