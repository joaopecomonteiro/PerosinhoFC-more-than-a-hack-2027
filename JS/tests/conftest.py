"""Shared fixtures. Nothing here touches the real match: synthetic games and toy objects only.

The caches go to a temporary folder (set before mqf is imported), so a test run never writes into
cache/ next to the arrays derived from the real data.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MQF_CACHE_DIR", tempfile.mkdtemp(prefix="mqf-test-cache-"))

import pytest  # noqa: E402

from mqf import config  # noqa: E402

N_FRAMES = 45_000            # synthetic.generate() defaults


@pytest.fixture(scope="session")
def official():
    if not (config.OFFICIAL_DIR / "games.py").is_file():
        pytest.exit("official repo missing: run  python bootstrap.py  first", returncode=2)
    return config.official()


@pytest.fixture(scope="session")
def syn_spec(tmp_path_factory, official):
    from mqf import synthetic
    return synthetic.generate(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="session")
def syn_submission(syn_spec, tmp_path_factory):
    from mqf import decode, make_submission
    out = tmp_path_factory.mktemp("results") / "syn.csv"
    make_submission.run(syn_spec, decode.Params(), out)
    return out
