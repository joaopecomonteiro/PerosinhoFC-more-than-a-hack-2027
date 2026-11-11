"""Both layouts: our code at the top of its own repo, or in JS/ of a fork of the official repo."""
from __future__ import annotations

import os

from mqf import config


def test_repo_root_detection(tmp_path):
    pkg = tmp_path / "JS"
    pkg.mkdir()
    assert config._repo_root(pkg) == pkg                               # own repo
    (tmp_path / "games.py").write_text("")
    assert config._repo_root(pkg) == pkg                               # games.py alone is not enough
    (tmp_path / "evaluation").mkdir()
    assert config._repo_root(pkg) == tmp_path                          # fork of the official repo


def test_paths_follow_the_layout():
    if "MQF_EXTERNAL_RAW" not in os.environ:
        assert config.EXTERNAL_RAW_DIR == config.REPO_ROOT / "external_data"
    if "MTAH_DATA_DIR" not in os.environ:
        assert config.DATA_DIR == config.REPO_ROOT / "data"
    if "MTAH_OFFICIAL_DIR" not in os.environ:
        assert config.OFFICIAL_DIR == config.ROOT / "vendor" / "mtah2027"
    assert config.CACHE_DIR != config.EXTERNAL_DIR
