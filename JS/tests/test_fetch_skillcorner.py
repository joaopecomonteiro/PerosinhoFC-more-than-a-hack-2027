"""LFS pointer -> real file, with the network replaced by fakes."""
from __future__ import annotations

import hashlib
import io

import pytest

from mqf.external import fetch_skillcorner as fs


def _pointer(path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version https://git-lfs.github.com/spec/v1\n"
                    f"oid sha256:{hashlib.sha256(payload).hexdigest()}\nsize {len(payload)}\n")


@pytest.fixture
def tree(tmp_path):
    payload = b'{"frame": 10, "player_data": []}\n' * 1000
    f = tmp_path / "data" / "matches" / "123" / "123_tracking_extrapolated.jsonl"
    _pointer(f, payload)
    (tmp_path / "data" / "matches" / "123" / "123_match.json").write_text("{}")
    return tmp_path, f, payload


def _fake_net(monkeypatch, body: bytes):
    monkeypatch.setattr(fs, "_post_json", lambda url, payload: {
        "objects": [{**payload["objects"][0], "actions": {"download": {"href": "https://x/y", "header": {}}}}]})
    monkeypatch.setattr(fs, "_open", lambda url, headers: io.BytesIO(body))


def test_parse_and_find(tree):
    root, f, payload = tree
    p = fs.parse_pointer(f)
    assert p.size == len(payload) and p.oid == hashlib.sha256(payload).hexdigest()
    assert [q.path for q in fs.find_pointers(root)] == [f]
    assert fs.find_pointers(root, matches=["999"]) == []
    assert fs.parse_pointer(root / "data" / "matches" / "123" / "123_match.json") is None


def test_fetch_replaces_pointer(tree, monkeypatch):
    root, f, payload = tree
    _fake_net(monkeypatch, payload)
    fs.fetch(fs.parse_pointer(f), log=lambda *_: None)
    assert f.read_bytes() == payload
    assert fs.find_pointers(root) == []                                # rerun: nothing left to do


def test_fetch_rejects_corrupt_download(tree, monkeypatch):
    root, f, payload = tree
    before = f.read_bytes()
    _fake_net(monkeypatch, payload[:-1] + b"X")
    with pytest.raises(SystemExit):
        fs.fetch(fs.parse_pointer(f), log=lambda *_: None)
    assert f.read_bytes() == before                                    # pointer kept
    assert not list(f.parent.glob("*.part"))
