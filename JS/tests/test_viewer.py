"""Viewer: the Original / Fixed tracking switch -- synthetic data only."""
from __future__ import annotations

import numpy as np

from mqf import synthetic
from mqf.harness.noise import NoiseConfig, corrupt
from mqf.viewer import fixed
from mqf.viewer import game as G


def _trk():
    m = synthetic.clean_match(n_frames=12_000, seed=11, halves=((1, 300, 5_699), (2, 6_300, 11_699)))
    trk = corrupt(m["players"], m["team_of_player"], m["ball"], m["labels"] != "X", m["fps"], NoiseConfig())
    return m, trk


def test_fixed_tracking_without_model_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fixed, "VIEWER_CACHE", tmp_path)
    m, trk = _trk()
    logs = []
    ts = fixed.fixed_tracking(trk, m["fps"], m["pitch"], m["halves"], model_path=tmp_path / "none.pt", log=logs.append)
    assert ts.key == "fixed" and ts.players.shape[0] == len(trk.ball) and len(ts.team_of_slot) == ts.players.shape[1]
    seen = ~np.isnan(trk.ball[:, 0])
    np.testing.assert_array_equal(ts.ball[seen], trk.ball[seen])                 # tracked ball untouched
    assert set(np.unique(ts.ball_source)) <= {0, 1, 3} and (ts.ball_source == 1).any()   # no model: interp only
    assert "only gaps <= 2 s" in ts.note
    ts2 = fixed.fixed_tracking(trk, m["fps"], m["pitch"], m["halves"], model_path=tmp_path / "none.pt", log=logs.append)
    assert any("cached" in l for l in logs) and np.array_equal(ts2.ball_source, ts.ball_source)


def test_bundle_lists_tracking_sets():
    m, trk = _trk()
    n = len(trk.ball)
    extra = G.TrackingSet("fixed", "Fixed", trk.players[:, :30], trk.team_of_slot[:30], trk.ball,
                          np.zeros(n, dtype=np.uint8), "note")
    g = G.ViewerGame("SYN", "A v B", "A", "B", m["fps"], n, m["pitch"], m["halves"], trk.players,
                     trk.team_of_slot, trk.ball, [], [], extra_tracking=[extra])
    meta, blobs = G.bundle(g)
    keys = [t["key"] for t in meta["tracking"]]
    assert keys == ["original", "fixed"]
    t = meta["tracking"][1]
    assert len(blobs[t["players"]]) == n * 30 * 4 and len(blobs[t["ball"]]) == n * 4 and len(blobs[t["ball_source"]]) == n
    assert meta["tracking"][0]["players"] == "players" and meta["n_slots"] == trk.players.shape[1]
