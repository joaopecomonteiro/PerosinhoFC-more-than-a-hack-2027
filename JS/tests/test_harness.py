"""Noise model against the published numbers, harness scoring, IDSSE conversion (toy objects)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mqf import decode, synthetic
from mqf.harness import evaluate, idsse, noise


def _clean():
    return synthetic.clean_match(seed=101)


def test_noise_matches_published_numbers():
    m = _clean()
    graded = m["labels"] != "X"
    trk = noise.corrupt(m["players"], m["team_of_player"], m["ball"], graded, m["fps"], noise.NoiseConfig(seed=0))
    assert trk.players.shape[1] == 46
    missing = np.isnan(trk.ball[graded, 0]).mean()
    assert abs(missing - 0.41) < 0.02, missing                         # DATA.md: 41 % of graded frames
    n_obj = (~np.isnan(trk.players[graded, :, 0])).sum(axis=1).mean()
    assert 19.0 < n_obj < 22.0, n_obj                                  # DATA.md: ~20.5 in play


def test_harness_scoring_uses_official_metric():
    m = _clean()
    pred = evaluate.predict(m, decode.Params(), noise.NoiseConfig(seed=0))
    r = evaluate.score(m["labels"], pred, m["fps"])
    for k in ("spellF1", "macroF1", "n_gt_spells", "transF1_25", "alive_dead_f1", "by_length"):
        assert k in r
    assert sum(v["n"] for v in r["by_length"].values()) == r["n_gt_spells"]
    assert 0.0 <= r["spellF1"] <= 1.0


def _xy(a):
    return SimpleNamespace(xy=a)


def test_idsse_from_floodlight_toy():
    T = 50
    rng = np.random.default_rng(0)
    xy, pos, stat = {}, {}, {}
    for seg in idsse.SEGMENTS:
        xy[seg] = {"Home": _xy(rng.uniform(-50, 50, (T, 2 * 3))), "Away": _xy(rng.uniform(-50, 50, (T, 2 * 2))),
                   "Ball": _xy(np.zeros((T, 2)))}
        stat[seg] = SimpleNamespace(code=np.r_[np.zeros(5), np.ones(T - 5)])
        pos[seg] = SimpleNamespace(code=np.r_[np.ones(T // 2), np.full(T - T // 2, 2.0)])
    pitch = SimpleNamespace(xlim=(-52.5, 52.5), ylim=(-34.0, 34.0))
    m = idsse.from_floodlight(xy, pos, stat, pitch, fps=25, gap_s=1.0)
    gap = 25
    assert m["halves"] == ((1, gap, gap + T - 1), (2, 2 * gap + T, 2 * gap + 2 * T - 1))
    assert m["players"].shape == (3 * gap + 2 * T, 5, 2)
    assert m["pitch"] == (105.0, 68.0)
    _, f1, l1 = m["halves"][0]
    lab = m["labels"][f1:l1 + 1]
    assert (lab[:5] == "D").all() and (lab[5:T // 2] == "H").all() and (lab[T // 2:] == "A").all()
    assert (m["labels"][:gap] == "X").all()
    np.testing.assert_allclose(m["ball"][f1], [52.5, 34.0])            # centred -> corner origin
    assert list(m["team_of_player"]) == [0, 0, 0, 1, 1]


def test_idsse_local_files(tmp_path):
    assert idsse.raw_files("J03WMX", tmp_path) is None
    (tmp_path / "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml").write_text("<x/>")
    assert idsse.raw_files("J03WMX", tmp_path) is None                 # matchinformation still missing
    (tmp_path / "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml").write_text("<x/>")
    pos, info = idsse.raw_files("J03WMX", tmp_path)
    assert "positions" in pos.name and "matchinformation" in info.name
    assert idsse.raw_files("J03WN1", tmp_path) is None
