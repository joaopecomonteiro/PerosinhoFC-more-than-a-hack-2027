"""Noise model against the published numbers, harness scoring, IDSSE conversion (toy objects)."""
from __future__ import annotations

from dataclasses import replace
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


def _toy_with_keepers(n=3000):
    rng = np.random.default_rng(3)
    outfield = rng.uniform([25, 5], [80, 63], (20, 2))
    base = np.concatenate([[[2.0, 34.0]], outfield[:10], [[103.0, 34.0]], outfield[10:]])   # keepers: index 0 and 11
    players = base[None] + np.cumsum(rng.normal(0, 0.05, (n, 22, 2)), axis=0)
    team = np.r_[np.zeros(11, int), np.ones(11, int)]
    ball = np.tile([52.0, 34.0], (n, 1))
    return players, team, ball


def test_goalkeepers_found_without_a_roster():
    players, team, _ = _toy_with_keepers()
    assert sorted(noise.goalkeepers(players, team).tolist()) == [0, 11]


def test_keeper_team_flips_only_move_pickups_to_the_other_team():
    players, team, ball = _toy_with_keepers()
    graded = np.ones(len(ball), dtype=bool)
    base = noise.NoiseConfig(seed=0, track_break_per_min=6.0)
    a = noise.corrupt(players, team, ball, graded, 25, base)
    b = noise.corrupt(players, team, ball, graded, 25, replace(base, gk_team_flip_p=1.0))
    pts = lambda t: np.sort(t.players[~np.isnan(t.players[..., 0])].ravel())
    np.testing.assert_array_equal(pts(a), pts(b))                     # same points, only their slots/teams differ
    near_goal = lambda t, tm: np.nansum((t.players[:, t.team_of_slot == tm, 0] < 6))
    assert near_goal(a, 1) == 0 and near_goal(b, 1) > 0               # the home keeper now also sits in away slots
    assert near_goal(b, 0) > 0                                        # ... but his first tracklet is still home


def test_hold_lost_repeats_the_last_place_and_is_off_by_default():
    import numpy as np
    from mqf import synthetic
    from mqf.harness.noise import NoiseConfig, corrupt
    m = synthetic.clean_match(n_frames=6_000, seed=5, halves=((1, 200, 5_799),))
    g = m["labels"] != "X"
    a = corrupt(m["players"], m["team_of_player"], m["ball"], g, m["fps"], NoiseConfig(seed=1))
    b = corrupt(m["players"], m["team_of_player"], m["ball"], g, m["fps"], NoiseConfig(seed=1, hold_lost_s=0.0, hold_lost_p=0.3))
    np.testing.assert_array_equal(np.nan_to_num(a.players, nan=-1), np.nan_to_num(b.players, nan=-1))
    h = corrupt(m["players"], m["team_of_player"], m["ball"], g, m["fps"], NoiseConfig(seed=1, hold_lost_s=2.0))
    step = np.linalg.norm(np.diff(h.players, axis=0), axis=-1)
    exact = np.nan_to_num(step, nan=np.inf) < 1e-6
    base = (np.nan_to_num(np.linalg.norm(np.diff(a.players, axis=0), axis=-1), nan=np.inf) < 1e-6).sum()
    assert exact.sum() > 100 * max(base, 1)                                       # held copies only with the knob on
    assert np.nanmax(h.player_speed[1:][exact]) == 0.0                           # a held copy reports no speed
