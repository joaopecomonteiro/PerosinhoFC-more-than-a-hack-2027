"""Bundles of other games: format round-trip, and the label rules of each source on toy inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mqf import atd
from mqf.external import bundle, metrica, skillcorner


def _toy_match(n=600, fps=25, P=6, seed=0):
    rng = np.random.default_rng(seed)
    players = rng.uniform([0, 0], [105, 68], (n, P, 2))
    players[100:130, 2] = np.nan
    ball = rng.uniform([0, 0], [105, 68], (n, 2))
    ball[200:260] = np.nan
    labels = np.array(["X"] * 50 + ["H"] * 200 + ["D"] * 50 + ["A"] * 250 + ["X"] * 50)
    return {"name": "TOY-1", "source": "toy", "fps": fps, "pitch": (105.0, 68.0), "players": players,
            "ball": ball, "team_of_player": np.r_[np.zeros(3), np.ones(3)].astype(np.int8),
            "labels": labels, "halves": ((1, 50, 549),), "home": "Home FC", "away": "Away & Co",
            "labels_from": "toy", "licence": ""}


def test_bundle_roundtrip(tmp_path):
    m = _toy_match()
    d = bundle.write_bundle(m, tmp_path)
    spec = bundle.spec(d)
    assert spec.n_frames == 600 and spec.fps == 25 and spec.halves == ((1, 50, 549),)
    trk = atd.load(spec, cache_dir=tmp_path / "cache")
    np.testing.assert_allclose(trk.players, m["players"], atol=1e-5 * 105, equal_nan=True)
    np.testing.assert_allclose(trk.ball, m["ball"], atol=1e-5 * 105, equal_nan=True)
    assert list(trk.team_of_slot) == [0, 0, 0, 1, 1, 1]
    gt = pd.read_csv(spec.gt_public, dtype={"label": str})
    assert list(gt["frame"]) == list(range(600)) and (gt["label"].to_numpy() == m["labels"]).all()
    first = spec.atd_raw.read_text().splitlines()[0]
    assert first.startswith("1:") and first.count(";") == 5                     # frame_id from 1, 6 slots
    sp = bundle.speed_from_positions(m["players"], 25)
    assert np.isnan(sp[100:130, 2]).all() and np.nanmax(sp) > 0                 # NaN where the player is missing
    assert bundle.find("toy", root=tmp_path) == [d] and bundle.find("idsse", root=tmp_path) == []


def test_bundle_keeps_frame_rate(tmp_path):
    m = _toy_match(fps=10)
    spec = bundle.spec(bundle.write_bundle(m, tmp_path))
    assert spec.fps == 10 and spec.n_frames == 600                              # no resampling


def _ev(rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "start", "end", "period"])


def test_metrica_event_rules():
    ev = _ev([
        (0, "SET PIECE", "KICK OFF", 1, 1, 1),
        (0, "PASS", "", 1, 20, 1),
        (0, "BALL OUT", "", 90, 100, 1),          # home kicks it out: dead from 100
        (1, "SET PIECE", "THROW IN", 150, 150, 1),  # away restarts at 150
        (1, "PASS", "", 150, 170, 1),
        (1, "BALL LOST", "INTERCEPTION", 200, 200, 1),
        (0, "CHALLENGE", "AERIAL-WON", 205, 205, 1),  # loose / contested: stays away
        (0, "RECOVERY", "", 220, 220, 1),         # home controls at 220
        (0, "PASS", "", 230, 260, 1),
        (1, "FAULT RECEIVED", "", 300, 300, 1),   # foul: dead until the free kick
        (1, "SET PIECE", "FREE KICK", 340, 340, 1),
        (1, "PASS", "", 340, 400, 1),
    ])
    frame = np.arange(1, 501)
    lab, halves = metrica.labels_from_events(ev, frame)
    assert halves == ((1, 0, 399),)
    at = lambda f: lab[f - 1]                     # frame id -> label
    assert at(1) == at(99) == "H"
    assert at(100) == at(149) == "D" and at(150) == "A"
    assert at(210) == "A" and at(220) == "H" and at(299) == "H"
    assert at(300) == at(339) == "D" and at(340) == "A"
    assert at(450) == "X"


def test_skillcorner_phase_rules():
    n = 100
    halves = ((1, 10, 89),)
    phases = pd.DataFrame({"frame_start": [10, 40], "frame_end": [30, 80], "team_in_possession_id": [7, 9]})
    group = np.full(n, -1, dtype=np.int8)
    group[15:20] = 0                              # home possessor
    group[25:28] = 1                              # away possessor
    group[50:55] = 0
    lab = skillcorner.labels_from_phases(n, halves, phases, group, home_id=7)
    assert (lab[:10] == "X").all() and (lab[90:] == "X").all()
    assert (lab[10:25] == "H").all()              # phase team, then the possessor
    assert (lab[25:31] == "A").all()              # carried forward to the phase end
    assert (lab[31:40] == "D").all()              # between phases: ball out of play
    assert (lab[40:50] == "A").all()              # phase team (away) until the first possessor
    assert (lab[50:81] == "H").all()
    assert (lab[81:90] == "D").all()


def test_idsse_bundle_from_cache(tmp_path, monkeypatch):
    """The IDSSE source goes through the existing harness cache, unchanged, into a bundle."""
    from types import SimpleNamespace
    from mqf.external import build
    from mqf.harness import idsse
    T, rng = 80, np.random.default_rng(1)
    xy = {s: {"Home": SimpleNamespace(xy=rng.uniform(-50, 50, (T, 4))), "Away": SimpleNamespace(xy=rng.uniform(-50, 50, (T, 4))),
              "Ball": SimpleNamespace(xy=np.zeros((T, 2)))} for s in idsse.SEGMENTS}
    code = {s: SimpleNamespace(code=np.ones(T)) for s in idsse.SEGMENTS}
    m = idsse.from_floodlight(xy, code, code, SimpleNamespace(xlim=(-52.5, 52.5), ylim=(-34, 34)), gap_s=1.0)
    cache = tmp_path / "J03WMX_v1.npz"
    idsse.save_cache(m, cache)
    monkeypatch.setattr(idsse, "_cache_path", lambda mid: cache)
    g = next(build.games("idsse", ["J03WMX"]))
    assert g["name"] == "IDSSE-J03WMX" and g["source"] == "idsse" and g["fps"] == 25
    spec = bundle.spec(bundle.write_bundle(g, tmp_path / "bundles"))
    assert spec.halves == m["halves"] and spec.pitch == (105.0, 68.0)
    trk = atd.load(spec, cache_dir=tmp_path / "c")
    np.testing.assert_allclose(trk.players, m["players"], atol=1e-5 * 105, equal_nan=True)


def test_splits_by_game():
    from mqf.external import splits
    s = splits.make()
    assert s == splits.make()                                                    # deterministic
    assert s == splits.load(), "splits.json is stale: python -m mqf.external.splits"
    for src, info in s["sources"].items():
        games = set(splits.SOURCES[src][0])
        assert set(info["train"]).isdisjoint(info["val"])
        assert set(info["train"]) | set(info["val"]) == games                   # every game, once
        assert info["val"]
    lg = splits.IDSSE_LEAGUE
    assert sorted(lg[g.split("-")[1]] for g in s["sources"]["idsse"]["val"]) == ["2. Bundesliga", "Bundesliga"]
    assert not s["sources"]["skillcorner"]["train_ok"]
    assert not any(g.startswith("SKC-") for g in splits.games("train", train_ok_only=True))
