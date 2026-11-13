"""Dead-ball detector: features, training, decoder use and the bundle harness -- synthetic data only."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mqf import dead, decode, features, synthetic
from mqf.ball import data
from mqf.harness.noise import NoiseConfig, corrupt


def _game(seed):
    m = synthetic.clean_match(n_frames=24_000, seed=seed, halves=((1, 500, 11_499), (2, 12_500, 23_499)))
    g = data.game_from_match(m, name=f"SYN-{seed}")
    return m, g


def test_features_shape_and_halves():
    m, g = _game(31)
    X = dead.features(g.players, g.team, g.fps, g.pitch, g.halves)
    assert len(X) == len(g.players) and X.shape[1] > 30
    assert X.iloc[:500].isna().all().all() and X.iloc[600].notna().sum() > 30
    sp = dead.player_speed(np.repeat(np.arange(100.0)[:, None, None], 2, axis=2) * np.array([0.08, 0.06]), 25)
    np.testing.assert_allclose(sp[20, 0], 2.5, rtol=1e-6)                      # 0.1 m / frame at 25 fps


def test_train_predict_and_decode(tmp_path, monkeypatch):
    games = {"train": [_game(41)[1], _game(42)[1]], "val": [_game(43)[1]]}
    monkeypatch.setattr(dead, "load_games", lambda part, sources, log=print: games[part])
    monkeypatch.setattr(dead, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(dead, "HARNESS_DIR", tmp_path)
    args = argparse.Namespace(tag="t", sources=["x"], val_sources=["x"], iters=40, dropout_copy=True, seed=0)
    assert dead.train(args) == 0
    rep = json.loads((tmp_path / "dead_t.json").read_text())
    model_auc = np.mean([r["auc"] for r in rep["rows"] if r["method"] == "model"])
    assert model_auc > 0.9
    mb = dead.load_model(tmp_path / "dead_t.joblib")
    m, g = _game(44)
    trk = corrupt(m["players"], m["team_of_player"], m["ball"], m["labels"] != "X", m["fps"], NoiseConfig())
    p = dead.predict(mb, trk.players, trk.team_of_slot, m["fps"], m["pitch"], m["halves"])
    assert np.isnan(p[:500]).all() and np.isfinite(p[600:11_000]).all()
    assert np.array_equal(p, dead.predict(mb, trk.players, trk.team_of_slot, m["fps"], m["pitch"], m["halves"]), equal_nan=True)
    feat = features.build(trk, m["pitch"], m["fps"], p_dead=p)
    lab = decode.decode(feat, m["halves"], len(p), decode.Params(), tags=None)
    assert set(np.unique(lab)) <= set("HADX") and (lab == "D").any()


def test_dead_who_max_never_favours_dead():
    m, g = _game(45)
    trk = corrupt(m["players"], m["team_of_player"], m["ball"], m["labels"] != "X", m["fps"], NoiseConfig())
    feat = features.build(trk, m["pitch"], m["fps"])
    zero = decode.emissions(feat, decode.Params())
    mx = decode.emissions(feat, decode.Params(dead_who="max"))
    np.testing.assert_allclose(mx[:, :2], zero[:, :2], rtol=1e-6)            # H / A untouched (float32 features)
    assert (mx[:, 2] <= zero[:, 2] + 1e-6).all()                                # D loses its free "who" advantage


def test_bundle_harness_grid(tmp_path, monkeypatch):
    from mqf.external import bundle, evaluate
    m1, _ = _game(51)
    m2, _ = _game(52)
    for k, m in enumerate((m1, m2)):
        bundle.write_bundle({**m, "name": f"METRICA-{k + 7}", "source": "metrica", "home": "A", "away": "B",
                             "team_of_player": m["team_of_player"], "labels_from": "synthetic"}, root=tmp_path)
    real_find = bundle.find
    monkeypatch.setattr(bundle, "find", lambda source=None, names=None, root=None: real_find(source, names, tmp_path))
    monkeypatch.setattr(evaluate, "HARNESS_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "EXTERNAL_DIR", tmp_path)
    # a tiny detector for the harness
    games = {"train": [_game(53)[1]], "val": [_game(54)[1]]}
    monkeypatch.setattr(dead, "load_games", lambda part, sources, log=print: games[part])
    monkeypatch.setattr(dead, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(dead, "HARNESS_DIR", tmp_path)
    dead.train(argparse.Namespace(tag="h", sources=["x"], val_sources=["x"], iters=30, dropout_copy=False, seed=0))
    assert evaluate.main(["--source", "metrica", "--dead-model", str(tmp_path / "dead_h.joblib"), "--players-drop",
                          "--hide-dead-ball", "--grid", "w_dead=0.5,1", "dead_who=zero,max", "--tag", "g"]) == 0
    out = json.loads((tmp_path / "bundles_g.json").read_text())
    assert len(out["grid"]) == 4 and len(out["runs"]) == 8 and out["hide_dead_ball"] and out["players_drop"]
