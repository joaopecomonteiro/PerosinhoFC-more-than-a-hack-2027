"""Who-has-the-ball model: features, swap symmetry, training, decoder use and the bundle harness --
synthetic data only."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mqf import decode, features, synthetic, who
from mqf.ball import data
from mqf.harness.noise import NoiseConfig, corrupt

HALVES = ((1, 500, 11_499), (2, 12_500, 23_499))


def _game(seed):
    m = synthetic.clean_match(n_frames=24_000, seed=seed, halves=HALVES)
    return m, data.game_from_match(m, name=f"SYN-{seed}")


def _noisy(m):
    return corrupt(m["players"], m["team_of_player"], m["ball"], m["labels"] != "X", m["fps"], NoiseConfig())


@pytest.fixture(scope="module")
def model_bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("who")
    games = {"train": [_game(61)[1], _game(62)[1]], "val": [_game(63)[1]]}
    mp = pytest.MonkeyPatch()
    mp.setattr(who, "load_games", lambda part, sources, log=print: games[part])
    mp.setattr(who, "MODELS_DIR", tmp)
    mp.setattr(who, "HARNESS_DIR", tmp)
    try:
        assert who.train(argparse.Namespace(tag="t", sources=["x"], val_sources=["x"], iters=40, seed=0)) == 0
    finally:
        mp.undo()
    return tmp, json.loads((tmp / "who_t.json").read_text()), who.load_model(tmp / "who_t.joblib")


def test_features_shape_and_halves():
    m, g = _game(31)
    X = who.features(g.players, g.team, g.ball, g.fps, g.pitch, g.halves)
    assert X.shape == (len(g.players), 59)
    assert X.iloc[:500].isna().all().all() and X.iloc[11_500:12_500].isna().all().all()
    assert X.iloc[600].notna().sum() > 40
    for c in ("since_h", "until_a", "since_ball"):                      # clocks are capped
        v = X[c].dropna()
        assert (v >= 0).all() and (v <= who.CAP_S).all()


def test_features_swap_teams_mirror():
    m, g = _game(32)
    X = who.features(g.players, g.team, g.ball, g.fps, g.pitch, g.halves)
    Xs = who.features(g.players, 1 - g.team, g.ball, g.fps, g.pitch, g.halves)
    np.testing.assert_allclose(Xs["d1_diff"], -X["d1_diff"], equal_nan=True)
    np.testing.assert_allclose(Xs["since_h"], X["since_a"], equal_nan=True)
    np.testing.assert_allclose(Xs["ball_x_att"], 1.0 - X["ball_x_att"], atol=1e-9, equal_nan=True)


def test_train_predict_symmetric(model_bundle):
    _, rep, mb = model_bundle
    auc = [r["auc"] for r in rep["rows"] if r["method"] == "model" and r["ball"] == "all" and r["auc"] is not None]
    assert auc and np.mean(auc) > 0.8
    m, _ = _game(64)
    trk = _noisy(m)
    p = who.predict(mb, trk.players, trk.team_of_slot, trk.ball, m["fps"], m["pitch"], m["halves"])
    ps = who.predict(mb, trk.players, 1 - trk.team_of_slot, trk.ball, m["fps"], m["pitch"], m["halves"])
    assert np.isnan(p[:500]).all() and np.isfinite(p[600:11_000]).all()
    inside = np.isfinite(p)
    np.testing.assert_allclose(p[inside] + ps[inside], 1.0, atol=1.5e-4)   # P(home) + P(away) = 1 (1e-4 rounding)
    again = who.predict(mb, trk.players, trk.team_of_slot, trk.ball, m["fps"], m["pitch"], m["halves"])
    assert np.array_equal(p, again, equal_nan=True)                       # deterministic


def test_decoder_unchanged_without_model_column():
    m, _ = _game(45)
    trk = _noisy(m)
    feat = features.build(trk, m["pitch"], m["fps"])
    assert "p_home_model" not in feat
    for params in (decode.Params(), decode.Params(w_who_model=0.3, who_clip=0.1)):   # who params idle without it
        base = decode.emissions(feat, decode.Params())
        np.testing.assert_array_equal(decode.emissions(feat, params), base)
        nan_col = feat.assign(p_home_model=np.nan)                        # a column of NaN = no model
        np.testing.assert_array_equal(decode.emissions(nan_col, params), base)
    lab = decode.decode(feat, m["halves"], len(feat), decode.Params())
    feat2 = features.build(trk, m["pitch"], m["fps"], p_home=None)
    assert np.array_equal(lab, decode.decode(feat2, m["halves"], len(feat), decode.Params()))


def test_decoder_uses_model_column():
    m, _ = _game(46)
    trk = _noisy(m)
    n = len(m["labels"])
    p_home = np.full(n, np.nan)
    for _, a, b in m["halves"]:
        p_home[a:b + 1] = 0.9
    feat = features.build(trk, m["pitch"], m["fps"], p_home=p_home)
    inside = np.isfinite(p_home)
    for w in (1.0, 0.5):
        ll = decode.emissions(feat, decode.Params(w_who_model=w))
        filled = feat["ball_filled"].to_numpy()
        expect = w * np.log(0.9 / 0.1) * np.where(filled, decode.Params().w_filled, 1.0)
        np.testing.assert_allclose((ll[:, 0] - ll[:, 1])[inside], expect[inside], rtol=1e-6)
    clipped = decode.emissions(feat.assign(p_home_model=np.where(inside, 1.0, np.nan)), decode.Params(who_clip=0.1))
    np.testing.assert_allclose((clipped[:, 0] - clipped[:, 1])[inside], np.log(9.0), rtol=1e-6)
    lab_h = decode.decode(feat, m["halves"], n, decode.Params())
    lab_a = decode.decode(feat.assign(p_home_model=np.where(inside, 0.1, np.nan)), m["halves"], n, decode.Params())
    alive_h, alive_a = lab_h[lab_h != "X"], lab_a[lab_a != "X"]
    assert (alive_h == "H").sum() > 0 and (alive_h == "A").sum() == 0
    assert (alive_a == "A").sum() > 0 and (alive_a == "H").sum() == 0


def test_bundle_harness_who_model(tmp_path, monkeypatch, model_bundle):
    from mqf.external import bundle, evaluate
    model_dir, _, _ = model_bundle
    for k, seed in enumerate((71, 72)):
        m, _ = _game(seed)
        bundle.write_bundle({**m, "name": f"METRICA-{k + 7}", "source": "metrica", "home": "A", "away": "B",
                             "team_of_player": m["team_of_player"], "labels_from": "synthetic"}, root=tmp_path)
    real_find = bundle.find
    monkeypatch.setattr(bundle, "find", lambda source=None, names=None, root=None: real_find(source, names, tmp_path))
    monkeypatch.setattr(evaluate, "HARNESS_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "EXTERNAL_DIR", tmp_path)
    args = ["--source", "metrica", "--players-drop", "--hide-dead-ball"]
    assert evaluate.main(args + ["--tag", "off"]) == 0
    assert evaluate.main(args + ["--who-model", str(model_dir / "who_t.joblib"),
                                 "--grid", "w_who_model=0.5,1", "--tag", "on"]) == 0
    off = json.loads((tmp_path / "bundles_off.json").read_text())
    on = json.loads((tmp_path / "bundles_on.json").read_text())
    assert off["who_model"] is None and on["who_model"].endswith("who_t.joblib")
    assert len(on["grid"]) == 2 and len(on["runs"]) == 4
