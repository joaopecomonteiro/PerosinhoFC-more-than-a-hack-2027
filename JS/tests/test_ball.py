"""Ball imputation: interpolation, windows and masks, models, inference, protocol -- toy data only."""
from __future__ import annotations

import numpy as np
import pytest

from mqf import synthetic
from mqf.ball import data, evaluate, impute, interp

FPS = 25


def _line(n=400, v=(8.0, 2.0)):
    t = np.arange(n) / FPS
    return np.stack([10 + v[0] * t, 20 + v[1] * t], axis=1)


def test_linear_and_hermite_exact_on_straight_line():
    true = _line()
    ball = true.copy()
    ball[100:150] = np.nan
    halves = ((1, 0, 399),)
    for method in ("linear", "hermite"):
        f, done = interp.fill(ball, halves, FPS, method)
        assert done[100:150].all() and not done[:100].any()
        np.testing.assert_allclose(f[100:150], true[100:150], atol=1e-6)


def test_interp_edges_and_max_gap():
    ball = _line()
    ball[:30] = np.nan                                        # gap at the half's start: hold the first seen
    ball[200:300] = np.nan                                    # 4 s gap
    f, done = interp.fill(ball, ((1, 0, 399),), FPS, "linear", max_gap_s=2.0)
    np.testing.assert_allclose(f[:30], np.repeat(ball[30][None], 30, axis=0))
    assert not done[200:300].any()                            # longer than max_gap_s: left for the model
    g = interp.gap_length_s(~np.isnan(ball[:, 0]), ((1, 0, 399),), FPS)
    assert g[250] == pytest.approx(4.0) and g[50] == 0


def _game(seed=101):
    m = synthetic.clean_match(n_frames=20_000, seed=seed, halves=((1, 500, 9_499), (2, 10_500, 19_499)))
    return data.game_from_match(m)


def test_sample_layout_and_alive_only_targets():
    g = _game()
    starts = data.window_starts(g)
    assert starts and all(s + 300 * 4 <= g.halves[h][2] + 1 for h, s in starts)
    rng = np.random.default_rng(0)
    for pm in (False, True):
        smp = data.make_sample(g, *starts[3], rng, player_mask=pm)
        assert smp["agents"].shape == (300, data.A_MAX, 4) and smp["present"].shape == (300, data.A_MAX)
        assert not (smp["ball_obs"] & (smp["w_masked"] > 0)).any()            # loss on hidden only
        assert (smp["ball_in"][~smp["ball_obs"]] == 0).all()
        idx = starts[3][1] + np.arange(300) * g.step
        alive = g.alive[idx]
        assert not ((smp["w_masked"] + smp["w_seen"]) > 0)[~alive].any()      # nothing on dead frames
    a = data.make_sample(g, *starts[3], np.random.default_rng(1), player_mask=False, train=False)
    b = data.make_sample(g, *starts[3], np.random.default_rng(1), player_mask=True, train=False)
    assert b["present"].sum() < a["present"].sum()                             # dropouts in the 'on' variant


def test_split_pieces():
    xy = np.full((10, 1, 2), 1.0)
    xy[4:6, 0] = np.nan
    out, tm = data.split_pieces(xy, np.array([1], dtype=np.int8))
    assert out.shape[1] == 2 and list(tm) == [1, 1]


def test_place_gaps_keeps_edges_seen():
    ok = np.ones(2000, dtype=bool)
    ok[500:600] = False
    gaps = evaluate.place_gaps(ok, [50] * 10, np.random.default_rng(0), sep=25)
    for a, b in gaps:
        assert ok[a - 1] and ok[b + 1] and ok[a:b + 1].all()
    occ = np.zeros(2000, dtype=bool)
    for a, b in gaps:
        assert not occ[a:b + 1].any()
        occ[a:b + 1] = True


def test_hybrid_routes_by_tau():
    ball = _line(800)
    ball[100:120] = np.nan                                    # 0.8 s
    ball[400:500] = np.nan                                    # 4 s
    mb = np.zeros_like(ball)                                  # a "model" that always says (0, 0)
    bf = impute.fill(None, None, ball, FPS, (105.0, 68.0), ((1, 0, 799),), "hybrid", tau_s=1.0, model_ball=mb)
    assert (bf.source[100:120] == impute.INTERP).all() and (bf.source[400:500] == impute.MODEL).all()
    assert (bf.source[~np.isnan(ball[:, 0])] == impute.SEEN).all()
    frames = np.ones(800, dtype=bool)
    frames[450:] = False                                      # dead from 450: never filled there
    bf = impute.fill(None, None, ball, FPS, (105.0, 68.0), ((1, 0, 799),), "model", frames=frames, model_ball=mb)
    assert (bf.source[450:500] == impute.EMPTY).all()


# ------------------------------------------------------------------------------------ torch

def _toy_batch(torch, B=4, T=60, A=6):
    rng = np.random.default_rng(0)
    xy = np.cumsum(rng.normal(0, 0.02, (B, T, A, 2)), axis=1) + rng.uniform(-0.5, 0.5, (B, 1, A, 2))
    agents = np.concatenate([xy, np.zeros_like(xy)], axis=-1).astype(np.float32)
    ball = xy[:, :, 0]                                        # the toy rule: the ball is on agent 0
    obs = np.ones((B, T), dtype=bool)
    obs[:, 20:40] = False
    pitch = np.tile(np.array([105.0, 68.0], dtype=np.float32), (B, 1))
    target = (ball + 1) / 2 * pitch[:, None, :]
    return {"agents": torch.tensor(agents), "present": torch.ones(B, T, A, dtype=torch.bool),
            "team": torch.tensor(np.r_[np.zeros(3), np.ones(3)].astype(np.int64)).repeat(B, 1),
            "ball_in": torch.tensor(np.where(obs[..., None], ball, 0).astype(np.float32)),
            "ball_obs": torch.tensor(obs), "target": torch.tensor(target.astype(np.float32)),
            "w_masked": torch.tensor((~obs).astype(np.float32)), "w_seen": torch.tensor(obs.astype(np.float32)),
            "pitch": torch.tensor(pitch)}


@pytest.mark.parametrize("arch", ["A", "B"])
def test_models_learn_a_toy_rule(arch):
    torch = pytest.importorskip("torch")
    from mqf.ball import models
    torch.manual_seed(0)
    m = models.build({"arch": arch, "d": 32, "heads": 2, "layers": 2, "dropout": 0.0})
    b = _toy_batch(torch)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        pred = m(b["agents"], b["present"], b["team"], b["ball_in"], b["ball_obs"])
        assert pred.shape == (4, 60, 2)
        loss, ade = models.loss_and_ade(models.to_metres(pred, b["pitch"]), b["target"], b["w_masked"], b["w_seen"])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_inference_keeps_seen_and_is_deterministic():
    torch = pytest.importorskip("torch")
    from mqf.ball import models
    torch.manual_seed(0)
    cfg = {"arch": "A", "d": 16, "heads": 2, "layers": 1, "dropout": 0.0, "hz": data.HZ,
           "window_s": data.WINDOW_S, "a_max": data.A_MAX}
    m = models.build(cfg).eval()
    g = _game()
    ball = g.ball.astype(np.float64).copy()
    ball[3000:3200] = np.nan
    halves = g.halves[:1]
    p1 = impute.predict_ball(m, cfg, g.players, g.team, ball, g.fps, g.pitch, halves)
    p2 = impute.predict_ball(m, cfg, g.players, g.team, ball, g.fps, g.pitch, halves)
    assert p1.tobytes() == p2.tobytes()
    _, a, b = halves[0]
    assert np.isfinite(p1[a:b + 1]).all()
    bf = impute.fill(g.players, g.team, ball, g.fps, g.pitch, halves, "model", model_ball=p1)
    seen = ~np.isnan(ball[:, 0])
    np.testing.assert_array_equal(bf.ball[seen], ball[seen])
    assert (bf.source[3000:3200] == impute.MODEL).all()


def test_train_smoke(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from mqf.ball import train
    games = [_game(101), _game(102)]
    monkeypatch.setattr(data, "load_split", lambda part, sources=None: games[:1] if part == "train" else games[1:])
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train, "HARNESS_DIR", tmp_path)
    assert train.main(["--arch", "A", "--d", "16", "--heads", "2", "--layers", "1", "--epochs", "1",
                       "--batch", "2", "--max-windows", "4", "--device", "cpu"]) == 0
    assert (tmp_path / "ball_A_off.pt").exists() and (tmp_path / "ball_A_off.json").exists()
