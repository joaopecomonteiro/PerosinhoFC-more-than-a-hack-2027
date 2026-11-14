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
            "pitch": torch.tensor(pitch), "nearest": torch.zeros(B, T, dtype=torch.long),
            **_toy_context(torch, ball, obs, pitch)}


def _toy_context(torch, ball, obs, pitch):
    """v3 inputs for the toy: the observed (normalised) ball interpolated across the gap."""
    an, cx = [], []
    for r in range(len(ball)):
        metres = (ball[r] + 1) / 2 * pitch[r]
        a, c = data.context_features(*data.ball_context(metres, obs[r]), tuple(pitch[r]))
        an.append(a)
        cx.append(c)
    return {"anchor": torch.tensor(np.stack(an)), "ctx": torch.tensor(np.stack(cx))}


@pytest.mark.parametrize("arch,ctx,ctx_dim,aux", [("A", False, 4, False), ("B", False, 4, False),
                                                  ("A", True, 4, False), ("B", True, 4, False), ("B", True, 8, True)])
def test_models_learn_a_toy_rule(arch, ctx, ctx_dim, aux):
    torch = pytest.importorskip("torch")
    from mqf.ball import models
    torch.manual_seed(0)
    m = models.build({"arch": arch, "d": 32, "heads": 2, "layers": 2, "dropout": 0.0, "ctx": ctx,
                      "ctx_dim": ctx_dim, "aux": aux})
    b = _toy_batch(torch)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    if aux:
        pred, logits = models.forward(m, b, with_logits=True)
        assert logits.shape == (4, 60, 6)
        assert models.nearest_loss(logits, b["nearest"], b["w_masked"]).detach().item() > 0
    for _ in range(60):
        pred = models.forward(m, b)
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
    monkeypatch.setattr(data, "load_split", lambda part, *a, **k: games[:1] if part == "train" else games[1:])
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train, "HARNESS_DIR", tmp_path)
    assert train.main(["--arch", "A", "--d", "16", "--heads", "2", "--layers", "1", "--epochs", "1",
                       "--batch", "2", "--max-windows", "4", "--device", "cpu"]) == 0
    assert (tmp_path / "ball_A_off_r.pt").exists() and (tmp_path / "ball_A_off_r.json").exists()


def test_anchor_meets_the_edges_and_fades():
    true = _line(2000, v=(3.0, 1.0))
    ball = true.copy()
    ball[100:125] = np.nan                                    # 1 s
    ball[500:1250] = np.nan                                   # 30 s
    model = true + np.array([3.0, -2.0])                      # a model with a constant bias
    halves = ((1, 0, 1999),)
    an = impute.anchor(model, ball, halves, FPS, tau_s=2.0)
    err = lambda f, s: np.linalg.norm(f[s] - true[s], axis=1).mean()
    assert err(model, slice(100, 125)) > 3.5 and err(an, slice(100, 125)) < 0.05
    assert err(an, slice(500, 510)) < 0.5                     # next to an edge: corrected
    np.testing.assert_allclose(an[860:890], model[860:890], atol=0.02)   # 15 s from both edges: the model
    seen = ~np.isnan(ball[:, 0])
    np.testing.assert_array_equal(an[seen], model[seen])      # only gap frames move
    out = impute.anchor(np.full_like(model, -5.0), ball, halves, FPS, pitch=(105.0, 68.0))
    assert (out >= 0).all()                                   # clipped to the pitch


def test_filled_ball_is_who_evidence_only():
    from mqf import decode, features
    from mqf.atd import Tracking
    from mqf.external.bundle import speed_from_positions
    g = _game()
    sp = speed_from_positions(g.players.astype(np.float64), g.fps)
    ball = g.ball.astype(np.float64).copy()
    ball[2000:2500] = np.nan
    trk_miss = Tracking(g.players, sp, ball.astype(np.float32), g.team)
    trk_fill = Tracking(g.players, sp, g.ball, g.team)
    filled = np.zeros(len(ball), dtype=bool)
    filled[2000:2500] = True
    f_miss = features.build(trk_miss, g.pitch, g.fps)
    f_none = features.build(trk_fill, g.pitch, g.fps)
    f_fill = features.build(trk_fill, g.pitch, g.fps, ball_filled=filled)
    assert not f_fill["ball_out"][filled].any() and (f_fill["ball_missing_run"][2499] == 500)
    p = decode.Params()
    e_miss, e_none, e_fill = (decode.emissions(f, p) for f in (f_miss, f_none, f_fill))
    np.testing.assert_allclose(e_fill[filled, 2], e_miss[filled, 2])          # dead evidence as if missing
    np.testing.assert_allclose(e_fill[~filled], e_none[~filled], rtol=1e-6)    # elsewhere unchanged
    assert not np.allclose(e_fill[filled, 0] - e_fill[filled, 1], e_miss[filled, 0] - e_miss[filled, 1])


def test_evaluate_end_to_end(tmp_path, monkeypatch):
    g = data.game_from_match(synthetic.clean_match(n_frames=24_000, seed=5, halves=((1, 500, 11_499), (2, 12_500, 23_499))))
    monkeypatch.setattr(data, "load_split", lambda part, *a, **k: [g])
    monkeypatch.setattr(evaluate, "HARNESS_DIR", tmp_path)
    models_arg = []
    try:
        import torch
        from mqf.ball import models
        cfg = {"arch": "B", "d": 16, "heads": 2, "layers": 1, "dropout": 0.0, "hz": data.HZ,
               "window_s": data.WINDOW_S, "a_max": data.A_MAX}
        torch.save({"config": cfg, "state": models.build(cfg).state_dict()}, tmp_path / "ball_B_off.pt")
        models_arg = [str(tmp_path / "ball_B_off.pt")]
    except ImportError:
        pass
    assert evaluate.main(["--models", *models_arg, "--seeds", "0", "--players", "clean", "--per-bucket", "2",
                          "--taus", "2", "--w-filled", "0.4", "--device", "cpu", "--tag", "t"]) == 0
    import json
    rows = json.loads((tmp_path / "ball_t.json").read_text())["rows"]
    methods = {r["method"] for r in rows}
    assert {"hold", "none", "true", "oracle_fill", "hermite_le2s"} <= methods
    if models_arg:
        assert {"ball_B_off+a", "hybrid_ball_B_off+a_tau2"} <= methods


def test_ball_context():
    ball = np.arange(20, dtype=np.float64)[:, None].repeat(2, axis=1)
    vis = np.zeros(20, dtype=bool)
    vis[[5, 15]] = True
    an, dp, dn = data.ball_context(ball, vis, hz=1.0)
    np.testing.assert_allclose(an[5:16, 0], np.arange(5, 16))            # linear between seen steps
    assert (an[:5, 0] == 5).all() and (an[16:, 0] == 15).all()            # held at one-sided gaps
    assert dp[10] == 5 and dn[10] == 5 and np.isinf(dp[2]) and np.isinf(dn[18]) and dp[5] == 0
    an, dp, dn = data.ball_context(ball, np.zeros(20, dtype=bool))
    assert np.isnan(an).all() and np.isinf(dp).all()
    a, c = data.context_features(an, dp, dn, (105.0, 68.0))
    assert (a == 0).all() and (c[:, 2:4] == 1).all() and (c[:, :2] == 0).all() and (c[:, 4:] == 0).all()


def test_window_context_reads_beyond_the_window():
    g = _game()
    h, s0 = data.window_starts(g)[5]
    T = int(round(data.WINDOW_S * data.HZ))
    an, dp, dn, vp, vn = data.window_context(g, h, s0, np.zeros(T, dtype=bool), T)   # whole window hidden
    assert np.isfinite(an).all() and np.isfinite(dp).all() and np.isfinite(dn).all()
    assert dp[0] > 0 and dn[-1] > 0 and dp[-1] >= data.WINDOW_S - 1 / data.HZ


@pytest.mark.parametrize("ctx_dim,aux", [(4, False), (8, True)])
def test_zero_correction_model_is_linear_interpolation(ctx_dim, aux):
    torch = pytest.importorskip("torch")
    from mqf.ball import models
    cfg = {"arch": "B", "d": 16, "heads": 2, "layers": 1, "dropout": 0.0, "hz": data.HZ,
           "window_s": data.WINDOW_S, "a_max": data.A_MAX, "ctx": True, "ctx_dim": ctx_dim, "aux": aux}
    m = models.zero_correction(models.build(cfg)).eval()
    g = _game()
    ball = g.ball.astype(np.float64).copy()
    ball[3001:3207] = np.nan                                  # edges off the 6.25 Hz grid on purpose
    halves = g.halves[:1]
    mb = impute.predict_ball(m, cfg, g.players, g.team, ball, g.fps, g.pitch, halves)
    lin, _ = interp.fill(ball, halves, g.fps, "linear", pitch=g.pitch)
    np.testing.assert_allclose(mb[3001:3207], lin[3001:3207], atol=2e-3)
    smp = data.make_sample(g, *data.window_starts(g)[3], np.random.default_rng(0), player_mask=False)
    assert smp["anchor"].shape == (300, 2) and smp["ctx"].shape == (300, data.CTX_DIM)


def test_edge_velocity_on_a_straight_line():
    t = np.arange(40) / data.HZ
    ball = np.stack([10 + 6.0 * t, 20 - 2.0 * t], axis=1)
    vis = np.ones(40, dtype=bool)
    vis[10:20] = False
    vp, vn = data.edge_velocity(ball, vis)
    np.testing.assert_allclose(vp[15], [6.0, -2.0]) and np.testing.assert_allclose(vn[15], [6.0, -2.0])
    vis2 = np.zeros(40, dtype=bool)
    vis2[[5, 30]] = True                                     # isolated seen steps: velocity unknown
    vp, vn = data.edge_velocity(ball, vis2)
    assert (vp[10] == 0).all() and (vn[10] == 0).all()


def test_nearest_target_and_mirror():
    g = _game()
    h, s0 = data.window_starts(g)[3]
    for seed in range(4):
        smp = data.make_sample(g, h, s0, np.random.default_rng(seed), player_mask=False)
        k, t = 120, smp["target"][120]
        if smp["nearest"][k] < 0:
            continue
        L, W = smp["pitch"]
        xy = (smp["agents"][k, :, :2] + 1) / 2 * np.array([L, W])
        xy[~smp["present"][k]] = np.nan
        if smp["flip"][0]:
            xy[:, 0] = L - xy[:, 0]
        if smp["flip"][1]:
            xy[:, 1] = W - xy[:, 1]
        d = np.linalg.norm(xy - np.array([L - t[0] if smp["flip"][0] else t[0], W - t[1] if smp["flip"][1] else t[1]]), axis=1)
        assert smp["nearest"][k] == np.nanargmin(d)


def test_route_linear_then_hermite_then_model():
    ball = _line(1200)
    ball[100:115] = np.nan                                    # 0.6 s -> linear
    ball[300:340] = np.nan                                    # 1.6 s -> Hermite
    ball[600:800] = np.nan                                    # 8 s -> model
    mb = np.zeros_like(ball)
    bf = impute.fill(None, None, ball, FPS, (105.0, 68.0), ((1, 0, 1199),), "hybrid", tau_s=2.0, tau_linear_s=1.0,
                     interp_method="hermite", model_ball=mb)
    lin, _ = interp.fill(ball, ((1, 0, 1199),), FPS, "linear")
    np.testing.assert_allclose(bf.ball[100:115], lin[100:115])
    assert (bf.source[300:340] == impute.INTERP).all() and (bf.source[600:800] == impute.MODEL).all()


def test_train_smoke_axial_v4(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from mqf.ball import train
    games = [_game(101), _game(102)]
    monkeypatch.setattr(data, "load_split", lambda part, *a, **k: games[:1] if part == "train" else games[1:])
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train, "HARNESS_DIR", tmp_path)
    assert train.main(["--arch", "B", "--d", "16", "--heads", "2", "--layers", "1", "--epochs", "1",
                       "--batch", "2", "--max-windows", "4", "--device", "cpu", "--name", "ball_t"]) == 0
    import json
    rep = json.loads((tmp_path / "ball_t.json").read_text())
    assert rep["config"]["aux"] and rep["config"]["ctx_dim"] == 8
    assert rep["history"][0]["val_nearest_acc"] is not None


def test_rate_ok_and_decimate():
    assert data.rate_ok(25) and data.rate_ok(29.97) and not data.rate_ok(10)
    g = _game()
    d = data.decimate(g)
    assert d.step == 1 and d.hz == g.hz == 6.25 and len(d.ball) == -(-len(g.ball) // 4)
    np.testing.assert_array_equal(d.ball[10], g.ball[40])
    h, s = data.window_starts(g)[2]
    a = data.make_sample(g, h, s, np.random.default_rng(3), player_mask=False, train=False)
    hd, sd = h, s // 4
    b = data.make_sample(d, hd, sd, np.random.default_rng(3), player_mask=False, train=False)
    for k in ("agents", "anchor", "ctx", "target", "w_masked"):
        np.testing.assert_allclose(a[k], b[k], equal_nan=True)


def test_submission_with_ball_fill_is_reproducible(tmp_path):
    torch = pytest.importorskip("torch")
    from mqf import decode, make_submission, synthetic
    from mqf.ball import models
    spec = synthetic.generate(tmp_path / "syn", n_frames=12_000, halves=((1, 300, 5_699), (2, 6_300, 11_699)))
    cfg = {"arch": "B", "d": 16, "heads": 2, "layers": 1, "dropout": 0.0, "hz": data.HZ, "window_s": data.WINDOW_S,
           "a_max": data.A_MAX, "ctx": True, "ctx_dim": 8, "aux": True}
    torch.manual_seed(0)
    torch.save({"config": cfg, "state": models.build(cfg).state_dict()}, tmp_path / "m.pt")
    outs = []
    for k in range(2):
        out = tmp_path / f"s{k}.csv"
        rep = make_submission.run(spec, decode.Params(), out, ball_fill=True, ball_model=tmp_path / "m.pt")
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    assert rep["ball_fill"]["model_pct_of_missing"] + rep["ball_fill"]["interpolated_pct_of_missing"] > 50
    assert rep["options"]["ball_fill"] is True
    base = make_submission.run(spec, decode.Params(), tmp_path / "base.csv")
    assert base["ball_fill"] is None and base["options"]["ball_fill"] is False


def test_float64_inference_matches_float32():
    torch = pytest.importorskip("torch")
    from mqf.ball import models
    cfg = {"arch": "B", "d": 16, "heads": 2, "layers": 1, "dropout": 0.0, "hz": data.HZ, "window_s": data.WINDOW_S,
           "a_max": data.A_MAX, "ctx": True, "ctx_dim": 8, "aux": False}
    torch.manual_seed(0)
    m32 = models.build(cfg).eval()
    m64 = models.build(cfg).eval()
    m64.load_state_dict(m32.state_dict())
    m64.double()
    g = _game()
    ball = g.ball.astype(np.float64).copy()
    ball[3000:3300] = np.nan
    p32 = impute.predict_ball(m32, cfg, g.players, g.team, ball, g.fps, g.pitch, g.halves[:1])
    p64 = impute.predict_ball(m64, cfg, g.players, g.team, ball, g.fps, g.pitch, g.halves[:1])
    np.testing.assert_allclose(p64[3000:3300], p32[3000:3300], atol=1e-3)
