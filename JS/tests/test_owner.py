"""Owner experiments (mqf.owner): rules on a toy scene, feature symmetry, cache, models, decoders,
hybrids and the experiment runner -- synthetic data only."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mqf import decode, synthetic
from mqf.ball import data as bdata
from mqf.owner import combine, data, feats, rules

HALVES = ((1, 500, 11_499), (2, 12_500, 23_499))
FPS = 25


def _match(seed):
    return synthetic.clean_match(n_frames=24_000, seed=seed, halves=HALVES)


def _toy_scene():
    """200 frames, 4 players (0, 1 home; 2, 3 away). Home 0 dribbles (0-49), the ball flies past away 2
    and is deflected by him (60-64, a touch), home 1 controls it (90-119), it flies to away 3 who
    dribbles (131-170, a control), away 3 passes to away 2 (190-199)."""
    n = 200
    P = np.zeros((n, 4, 2))
    P[:, 0] = np.c_[10 + 2.0 * np.arange(n) / FPS, np.full(n, 10.0)]       # home 0 runs +x at 2 m/s
    P[:, 1] = (30.0, 30.0)
    P[:, 2] = (20.0, 20.0)
    P[:, 3] = (50.0, 20.0)
    P[131:171, 3] = np.c_[50 + 3.0 * np.arange(40) / FPS, np.full(40, 20.0)]   # away 3 dribbles
    P[171:, 3] = P[170, 3]
    b = np.full((n, 2), np.nan)
    b[0:50] = P[0:50, 0] + (0.3, 0.0)
    b[50:60] = np.linspace(b[49], (19.5, 19.5), 10)                       # towards away 2
    b[60:65] = np.linspace((19.5, 19.5), (21.5, 22.5), 5)                 # deflected at speed: no control
    b[65:90] = np.linspace((21.5, 22.5), (29.7, 29.7), 25)                # towards home 1
    b[90:120] = (29.7, 29.7)                                              # home 1 holds it, nobody near
    b[120:131] = np.linspace((29.7, 29.7), (50.0, 20.0), 11)              # long pass, intercepted
    b[131:171] = P[131:171, 3] + (0.3, 0.0)
    b[171:190] = np.linspace(b[170], (20.3, 20.0), 19)
    b[190:200] = (20.3, 20.0)
    return P, np.array([0, 0, 1, 1]), b, ((1, 0, n - 1),)


def test_rules_toy_touch_ignored_control_switches():
    P, team, b, halves = _toy_scene()
    r = rules.owner(P, team, b, FPS, halves)
    own = r["owner"]
    assert (own[:130] == 0).all(), "a deflection by away 2 must not change possession"
    assert (own[130:] == 1).all()                    # from the first frame of away 3's contact (ball arrives at 130)
    ev = r["events"]
    touch = ev[(ev.team == 1) & (ev.start < 70)]
    assert len(touch) == 1 and int(touch.kind.iloc[0]) == rules.KIND_TOUCH
    sw = ev[ev.kind == rules.KIND_CONTROL]
    assert len(sw) == 1 and int(sw.start.iloc[0]) == 130 and int(sw.team.iloc[0]) == 1
    assert np.all(r["p_home"][:130] > 0.5) and np.all(r["p_home"][130:] < 0.5)
    assert r["in_ctrl"][10] and not r["in_ctrl"][62]


def test_rules_pass_chain_and_settings():
    P, team, b, halves = _toy_scene()
    # without the free-ball condition the dribble of away 3 is still a control; with an impossible
    # velocity match nobody ever controls -> the switch can only come from a pass (away 3 -> away 2)
    r = rules.owner(P, team, b, FPS, halves, rules.RuleConfig(v_match_ms=-1.0, chain_gap_s=3.0))
    k = r["events"].kind.to_numpy()
    assert (k == rules.KIND_CONTROL).sum() == 0 and (k == rules.KIND_CHAIN).sum() == 1
    assert r["owner"][-1] == 1 and r["owner"][100] == 0
    r0 = rules.owner(P, team, b, FPS, halves, rules.RuleConfig(v_match_ms=-1.0, pass_chain=0))
    assert (r0["owner"] == 0).all()                                     # no control, no pass: home keeps it
    assert rules.parse_rule_cfg(["r_touch_m=1.2", "filled_can_control=1"]) == \
        rules.RuleConfig(r_touch_m=1.2, filled_can_control=True)
    with pytest.raises(SystemExit):
        rules.parse_rule_cfg(["nope=1"])


def test_rules_on_synthetic_and_alive_reset():
    m = _match(5)
    alive = np.isin(m["labels"], ["H", "A"])
    y = m["labels"][alive] == "H"
    r = rules.owner(m["players"], m["team_of_player"], m["ball"], FPS, m["halves"])
    assert np.mean((r["p_home"][alive] > 0.5) == y) > 0.9
    rs = rules.owner(m["players"], m["team_of_player"], m["ball"], FPS, m["halves"], alive=alive)
    assert len(rs["segments"]) > len(r["segments"])
    assert np.isnan(r["p_home"][:500]).all() and np.isnan(rs["p_home"][~alive]).all()
    sw = rules.owner(m["players"], 1 - m["team_of_player"], m["ball"], FPS, m["halves"])
    inside = np.isfinite(r["p_home"])
    np.testing.assert_allclose(r["p_home"][inside] + sw["p_home"][inside], 1.0, atol=1e-12)


def test_feature_sets_and_swap_symmetry():
    m = _match(6)
    args = (m["players"], m["team_of_player"], m["ball"], FPS, m["pitch"], m["halves"])
    Xw = feats.features(*args, feature_set="who")
    X = feats.features(*args, feature_set="full_rules")
    assert list(X.columns[:Xw.shape[1]]) == list(Xw.columns) == feats.who_columns()
    np.testing.assert_array_equal(X[Xw.columns].to_numpy(), Xw.to_numpy())
    Xs = feats.features(m["players"], 1 - m["team_of_player"], *args[2:], feature_set="full_rules")
    for a, b, f in (("line_def_h", "line_def_a", 1), ("rel_v1_h", "rel_v1_a", 1), ("pin_diff", "pin_diff", -1),
                    ("d1_h_w3min", "d1_a_w3min", 1), ("rule_since_ctrl_h", "rule_since_ctrl_a", 1)):
        np.testing.assert_allclose(X[a].to_numpy(), f * Xs[b].to_numpy(), equal_nan=True, atol=1e-9)
    np.testing.assert_allclose(X["rule_p_home"], 1 - Xs["rule_p_home"], equal_nan=True, atol=1e-12)
    assert X.iloc[:500].isna().all().all()
    S, names = feats.seq_inputs(*args, X_full=X)
    assert S.shape == (len(X), len(names)) and not any(n.startswith("rule_") or "since" in n for n in names)
    S2, all_names = feats.seq_inputs(*args, X_full=X, columns="all")
    assert all_names[:X.shape[1]] == list(X.columns) and feats.gru_columns(all_names) == names


def test_game_tables_cache(tmp_path):
    g = bdata.game_from_match(_match(7), "SYN-7")
    t1 = data.game_tables(g, 0, cache_dir=tmp_path)
    t2 = data.game_tables(g, 0, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    for k in ("X", "y", "S", "sy"):
        np.testing.assert_array_equal(t1[k], t2[k])
    assert t1["columns"] == t2["columns"] and set(np.unique(t1["view"])) == {0, 1, 2, 3}
    assert set(np.unique(t1["sy"])) <= {-1, 0, 1} and 0.3 < t1["y"].mean() < 0.7
    full = data.columns_of("full", t1["columns"])
    assert not any(c.startswith("rule_") for c in full) and len(data.columns_of("who", t1["columns"])) == 59


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    pytest.importorskip("torch")
    from mqf.owner import train
    tmp = tmp_path_factory.mktemp("owner")
    mp = pytest.MonkeyPatch()
    mp.setattr(train, "MODELS_DIR", tmp)
    mp.setattr(train, "HARNESS_DIR", tmp)
    games = [bdata.game_from_match(_match(s), f"SYN-{s}") for s in (61, 62)]
    try:
        T = data.load_tables("train", ["x"], 0, use_cache=False, games=games, log=None)
        paths = {"gbm": train.fit("gbm", T, "t", "full", iters=30, log=lambda m: None)[0],
                 "stack": train.fit("gbm", T, "ts", "full_rules", iters=30, log=lambda m: None)[0],
                 "mlp": train.fit("mlp", T, "t", "full", epochs=1, device="cpu", log=lambda m: None)[0],
                 "gru": train.fit("gru", T, "t", "full", epochs=1, device="cpu", log=lambda m: None)[0]}
    finally:
        mp.undo()
    return paths


def _views(m):
    out = []
    for k in (0, 1):
        team = m["team_of_player"] if k == 0 else 1 - m["team_of_player"]
        X = feats.features(m["players"], team, m["ball"], FPS, m["pitch"], m["halves"], "full_rules")
        S, sc = feats.seq_inputs(m["players"], team, m["ball"], FPS, m["pitch"], m["halves"], X_full=X, columns="all")
        out.append({"X": X, "S": S, "seq_columns": sc})
    return out


def test_models_predict_symmetric_and_deterministic(tiny_models):
    from mqf.owner.models import OwnerModel
    m = _match(63)
    v = _views(m)
    n = len(m["labels"])
    alive = np.isin(m["labels"], ["H", "A"])
    for name, path in tiny_models.items():
        mod = OwnerModel.load(path)
        p = mod.p_home(v, m["halves"], n, FPS, "cpu")
        ps = mod.p_home(v[::-1], m["halves"], n, FPS, "cpu")
        inside = np.isfinite(p)
        assert np.isnan(p[:500]).all() and inside[600:11_000].all(), name
        np.testing.assert_allclose(p[inside] + ps[inside], 1.0, atol=1.5e-4)
        assert np.array_equal(p, mod.p_home(v, m["halves"], n, FPS, "cpu"), equal_nan=True), name
        if name in ("gbm", "stack"):
            assert np.mean((p[alive] > 0.5) == (m["labels"][alive] == "H")) > 0.9, name


def test_who_joblib_loads_as_owner_model(tmp_path):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from mqf import who
    from mqf.owner.models import OwnerModel
    m = _match(64)
    Xw = who.features(m["players"], m["team_of_player"], m["ball"], FPS, m["pitch"], m["halves"])
    ok = np.isin(m["labels"], ["H", "A"])
    clf = HistGradientBoostingClassifier(max_iter=10).fit(Xw[ok].to_numpy(np.float32), (m["labels"][ok] == "H"))
    joblib.dump({"model": clf, "columns": list(Xw.columns), "meta": {}}, tmp_path / "who_t.joblib")
    mod = OwnerModel.load(tmp_path / "who_t.joblib")
    p = mod.p_home(_views(m), m["halves"], len(ok), FPS)
    ref = who.predict(who.load_model(tmp_path / "who_t.joblib"), m["players"], m["team_of_player"], m["ball"], FPS,
                      m["pitch"], m["halves"])
    np.testing.assert_array_equal(p, ref)                                # same numbers as who.predict


def test_viterbi_fast_matches_decode():
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(2, 800))
        ll = rng.normal(0, 3, (n, 3)).round(int(rng.integers(0, 2)))
        p = decode.Params(switch_cost=float(rng.choice([0, 3, 18])), to_dead_cost=float(rng.choice([0, 10])))
        assert np.array_equal(decode.viterbi(ll, p), combine.viterbi3(ll, p))
        big = decode.Params(switch_cost=p.switch_cost, to_dead_cost=1e9, from_dead_cost=1e9)
        ll2 = np.c_[ll[:, :2], np.full(n, -1e12)]
        assert np.array_equal(decode.viterbi(ll2, big), combine.viterbi2(ll[:, 0], ll[:, 1], p.switch_cost))


def test_decoders_hard_and_hybrids():
    n = 1000
    halves = ((1, 100, 899),)
    alive = np.zeros(n, dtype=bool)
    alive[150:400] = alive[500:880] = True
    p = np.full(n, np.nan)
    p[100:900] = 0.8
    p[300:310] = 0.1                                                      # a 10-frame blip
    lab = combine.decode_given_alive(p, alive, halves, n)
    assert (lab[:100] == "X").all() and (lab[100:150] == "D").all() and (lab[400:500] == "D").all()
    assert (lab[150:400] == "H").all()                                    # the blip costs less than a switch
    raw = combine.decode_raw(p, alive, halves, n)
    assert (raw[300:310] == "A").all()
    h = combine.hard(np.array([0.7, 0.5, 0.2, np.nan]))
    np.testing.assert_array_equal(h[:3], [0.98, 0.5, 0.02]) and np.isnan(h[3])
    pr = combine.product(np.array([0.7, np.nan]), np.array([0.7, 0.9]))
    assert pr[0] > 0.7 and np.isnan(pr[1])
    # veto: an A spell inside H with the rule owner H all along and the ball tracked -> merged back
    labs = np.array(["H"] * 50 + ["A"] * 20 + ["H"] * 50)
    rr = {"owner": np.zeros(120, dtype=int)}
    out = combine.veto(labs, rr, np.ones(120, dtype=bool), FPS)
    assert (out == "H").all()
    rr2 = {"owner": np.r_[np.zeros(50, int), np.ones(20, int), np.zeros(50, int)]}
    assert (combine.veto(labs, rr2, np.ones(120, dtype=bool), FPS) == labs).all()
    assert (combine.veto(labs, rr, np.zeros(120, dtype=bool), FPS) == labs).all()     # no ball: no veto


def test_smooth50_layer_uses_official_baseline(official):
    m = _match(8)
    lab, alive = combine.smooth50(m["players"], m["team_of_player"], m["ball"], FPS, m["halves"], len(m["labels"]))
    assert (lab[:500] == "X").all() and set(np.unique(lab[600:11_000])) <= {"H", "A", "D"}
    assert alive[600:11_000].mean() > 0.9                               # synthetic ball is always tracked
    codes = combine.nearest_codes(m["players"], m["team_of_player"], m["ball"])
    assert set(np.unique(codes)) <= {-1, 0, 1}


def test_experiment_runner_on_bundles(tmp_path, monkeypatch, tiny_models, official):
    from mqf.external import bundle
    from mqf.owner import experiment
    for k, seed in enumerate((71, 72)):
        m = _match(seed)
        bundle.write_bundle({**m, "name": f"METRICA-{k + 7}", "source": "metrica", "home": "A", "away": "B",
                             "labels_from": "synthetic"}, root=tmp_path)
    real_find = bundle.find
    monkeypatch.setattr(bundle, "find", lambda source=None, names=None, root=None: real_find(source, None, tmp_path))
    monkeypatch.setattr(experiment, "HARNESS_DIR", tmp_path)
    monkeypatch.setattr(experiment, "EVIDENCE_CACHE", tmp_path / "ev")
    args = ["--source", "metrica", "--no-ball-fill", "--layers", "oracle", "smooth50", "--decoders", "hmm", "raw",
            "--models", f"gbm={tiny_models['gbm']}", f"gru={tiny_models['gru']}", "--hybrid-base", "gbm",
            "--ref", "gbm", "--device", "cpu", "--tag", "t"]
    assert experiment.main(args) == 0
    out = json.loads((tmp_path / "owner_t.json").read_text())
    methods = {r["method"] for r in out["runs"]}
    assert {"nearest", "hold", "smooth50", "rule", "rule_hard", "rule_seg", "gbm", "gbm_hard", "gru", "gru_hard",
            "gbm+prod", "gbm+prod0.5", "gbm+override", "gbm+veto"} <= methods
    assert {r["layer"] for r in out["runs"]} == {"oracle", "smooth50"} and {r["game"] for r in out["runs"]} == {"METRICA-7", "METRICA-8"}
    best = max(r["spellF1"] for r in out["runs"] if r["layer"] == "oracle" and r["method"] == "gbm")
    assert best > 0.5
    assert (tmp_path / "owner_t.md").read_text().count("## Layer") == 2


def test_tcn_boundary_target_and_switch_costs():
    from mqf.owner import models
    sy = np.array([1, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1], dtype=np.int8)
    b = models.boundary_target(sy, sigma=1.0)
    assert b[2] == pytest.approx(1.0) and b[3] == pytest.approx(1.0) and b[8] == pytest.approx(1.0)
    assert b[9] == pytest.approx(1.0) and b[1] == pytest.approx(b[4]) and b[1] < 1.0   # symmetric around the change
    assert b[6] < 0.2 and b[0] < b[1] < b[2]                             # the restart at 7 is not a boundary
    c = combine.switch_costs(np.array([0.0, 0.5, 1.0, np.nan]), 18.0, 0.9)
    np.testing.assert_allclose(c, [18.0, 18 * 0.55, 18 * 0.1, 18.0])
    # a cheaper switch where the boundary says so lets a short spell through
    n = 600
    halves = ((1, 0, n - 1),)
    p = np.full(n, 0.8)
    p[300:310] = 0.2
    alive = np.ones(n, dtype=bool)
    assert (combine.decode_given_alive(p, alive, halves, n, 18.0)[300:310] == "H").all()
    pb = np.zeros(n)
    pb[[300, 310]] = 1.0
    lab = combine.decode_given_alive(p, alive, halves, n, combine.switch_costs(pb, 18.0, 0.9))
    assert (lab[300:310] == "A").all() and (lab[:300] == "H").all() and (lab[310:] == "H").all()


def test_tcn_trains_and_returns_a_boundary(tiny_models):
    from mqf.owner import train
    from mqf.owner.models import OwnerModel
    games = [bdata.game_from_match(_match(s), f"SYN-{s}") for s in (61, 62)]
    T = data.load_tables("train", ["x"], 0, use_cache=False, games=games, log=None)
    mp = pytest.MonkeyPatch()
    mp.setattr(train, "MODELS_DIR", tiny_models["gbm"].parent)
    mp.setattr(train, "HARNESS_DIR", tiny_models["gbm"].parent)
    try:
        path, info = train.fit("tcn", T, "t", "full", epochs=1, device="cpu", log=lambda m: None)
    finally:
        mp.undo()
    mod = OwnerModel.load(path)
    assert mod.kind == "tcn" and not any(c.startswith("rule_") for c in mod.columns)
    m = _match(63)
    v = _views(m)
    n = len(m["labels"])
    p, pb = mod.p_home_and_boundary(v, m["halves"], n, FPS, "cpu")
    ps, pbs = mod.p_home_and_boundary(v[::-1], m["halves"], n, FPS, "cpu")
    inside = np.isfinite(p)
    np.testing.assert_allclose(p[inside] + ps[inside], 1.0, atol=1.5e-4)
    np.testing.assert_allclose(pb[inside], pbs[inside], atol=1.5e-4)   # the boundary does not depend on who is home
    assert ((pb[inside] >= 0) & (pb[inside] <= 1)).all() and np.isnan(pb[:500]).all()


def test_pathcrf_bridge_helpers():
    from mqf.owner import pathcrf_bridge as pb
    team = pb.edges_to_team(["home_7", "home_7", "out_left", "away_3", None], ["home_7", "away_2", "away_9", "out_top", None])
    np.testing.assert_array_equal(team, [0, 0, 1, 1, -1])
    rng = np.random.default_rng(0)
    ours = np.convolve(rng.random(6000), np.ones(25) / 25, mode="same")
    o, r = pb.best_offset(ours, ours[37:4037], max_shift=60)
    assert o == 37 and r > 0.999
    src = "a\n            if args.lomocv:\n                pass\n"
    assert src.count(pb.SPLIT_ANCHOR) == 1


def test_pathcrf_align_on_a_toy_bundle(tmp_path, monkeypatch):
    """Fake PathCRF output for a bundle (their frames start 7 frames later, home and away swapped):
    align recovers the offset, flips the teams and writes P(home) on the bundle's timeline."""
    import argparse
    from mqf.external import bundle
    from mqf.owner import pathcrf_bridge as pb
    m = _match(91)
    root = tmp_path / "bundles"
    bundle.write_bundle({**m, "name": "IDSSE-J03WMX", "source": "idsse", "home": "A", "away": "B",
                         "labels_from": "synthetic"}, root=root)
    real_find = bundle.find
    monkeypatch.setattr(bundle, "find", lambda source=None, names=None, root_=None: real_find(source, names, root))
    lab = m["labels"]
    fid, per, team, alive, gtt, bx, by = [], [], [], [], [], [], []
    for half, a, b in m["halves"]:
        sl = slice(a + 7, b + 1)
        k = np.arange(b + 1 - (a + 7))
        fid.append((10_000 if half == 1 else 100_000) + k)
        per.append(np.full(len(k), half))
        t = np.where(lab[sl] == "H", 1, np.where(lab[sl] == "A", 0, -1))          # their home = our away
        team.append(t)
        gtt.append(t)
        alive.append(np.isin(lab[sl], ["H", "A"]))
        bx.append(m["ball"][sl, 0])
        by.append(m["ball"][sl, 1])
    out = tmp_path / "pcrf" / pb.OUT_DIR
    out.mkdir(parents=True)
    np.savez_compressed(out / "J03WMX.npz", frame_id=np.concatenate(fid), period_id=np.concatenate(per),
                        team=np.concatenate(team).astype(np.int8), alive=np.concatenate(alive),
                        gt_team=np.concatenate(gtt).astype(np.int8), ball_x=np.concatenate(bx), ball_y=np.concatenate(by))
    a = argparse.Namespace(pathcrf=tmp_path / "pcrf", games=["J03WMX"], out=tmp_path / "ext", max_shift=50)
    assert pb.cmd_align(a) == 0
    z = np.load(tmp_path / "ext" / "IDSSE-J03WMX.npz")
    info = json.loads(str(z["info"]))
    assert info["half1"]["offset"] == 7 and info["half2"]["offset"] == 7 and info["home_away_flipped"]
    p = z["p_home"]
    ok = np.isin(lab, ["H", "A"]) & np.isfinite(p)
    assert ok.sum() > 10_000 and np.mean((p[ok] > 0.5) == (lab[ok] == "H")) == 1.0


def test_runner_cleaner_noise_several_boundaries_and_ensembles(tmp_path, monkeypatch, tiny_models, official):
    """--clean-tracks + --noise on the atd condition, two TCN boundaries and their ensemble, end to end."""
    from mqf.external import bundle
    from mqf.owner import experiment, train
    games = [bdata.game_from_match(_match(s), f"SYN-{s}") for s in (61, 62)]
    T = data.load_tables("train", ["x"], 0, use_cache=False, games=games, log=None)
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train, "HARNESS_DIR", tmp_path)
    tcn = {f"tcn{k}": train.fit("tcn", T, f"t{k}", "full", epochs=1, seed=k, device="cpu", log=lambda m: None,
                                table_seed=0)[0] for k in (0, 1)}
    assert json.loads((tmp_path / "owner_train_tcn_t1.json").read_text())["table_seed"] == 0
    m = _match(73)
    bundle.write_bundle({**m, "name": "METRICA-9", "source": "metrica", "home": "A", "away": "B",
                         "labels_from": "synthetic"}, root=tmp_path)
    real_find = bundle.find
    monkeypatch.setattr(bundle, "find", lambda source=None, names=None, root=None: real_find(source, None, tmp_path))
    monkeypatch.setattr(experiment, "HARNESS_DIR", tmp_path)
    monkeypatch.setattr(experiment, "EVIDENCE_CACHE", tmp_path / "ev")
    args = ["--source", "metrica", "--no-ball-fill", "--layers", "oracle", "--decoders", "hmm", "--condition", "atd",
            "--seeds", "0", "--clean-tracks", "--noise", "team_flip_p=0.05", "--models", f"gbm={tiny_models['gbm']}",
            *[f"{k}={v}" for k, v in tcn.items()], "--hard", "--boundary-from", "tcn0", "tcn1",
            "--ensemble", "ens=tcn0,tcn1", "--bnd-alpha", "0.5", "--ref", "gbm", "--device", "cpu", "--tag", "t2"]
    assert experiment.main(args) == 0
    out = json.loads((tmp_path / "owner_t2.json").read_text())
    methods = {r["method"] for r in out["runs"]}
    assert {"gbm", "tcn0", "tcn1", "ens", "tcn0+bnd0.5", "tcn1+bnd0.5", "ens+bnd0.5", "gbm+tcn0bnd0.5",
            "gbm+tcn1bnd0.5", "gbm+ensbnd0.5"} <= methods
    assert not any(x.endswith("_hard") for x in methods - {"rule_hard"})      # --hard with no names: none
    assert out["config"]["clean_tracks"] and out["config"]["noise"] == ["team_flip_p=0.05"]
    with pytest.raises(SystemExit):                                           # a member that is no TCN
        experiment.main(args[:-2] + ["--ensemble", "bad=gbm,tcn0", "--tag", "t3"])


def test_tcn_check_report_rule(tmp_path, monkeypatch):
    """The driver's verdict: a candidate passes only with > 2 se and most wins in every condition."""
    from mqf.owner import tcn_check
    monkeypatch.setattr(tcn_check, "HARNESS_DIR", tmp_path)
    rng = np.random.default_rng(0)
    good, flat = "who_v1+ens_imbnd0.5", "ens_im"
    for cname, seeds in (("bar", [0]), ("atd", [0, 1])):
        rows = []
        for g in range(8):
            for sd in seeds:
                base = 0.5 + 0.01 * g
                for meth, v in ((tcn_check.REF, base), (good, base + 0.01 + 0.001 * rng.standard_normal()),
                                (flat, base + 0.01 * rng.standard_normal())):
                    rows.append({"method": meth, "layer": "dead_v1", "decoder": "hmm", "game": f"G{g}", "seed": sd,
                                 "source": "idsse" if g < 4 else "pff", "spellF1": v})
        (tmp_path / f"owner_t_{cname}.json").write_text(json.dumps({"runs": rows}))
    assert tcn_check.main(["--stage", "report", "--sets", "im", "--conditions", "bar", "atd", "--tag", "t",
                           "--no-clean"]) == 0
    rep = json.loads((tmp_path / "owner_t_report.json").read_text())
    assert rep["passing"] == ["who_v1 + ens_im boundary"]
    text = (tmp_path / "owner_t_report.md").read_text()
    assert "PASS  who_v1 + ens_im boundary" in text and "fail  ens_im " in text


def test_tcn_check_commands_parse(tmp_path, monkeypatch):
    """The driver's runner command lines are valid for the runner's own parser (no night lost to a typo)."""
    from mqf.owner import experiment, tcn_check, train
    for s in tcn_check.SETS:
        for k in (0, 1, 2):
            (tmp_path / f"owner_tcn_{tcn_check.tcn_tag(s, k)}.pt").write_bytes(b"")
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(tcn_check, "HARNESS_DIR", tmp_path)
    seen = []
    monkeypatch.setattr(experiment, "main", lambda args: seen.append(experiment.parser().parse_args(args)) or 0)
    assert tcn_check.main(["--stage", "eval", "--tag", "t"]) == 0
    assert [a.condition for a in seen] == ["bar", "atd", "atd"]
    bar, atd, flip = seen
    assert all(a.clean_tracks and a.hard == [] and a.bnd_alpha == [0.5] and a.ref == "who_v1" for a in seen)
    assert len(bar.boundary_from) == 6 and bar.ensemble == [
        "ens_im=tcn_full_im_s0,tcn_full_im_s1,tcn_full_im_s2",
        "ens_impff=tcn_full_impff_s0,tcn_full_impff_s1,tcn_full_impff_s2"]
    assert atd.seeds == [0, 1, 2] and atd.noise == [] and flip.seeds == [0, 1]
    assert flip.noise == ["team_flip_p=0.05", "gk_team_flip_p=0.5"]
    assert {m.split("=")[0] for m in bar.models} == {"who_v1", *bar.boundary_from}
