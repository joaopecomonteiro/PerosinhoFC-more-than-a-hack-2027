"""Dead-ball experiments (mqf.alive): features, rules on a toy scene, ball balancing, models, decoder
variants (the 'hmm' one = production, bit for bit), tags, diagnostics, the runners, the suite and the
make_submission options -- synthetic data only."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mqf import decode, features, synthetic
from mqf.ball import data as bdata
from mqf.alive import combine, data, feats, rules

HALVES = ((1, 500, 11_499), (2, 12_500, 23_499))
FPS = 25
PITCH = (105.3, 68.0)


def _match(seed):
    return synthetic.clean_match(n_frames=24_000, seed=seed, halves=HALVES)


def _games(seeds):
    return [bdata.game_from_match(_match(s), f"SYN-{s}") for s in seeds]


# ------------------------------------------------------------------------------------ features / rules

def test_feature_table_sets_and_no_missingness_columns():
    m = _match(1)
    ball = m["ball"].copy()
    ball[3000:3500] = np.nan
    X = feats.features(m["players"], m["team_of_player"], ball, FPS, PITCH, HALVES)
    assert len(X) == 24_000 and X.iloc[:500].isna().all().all() and X.iloc[11_600:12_400].isna().all().all()
    v1 = feats.v1_columns()
    assert list(X.columns[:len(v1)]) == v1 and len(v1) == 36
    sizes = [len(feats.columns_of(s, list(X.columns))) for s in feats.FEATURE_SETS]
    assert sizes == sorted(sizes) and sizes[0] == 36 and len(set(sizes)) == len(sizes)
    assert not any("seen" in c or "missing" in c for c in X.columns)           # absence is never a feature
    assert X.loc[3100:3400, "b_out"].isna().all() and X.loc[3100:3400, "b_speed"].isna().all()
    assert X.loc[3100:3400, "p_speed_p95"].notna().all()                        # players unaffected
    sub = feats.features(m["players"], m["team_of_player"], ball, FPS, PITCH, HALVES, "formation")
    assert list(sub.columns) == feats.columns_of("formation", list(X.columns))


def _toy():
    """22 players jog (alive), the ball goes out over the touchline at frame 400 and play stands still;
    a player walks behind the touchline (throw-in); at 900 the ball is thrown in (kick) and players run."""
    n = 1400
    rng = np.random.default_rng(0)
    base = np.c_[np.linspace(20, 85, 22), np.tile([20.0, 48.0], 11)]
    t = np.arange(n)[:, None, None]
    P = base[None] + np.zeros((n, 22, 2))
    run = np.r_[np.ones(400), np.zeros(500), np.ones(500)].astype(bool)
    vel = np.where(run[:, None, None], 4.5 / FPS, 0.0) * np.stack([np.cos(t[..., 0] * 0.02 + np.arange(22)),
                                                                  np.sin(t[..., 0] * 0.02 + np.arange(22))], -1)
    P = P + np.cumsum(vel, axis=0) + rng.normal(0, 0.01, P.shape)
    P[450:900, 0] = (40.0, -0.8)                                             # the thrower, behind the line
    b = np.full((n, 2), np.nan)
    b[:400] = np.c_[np.linspace(50, 52, 400), np.linspace(30, 67.5, 400)]
    b[400:430] = np.c_[np.full(30, 52.0), np.linspace(67.5, 69.5, 30)]          # out over the touchline
    b[880:900] = (40.0, -0.3)                                                # in the thrower's hands
    b[900:920] = np.c_[np.full(20, 40.0), np.linspace(-0.3, 8.0, 20)]           # thrown in: > 7 m/s
    b[920:] = np.c_[np.full(n - 920, 40.0), np.linspace(8.0, 30.0, n - 920)]
    team = np.repeat(np.array([0, 1], dtype=np.int8), 11)
    return P, team, b, ((1, 0, n - 1),)


def test_rules_toy_out_throw_in_and_restart():
    P, team, b, halves = _toy()
    q = feats.frame_quantities(P, team, b, FPS, PITCH, halves)
    rr = rules.apply(q, FPS, PITCH, halves, len(P))
    d = rr["dead"]
    assert d[:5].all(), "every half starts dead (kickoff)"
    assert not d[100:390].any(), "players running, ball moving: alive"
    assert d[416:880].all(), "ball out (> 0.5 m over the line from frame 415), then a throw-in shape: dead"
    assert not d[905:1300].any(), "the throw (ball speed jump) restarts play"
    assert rr["out"][416:428].all() and rr["kick"][895:905].any() and rr["throw"][500:880].all()
    cols = rules.columns(rr, halves, len(P))
    assert cols.shape[1] == 14 and cols["r_dead"].sum() == d.sum()


def test_balance_ball_makes_missingness_label_independent():
    m = _match(2)
    lab = m["labels"]
    ball = m["ball"].copy()
    rng = np.random.default_rng(0)
    dead_ = lab == "D"
    ball[dead_ & (rng.random(len(lab)) < 0.85)] = np.nan                     # a PFF-like habit: lost when dead
    out = data.balance_ball(ball, lab, HALVES, FPS, np.random.default_rng(1))
    ins = lab != "X"
    seen = ~np.isnan(out[:, 0])
    r_alive = seen[ins & np.isin(lab, ["H", "A"])].mean()
    r_dead = seen[ins & dead_].mean()
    assert abs(r_alive - r_dead) < 0.05 and r_dead > 0.1
    assert np.array_equal(np.isnan(out[dead_, 0]), np.isnan(ball[dead_, 0]))   # the worse-seen class untouched


def test_tables_cache_and_views(tmp_path):
    g = _games([3])[0]
    t1 = data.game_tables(g, 0, cache_dir=tmp_path)
    t2 = data.game_tables(g, 0, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1 and np.array_equal(t1["X"], t2["X"], equal_nan=True)
    assert set(np.unique(t1["view"])) == {0, 1} and set(np.unique(t1["y"])) <= {-1, 0, 1}
    assert t1["columns"] == t2["columns"] and 0.2 < (t1["y"] == 1).mean() < 0.6
    seqs = data.sequences([("g", "x", t1)], feats.columns_of("full", t1["columns"]))
    assert len(seqs) == 4 and all(np.all(np.diff(t1["frame"][t1["view"] == 0][:100]) > 0) for _ in [0])


# ------------------------------------------------------------------------------------ models

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from mqf.alive import train
    tmp = tmp_path_factory.mktemp("alive")
    mp = pytest.MonkeyPatch()
    mp.setattr(train, "MODELS_DIR", tmp)
    mp.setattr(train, "HARNESS_DIR", tmp)
    T = data.load_tables("train", ["x"], 0, use_cache=False, games=_games([61, 62]), log=None)
    out = {"gbm": train.fit("gbm", T, "t", "full", iters=40, log=lambda m: None)[0],
           "stack": train.fit("gbm", T, "ts", "full_rules", iters=40, log=lambda m: None)[0],
           "tmp": tmp, "T": T}
    try:
        import torch  # noqa: F401
        out["mlp"] = train.fit("mlp", T, "t", "full", epochs=1, device="cpu", log=lambda m: None)[0]
        out["tcn"] = train.fit("tcn", T, "t", "full", epochs=1, device="cpu", log=lambda m: None)[0]
        out["asf"] = train.fit("asf", T, "t", "full", epochs=1, device="cpu", log=lambda m: None)[0]
    except ImportError:
        pass
    yield out
    mp.undo()


def test_models_predict_and_dead_v1_wrapper(tiny):
    from mqf import dead
    from mqf.alive.models import AliveModel
    g = _games([63])[0]
    X = feats.features(g.players, g.team, g.ball, g.fps, g.pitch, g.halves)
    ins = np.isin(g.labels, ["H", "A", "D"])
    for k in ("gbm", "stack", "mlp", "tcn", "asf"):
        if k not in tiny:
            continue
        m = AliveModel.load(tiny[k])
        p, pb = m.predict(X, g.halves, len(X), g.fps, "cpu")
        assert np.isnan(p[:500]).all() and np.isfinite(p[ins]).all() and ((p[ins] >= 0) & (p[ins] <= 1)).all()
        assert (pb is not None) == (k in ("tcn", "asf"))
        p2, _ = m.predict(X, g.halves, len(X), g.fps, "cpu")
        assert np.array_equal(p, p2, equal_nan=True)
    m = AliveModel.load(tiny["gbm"])
    p, _ = m.predict(X, g.halves, len(X), g.fps)
    assert ((p[ins] >= 0.5) == (g.labels[ins] == "D")).mean() > 0.85
    d1 = AliveModel.load(dead.__file__.replace("dead.py", "../models/dead_v1.joblib"))
    a, _ = d1.predict(X, g.halves, len(X), g.fps)
    b = dead.predict(dead.load_model(dead.__file__.replace("dead.py", "../models/dead_v1.joblib")), g.players, g.team,
                     g.fps, g.pitch, g.halves)
    assert np.array_equal(a, b, equal_nan=True)


# ------------------------------------------------------------------------------------ decoders

def _feat_and_p(seed=64):
    from mqf.harness.noise import NoiseConfig, corrupt
    m = _match(seed)
    trk = corrupt(m["players"], m["team_of_player"], m["ball"], m["labels"] != "X", FPS, NoiseConfig())
    from mqf import dead
    pdd = dead.predict(dead.load_model(dead.__file__.replace("dead.py", "../models/dead_v1.joblib")), trk.players,
                       trk.team_of_slot, FPS, PITCH, HALVES)
    feat = features.build(trk, PITCH, FPS, p_dead=pdd)
    q = feats.frame_quantities(trk.players, trk.team_of_slot, trk.ball, FPS, PITCH, HALVES)
    rr = rules.apply(q, FPS, PITCH, HALVES, len(pdd))
    return m, trk, feat, pdd, rr


def test_hmm_variant_equals_production_decoder():
    m, trk, feat, pdd, rr = _feat_and_p()
    prod = decode.decode(feat, HALVES, len(pdd), decode.Params(dead_who="max"), None)
    mine = combine.decode_variant("hmm", feat, pdd, HALVES, len(pdd), FPS, decode.Params(dead_who="max"), rr)
    assert np.array_equal(prod, mine)
    v0 = decode.decode(features.build(trk, PITCH, FPS), HALVES, len(pdd), decode.Params(), None)
    mine0 = combine.decode_hmm(features.build(trk, PITCH, FPS), None, HALVES, len(pdd), decode.Params())
    assert np.array_equal(v0, mine0)


def test_variants_post_processing():
    lab = np.array(list("HHHHHDDHHHHAAAADDDDDDDDDDAAAA"))
    halves = ((1, 0, len(lab) - 1),)
    out = combine.min_dead(lab, halves, 1.0, 3)
    assert "".join(out) == "HHHHHHHHHHHAAAADDDDDDDDDDAAAA"
    lab2 = np.array(list("HHHHDDAAAA"))
    assert "".join(combine.min_dead(lab2, halves, 1.0, 3)) == "HHHHHAAAAA"      # two teams: split in the middle
    n = len(lab)
    rr = {"restart_cand": np.zeros(n, bool), "stop_cand": np.zeros(n, bool), "p_dead": np.full(n, 0.1)}
    rr["restart_cand"][23] = True
    rr["stop_cand"][16] = True
    s = combine.snap(lab, rr, halves, 1.0, 2.0)
    assert "".join(s) == "HHHHHDDHHHHAAAAADDDDDDDAAAAAA"
    rr["p_dead"][15:25] = 0.9
    v = combine.veto(lab, rr, halves, 1.0, max_s=6.0)
    assert "".join(v) == "HHHHHHHHHHHAAAADDDDDDDDDDAAAA"                     # the unsupported 2 s stoppage goes
    with pytest.raises(SystemExit):
        combine.parse_variant("start+wobble")
    m, trk, feat, pdd, rr = _feat_and_p()
    P = decode.Params(dead_who="max")
    for v in ("start", "min2", "snap1", "veto", "start+min2+snap1"):
        lab = combine.decode_variant(v, feat, pdd, HALVES, len(pdd), FPS, P, rr)
        assert set(np.unique(lab)) <= set("HADX") and (lab[600:11_000] != "X").all()
    assert combine.decode_variant("bnd0.5", feat, pdd, HALVES, len(pdd), FPS, P, rr) is None    # no boundary
    pb = np.zeros(len(pdd))
    same = combine.decode_variant("bnd0.5", feat, pdd, HALVES, len(pdd), FPS, P, rr, p_bnd=pb)
    assert np.array_equal(same, combine.decode_variant("hmm", feat, pdd, HALVES, len(pdd), FPS, P, rr))


def test_tag_windows_and_prior():
    m = _match(5)
    gt = m["labels"]
    rng = np.random.default_rng(0)
    w1 = combine.simulate_tag_windows(gt, HALVES, FPS, rng, recall=1.0, fp_per90=0.0)
    restarts = sum(len(np.flatnonzero((gt[a + 1:b + 1] != "D") & (gt[a:b] == "D"))) for _, a, b in HALVES)
    assert len(w1) == restarts and all(e > s for s, e in w1)
    b = combine.tag_bonus(len(gt), w1, FPS)
    assert (b[[s for s, _ in w1]] == 1).all() and b.max() == 1.0
    assert (gt[b > 0] == "D").mean() > 0.8                                    # the lead sits before the restart
    p = np.full(len(gt), 0.3)
    q = combine.with_prior(p, b, 1.0)
    assert (q[b > 0] > p[b > 0]).all() and np.array_equal(q[b == 0], p[b == 0])


def test_stoppage_stats_toy():
    gt = np.array(list("HHHHHDDDDDDHHHHHHHHHHDDDDDDDDDDDDDDDDDDDDAAAAA"))
    pred = gt.copy()
    pred[[12, 13]] = "D"                                                     # a spurious 2-frame stoppage
    pred[5] = "H"                                                            # the first one starts 1 frame late
    st = combine.stoppage_stats(gt, pred, ((1, 0, len(gt) - 1),), 1.0)
    assert st["rec_5-12s"] == 1.0 and st["rec_12-30s"] == 1.0 and st["spurious_per90"] > 0
    assert st["start_bias_s"] == 0.5 and st["end_med_s"] == 0.0


# ------------------------------------------------------------------------------------ runners

def _bundles(tmp_path, monkeypatch, names_by_part):
    from mqf.external import bundle, splits
    seed = 80
    for part, names in names_by_part.items():
        for name in names:
            seed += 1
            src = name.split("-")[0].lower()
            bundle.write_bundle({**_match(seed), "name": name, "source": src, "home": "A", "away": "B",
                                 "labels_from": "synthetic"}, root=tmp_path / "bundles")
    real_find = bundle.find
    monkeypatch.setattr(bundle, "find", lambda source=None, names=None, root=None:
                        real_find(source, names, tmp_path / "bundles"))
    monkeypatch.setattr(splits, "games", lambda part, sources=None, **k:
                        [n for n in names_by_part.get(part, []) if not sources or n.split("-")[0].lower() in sources])


def test_experiment_and_sensitivity_end_to_end(tmp_path, monkeypatch, tiny, official):
    from mqf.alive import experiment, sensitivity
    _bundles(tmp_path, monkeypatch, {"val": ["METRICA-9"]})
    monkeypatch.setattr(experiment, "HARNESS_DIR", tmp_path)
    monkeypatch.setattr(experiment, "EVIDENCE_CACHE", tmp_path / "ev")
    monkeypatch.setattr(sensitivity, "HARNESS_DIR", tmp_path)
    models = [f"gbm={tiny['gbm']}", f"stack={tiny['stack']}"] + ([f"tcn={tiny['tcn']}"] if "tcn" in tiny else [])
    args = ["--source", "metrica", "--no-ball-fill", "--condition", "atd", "--seeds", "0", "--ball-modes", "exact",
            "fuzzy", "random", "--clean-tracks", "--models", *models, "--ensemble", "e=gbm,stack", "--hybrid-base", "gbm",
            "--variants", "hmm", "start+min2", "snap1", "tags", "bnd0.5", "--variant-methods", "gbm",
            "--tag-recall", "0.6", "--device", "cpu", "--tag", "t"]
    if "tcn" in tiny:
        args += ["--bnd-from", "tcn"]
    assert experiment.main(args) == 0
    out = json.loads((tmp_path / "alive_t.json").read_text())
    got = {(r["method"], r["variant"]) for r in out["runs"]}
    assert {("oracle", "-"), ("v0", "hmm"), ("dead_v1", "hmm"), ("dead_v1", "start+min2"), ("rules", "hmm"),
            ("gbm", "tags"), ("e", "hmm"), ("gbm+prod", "hmm"), ("gbm+override", "hmm")} <= got
    assert ("stack", "start+min2") not in got                                  # not a variant method
    assert {r["ball_mode"] for r in out["runs"]} == {"exact", "fuzzy", "random"}
    orc = [r["spellF1"] for r in out["runs"] if r["method"] == "oracle"]
    assert min(orc) > 0.5 and all(r["alive_dead_f1"] == 1.0 for r in out["runs"] if r["method"] == "oracle")
    assert experiment.main(args[:-2] + ["--tag", "t2"]) == 0                    # evidence from the cache
    assert sensitivity.main(["--source", "metrica", "--no-ball-fill", "--clean-tracks", "--device", "cpu",
                             "--tag", "s"]) == 0
    sens = json.loads((tmp_path / "alive_sens_s.json").read_text())["rows"]
    o = next(r["spellF1"] for r in sens if r["case"] == "oracle")
    late = next(r["spellF1"] for r in sens if r["case"] == "lag 2s")
    assert late < o


def test_ball_modes():
    from mqf.alive import experiment
    m = _match(7)
    gt = m["labels"]
    rng = np.random.default_rng(0)
    ex = experiment.apply_ball_mode(m["ball"], gt, HALVES, FPS, "exact", rng)
    assert np.isnan(ex[gt == "D", 0]).all() and not np.isnan(ex[np.isin(gt, ["H", "A"]), 0]).any()
    ins = gt != "X"
    for mode in ("fuzzy", "random"):
        b = experiment.apply_ball_mode(m["ball"], gt, HALVES, FPS, mode, np.random.default_rng(1))
        assert np.isnan(b[ins, 0]).mean() >= experiment.MISSING_TARGET - 0.01
    fz = experiment.apply_ball_mode(m["ball"], gt, HALVES, FPS, "fuzzy", np.random.default_rng(1))
    rd = experiment.apply_ball_mode(m["ball"], gt, HALVES, FPS, "random", np.random.default_rng(1))
    dead_ = ins & (gt == "D")
    assert np.isnan(fz[dead_, 0]).mean() > np.isnan(rd[dead_, 0]).mean() + 0.2


def test_suite_quick_end_to_end(tmp_path, monkeypatch, official):
    pytest.importorskip("torch")
    from mqf.alive import data as adata, experiment, sensitivity, suite, train
    _bundles(tmp_path, monkeypatch, {"train": ["IDSSE-T1", "METRICA-T1", "PFF-T1"], "val": ["METRICA-V1"]})
    for mod in (suite, train, experiment, sensitivity):
        monkeypatch.setattr(mod, "HARNESS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(experiment, "EVIDENCE_CACHE", tmp_path / "ev")
    monkeypatch.setattr(adata, "CACHE_DIR", tmp_path / "cache")
    assert suite.main(["--quick", "--no-ball-fill", "--device", "cpu", "--tag", "q", "--skip", "asf"]) == 0
    assert (tmp_path / "alive_q_report.md").exists()
    rep = (tmp_path / "alive_q_report.md").read_text()
    assert "Q0 replica" in rep and "Verdict" in rep and "Q6 Smart Tagging" in rep
    assert suite.main(["--quick", "--no-ball-fill", "--device", "cpu", "--tag", "q", "--skip", "asf", "--resume"]) == 0


def test_make_submission_alive_options(tmp_path, tiny, official):
    import subprocess
    base = tmp_path / "base.csv"
    same = tmp_path / "same.csv"
    alive = tmp_path / "alive.csv"
    run = lambda out, *extra: subprocess.run([sys.executable, "-m", "mqf.make_submission", "--synthetic", "--out",
                                              str(out), *extra], capture_output=True, text=True)
    r1 = run(base, "--dead-model", "models/dead_v1.joblib")
    r2 = run(same, "--dead-model", "models/dead_v1.joblib", "--alive-variant", "hmm")
    assert r1.returncode == 0 and r2.returncode == 0 and base.read_bytes() == same.read_bytes()
    r3 = run(alive, "--alive-model", str(tiny["stack"]), "--alive-variant", "start+min2", "--alive-tag-w", "1")
    assert r3.returncode == 0, r3.stderr[-500:]
    rep = json.loads(alive.with_suffix(".json").read_text())
    assert rep["options"]["alive_variant"] == "start+min2" and rep["alive"]["set_piece_windows"] > 0
    r4 = run(tmp_path / "bad.csv", "--alive-model", str(tiny["stack"]), "--dead-model", "models/dead_v1.joblib")
    assert r4.returncode != 0
    ens = tmp_path / "ens.csv"                                                # several models: their mean
    r5 = run(ens, "--alive-model", str(tiny["gbm"]), str(tiny["stack"]))
    assert r5.returncode == 0, r5.stderr[-500:]
    rep = json.loads(ens.with_suffix(".json").read_text())
    assert isinstance(rep["options"]["alive_model"], list) and len(rep["options"]["alive_model"]) == 2
    import argparse
    from mqf import make_submission
    from mqf.config import SYNTHETIC_DIR
    opts, params = make_submission.settings(argparse.Namespace(final=False), rep)    # what --reproduce does
    again = tmp_path / "ens_again.csv"
    make_submission.run(synthetic.generate(SYNTHETIC_DIR), params, again, **opts)
    assert again.read_bytes() == ens.read_bytes()


def test_predict_ensemble_matches_experiment_mean(tiny):
    from mqf.alive import predict
    from mqf.alive.models import AliveModel
    g = _games([66])[0]
    p, pb, rr = predict.p_dead(f"{tiny['gbm']},{tiny['stack']}", g.players, g.team, g.ball, g.fps, g.pitch, g.halves)
    X = feats.features(g.players, g.team, g.ball, g.fps, g.pitch, g.halves)
    a = AliveModel.load(tiny["gbm"]).predict(X, g.halves, len(X), g.fps)[0]
    b = AliveModel.load(tiny["stack"]).predict(X, g.halves, len(X), g.fps)[0]
    assert np.array_equal(p, np.round(np.mean(np.stack([a, b]), axis=0), 4), equal_nan=True) and pb is None
    assert rr is not None and rr["dead"].shape == p.shape


def test_final_style_owner_tcn_plus_alive_ensemble(tmp_path, tiny, official):
    """The new FINAL's code path (owner TCN ensemble + its boundary, an alive torch ensemble, float64 CPU,
    ball fill, cleaner, tags) on a short synthetic match: twice byte-identical, and again from the saved
    report (what --reproduce does). The real asf files live on the team's machine; tiny ones stand in."""
    import argparse
    pytest.importorskip("torch")
    from mqf import make_submission
    from mqf.config import ROOT
    fin = make_submission.FINAL["options"]
    if not all((ROOT / m).is_file() for m in [*fin["who_model"], "models/ball_B_off_v5.pt"]):
        pytest.skip("owner TCN / ball model not in models/")
    if "asf" not in tiny:
        pytest.skip("torch models not trained")
    spec = synthetic.generate(tmp_path / "short", n_frames=9_000, halves=((1, 300, 4_199), (2, 4_700, 8_699)))
    opts, params = make_submission.settings(argparse.Namespace(final=True, no_tags=False, clean_tracks=False,
                                                               ball_fill=False, ball_model=None, dead_model=None,
                                                               who_model=None, params=[]))
    opts = {**opts, "alive_model": [str(tiny["asf"]), str(tiny["tcn"])]}
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    rep = make_submission.run(spec, params, a, **opts)
    make_submission.run(spec, params, b, **opts)
    assert a.read_bytes() == b.read_bytes()
    assert rep["used_tags"] and rep["options"]["who_bnd_alpha"] == 0.5 and len(rep["options"]["who_model"]) == 3
    saved_opts, saved_params = make_submission.settings(argparse.Namespace(final=False), rep)
    c = tmp_path / "c.csv"
    make_submission.run(spec, saved_params, c, **saved_opts)
    assert c.read_bytes() == a.read_bytes()
    v1 = tmp_path / "v1.csv"                                                  # the old recipe still runs
    o1 = {**make_submission.FINAL_V1["options"]}
    make_submission.run(spec, replace_params(make_submission.FINAL_V1["params"]), v1, **o1)
    assert v1.read_bytes() != a.read_bytes()


def replace_params(over):
    from dataclasses import replace
    return replace(decode.Params(), **over)


def test_final_check_quick(tmp_path, monkeypatch, tiny, official):
    """The old-vs-new FINAL harness table runs end to end (tiny alive models stand in for the asf files)."""
    pytest.importorskip("torch")
    from mqf import make_submission
    from mqf.alive import experiment, final_check
    from mqf.config import ROOT
    if not all((ROOT / m).is_file() for m in make_submission.FINAL["options"]["who_model"]):
        pytest.skip("owner TCN not in models/")
    if "asf" not in tiny:
        pytest.skip("torch models not trained")
    _bundles(tmp_path, monkeypatch, {"val": ["METRICA-V1"]})
    fin = {**make_submission.FINAL, "options": {**make_submission.FINAL["options"],
                                                "alive_model": [str(tiny["asf"]), str(tiny["tcn"])]}}
    monkeypatch.setattr(make_submission, "FINAL", fin)
    for mod in (experiment, final_check):
        monkeypatch.setattr(mod, "HARNESS_DIR", tmp_path)
    monkeypatch.setattr(experiment, "EVIDENCE_CACHE", tmp_path / "ev")
    assert final_check.main(["--quick", "--no-ball-fill", "--device", "cpu", "--tag", "fc"]) == 0
    text = (tmp_path / "final_check_fc.md").read_text(encoding="utf-8")
    assert "new − old (spellF1)" in text and "who_v1 × dead_v1" in text and "tcn × asf" in text
    assert "PASS" in text or "FAIL" in text
