"""mqf.fusion / mqf.fusion_export: kernels, production faithfulness, export, selection (synthetic only).

Nothing here reads the competition match or another game's data: arrays are generated, and the
pipeline pieces run on mqf.synthetic's match.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mqf import fusion as fu


# ---------------------------------------------------------------------------------- kernels

def _objective(E, C, pen, path):
    T = len(path)
    s = sum(E[t, path[t]] for t in range(T))
    s -= sum(C[path[t - 1], path[t]] for t in range(1, T) if path[t] != path[t - 1])
    t = 0
    while t < T:
        u = t
        while u + 1 < T and path[u + 1] == path[t]:
            u += 1
        d = u - t + 1
        s -= pen[path[t], min(d, pen.shape[1]) - 1]
        t = u + 1
    return s


def _brute(E, C, pen):
    T, K = E.shape
    return max(_objective(E, C, pen, p) for p in itertools.product(range(K), repeat=T))


@pytest.mark.parametrize("seed", range(6))
def test_semimarkov_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    T, K = 7, 3
    E = rng.normal(0, 1.5, (T, K))
    C = rng.uniform(0, 2, (K, K))
    np.fill_diagonal(C, 0)
    pen = fu.stack_pens([fu.soft_pen(3, 2.0, 1.0), fu.soft_pen(2, 1.0, 1.0), fu.soft_pen(4, 3.0, 2.0)])
    path = fu.semimarkov_viterbi(E, C, pen=pen)
    assert _objective(E, C, pen, path) == pytest.approx(_brute(E, C, pen), abs=1e-9)


def test_non_monotone_prior_is_made_exact():
    """A raw duration penalty that rises with d would make the recursion inexact; stack_pens takes
    the non-increasing envelope, and then the kernel's optimum is the brute-force optimum."""
    rng = np.random.default_rng(3)
    E = rng.normal(0, 1.0, (7, 2))
    C = np.array([[0.0, 0.5], [0.5, 0.0]])
    pen = fu.stack_pens([[3.0, 0.5, 2.0, 0.0], [1.0, 2.5, 0.0]])
    assert np.all(np.diff(pen, axis=1) <= 0)
    path = fu.semimarkov_viterbi(E, C, pen=pen)
    assert _objective(E, C, pen, path) == pytest.approx(_brute(E, C, pen), abs=1e-9)
    prior = fu.duration_prior([(np.repeat(np.array([0, 2, 1, 2] * 30, np.int8), 40), 25.0)])
    assert np.all(np.diff(fu.emp_pen(prior, "spell", 25.0, 1.0, 4.0)) <= 0)


@pytest.mark.parametrize("seed", range(4))
def test_numpy_fallback_equals_fast_kernel(seed):
    rng = np.random.default_rng(100 + seed)
    T = 400
    E = rng.normal(0, 1, (T, 3))
    C, Bm = fu.costs3({"c_ha": 3.0, "c_ad": 4.0, "c_da": 2.0})
    mult = np.stack([rng.uniform(0.2, 1.0, T), rng.uniform(0.3, 1.0, T)], 1)
    pen = fu.stack_pens([fu.soft_pen(9, 5.0, 1.0)] * 2 + [fu.soft_pen(15, 5.0, 2.0)])
    E_, C_, B_, m_ = fu._prep(E, C, Bm, mult)
    assert np.array_equal(fu.semimarkov_viterbi(E, C, Bm, mult, pen), fu._smk_numpy(E_, C_, B_, m_, pen, pen.shape[1]))
    assert np.array_equal(fu.viterbi(E, C, Bm, mult), fu._vit_numpy(E_, C_, B_, m_))
    p1 = fu.forward_backward(E, C, Bm, mult)
    assert np.allclose(p1, fu._fb_numpy(E_, C_, B_, m_), atol=1e-9)
    assert np.allclose(np.exp(p1).sum(1), 1.0)


def test_hard_min_duration_is_respected():
    rng = np.random.default_rng(7)
    E = rng.normal(0, 3, (300, 3))
    path = fu.semimarkov_viterbi(E, np.zeros((3, 3)), pen=fu.stack_pens([fu.soft_pen(10, 1e6, 1.0)] * 3))
    edges = np.flatnonzero(np.diff(np.concatenate(([-1], path, [-1]))) != 0)
    assert np.diff(edges).min() >= 10


@pytest.mark.parametrize("seed", range(3))
def test_viterbi_is_production_viterbi3_bit_for_bit(seed):
    """Same float operations and tie-breaking as alive.combine.viterbi3 (also on exact ties)."""
    from mqf.alive import combine as acombine
    rng = np.random.default_rng(seed)
    T = 3000
    E = np.round(rng.normal(-1, 1, (T, 3)), 1)                     # coarse values -> real ties
    sw = 18.0 * (1.0 - 0.5 * np.clip(np.round(rng.random(T), 1), 0, 1))
    C, Bm = fu.costs3({"c_ha": 18.0, "c_ad": 10.0, "c_da": 10.0})
    mult = np.stack([sw / 18.0, np.ones(T)], 1)
    ref = acombine.viterbi3(E, sw, 10.0, 10.0)
    assert np.array_equal(fu.viterbi(E, C, Bm, mult), ref)
    assert np.array_equal(fu._vit_numpy(*fu._prep(E, C, Bm, mult)), ref)


# ---------------------------------------------------------------------------------- metric, files

def test_score_matches_the_official_evaluate(official):
    from mqf.harness.evaluate import score as official_score
    g = fu.synth_game(np.random.default_rng(4), half_min=3.0)
    pred = fu.decode("hsmm_soft", g, fu.sample_configs("hsmm_soft", 3, {})[2])
    lab = np.array(["H", "A", "D", "X"])
    a, b = fu.score(g["y"], pred), official_score(lab[g["y"]], lab[pred], 25)
    assert a["spellF1"] == b["spellF1"] and a["macroF1"] == b["macroF1"]
    assert a["adF1"] == b["alive_dead_f1"] and a["cnt_err"] == b["n_pred_spells"] - b["n_gt_spells"]


def test_save_load_roundtrip_and_guards(tmp_path):
    g = fu.synth_game(np.random.default_rng(1), half_min=2.0, source="pff")
    feat = {c: g["feat_" + c] for c in fu.FEAT_COLS}
    fu.save_game(tmp_path, "bar_exact", "G1", 0, g["y"], 25, source="pff", halves=[(1, *h) for h in g["halves"]],
                 p_dead=g["p_dead"], p_dead_asf=g["p_dead_asf"], p_dead_v8=g["p_dead_v8"], p_home=g["p_home"],
                 p_home_seeds=g["p_home_seeds"], p_bnd=g["p_bnd"], ball_out=g["ball_out"],
                 ball_filled=g["ball_filled"], feat=feat, pred_final=g["pred_final"],
                 params_json=g["params_json"], who_alpha=0.5)
    h = fu.load_npz(tmp_path / "bar_exact" / "G1__s0.npz")
    assert h["source"] == "pff" and h["fps"] == 25.0
    for k in ("p_dead", "p_home", "p_home_seeds", "p_bnd", "pred_final", "ball_out", "halves"):
        assert np.array_equal(h[k], g[k]), k                                   # float64, bit for bit
    assert np.array_equal(fu.decode("hmm3_bnd", h, fu.CENTRE), fu.decode("hmm3_bnd", g, fu.CENTRE))
    with pytest.raises(ValueError):
        fu.save_game(tmp_path, "c", "BAR-ATH", 0, g["y"], 25, p_dead=g["p_dead"], p_home=g["p_home"])
    with pytest.raises(ValueError):
        fu.save_game(tmp_path, "c", "G2", 0, g["y"], 25, p_dead=g["p_dead"][:-1], p_home=g["p_home"])
    with pytest.raises(ValueError):
        fu.save_game(tmp_path, "c", "G2", 0, g["y"], 25, p_dead=g["p_dead"] * 3, p_home=g["p_home"])


def test_decoders_end_to_end_on_synthetic():
    g = fu.synth_game(np.random.default_rng(0), fps=25.0, half_min=3.0)
    meta = {"has_dead_pair": True, "has_seeds": True, "has_bnd": True, "has_bnd_dead": True, "has_v1": True}
    g["_dur_prior"] = fu.duration_prior([(g["y"], 25.0)])
    inside = np.zeros(len(g["y"]), bool)
    for a, b in g["halves"]:
        inside[a:b + 1] = True
    for name in fu.DECODERS:
        for cfg in fu.sample_configs(name, 3, meta)[1:]:
            pred = fu.decode(name, g, cfg)
            assert len(pred) == len(g["y"])
            assert not (pred[inside] == fu.X).any() and (pred[~inside] == fu.X).all()
            assert np.array_equal(pred, fu.decode(name, g, cfg))                   # deterministic
            assert 0.0 <= fu.score(g["y"], pred)["spellF1"] <= 1.0


# ---------------------------------------------------------------------------------- production

@pytest.fixture(scope="module")
def prod_case(syn_spec):
    """FINAL's decoder on the synthetic match, built as experiment.run_game builds it: fake model
    streams (P(dead), P(home), boundary), a filled ball, the Smart Tagging votes."""
    from mqf import atd, decode, features, smart_tagging
    from mqf.alive import combine as acombine
    from mqf.owner import combine as ocombine
    spec = syn_spec
    gt = pd.read_csv(spec.gt_public)["label"].to_numpy().astype("<U1")
    n = spec.n_frames
    rng = np.random.default_rng(11)
    inside = np.zeros(n, bool)
    for _, a, b in spec.halves:
        inside[a:b + 1] = True
    p_dead = np.where(inside, np.round(np.clip(np.where(gt == "D", 0.8, 0.15) + rng.normal(0, 0.15, n), 0, 1), 4),
                      np.nan)
    p_home = np.where(inside, np.round(np.clip(np.where(gt == "H", 0.75, 0.3) + rng.normal(0, 0.2, n), 0, 1), 4),
                      np.nan)
    pb = np.round(np.clip(rng.random(n) ** 6, 0, 1), 4)
    trk = atd.load(spec)
    ball = trk.ball.astype(np.float64).copy()
    miss = np.isnan(ball[:, 0])
    t = np.arange(n)
    for k in (0, 1):                                    # a crude fill, so w_filled is exercised
        ball[miss, k] = np.interp(t[miss], t[~miss], ball[~miss, k])
    filled = miss & inside
    tracking = atd.Tracking(trk.players, trk.player_speed, ball.astype(np.float32), trk.team_of_slot)
    feat = features.build(tracking, spec.pitch, spec.fps, ball_filled=filled, p_home=p_home)
    params = replace(decode.Params(), dead_who="max")
    sw_t = ocombine.switch_costs(pb, params.switch_cost, 0.5)
    tags = smart_tagging.evidence(spec.smart_tagging, spec.atd_meta, n, spec.fps)
    lab = acombine.decode_variant("hmm", feat, p_dead, spec.halves, n, spec.fps, params, None, None, None, 0.0, sw_t)
    lab_tags = acombine.decode_hmm(feat, p_dead, spec.halves, n, params, None, 0.0, False, sw_t, tags)
    return dict(spec=spec, gt=gt, trk=trk, ball=ball, filled=filled, feat=feat, params=params, p_dead=p_dead,
                p_home=p_home, pb=pb, tags=tags, lab=lab, lab_tags=lab_tags)


def test_centre_and_hook_are_final_bit_for_bit(prod_case, official):
    """hmm3_bnd config 0 and the --baseline-hook reproduce FINAL's labels exactly, with and without
    the Smart Tagging votes (BAR-ATH's extra stream)."""
    from mqf import decode
    from mqf.fusion_export import submission_g, to_codes
    c = prod_case
    g = submission_g(c["spec"], c["feat"], c["p_dead"], c["p_home"], c["pb"], 0.5, c["params"])
    want = to_codes(c["lab"])
    assert (want != fu.X).sum() > 0.9 * sum(b - a + 1 for _, a, b in c["spec"].halves)
    assert np.array_equal(fu.decode("hmm3_bnd", g, fu.sample_configs("hmm3_bnd", 1, {})[0]), want)
    assert np.array_equal(decode.fusion_baseline(g), want)
    gt_ = submission_g(c["spec"], c["feat"], c["p_dead"], c["p_home"], c["pb"], 0.5, c["params"], tags=c["tags"])
    assert np.array_equal(fu.decode("hmm3_bnd", gt_, fu.CENTRE), to_codes(c["lab_tags"]))
    assert np.array_equal(decode.fusion_baseline(gt_), to_codes(c["lab_tags"]))
    assert not np.array_equal(c["lab"], c["lab_tags"])                       # the votes did something
    # a perturbed config is a different decoder (the knobs reach the arithmetic)
    assert not np.array_equal(fu.decode("hmm3_bnd", g, {**fu.CENTRE, "c_ha": 6.0}), want)


def test_export_reproduces_the_run_and_refuses_a_mismatch(prod_case, tmp_path, official):
    from mqf import fusion_export
    from mqf.harness.evaluate import score
    c = prod_case
    spec = c["spec"]
    P = {"asf": c["p_dead"], "v8": c["p_dead"], "asf_v8": c["p_dead"]}
    PB = {"asf": c["pb"]}
    s = score(c["gt"], c["lab"], spec.fps)["spellF1"]
    trk = c["trk"]
    path = fusion_export.export(tmp_path, "bar_exact", spec, "syn", 0, c["gt"], trk, c["ball"], c["filled"],
                                c["p_home"], c["pb"], P, PB, c["params"], 0.5, "asf_v8", s,
                                p_home_seeds=np.stack([c["p_home"]] * 3))
    g = fu.load_npz(path)
    assert int(g["nan_dead_in"]) == 0 and int(g["nan_home_in"]) == 0
    assert np.array_equal(g["pred_final"], fusion_export.to_codes(c["lab"]))
    assert np.array_equal(fusion_export.prod(g), g["pred_final"])
    assert np.array_equal(fu.decode("hmm3_bnd", g, fu.CENTRE), g["pred_final"])
    assert fu.score(g["y"], g["pred_final"])["spellF1"] == s
    with pytest.raises(SystemExit):
        fusion_export.export(tmp_path, "bar_exact", spec, "syn", 1, c["gt"], trk, c["ball"], c["filled"],
                             c["p_home"], c["pb"], P, PB, c["params"], 0.5, "asf_v8", s + 1e-3)


# ---------------------------------------------------------------------------------- selection

def _rows(effects, n_games=6, seeds=("0", "1"), conds=("bar_exact", "atd_fuzzy")):
    """Synthetic score rows: decoder -> per-game effect on spellF1 over the baseline 'final'."""
    rng = np.random.default_rng(0)
    rows = []
    for k in range(n_games):
        src = ("idsse", "metrica", "pff", "pff", "pff", "pff")[k % 6]
        for c in conds:
            for s in seeds:
                base = 0.6 + rng.normal(0, 0.03)
                rows.append({"decoder": "final", "cfg": 0, "cond": c, "game": f"G{k}", "seed": s, "source": src,
                             "spellF1": base, "macroF1": 0.8, "adF1": 0.9, "cnt_err": 0, "same_final": -1})
                for (d, ci), eff in effects.items():
                    e = eff(k) + rng.normal(0, 0.002)
                    rows.append({"decoder": d, "cfg": ci, "cond": c, "game": f"G{k}", "seed": s, "source": src,
                                 "spellF1": base + e, "macroF1": 0.8, "adF1": 0.9, "cnt_err": 0, "same_final": -1})
    return rows


def test_logo_rule_and_pipeline():
    effects = {("good", 0): lambda k: 0.0, ("good", 1): lambda k: 0.03,          # better everywhere
               ("lucky", 0): lambda k: 0.0, ("lucky", 1): lambda k: 0.2 if k == 2 else -0.01}  # one game
    tab, res = fu.analyse(_rows(effects), "final")
    assert res["good"]["passes"] and all(p == ("good", 1) for p in res["good"]["picks"])
    assert not res["lucky"]["passes"]
    assert res["PIPELINE"]["passes"]
    assert res["PIPELINE"]["final"] == ("good", 1)
    assert res["good"]["paired"][0]["n"] == 12                                   # 6 games x 2 seeds
    # columns are averaged per source then over sources, as final_check
    assert res["final"]["cols"]["spellF1"][0] == pytest.approx(
        np.mean([np.mean([np.nanmean(tab.v["spellF1"][tab.keys.index(("final", 0)), g, 0])
                          for g in range(6) if tab.sources[tab.src_of[g]] == s]) for s in tab.sources]))


def test_suite_runs_end_to_end(tmp_path, official):
    out = tmp_path / "preds"
    fu.main(["synth", "--out", str(out), "--games", "3", "--n-seeds", "1", "--half-min", "2"])
    args = argparse.Namespace(preds=str(out), decoders=["hmm3_bnd", "hsmm_soft", "asrf"], conds=None, seeds=None,
                              budget=4, quick=False, sample_seed=0, local_frac=0.5, baseline=None,
                              baseline_hook="mqf.decode:fusion_baseline", duration_dir=None, official=None,
                              jobs=1, tag="t", out=str(tmp_path), no_buckets=False)
    run = fu.run_suite(args)
    res = run["res"]["res"]
    assert run["baseline"] == "final"
    assert all(v == 1 for v in run["res"]["tab"].same[("prod", 0)])           # hook = exported labels
    assert all(v == 1 for v in run["res"]["tab"].same[("hmm3_bnd", 0)])       # centre = FINAL
    assert res["prod"]["paired"][0]["mean"] == 0.0
    assert {"PIPELINE", "hmm3_bnd@0", "hsmm_soft"} <= set(res)
    assert (tmp_path / "fusion_t_report.md").read_text(encoding="utf-8").count("PIPELINE") >= 2


def test_make_submission_fusion_options(tmp_path):
    from mqf import make_submission as ms
    assert ms.load_fusion(None, None) == (None, None)
    cfgf = tmp_path / "c.json"
    cfgf.write_text(json.dumps({"decoder": "hmm3_bnd", "config": fu.CENTRE}))
    assert ms.load_fusion("hmm3_bnd", cfgf) == ("hmm3_bnd", fu.CENTRE)
    for bad in (("nope", cfgf), ("hmm3_bnd", None), ("hsmm_emp", cfgf)):
        with pytest.raises(SystemExit):
            ms.load_fusion(*bad)
    with pytest.raises(SystemExit):                                         # --final stays frozen
        ms.settings(argparse.Namespace(final=True, fusion_decoder="hmm3_bnd", fusion_config=cfgf))


# ---------------------------------------------------------------------------------- the submission's decoder (#48)

def _segs(codes):
    c = np.asarray(codes)
    e = np.flatnonzero(np.diff(np.concatenate(([-1], c, [-1]))) != 0)
    return [(int(c[a]), int(b - a)) for a, b in zip(e[:-1], e[1:])]


def _stoppage_legal(codes, m, md, mb):
    """The stoppage decoder's rules: every run >= m, except a possession with a dead ball on BOTH sides (>= mb);
    every dead run >= max(m, md)."""
    seg = _segs(codes)
    for k, (lab, d) in enumerate(seg):
        between = lab != fu.D and 0 < k < len(seg) - 1 and seg[k - 1][0] == fu.D and seg[k + 1][0] == fu.D
        if lab == fu.D and d < max(m, md):
            return False
        if lab != fu.D and d < (mb if between else m):
            return False
    return True


@pytest.mark.parametrize("seed", range(4))
def test_stoppage_exact_when_between_is_shorter_than_min(seed):
    """Tuned lengths put the between-dead-balls minimum BELOW the general one (a quick restart then out again):
    still the brute-force best labelling under the rules."""
    import itertools
    rng = np.random.default_rng(40 + seed)
    E = rng.normal(0, 1, (10, 3))
    E[:, fu.D] += 0.4 * np.cos(np.arange(10))
    m, md, mb = 3, 2, 1
    out = fu.stoppage_path(E, m, md, mb)
    best = max(sum(E[t, p[t]] for t in range(10)) for p in itertools.product(range(3), repeat=10)
               if _stoppage_legal(p, m, md, mb))
    assert _stoppage_legal(out, m, md, mb)
    assert sum(E[t, out[t]] for t in range(10)) == pytest.approx(best)


@pytest.mark.parametrize("seed", range(3))
def test_stoppage_numpy_fallback_equals_fast_kernel(seed, monkeypatch):
    rng = np.random.default_rng(70 + seed)
    E = rng.normal(0, 1.5, (3000, 3))
    fast = fu.stoppage_path(E, 13, 50, 15)
    monkeypatch.setattr(fu, "_smk_fast", fu._smk_numpy)
    assert np.array_equal(fast, fu.stoppage_path(E, 13, 50, 15))


def test_final_is_the_stoppage_decoder_on_final_hmm_evidence(official):
    from mqf import make_submission as ms
    fo, ho = ms.FINAL["options"], ms.FINAL_HMM["options"]
    assert fo["fusion_decoder"] == "stoppage" and fo["fusion_config"] is ms.STOPPAGE
    assert {k: v for k, v in fo.items() if not k.startswith("fusion_")} == ho and ms.FINAL["params"] == ms.FINAL_HMM["params"]
    assert set(ms.STOPPAGE) == {"min_s", "min_dead_s", "min_between_s"}
    assert all(isinstance(v, float) and 0 < v < 10 for v in ms.STOPPAGE.values())
    g = fu.synth_game(np.random.default_rng(8), half_min=3.0)
    out = fu.decode("stoppage", g, ms.STOPPAGE)
    m, md, mb = (fu.frames(ms.STOPPAGE[k], g["fps"]) for k in ("min_s", "min_dead_s", "min_between_s"))
    for a, b in g["halves"]:
        assert _stoppage_legal(out[a:b + 1], m, md, mb) and not (out[a:b + 1] == fu.X).any()
    # the evidence is FINAL_HMM's: the config carries no fusion knob, so every emission term is the centre's
    assert np.array_equal(fu.emissions3(fu.evidence(g, ms.STOPPAGE)), fu.emissions3(fu.evidence(g, fu.CENTRE)))
