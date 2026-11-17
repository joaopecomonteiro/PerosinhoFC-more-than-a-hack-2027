"""Viewer: the Original / Fixed tracking switch -- synthetic data only."""
from __future__ import annotations

import numpy as np
import pytest

from mqf import synthetic
from mqf.harness.noise import NoiseConfig, corrupt
from mqf.viewer import fixed
from mqf.viewer import game as G


def _trk():
    m = synthetic.clean_match(n_frames=12_000, seed=11, halves=((1, 300, 5_699), (2, 6_300, 11_699)))
    trk = corrupt(m["players"], m["team_of_player"], m["ball"], m["labels"] != "X", m["fps"], NoiseConfig())
    return m, trk


def test_fixed_tracking_without_model_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fixed, "VIEWER_CACHE", tmp_path)
    m, trk = _trk()
    logs = []
    ts = fixed.fixed_tracking(trk, m["fps"], m["pitch"], m["halves"], model_path=tmp_path / "none.pt", log=logs.append)
    assert ts.key == "fixed" and ts.players.shape[0] == len(trk.ball) and len(ts.team_of_slot) == ts.players.shape[1]
    seen = ~np.isnan(trk.ball[:, 0])
    np.testing.assert_array_equal(ts.ball[seen], trk.ball[seen])                 # tracked ball untouched
    assert set(np.unique(ts.ball_source)) <= {0, 1, 3} and (ts.ball_source == 1).any()   # no model: interp only
    assert "only gaps <= 2 s" in ts.note
    ts2 = fixed.fixed_tracking(trk, m["fps"], m["pitch"], m["halves"], model_path=tmp_path / "none.pt", log=logs.append)
    assert any("cached" in l for l in logs) and np.array_equal(ts2.ball_source, ts.ball_source)


def test_bundle_lists_tracking_sets():
    m, trk = _trk()
    n = len(trk.ball)
    extra = G.TrackingSet("fixed", "Fixed", trk.players[:, :30], trk.team_of_slot[:30], trk.ball,
                          np.zeros(n, dtype=np.uint8), "note")
    g = G.ViewerGame("SYN", "A v B", "A", "B", m["fps"], n, m["pitch"], m["halves"], trk.players,
                     trk.team_of_slot, trk.ball, [], [], extra_tracking=[extra])
    meta, blobs = G.bundle(g)
    keys = [t["key"] for t in meta["tracking"]]
    assert keys == ["original", "fixed"]
    t = meta["tracking"][1]
    assert len(blobs[t["players"]]) == n * 30 * 4 and len(blobs[t["ball"]]) == n * 4 and len(blobs[t["ball_source"]]) == n
    assert meta["tracking"][0]["players"] == "players" and meta["n_slots"] == trk.players.shape[1]


def test_fixed_is_its_own_copy_and_carries_flags(tmp_path, monkeypatch):
    """Original stays exactly as delivered; Fixed is another array with its own slots and flags."""
    monkeypatch.setattr(fixed, "VIEWER_CACHE", tmp_path)
    m, trk = _trk()
    before = trk.players.copy(), trk.ball.copy()
    logs = []
    ts = fixed.fixed_tracking(trk, m["fps"], m["pitch"], m["halves"], model_path=tmp_path / "none.pt", log=logs.append)
    np.testing.assert_array_equal(trk.players, before[0])                         # cleaning never writes into it
    np.testing.assert_array_equal(trk.ball, before[1])
    assert not np.may_share_memory(ts.players, trk.players)
    assert ts.flags is not None and ts.flags.shape == ts.players.shape[:2]
    assert any(l.startswith("fixed tracking: vs Original:") for l in logs)        # the change summary is logged
    n = len(trk.ball)
    g = G.ViewerGame("SYN", "A v B", "A", "B", m["fps"], n, m["pitch"], m["halves"], trk.players,
                     trk.team_of_slot, trk.ball, [], [], extra_tracking=[ts])
    meta, blobs = G.bundle(g)
    orig, fx = meta["tracking"]
    assert orig["flags"] is None and len(blobs[fx["flags"]]) == n * fx["n_slots"]
    assert blobs[orig["players"]] != blobs[fx["players"]]
    ts2 = fixed.fixed_tracking(trk, m["fps"], m["pitch"], m["halves"], model_path=tmp_path / "none.pt", log=logs.append)
    np.testing.assert_array_equal(ts2.flags, ts.flags)                            # flags survive the cache


def test_fixed_cache_key_follows_the_cleaner_code(tmp_path, monkeypatch):
    m, trk = _trk()
    k1 = fixed._key(trk, m["fps"], m["pitch"], m["halves"], None, None)
    src = tmp_path / "tracks_copy.py"
    src.write_text(open(fixed.tracks.__file__).read() + "\n# edited\n")
    monkeypatch.setattr(fixed.tracks, "__file__", str(src))
    assert fixed._key(trk, m["fps"], m["pitch"], m["halves"], None, None) != k1


# ---------------------------------------------------------------------------------- Best so far lane

def _halves_inside(halves, n):
    m = np.zeros(n, dtype=bool)
    for _, a, b in halves:
        m[a:b + 1] = True
    return m


def test_best_decode_without_dead_model_is_h_a_only():
    """dead=none: every frame of the halves is H or A, X outside; the boundary lowers the switch cost."""
    from mqf.viewer import best
    n, halves = 3_000, ((1, 100, 1_399), (2, 1_600, 2_899))
    p = np.full(n, np.nan)
    inside = _halves_inside(halves, n)
    p[inside] = 0.9
    p[700:715] = 0.1                                            # a 0.6 s away blip inside a home spell
    no_bnd = np.where(inside, 0.0, np.nan)
    lab = best.decode(p, no_bnd, halves, n, best.DEAD_NONE)
    assert set(lab[inside]) <= {"H", "A"} and (lab[~inside] == "X").all()
    assert (lab[700:715] == "H").all()                          # too short for the full switch cost
    bnd = no_bnd.copy()
    bnd[698:702] = bnd[713:717] = 1.0                           # the boundary head sees both changes
    lab_b = best.decode(p, bnd, halves, n, best.DEAD_NONE)
    assert (lab_b[702:713] == "A").all() and (lab_b[600:690] == "H").all()


def test_bundle_carries_lane_probabilities():
    m, trk = _trk()
    n = len(trk.ball)
    p = np.full(n, np.nan)
    p[1000:1010] = np.linspace(0, 1, 10)
    lane = G.Lane("Best so far", "model", np.full(n, 3, np.uint8), "x", p_home=p, p_boundary=None, no_dead=True,
                  note="agg")
    g = G.ViewerGame("SYN", "A v B", "A", "B", m["fps"], n, m["pitch"], m["halves"], trk.players,
                     trk.team_of_slot, trk.ball, [lane], [])
    meta, blobs = G.bundle(g)
    e = meta["lanes"][0]
    assert e["no_dead"] is True and e["p_boundary"] is None and e["note"] == "agg"
    enc = np.frombuffer(blobs[e["p_home"]], dtype=np.uint8)
    assert len(enc) == n and enc[0] == 255 and enc[1000] == 0 and enc[1009] == 250
    np.testing.assert_allclose(enc[1000:1010] / 250, np.linspace(0, 1, 10), atol=0.002)


def test_best_lane_never_blocks_the_viewer(tmp_path):
    """No Fixed set, missing models or a bad option: (None, why) -- the viewer opens without the lane."""
    from mqf.viewer import best
    m, trk = _trk()
    n = len(trk.ball)
    ts = G.TrackingSet("fixed", "Fixed", trk.players, trk.team_of_slot, trk.ball)
    for args, kw, word in (((None,), {}, "Fixed"),
                           ((ts,), {"model_paths": [tmp_path / "none.pt"]}, "missing"),
                           ((ts,), {"dead_mode": "oracle"}, "--best-dead")):
        lane, why = best.best_lane(*args, m["fps"], m["pitch"], m["halves"], n, log=lambda s: None, **kw)
        assert lane is None and word in why


def test_best_lane_end_to_end_on_synthetic(tmp_path, monkeypatch):
    """The three ens_impff TCNs on a cleaned synthetic match: H/A only inside the halves, cached."""
    import pytest
    from mqf.viewer import best
    pytest.importorskip("torch")
    if not all(p.is_file() for p in best.ENSEMBLE):
        pytest.skip("models/owner_tcn_full_impff_s*.pt not here")
    monkeypatch.setattr(fixed, "VIEWER_CACHE", tmp_path)
    monkeypatch.setattr(best, "VIEWER_CACHE", tmp_path)
    m, trk = _trk()
    n = len(trk.ball)
    ts = fixed.fixed_tracking(trk, m["fps"], m["pitch"], m["halves"], model_path=tmp_path / "none.pt",
                              log=lambda s: None)
    logs = []
    lane, why = best.best_lane(ts, m["fps"], m["pitch"], m["halves"], n, device="cpu", log=logs.append)
    assert why == "" and lane.no_dead and lane.kind == "model"
    inside = _halves_inside(m["halves"], n)
    assert set(np.unique(lane.codes[inside])) <= {0, 1} and (lane.codes[~inside] == 3).all()
    assert np.isfinite(lane.p_home[inside]).all() and np.isnan(lane.p_home[~inside]).all()
    alive = inside & np.isin(m["labels"], ["H", "A"])
    truth = np.where(m["labels"] == "H", 0, 1)
    assert np.mean(lane.codes[alive] == truth[alive]) > 0.6   # far above chance on a toy match
    lane2, _ = best.best_lane(ts, m["fps"], m["pitch"], m["halves"], n, device="cpu", log=logs.append)
    assert any("cached" in l for l in logs) and np.array_equal(lane2.codes, lane.codes)


# ---------------------------------------------------------------------------------- fusion lanes (last stage)

def test_fusion_lanes_no_decoder_final_and_best(official):
    from mqf import fusion as fu
    from mqf.viewer import fusion_lanes as FL
    g = fu.synth_game(np.random.default_rng(2), half_min=3.0)
    best = {"decoder": "hsmm_soft", "config": fu.sample_configs("hsmm_soft", 4, {})[3], "label": "hsmm_soft (fusion)"}
    ls = FL.lanes(g, best, g["y"])
    assert [l.name for l in ls] == [FL.SUB, FL.RAW, FL.FIN, FL.MIN, FL.STOP, FL.V8_DEC, "hsmm_soft (fusion)"]
    from mqf import fusion_export, make_submission
    fo = make_submission.FINAL["options"]
    assert np.array_equal(ls[0].codes, fu.decode(fo["fusion_decoder"], g, fo["fusion_config"]))   # the submission
    assert ls[0].compare and not any(l.compare for l in ls[1:]) and "spellF1" in ls[0].note
    assert np.array_equal(ls[3].codes, FL.min_run_decode(g))                     # best so far, runs >= 0.5 s
    assert np.array_equal(ls[4].codes, FL.stoppage_decode(g))
    assert np.array_equal(ls[5].codes, fusion_export.prod({**g, "p_dead": g["p_dead_v8"]}))   # v8, production HMM
    assert not np.array_equal(ls[5].codes, ls[2].codes)
    assert np.array_equal(ls[1].codes, fu.decode("argmax", g, fu.CENTRE))          # each frame on its own
    assert np.array_equal(ls[2].codes, g["pred_final"])                            # FINAL_HMM (synth_game's)
    assert "spellF1" in ls[2].note and ls[2].p_boundary is not None
    assert (np.diff(ls[1].codes.astype(int)) != 0).sum() > (np.diff(ls[2].codes.astype(int)) != 0).sum()
    meta, blobs = G.bundle(G.ViewerGame("k", "t", "H", "A", 25.0, len(g["y"]), (105.0, 68.0), ((1, 0, 9),),
                                        np.zeros((len(g["y"]), 1, 2)), np.zeros(1, int), np.zeros((len(g["y"]), 2)),
                                        ls, []))
    assert [l["compare"] for l in meta["lanes"]] == [True, False, False, False, False, False, False]


def test_val_game_full_truth_and_tracking(tmp_path, monkeypatch, official):
    """--val: the exported file's truth, the lanes, and the tracking final_check read for that column."""
    from dataclasses import replace
    from mqf import decode, fusion, fusion_export
    from mqf.alive import experiment
    from mqf.external import bundle
    from mqf.viewer import fusion_lanes as FL
    halves = ((1, 500, 11_499), (2, 12_500, 23_499))
    bundle.write_bundle({**synthetic.clean_match(n_frames=24_000, seed=5, halves=halves), "name": "METRICA-V9",
                         "source": "metrica", "home": "A", "away": "B", "labels_from": "synthetic"},
                        root=tmp_path / "bundles")
    real_find = bundle.find
    monkeypatch.setattr(bundle, "find", lambda source=None, names=None, root=None:
                        real_find(source, names, tmp_path / "bundles"))
    d = bundle.find(names=["METRICA-V9"])[0]
    spec, trk, gt = experiment.load_condition(d, "bar", 0, "exact", True)
    rng = np.random.default_rng(0)
    n = spec.n_frames
    inside = np.zeros(n, bool)
    for _, a, b in spec.halves:
        inside[a:b + 1] = True
    pd_ = np.where(inside, np.round(np.clip(np.where(gt == "D", 0.8, 0.2) + rng.normal(0, 0.2, n), 0, 1), 4), np.nan)
    ph = np.where(inside, np.round(np.clip(np.where(gt == "H", 0.7, 0.3) + rng.normal(0, 0.25, n), 0, 1), 4), np.nan)
    pb = np.round(rng.random(n) ** 8, 4)
    params = replace(decode.Params(), dead_who="max")
    fusion_export.export(tmp_path / "fp", "bar_exact", spec, "metrica", 0, gt, trk, trk.ball, None, ph, pb,
                         {"asf": pd_, "v8": pd_, "asf_v8": pd_}, {"asf": pb}, params, 0.5)
    assert "METRICA-V9: bar_exact s0" in FL.val_list(tmp_path / "fp")
    game = FL.val_game(tmp_path / "fp", "METRICA-V9", "bar_exact", "0", log=lambda m: None)
    g = fusion.load_npz(tmp_path / "fp" / "bar_exact" / "METRICA-V9__s0.npz")
    assert [l.name for l in game.lanes] == ["Ground truth", FL.SUB, FL.RAW, FL.FIN, FL.MIN, FL.STOP, FL.V8_DEC]
    assert np.array_equal(game.lanes[0].codes, fusion_export.to_codes(gt))         # full ground truth
    assert np.array_equal(game.lanes[3].codes, g["pred_final"])                  # FINAL_HMM's labels, exported
    assert game.players.shape == trk.players.shape and np.array_equal(game.ball, trk.ball, equal_nan=True)
    assert not game.notes and game.videos[0].path is None
    meta, _ = G.bundle(game)
    assert meta["lanes"][1]["compare"] and meta["lanes"][3]["p_boundary"]
    with pytest.raises(SystemExit):
        FL.val_game(tmp_path / "fp", "METRICA-V9", "atd_fuzzy", "0", log=lambda m: None)


def test_dump_arrays_stays_out_of_the_repo(tmp_path):
    from mqf import make_submission
    from mqf.config import ROOT
    with pytest.raises(SystemExit):
        make_submission._dump_arrays(ROOT / "results" / "x.npz", *([None] * 12))


def test_final_dump_arrays_feed_the_lanes(official, tmp_path):
    """make_submission --dump-arrays on FINAL: the saved labels are the CSV's, the fusion centre reproduces
    them, and the BAR-ATH lane builder reads the file."""
    import argparse
    from mqf import fusion, make_submission
    from mqf.config import ROOT
    from mqf.viewer import fusion_lanes as FL
    pytest.importorskip("torch")
    fin = make_submission.FINAL["options"]
    needed = [*fin["who_model"], *fin["alive_model"], fin["deadball_model"], "models/ball_B_off_v5.pt"]
    if not all((ROOT / m).is_file() for m in needed):
        pytest.skip("submitted models not in models/")
    spec = synthetic.generate(tmp_path / "short", n_frames=9_000, halves=((1, 300, 4_199), (2, 4_700, 8_699)))
    ns = argparse.Namespace(final=True, synthetic=False, no_tags=False, clean_tracks=False, ball_fill=False,
                            ball_model=None, dead_model=None, who_model=None, who_bnd_alpha=None, params=[],
                            alive_model=None, alive_variant="hmm", alive_tag_w=0.0, alive_goal_w=0.0,
                            deadball_model=None, deadball_decoder="hsmm", deadball_prior=0.5,
                            fusion_decoder=None, fusion_config=None)
    opts, params = make_submission.settings(ns)
    out, dump = tmp_path / "f.csv", tmp_path / "arrays.npz"
    make_submission.run(spec, params, out, **opts, dump_arrays=dump)
    g = fusion.load_npz(dump)
    import pandas as pd
    lab = pd.read_csv(out)["label"].to_numpy().astype(str)
    assert np.array_equal(FL.LAB[g["pred_final"]], lab)
    fo = make_submission.FINAL["options"]
    assert str(g["submission_decoder"]) == fo["fusion_decoder"] == "stoppage"
    assert np.array_equal(fusion.decode(fo["fusion_decoder"], g, fo["fusion_config"]), g["pred_final"])
    ls, why = FL.arrays_lanes(dump, spec.n_frames, log=lambda m: None)
    assert np.array_equal(ls[0].codes, g["pred_final"]) and ls[0].compare and not ls[2].compare
    assert not why and [l.name for l in ls] == [FL.SUB, FL.RAW, FL.FIN, FL.MIN, FL.STOP, FL.V8_DEC] and \
        "spellF1" not in ls[1].note


def test_three_bars_by_default_with_the_arrays():
    import argparse
    from mqf.viewer.__main__ import lane_plan
    best = {"dead_mode": "none"}
    ns = lambda **k: argparse.Namespace(**{"all_lanes": False, **k})
    assert lane_plan(ns(), {"arrays": "a.npz"}, best) == (None, False)        # GT + no decoder + decoder only
    assert lane_plan(ns(all_lanes=True), {"arrays": "a.npz"}, best) == (best, True)
    assert lane_plan(ns(), None, best) == (best, True)                         # no arrays yet: older lanes


def _runs(codes):
    e = np.flatnonzero(np.diff(np.concatenate(([-1], np.asarray(codes), [-1]))) != 0)
    return np.diff(e)


def test_min_run_decoder_is_exact_and_respects_the_minimum(official):
    """Every run >= 0.5 s inside each half (13 frames at 25 fps), and on a tiny case the brute-force best
    labelling under that one rule."""
    import itertools
    from mqf import fusion as fu
    from mqf.viewer import fusion_lanes as FL
    g = fu.synth_game(np.random.default_rng(3), half_min=2.0)                  # best so far's evidence (the mix)
    out = FL.min_run_decode(g)
    m = int(np.ceil(0.5 * g["fps"]))
    assert m == 13
    for a, b in g["halves"]:
        assert _runs(out[a:b + 1]).min() >= m
    assert (out[: g["halves"][0][0]] == fu.X).all()
    a, b = g["halves"][0]
    assert (_runs(fu.decode("argmax", g, fu.CENTRE)[a:b + 1]) < m).any()     # the rule did something
    rng = np.random.default_rng(1)
    E = rng.normal(0, 1, (9, 3))
    pen = np.zeros((3, 3))
    pen[:, :2] = fu.HARD
    path = fu.semimarkov_viterbi(E, np.zeros((3, 3)), pen=pen)
    legal = [p for p in itertools.product(range(3), repeat=9) if _runs(p).min() >= 3]
    best = max(sum(E[t, p[t]] for t in range(9)) for p in legal)
    assert _runs(path).min() >= 3 and sum(E[t, path[t]] for t in range(9)) == pytest.approx(best)


def _segments(codes):
    c = np.asarray(codes)
    e = np.flatnonzero(np.diff(np.concatenate(([-1], c, [-1]))) != 0)
    return [(int(c[a]), int(b - a)) for a, b in zip(e[:-1], e[1:])]


def _legal(codes, m, md, mb):
    seg = _segments(codes)
    for k, (lab, d) in enumerate(seg):
        if d < m or (lab == 2 and d < md):
            return False
        if lab != 2 and 0 < k < len(seg) - 1 and seg[k - 1][0] == 2 and seg[k + 1][0] == 2 and d < mb:
            return False
    return True


@pytest.mark.parametrize("seed", range(5))
def test_stoppage_rules_are_exact(seed):
    """The 7-state decoder finds the brute-force best labelling under: runs >= m, dead runs >= md, a possession
    with a dead ball on both sides >= mb (tiny case, every labelling enumerated)."""
    import itertools
    from mqf.viewer import fusion_lanes as FL
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 1, (11, 3))
    E[:, 2] += 0.3 * np.sin(np.arange(11))                 # dead / alive alternation worth exploring
    m, md, mb = 1, 3, 3
    out = FL.stoppage_path(E, m, md, mb)
    best = max(sum(E[t, p[t]] for t in range(11)) for p in itertools.product(range(3), repeat=11)
               if _legal(p, m, md, mb))
    assert _legal(out, m, md, mb)
    assert sum(E[t, out[t]] for t in range(11)) == pytest.approx(best)


def test_stoppage_lane_on_synthetic(official):
    from mqf import fusion as fu
    from mqf.viewer import fusion_lanes as FL
    g = fu.synth_game(np.random.default_rng(4), half_min=2.0)
    out = FL.stoppage_decode(g)
    m, md, mb = 13, 50, 50                                  # 0.5 s, 2 s, 2 s at 25 fps
    for a, b in g["halves"]:
        assert _legal(out[a:b + 1], m, md, mb)
    assert (out[: g["halves"][0][0]] == fu.X).all()
    assert not np.array_equal(out, FL.min_run_decode(g))    # the 2 s rules changed something
