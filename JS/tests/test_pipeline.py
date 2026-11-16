"""End to end on the synthetic match, scored and validated by the OFFICIAL code from vendor/."""
from __future__ import annotations

import subprocess
import sys

import numpy as np

from mqf import atd, decode, make_submission, synthetic
from mqf.config import OFFICIAL_DIR
from mqf.harness import noise


def graded_mask(spec):
    m = np.zeros(spec.n_frames, dtype=bool)
    for _, first, last in spec.halves:
        m[first:last + 1] = True
    return m


def test_submission_contract(syn_spec, syn_submission, official):
    _, pm = official
    table = pm.read_label_csv(str(syn_submission))
    assert pm.validate_submission(table, syn_spec.n_frames) is None      # the official check
    codes = pm.label_codes(table)
    assert len(codes) == syn_spec.n_frames
    assert not np.any(codes[graded_mask(syn_spec)] == pm.X), "X on a graded frame never helps"


def test_official_validator_cli(syn_spec, syn_submission):
    exe = OFFICIAL_DIR / "evaluation" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(exe), str(syn_submission), "--rows", str(syn_spec.n_frames)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("OK")


def test_official_scorer_sanity(syn_spec, syn_submission, official):
    _, pm = official
    gt = pm.label_codes(pm.read_label_csv(str(syn_spec.gt_public)))
    pred = pm.label_codes(pm.read_label_csv(str(syn_submission)))
    r = pm.evaluate(gt, pred)
    # a sanity floor on an easy match, not a target: v0 scores ~0.7 here
    assert r["spell_f1_mean"] > 0.3 and r["macro_f1"] > 0.5, r["spell_f1_mean"]


def test_deterministic(syn_spec, syn_submission, tmp_path):
    again = tmp_path / "again.csv"
    make_submission.run(syn_spec, decode.Params(), again)
    assert again.read_bytes() == syn_submission.read_bytes()


def test_atd_roundtrip(syn_spec):
    """The writer and the reader agree: file -> arrays equals what the noise model produced."""
    m = synthetic.clean_match(syn_spec.n_frames, syn_spec.fps, 7, syn_spec.halves, syn_spec.pitch)
    want = noise.corrupt(m["players"], m["team_of_player"], m["ball"], m["labels"] != "X", syn_spec.fps,
                         noise.NoiseConfig(seed=7))
    got = atd.load(syn_spec, use_cache=False)
    L, _ = syn_spec.pitch
    assert got.players.shape == want.players.shape == (syn_spec.n_frames, 46, 2)
    np.testing.assert_array_equal(np.isnan(got.players[..., 0]), np.isnan(want.players[..., 0]))
    np.testing.assert_allclose(got.players, want.players, atol=1e-5 * L, equal_nan=True)
    np.testing.assert_allclose(got.ball, want.ball, atol=1e-5 * L, equal_nan=True)
    np.testing.assert_allclose(got.player_speed, want.player_speed, atol=6e-3, equal_nan=True)
    assert (got.team_of_slot == 0).sum() == 23 and (got.team_of_slot == 1).sum() == 23


def test_profile_identity_runs(syn_spec):
    from mqf import profile
    r = profile.profile_identity(atd.load(syn_spec, use_cache=False), syn_spec, graded_mask(syn_spec))
    assert r["share_frames_over_11"]["home"] is not None and r["births_per_min"] > 0
    assert 0 <= r["births_with_lost_track_in_gate"]["within_1s"] <= r["births_with_lost_track_in_gate"]["within_10s"] <= 1


def test_compare_two_submissions(syn_spec, syn_submission, tmp_path):
    from mqf import compare
    a = compare._labels(syn_submission)
    same = compare.compare(a, a, syn_spec.halves, syn_spec.fps)
    assert same["graded_frames_changed"] == 0.0 and same["changed_stretches"]["n"] == 0
    b = a.copy()
    _, first, _ = syn_spec.halves[0]
    b[first:first + 50] = "D"
    r = compare.compare(a, b, syn_spec.halves, syn_spec.fps)
    assert r["changed_stretches"]["n"] >= 1 and 0 < r["graded_frames_changed"] < 0.01


def test_compare_resolves_names_and_reports_missing(tmp_path, monkeypatch):
    import pytest
    from mqf import compare
    res = tmp_path / "res"
    res.mkdir()
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")          # relative paths must not hit the real results/ next to JS/
    (res / "base_v0.csv").write_text("frame,label\n0,H\n")
    assert compare.resolve("base_v0", res) == res / "base_v0.csv"
    assert compare.resolve("results/base_v0.csv", res) == res / "base_v0.csv"
    with pytest.raises(SystemExit) as e:
        compare.resolve("results/Perosinho_FC.csv", res)
    assert "base_v0.csv" in str(e.value) and "tasks.py submit" in str(e.value)


# --- the submitted recipe: --final, --params, and --reproduce from a saved report -------------------------

def _args(**kw):
    import argparse
    base = dict(final=False, no_tags=False, clean_tracks=False, ball_fill=False, ball_model=None, dead_model=None,
                who_model=None, params=[])
    return argparse.Namespace(**{**base, **kw})


def test_parse_params_types_and_errors():
    import pytest
    p = make_submission.parse_params(["dead_who=max", "switch_cost=20", "w_tags=0.25"])
    assert p == {"dead_who": "max", "switch_cost": 20.0, "w_tags": 0.25}
    assert isinstance(p["switch_cost"], float)
    for bad in (["nope=1"], ["dead_who"], ["dead_who=both"]):
        with pytest.raises(SystemExit):
            make_submission.parse_params(bad)


def test_settings_final_and_flags():
    import pytest
    from pathlib import Path
    opts, params = make_submission.settings(_args(final=True))
    assert params.dead_who == "max" and params.switch_cost == decode.Params().switch_cost
    assert opts["clean_tracks"] and opts["ball_fill"] and opts["use_tags"]
    assert opts["who_model"] == [f"models/owner_tcn_full_impff_s{k}.pt" for k in range(3)] and opts["dead_model"] is None
    assert opts["alive_model"] == [f"models/alive_asf_full_impff_s{k}.pt" for k in range(3)]
    assert opts["who_bnd_alpha"] == 0.5 and opts["alive_variant"] == "hmm" and opts["alive_tag_w"] == 0.0
    assert opts["deadball_model"] == Path("models/deadball_v8.joblib") and opts["deadball_decoder"] == "hmm"
    with pytest.raises(SystemExit):                      # the recipe is frozen: no mixing with option flags
        make_submission.settings(_args(final=True, clean_tracks=True))
    with pytest.raises(SystemExit):
        make_submission.settings(_args(final=True, params=["dead_who=zero"]))
    opts, params = make_submission.settings(_args(ball_fill=True, params=["dead_who=max"]))
    assert opts["ball_fill"] and not opts["clean_tracks"] and params.dead_who == "max"


def test_reproduce_reuses_saved_params(syn_spec, tmp_path):
    """--reproduce rebuilds with the reference report's decoder parameters, not the defaults."""
    import json
    import pytest
    from dataclasses import replace
    ref = tmp_path / "ref.csv"
    params = replace(decode.Params(), switch_cost=30.0, to_dead_cost=6.0)     # visibly different labels
    make_submission.run(syn_spec, params, ref)
    saved = json.loads(ref.with_suffix(".json").read_text())
    opts, again_params = make_submission.settings(_args(final=True), saved)  # the report wins over the flags
    assert again_params == params and opts["clean_tracks"] is False
    again = tmp_path / "again.csv"
    make_submission.run(syn_spec, again_params, again, **opts)
    assert again.read_bytes() == ref.read_bytes()
    default = tmp_path / "default.csv"
    make_submission.run(syn_spec, decode.Params(), default, **opts)
    assert default.read_bytes() != ref.read_bytes()      # ... so reusing the defaults would have failed
    saved["params"]["not_a_param"] = 1
    with pytest.raises(SystemExit):
        make_submission.settings(_args(), saved)


def test_final_recipe_runs_and_is_deterministic(official, tmp_path):
    """FINAL end to end (models, ball fill, cleaner, tags) on a short synthetic match, twice, byte-identical.
    Skips until the alive asf models are committed (they are trained on the team's machine)."""
    import json
    import pytest
    from mqf.config import ROOT
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    fin = make_submission.FINAL["options"]
    needed = [*fin["who_model"], *fin["alive_model"], fin["deadball_model"], "models/ball_B_off_v5.pt"]
    if not all((ROOT / m).is_file() for m in needed):
        pytest.skip(f"submitted models not in models/: {[m for m in needed if not (ROOT / m).is_file()]}")
    spec = synthetic.generate(tmp_path / "short", n_frames=9_000, halves=((1, 300, 4_199), (2, 4_700, 8_699)))
    opts, params = make_submission.settings(_args(final=True))
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    rep = make_submission.run(spec, params, a, **opts)
    make_submission.run(spec, params, b, **opts)
    assert a.read_bytes() == b.read_bytes()
    saved = json.loads(a.with_suffix(".json").read_text())
    assert saved["options"]["who_model"] == fin["who_model"]                 # recorded as given, portable
    assert saved["options"]["alive_model"] == fin["alive_model"] and saved["options"]["who_bnd_alpha"] == 0.5
    assert saved["options"]["deadball_model"] == fin["deadball_model"] and saved["options"]["deadball_decoder"] == "hmm"
    assert saved["params"]["dead_who"] == "max" and rep["used_tags"]
    assert make_submission.settings(_args(), saved) == (opts, params)
