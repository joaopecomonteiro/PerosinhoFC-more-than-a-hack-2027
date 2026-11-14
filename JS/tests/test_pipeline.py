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
