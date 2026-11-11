"""Team-level tracklet cleaning on hand-made tracks (no real data)."""
from __future__ import annotations

import numpy as np

from mqf import tracks
from mqf.atd import Tracking

FPS = 25


def _trk(xy, team):
    xy = np.asarray(xy, dtype=float)
    return Tracking(players=xy, player_speed=np.full(xy.shape[:2], np.nan), ball=np.full((len(xy), 2), np.nan),
                    team_of_slot=np.asarray(team, dtype=np.int8))


def _walk(n, x0=10.0, y=30.0, v=2.0):
    return np.stack([x0 + v * np.arange(n) / FPS, np.full(n, y)], axis=1)


def test_clean_tracks_untouched():
    n = 200
    xy = np.stack([_walk(n, 10 + 5 * k, 5 + 3 * k) for k in range(22)], axis=1)
    out, rep = tracks.clean(_trk(xy, [0] * 11 + [1] * 11), FPS)
    assert rep["cuts_at_jumps"] == 0 and rep["duplicate_episodes"] == 0 and rep["cap_points_hidden"] == 0
    assert out.players.shape[1] == 22
    got = np.sort(out.players[100, :, 0]); want = np.sort(xy[100, :, 0])
    np.testing.assert_allclose(got, want, atol=1e-4)
    np.testing.assert_allclose(np.nanmedian(out.player_speed), 2.0, atol=1e-3)   # speeds from positions


def test_jump_is_cut_and_no_speed_spike():
    n = 100
    a = _walk(n)
    a[50:] += [20.0, 0.0]                                                         # 20 m in one frame
    out, rep = tracks.clean(_trk(a[:, None, :], [0]), FPS)
    assert rep["cuts_at_jumps"] == 1
    assert np.nanmax(out.player_speed) < 3.0                                      # no 500 m/s artefact
    assert out.players.shape[1] == 2                                              # two pieces, two lanes


def test_duplicate_older_hidden_newer_continues():
    n = 120
    a = _walk(n)
    b = np.full((n, 2), np.nan)
    b[50:] = a[50:] + [0.3, 0.2]                                                  # tracker re-acquires the same player
    a[100:] = np.nan                                                              # old track dies later
    out, rep = tracks.clean(_trk(np.stack([a, b], axis=1), [0, 0]), FPS)
    assert rep["duplicate_episodes"] == 1 and rep["episodes_handover_old_died"] == 1
    n_pts = (~np.isnan(out.players[..., 0])).sum(axis=1)
    assert (n_pts[:n] == 1).all()                                                 # one point per frame, no gap
    assert out.players.shape[1] == 1                                              # stitched into one lane


def test_duplicate_newer_dies_first_older_resumes():
    n = 120
    a = _walk(n)
    b = np.full((n, 2), np.nan)
    b[50:70] = a[50:70] + [0.3, 0.0]
    out, rep = tracks.clean(_trk(np.stack([a, b], axis=1), [0, 0]), FPS)
    assert rep["episodes_newer_died_first"] == 1
    n_pts = (~np.isnan(out.players[..., 0])).sum(axis=1)
    assert (n_pts == 1).all()


def test_cap_at_11():
    n = 60
    xy = np.stack([_walk(n, 10 + 5 * k, 5 + 3 * k) for k in range(12)], axis=1)
    xy[:, 11] = xy[:, 0] + [0.6, 0.0]                                             # 12th track next to the 1st ...
    xy[:, 11, 1] += np.sin(np.arange(n) / 3.0) * 0.3                              # ... but not moving alike
    out, rep = tracks.clean(_trk(xy, [0] * 12), FPS)
    assert ((~np.isnan(out.players[..., 0])).sum(axis=1) <= 11).all()
    assert rep["share_frames_over_11"]["after"]["home"] == 0.0


def test_duplicate_pair_that_separates_keeps_both():
    n = 150
    a = _walk(n)
    b = np.full((n, 2), np.nan)
    b[50:] = a[50:] + [0.3, 0.0]
    b[80:, 1] += np.linspace(0, 15, n - 80)                                       # they split up at 80
    out, rep = tracks.clean(_trk(np.stack([a, b], axis=1), [0, 0]), FPS)
    n_pts = (~np.isnan(out.players[..., 0])).sum(axis=1)
    assert (n_pts[50:80] == 1).all()                                              # one point while together
    assert (n_pts[95:] == 2).all()                                                # both back once apart


def test_cap_prefers_off_pitch_track():
    n = 60
    xy = np.stack([_walk(n, 10 + 5 * k, 5 + 3 * k) for k in range(12)], axis=1)
    xy[:, 11] = np.stack([np.full(n, 50.0), np.full(n, 71.0)], axis=1)            # 3 m beyond the touchline
    out, rep = tracks.clean(_trk(xy, [0] * 12), FPS, pitch=(105.0, 68.0))
    assert rep["cap_reasons"]["home"]["off_pitch"] == n
    assert np.nanmax(out.players[..., 1]) < 68.0


def test_goalkeeper_is_never_frozen():
    n = 120
    xy = np.stack([_walk(n, 20 + 5 * k, 5 + 3 * k, v=3.0) for k in range(12)], axis=1)
    xy[:, 0] = [2.0, 34.0]                                                        # keeper standing in his box
    xy[:, 11] = xy[:, 3] + [6.0, 4.0]                                             # 12th, moving, not next to anyone
    out, rep = tracks.clean(_trk(xy, [0] * 12), FPS, pitch=(105.0, 68.0))
    assert rep["cap_reasons"]["home"]["frozen"] == 0
    assert (np.abs(out.players[:, :, 0] - 2.0) < 1e-6).any(axis=1).all()        # keeper kept on every frame
