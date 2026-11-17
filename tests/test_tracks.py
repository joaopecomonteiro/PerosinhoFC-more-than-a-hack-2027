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


# ---------------------------------------------------------------- v4: speed over a window, team repair
PITCH = (105.0, 68.0)


def _trk_ball(xy, team, ball):
    t = _trk(xy, team)
    return Tracking(t.players, t.player_speed, np.asarray(ball, dtype=float), t.team_of_slot)


def _sway(n, x, y, amp=0.3, period=40.0, phase=0.0):
    """A player standing more or less still: sways, so never 'frozen'."""
    t = np.arange(n)
    return np.stack([x + amp * np.sin(2 * np.pi * t / period + phase), np.full(n, y) + amp * np.cos(2 * np.pi * t / period + phase)], axis=1)


def _teams(n, keeper=True):
    """11 home in the left half, 11 away in the right half, all swaying, keepers at both ends."""
    home = [_sway(n, 20 + 3 * k, 6 + 6 * k, phase=k) for k in range(10)]
    away = [_sway(n, 60 + 3 * k, 6 + 6 * k, phase=k + 0.5) for k in range(10)]
    home.insert(0, _sway(n, 3.0, 34.0) if keeper else _sway(n, 45, 60))
    away.insert(0, _sway(n, 102.0, 34.0))
    return np.stack(home + away, axis=1), [0] * 11 + [1] * 11


def _points_of_team(out, frame, team):
    return out.players[frame, out.team_of_slot == team]


def test_slide_over_several_frames_is_cut_and_hidden():
    n = 150
    a = _walk(n)                                                                  # 2 m/s
    a[60:85, 0] += 1.2 * np.arange(1, 26)                                         # +1.2 m per frame: never one big hop ...
    a[85:, 0] += 30.0                                                             # ... but 30 m in 1 s
    out, rep = tracks.clean(_trk(a[:, None, :], [0]), FPS)
    assert rep["cuts_at_jumps"] == 0 and rep["fast_window_cuts"] >= 1
    assert 0 < rep["fast_points_hidden"] < 40
    kept = ~np.isnan(out.players[..., 0]).all(axis=1)
    assert kept[:50].all() and kept[100:].all()                                   # only the slide (and its edges) goes
    assert np.nanmax(out.player_speed) < 3.0


def test_sprint_at_top_speed_is_not_cut():
    n = 100
    a = _walk(n, v=10.0)
    out, rep = tracks.clean(_trk(a[:, None, :], [0]), FPS)
    assert rep["fast_window_cuts"] == 0 and rep["fast_points_hidden"] == 0 and rep["cuts_at_jumps"] == 0


def test_keeper_picked_up_on_the_wrong_team_takes_the_old_team():
    n = 200
    xy, team = _teams(n)
    xy[100:, 0] = np.nan                                                          # home keeper's track ends ...
    new = np.full((n, 2), np.nan)
    new[110:] = _sway(n - 110, 3.4, 34.2)                                         # ... picked up in an away slot
    xy = np.concatenate([xy, new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH)
    assert rep["team_inherit"]["keeper_box_pickups_changed"] == 1 and rep["tracklets_team_changed"] == 1
    for f in (120, 190):
        assert (_points_of_team(out, f, 0)[:, 0] < 5).sum() == 1                  # the keeper is home again
        assert (_points_of_team(out, f, 1)[:, 0] < 5).sum() == 0
    assert rep["share_frames_over_11"]["after"]["away"] == 0.0


def test_frozen_copy_is_hidden_and_the_pickup_takes_its_team():
    n = 200
    xy, team = _teams(n)
    xy[100:, 0] = xy[99, 0]                                                       # tracker holds the keeper's last position
    new = np.full((n, 2), np.nan)
    new[130:] = _walk(n - 130, x0=4.0, y=34.5, v=1.0)                            # the keeper, picked up as away
    xy = np.concatenate([xy, new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH)
    ti = rep["team_inherit"]
    assert ti["pickups_team_changed"] + ti["keeper_box_pickups_changed"] == 1   # v4g: the box rule (copy ignored)
    assert rep["frozen_copies_hidden_points"] == n - 130
    home_near_goal = (_points_of_team(out, 170, 0)[:, 0] < 10).sum()
    assert home_near_goal == 1                                                    # the live keeper only, no frozen copy
    assert (_points_of_team(out, 170, 1)[:, 0] < 10).sum() == 0


def test_pickup_next_to_a_moving_opponent_keeps_its_team():
    n = 200
    xy, team = _teams(n)
    xy[:, 5] = _walk(n, x0=30.0, y=30.0, v=2.0)                                   # home player running ...
    xy[100:, 5] = np.nan                                                          # ... lost at x = 38
    xy[:, 15] = _walk(n, x0=31.5, y=31.0, v=2.0)                                  # away player alongside him (a duel)
    new = np.full((n, 2), np.nan)
    new[105:] = _walk(n - 105, x0=38.6, y=30.0, v=2.0)                            # someone picked up as away right there
    xy = np.concatenate([xy, new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH)
    assert rep["team_inherit"]["crowded"] >= 1
    assert rep["tracklets_team_changed"] == 0


def test_pickup_with_two_candidates_is_left_alone():
    n = 200
    xy, team = _teams(n)
    xy[:, 5] = _sway(n, 50.0, 34.0)                                               # a home player in midfield ...
    xy[100:, 5] = np.nan
    xy[:, 15] = _sway(n, 53.0, 37.0)                                              # ... an away player next to him ...
    xy[100:, 15] = np.nan                                                         # ... both lost at the same moment
    new = np.full((n, 2), np.nan)
    new[105:] = _sway(n - 105, 51.5, 35.5)
    xy = np.concatenate([xy, new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH)
    assert rep["team_inherit"]["ambiguous"] >= 1 and rep["tracklets_team_changed"] == 0


def test_short_predecessor_does_not_pass_its_team_on():
    n = 200
    xy, team = _teams(n, keeper=False)
    old = np.full((n, 2), np.nan)
    old[60:85] = _sway(25, 50.0, 64.0)                                            # a 1 s home track in midfield
    new = np.full((n, 2), np.nan)
    new[90:] = _sway(n - 90, 50.3, 64.2)
    xy = np.concatenate([xy, old[:, None], new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [0, 1]), FPS, pitch=PITCH)
    assert rep["team_inherit"]["short_predecessor"] >= 1 and rep["tracklets_team_changed"] == 0


def _kickoff_scene(n=3000, ball_on_spot=True):
    """A kickoff at the start of a 2-minute scene: a home player is tracked as away for the first 12 s,
    standing in the home half, then picked up correctly (the rest of the scene is 11 v 11)."""
    xy, team = _teams(n)
    xy[:, 11] = _sway(n, 51.0, 34.0)                                              # away kicker at the ball (inside the circle)
    wrong = np.full((n, 2), np.nan)
    wrong[:300] = _sway(300, 30.0, 20.0)                                          # a home player tracked as away ...
    xy[:300, 1] = np.nan                                                          # ... (his own home track starts later)
    xy[300:, 1] = _sway(n - 300, 30.2, 20.1)
    xy = np.concatenate([xy, wrong[:, None]], axis=1)
    team = team + [1]
    ball = np.full((n, 2), np.nan)
    if ball_on_spot:
        ball[:50] = (52.5, 34.0)                                                  # still on the spot for 2 s ...
        ball[50:] = np.stack([52.5 - 0.4 * np.arange(n - 50), np.full(n - 50, 34.0)], axis=1)   # ... then played
    return xy, team, ball


def test_kickoff_flips_a_track_in_the_wrong_half():
    xy, team, ball = _kickoff_scene()
    out, rep = tracks.clean(_trk_ball(xy, team, ball), FPS, pitch=PITCH)
    assert rep["kickoffs_found"] == 1 and rep["kickoff_team_flips"] == 1
    assert rep["kickoff_flips_detail"][0]["from"] == "away" and rep["kickoff_flips_detail"][0]["to"] == "home"
    home = _points_of_team(out, 70, 0)
    away = _points_of_team(out, 70, 1)
    assert (np.linalg.norm(home - [30.0, 20.0], axis=1) < 1).sum() == 1           # now home
    assert (np.linalg.norm(away - [30.0, 20.0], axis=1) < 1).sum() == 0
    assert (np.linalg.norm(away - [51.0, 34.0], axis=1) < 1).sum() == 1           # the kicker stays away


def test_kickoff_found_without_the_ball_from_a_tagged_window():
    xy, team, ball = _kickoff_scene(ball_on_spot=False)                           # this tracker often loses a still ball
    out, rep = tracks.clean(_trk_ball(xy, team, ball), FPS, pitch=PITCH, kickoff_windows=[(10, 60)])
    assert rep["kickoffs_found"] == 1 and rep["kickoff_team_flips"] == 1
    assert (np.linalg.norm(_points_of_team(out, 70, 0) - [30.0, 20.0], axis=1) < 1).sum() == 1


def test_kickoff_shape_alone_is_not_a_kickoff():
    xy, team, ball = _kickoff_scene(ball_on_spot=False)                           # no ball, no tag: a slow build-up
    out, rep = tracks.clean(_trk_ball(xy, team, ball), FPS, pitch=PITCH)           # against a mid-block looks the same
    assert rep["kickoffs_found"] == 0 and rep["kickoff_team_flips"] == 0


def test_no_kickoff_without_a_kicker_at_the_spot():
    xy, team, ball = _kickoff_scene(ball_on_spot=False)
    xy[:, 11] = _sway(len(xy), 75.0, 40.0)                                        # nobody at the centre spot
    out, rep = tracks.clean(_trk_ball(xy, team, ball), FPS, pitch=PITCH, kickoff_windows=[(10, 60)])
    assert rep["kickoffs_found"] == 0 and rep["kickoff_team_flips"] == 0


def test_no_kickoff_with_the_ball_seen_elsewhere_or_players_running():
    xy, team, ball = _kickoff_scene(ball_on_spot=False)
    ball[:] = (80.0, 20.0)                                                        # the ball is in play elsewhere
    out, rep = tracks.clean(_trk_ball(xy, team, ball), FPS, pitch=PITCH, kickoff_windows=[(10, 60)])
    assert rep["kickoffs_found"] == 0
    xy, team, ball = _kickoff_scene(ball_on_spot=False)
    xy[:, :, 1] += 3.0 * np.arange(len(xy))[:, None] / FPS                        # everybody running at 3 m/s
    out, rep = tracks.clean(_trk_ball(xy, team, ball), FPS, pitch=PITCH, kickoff_windows=[(10, 60)])
    assert rep["kickoffs_found"] == 0


def test_kickoff_windows_from_smart_tagging(tmp_path):
    from mqf import smart_tagging
    xml = """<file><ALL_INSTANCES>
<instance><ID>1</ID><start>10.0</start><end>16.0</end><code>SET PIECES</code>
  <label><group>Type</group><text>Kick Off (Start)</text></label><label><group>Team</group><text>Barcelona</text></label></instance>
<instance><ID>2</ID><start>20.0</start><end>24.0</end><code>SET PIECES</code>
  <label><group>Type</group><text>Throw-in</text></label><label><group>Team</group><text>N/A</text></label></instance>
<instance><ID>3</ID><start>40.0</start><end>45.0</end><code>SET PIECES</code>
  <label><group>Type</group><text>Kick Off (After Goal)</text></label></instance>
</ALL_INSTANCES></file>"""
    p = tmp_path / "tags.xml"
    p.write_text(xml)
    assert smart_tagging.kickoff_windows(p, 25, 2000) == [(250, 400), (1000, 1125)]


def test_player_back_at_his_saved_place_takes_his_team():
    n = 3000                                                                      # long enough for the usual 11 v 11
    xy, team = _teams(n)
    xy[:, 5] = _sway(n, 50.0, 64.0)                                               # a home player out wide ...
    xy[2000:, 5] = np.nan                                                         # ... unseen ...
    new = np.full((n, 2), np.nan)
    new[2225:] = _sway(n - 2225, 50.5, 64.2)                                      # ... back 9 s later, as away
    xy = np.concatenate([xy, new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH)
    assert rep["team_inherit"]["from_saved_place"] == 1 and rep["tracklets_team_changed"] == 1
    assert (np.linalg.norm(_points_of_team(out, 2500, 0) - [50.5, 64.2], axis=1) < 1).sum() == 1
    assert (np.linalg.norm(_points_of_team(out, 2500, 1) - [50.5, 64.2], axis=1) < 1).sum() == 0


def test_track_alone_in_a_box_is_that_boxs_keeper():
    n = 3000                                                                      # long enough for the usual 11 v 11
    xy, team = _teams(n)
    xy[2000:, 11] = np.nan                                                        # away keeper (right goal) lost ...
    new = np.full((n, 2), np.nan)
    new[2080:] = _sway(n - 2080, 101.0, 30.0)                                     # ... a track alone in that box, as home
    xy = np.concatenate([xy, new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [0]), FPS, pitch=PITCH)
    assert rep["team_inherit"]["keeper_box_pickups_changed"] == 1
    assert (_points_of_team(out, 2500, 1)[:, 0] > 95).sum() == 1                  # away again
    assert (_points_of_team(out, 2500, 0)[:, 0] > 95).sum() == 0


def test_box_rule_takes_the_side_not_the_counts():
    """v4g (Sigmar): alone and deepest in a box, staying there: the defending side's keeper, whatever the counts
    (here away has its 11 with the keeper far off his line). v4c-v4f refused it on the counts."""
    n = 3000
    xy, team = _teams(n)
    xy[:, 11] = _sway(n, 80.0, 60.0)                                              # away keeper far off his line (11 away)
    xy[:, 5] = _sway(n, 60.0, 10.0)
    xy[2000:, 5] = np.nan                                                         # a home player lost ...
    new = np.full((n, 2), np.nan)
    new[2080:] = _sway(n - 2080, 100.0, 36.0)                                     # ... appears alone, deepest, in the away box
    xy = np.concatenate([xy, new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [0]), FPS, pitch=PITCH)
    assert rep["team_inherit"]["keeper_box_pickups_changed"] == 1
    _, rep = tracks.clean(_trk(xy, team + [0]), FPS, pitch=PITCH, cfg=tracks.CleanConfig(keeper_box_count_check=True))
    assert rep["team_inherit"]["keeper_box_counts_disagree"] == 1
    assert rep["team_inherit"]["keeper_box_pickups_changed"] == 0 and rep["tracklets_team_changed"] == 0


def test_kickoff_does_not_flip_an_official_into_a_full_team():
    xy, team, ball = _kickoff_scene()
    xy[:, 1] = _sway(len(xy), 25.0, 60.0)                                         # home has its 11 (the "wrong" track too
                                                                                  # is then an extra, not a home player) ...
    ref = _sway(len(xy), 47.0, 50.0)                                              # ... and an away-labelled official
    xy = np.concatenate([xy, ref[:, None]], axis=1)                               #     stands in the home half
    out, rep = tracks.clean(_trk_ball(xy, team + [1], ball), FPS, pitch=PITCH)
    assert rep["kickoffs_found"] == 1 and rep["kickoff_team_flips"] == 0


def test_box_track_next_to_someone_or_not_deepest_is_not_the_keeper():
    n = 300
    xy, team = _teams(n)
    xy[:, 5] = _sway(n, 92.0, 30.0)                                               # a home attacker in the away box
    new = np.full((n, 2), np.nan)
    new[180:] = _sway(n - 180, 93.5, 31.0)                                        # someone appears 1.8 m from him, as home
    xy = np.concatenate([xy, new[:, None]], axis=1)
    _, rep = tracks.clean(_trk(xy, team + [0]), FPS, pitch=PITCH)
    assert rep["team_inherit"]["keeper_box_pickups_changed"] == 0
    xy, team = _teams(n)                                                          # the away keeper is still there (deeper)
    new = np.full((n, 2), np.nan)
    new[180:] = _sway(n - 180, 95.0, 20.0)
    xy = np.concatenate([xy, new[:, None]], axis=1)
    _, rep = tracks.clean(_trk(xy, team + [0]), FPS, pitch=PITCH)
    assert rep["team_inherit"]["keeper_box_pickups_changed"] == 0


def test_team_short_of_11_gets_its_keeper_at_his_saved_place():
    n = 400
    xy, team = _teams(n)
    xy[200:, 0] = np.nan                                                          # home keeper unseen, nobody in his box
    out, rep = tracks.clean(_trk(xy, team), FPS, pitch=PITCH)
    assert rep["keeper_fill_points"]["home"] == 200 and rep["keeper_fill_points"]["away"] == 0
    home = _points_of_team(out, 300, 0)
    assert (~np.isnan(home[:, 0])).sum() == 11                                    # 10 tracked + the keeper
    np.testing.assert_allclose(home[np.nanargmin(home[:, 0])], xy[199, 0], atol=0.01)   # at his last place


def test_keeper_fill_uses_the_6_yard_line_before_he_is_seen():
    n = 300
    xy, team = _teams(n)
    xy[:, 0] = np.nan                                                             # home keeper never tracked
    out, rep = tracks.clean(_trk(xy, team), FPS, pitch=PITCH)
    assert rep["keeper_fill_points"]["home"] == n
    home = _points_of_team(out, 100, 0)
    np.testing.assert_allclose(home[np.nanargmin(home[:, 0])], [5.5, 34.0], atol=1e-4)


def test_no_keeper_fill_when_the_team_has_11():
    n = 300
    xy, team = _teams(n)
    xy[:, 0] = _sway(n, 30.0, 50.0)                                               # keeper out of his box, 11 tracked
    out, rep = tracks.clean(_trk(xy, team), FPS, pitch=PITCH)
    assert rep["keeper_fill_points"]["home"] == 0


def test_held_copy_shows_again_when_its_pickup_ends():
    n = 450
    xy, team = _teams(n)
    xy[100:, 0] = xy[99, 0]                                                       # the tracker holds the keeper's place
    flick = np.full((n, 2), np.nan)
    flick[130:140] = xy[99, 0] + [0.8, 0.0]                                       # a 0.4 s flicker next to it
    new = np.full((n, 2), np.nan)
    new[300:] = _sway(n - 300, 4.0, 34.5)                                         # the real pickup, as away
    xy = np.concatenate([xy, flick[:, None], new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [0, 1]), FPS, pitch=PITCH)
    assert (_points_of_team(out, 200, 0)[:, 0] < 6).sum() == 1                    # the saved place is shown again
    assert (_points_of_team(out, 380, 0)[:, 0] < 6).sum() == 1                    # then the keeper, home, alone
    assert (_points_of_team(out, 380, 1)[:, 0] < 6).sum() == 0
    ti = rep["team_inherit"]
    assert ti["pickups_team_changed"] + ti["keeper_box_pickups_changed"] == 1


def test_team_repair_can_be_switched_off():
    n = 200
    xy, team = _teams(n)
    xy[100:, 0] = np.nan
    new = np.full((n, 2), np.nan)
    new[110:] = _sway(n - 110, 3.4, 34.2)
    xy = np.concatenate([xy, new[:, None]], axis=1)
    cfg = tracks.CleanConfig(inherit_team=False, kickoff_check=False, fast_windows_s=())
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH, cfg=cfg)
    assert rep["tracklets_team_changed"] == 0 and rep["team_inherit"] is None


def test_pickup_keeps_its_team_when_the_counts_are_already_even():
    n = 600
    xy, team = _teams(n)
    mis = np.full((n, 2), np.nan)
    mis[:100] = _sway(100, 70.0, 50.0)                                            # an away player tracked as home ...
    xy[:, 20] = np.nan                                                            # (his own away slot unused)
    new = np.full((n, 2), np.nan)
    new[105:] = _sway(n - 105, 70.4, 50.2)                                        # ... picked up correctly as away
    xy = np.concatenate([xy, mis[:, None], new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [0, 1]), FPS, pitch=PITCH)
    assert rep["team_inherit"]["counts_disagree"] == 1                            # 11 v 11 already: no change
    assert rep["tracklets_team_changed"] == 0


def test_cap_can_hide_a_held_copy_before_the_live_keeper():
    n = 300
    xy, team = _teams(n)
    xy[:, 0] = xy[0, 0]                                                           # a held copy at the keeper's place (exact)
    live = _sway(n, 6.0, 30.0)                                                    # the keeper, live, in a newer track
    live[:100] = np.nan
    xy = np.concatenate([xy, live[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [0]), FPS, pitch=PITCH)              # v4g: the box rule hides the copy
    assert rep["team_inherit"]["keeper_box_still_hidden"] == 1 and rep["cap_reasons"]["home"]["youngest"] == 0
    assert (np.linalg.norm(_points_of_team(out, 250, 0) - [6.0, 30.0], axis=1) < 1).sum() == 1
    old = tracks.CleanConfig(keeper_box_ignore_still=False)                       # v4f: the cap decides
    _, rep = tracks.clean(_trk(xy, team + [0]), FPS, pitch=PITCH, cfg=old)
    assert rep["cap_reasons"]["home"]["youngest"] > 0                             # ... and the live keeper goes
    out, rep = tracks.clean(_trk(xy, team + [0]), FPS, pitch=PITCH,
                            cfg=tracks.CleanConfig(keeper_box_ignore_still=False, cap_held_first=True))
    assert rep["cap_reasons"]["home"]["held"] > 0 and rep["cap_reasons"]["home"]["youngest"] == 0
    home = _points_of_team(out, 250, 0)
    assert (np.linalg.norm(home - [6.0, 30.0], axis=1) < 1).sum() == 1           # the live keeper stays


def test_trackdiag_on_a_toy_bundle(tmp_path, monkeypatch):
    import json
    from mqf import trackdiag
    from mqf.external import bundle
    n = 400
    xy, team = _teams(n)
    xy[200:, 0] = xy[199, 0]
    m = {"name": "TOY-2", "source": "toy", "fps": FPS, "pitch": PITCH, "players": xy,
         "ball": np.full((n, 2), np.nan), "team_of_player": np.array(team, dtype=np.int8),
         "labels": np.array(["X"] * 10 + ["H"] * 380 + ["X"] * 10), "halves": ((1, 10, 389),),
         "home": "Home", "away": "Away", "labels_from": "toy", "licence": ""}
    bundle.write_bundle(m, tmp_path)
    monkeypatch.setattr(bundle, "BUNDLES_DIR", tmp_path)
    out = tmp_path / "diag.json"
    assert trackdiag.main(["--bundle", "TOY-2", "--json", str(out)]) == 0
    d = json.loads(out.read_text())
    assert set(d["variants"]) == set(trackdiag.VARIANTS)
    assert d["delivered_tracking"]["exact_repeat_runs"]["runs"] >= 1                # the held copy is found


def test_kickoff_flip_refused_when_the_counts_do_not_agree():
    xy, team, ball = _kickoff_scene()                                             # the away-labelled track in the home half
    xy[60:300, 1] = _sway(240, 25.0, 60.0)                                        # is an extra: home is short only during the
    out, rep = tracks.clean(_trk_ball(xy, team, ball), FPS, pitch=PITCH)           # wait, 11 for the rest of its life
    assert rep["kickoffs_found"] == 1
    assert rep["team_inherit"]["kickoff_counts_disagree"] == 1
    assert rep["kickoff_team_flips"] == 0 and rep["kickoff_flips_detail"] == []


# ---------------------------------------------------------------- v4d: keeper takeover (Sigmar's rule)
def _held_keeper_scene(pickup_at, n=3000, lost_at=2000, hold=78):
    """The home keeper is lost at `lost_at`; the tracker holds his last place for ~3.1 s (78 frames,
    as delivered on BAR-ATH) with an away attacker 2.5 m from him; the keeper is picked up again at
    `pickup_at` in an AWAY slot, right next to his held copy."""
    xy, team = _teams(n)
    xy[:, 15] = _sway(n, 5.5, 35.5)                                                # away attacker on the keeper
    xy[lost_at:lost_at + hold, 0] = xy[lost_at - 1, 0]                            # held copy (exact repeat) ...
    xy[lost_at + hold:, 0] = np.nan                                               # ... then dropped
    new = np.full((n, 2), np.nan)
    new[pickup_at:] = _sway(n - pickup_at, 3.8, 33.3)                             # the keeper, as away
    return np.concatenate([xy, new[:, None]], axis=1), team + [1]


def _near(out, frame, team, p, r=1.0):
    return int((np.linalg.norm(_points_of_team(out, frame, team) - p, axis=1) < r).sum())


def test_new_track_next_to_the_other_teams_held_keeper_becomes_that_keeper():
    for pickup_at in (2005, 2040):                                                # before the copy counts as held / crowded
        xy, team = _held_keeper_scene(pickup_at)
        out, rep, flags = tracks.clean_with_flags(_trk(xy, team), FPS, pitch=PITCH)
        ti = rep["team_inherit"]
        assert ti["keeper_takeover_held"] == 1 and rep["tracklets_keeper_takeover"] == 1, ti
        assert ti["keeper_box_pickups_changed"] == 0 and ti["pickups_team_changed"] == 0   # the older rules skip it
        for f in (2060, 2500):
            p = xy[f, -1]
            assert _near(out, f, 0, p) == 1 and _near(out, f, 1, p) == 0          # the keeper is home ...
            assert _near(out, f, 0, [3.0, 34.0], r=2.5) == 1                      # ... alone: no held copy
            assert _near(out, f, 1, [5.5, 35.5]) == 1                             # the attacker is untouched
        col = np.flatnonzero(np.linalg.norm(out.players[2500] - xy[2500, -1], axis=1) < 1e-3)
        assert flags[2500, col[0]] == tracks.FLAG_TAKEOVER
        assert rep["keeper_fill_points"]["home"] == 0                             # no saved keeper on top of him


def test_new_track_at_the_other_teams_saved_keeper_place_becomes_that_keeper():
    n = 3000
    xy, team = _teams(n)
    xy[2000:, 0] = np.nan                                                         # home keeper unseen (no held copy)
    xy[:, 15] = _sway(n, 6.0, 34.0)                                               # an away attacker near his place
    new = np.full((n, 2), np.nan)
    new[2060:] = _sway(n - 2060, 4.5, 32.5)                                       # 2.4 s later, 2.1 m away, as away
    xy = np.concatenate([xy, new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH)
    ti = rep["team_inherit"]
    assert ti["keeper_takeover_saved"] == 1 and rep["tracklets_keeper_takeover"] == 1, ti
    assert _near(out, 2500, 0, xy[2500, -1]) == 1 and _near(out, 2500, 1, xy[2500, -1]) == 0
    assert _near(out, 2500, 1, [6.0, 34.0]) == 1


def test_attacker_picked_up_next_to_a_live_keeper_keeps_his_team():
    n = 3000
    xy, team = _teams(n)
    xy[:, 15] = _sway(n, 20.0, 50.0)
    xy[1900:, 15] = np.nan                                                        # an away attacker lost far away ...
    new = np.full((n, 2), np.nan)
    new[2010:] = _sway(n - 2010, 4.5, 35.0)                                       # ... back 1.8 m from the live keeper
    xy = np.concatenate([xy, new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH)
    assert rep["team_inherit"]["keeper_takeover_counts_disagree"] >= 1            # 11 v 11: a real attacker
    assert rep["tracklets_team_changed"] == 0 and rep["tracklets_keeper_takeover"] == 0
    assert _near(out, 2500, 0, [3.0, 34.0]) == 1                                  # keeper still there, home
    assert _near(out, 2500, 1, [4.5, 35.0]) == 1                                  # attacker still there, away
    strict = tracks.CleanConfig(keeper_live_test="held_or_duplicate")             # v4d refused it on geometry
    _, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH, cfg=strict)
    assert rep["team_inherit"]["keeper_takeover_apart"] >= 1 and rep["tracklets_team_changed"] == 0


def test_new_track_next_to_a_live_keeper_becomes_the_keeper():
    """v4e: a second track of the other colour appears 1.6 m from the live keeper and stays with him
    (a duplicate the tracker put on the other team) -- it is the keeper; his old track is hidden while
    the two stay within 3 m. v4d refused it (not within 1 m)."""
    n = 3000
    xy, team = _teams(n)
    new = np.full((n, 2), np.nan)
    new[2040:] = xy[2040:, 0] + [1.2, -1.0]                                       # 1.6 m off the keeper, moving with him
    xy = np.concatenate([xy, new[:, None]], axis=1)                               # (away 12 while it lives)
    out, rep, flags = tracks.clean_with_flags(_trk(xy, team + [1]), FPS, pitch=PITCH)
    ti = rep["team_inherit"]
    assert ti["keeper_takeover_near"] == 1 and rep["tracklets_keeper_takeover"] == 1, ti
    for f in (2100, 2900):
        assert _near(out, f, 0, [3.0, 34.0], r=3.0) == 1                          # one home keeper ...
        assert _near(out, f, 0, xy[f, -1], r=0.1) == 1                            # ... drawn at the new track
        assert _near(out, f, 1, xy[f, -1], r=1.0) == 0                            # no away track on him
    strict = tracks.CleanConfig(keeper_live_test="held_or_duplicate")
    _, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH, cfg=strict)
    assert rep["tracklets_keeper_takeover"] == 0


def test_old_keeper_track_shows_again_when_the_new_one_moves_away():
    n = 3000
    xy, team = _teams(n)
    new = np.full((n, 2), np.nan)
    new[2040:2200] = xy[2040:2200, 0] + [1.2, -1.0]                               # next to the keeper ...
    new[2200:] = np.linspace(xy[2199, 0] + [1.2, -1.0], [30.0, 20.0], n - 2200)   # ... then walks off (< 10 m/s)
    xy = np.concatenate([xy, new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH)
    if rep["tracklets_keeper_takeover"]:
        assert _near(out, 2900, 0, [3.0, 34.0], r=1.0) == 1                       # the keeper's track is back


def test_attacker_at_the_saved_keeper_place_keeps_his_team_when_the_counts_say_so():
    n = 3000
    xy, team = _teams(n)
    xy[2000:, 0] = np.nan                                                         # home keeper unseen ...
    xy[:, 15] = _sway(n, 20.0, 50.0)
    xy[1900:, 15] = np.nan                                                        # ... an away attacker lost far away
    new = np.full((n, 2), np.nan)
    new[2060:] = np.linspace([4.5, 32.5], [40.0, 30.0], n - 2060)                 # ... picked up at the keeper's place,
    xy = np.concatenate([xy, new[:, None]], axis=1)                               # then out of the box (away 11, home
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH)               # 10: +1, not +2; not keeper-like)
    assert rep["team_inherit"]["keeper_takeover_counts_disagree"] == 1
    assert rep["tracklets_team_changed"] == 0


def test_keeper_takeover_can_be_switched_off():
    xy, team = _held_keeper_scene(2040)
    out, rep = tracks.clean(_trk(xy, team), FPS, pitch=PITCH, cfg=tracks.CleanConfig(keeper_takeover=False))
    assert rep["tracklets_keeper_takeover"] == 0 and rep["tracklets_team_changed"] == 0
    assert _near(out, 2500, 0, xy[2500, -1]) == 0                                 # the v4c result: never home (away,
                                                                                  # and here capped as the 12th)


def test_flags_mark_repairs_and_the_filled_keeper():
    n = 3000
    xy, team = _teams(n)
    xy[200:, 0] = np.nan                                                          # home keeper unseen, nobody in his box
    out, rep, flags = tracks.clean_with_flags(_trk(xy[:400], team), FPS, pitch=PITCH)
    assert flags.shape == out.players.shape[:2]
    filled = flags == tracks.FLAG_KEEPER_FILL
    assert filled.sum() == rep["keeper_fill_points"]["home"] > 0
    assert np.isnan(out.players[..., 0][flags > 0]).sum() == 0                    # flags only on drawn points
    assert ((flags == tracks.FLAG_NONE) | filled).all()                           # nothing else was changed


def test_trackdiag_explain_on_a_toy_bundle(tmp_path, monkeypatch, capsys):
    """The held-keeper scene as a bundle: --explain names the intruder and the rule's decision."""
    import json
    from mqf import trackdiag
    from mqf.external import bundle
    xy, team = _held_keeper_scene(2040)
    n = len(xy)
    m = {"name": "TOY-3", "source": "toy", "fps": FPS, "pitch": PITCH, "players": xy,
         "ball": np.full((n, 2), np.nan), "team_of_player": np.array(team, dtype=np.int8),
         "labels": np.array(["X"] * 10 + ["H"] * (n - 20) + ["X"] * 10), "halves": ((1, 10, n - 11),),
         "home": "Home", "away": "Away", "labels_from": "toy", "licence": ""}
    bundle.write_bundle(m, tmp_path)
    monkeypatch.setattr(bundle, "BUNDLES_DIR", tmp_path)
    out = tmp_path / "explain.json"
    assert trackdiag.main(["--bundle", "TOY-3", "--explain", "1:01:22", "--span", "5", "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    d = json.loads(out.read_text())
    orig, v4c, v4d = (d["table"][k] for k in ("Original (as delivered)", "v4c (no takeover)", "v4g (default)"))
    assert float(orig[0].split()[0]) > 0                                          # delivered: away track on the home keeper
    assert float(v4d[0].split()[0]) == 0.0                                        # v4d: gone
    assert all(it["shown_as"] != "Away" or "takeover" not in " ".join(it["decisions"]) for it in d["intruders"])
    import re
    body = "\n".join(l for l in printed.splitlines() if not l.startswith("wrote "))
    assert not re.search(r"\d{4,}", body)                                        # no frame ids or coordinates


# ---------------------------------------------------------------- v4f: keeper behaviour when counts are ambiguous
def _ambiguous_counts(xy, team):
    """An away player unseen from frame 1900 on: the wrong-team keeper no longer makes away one too many."""
    xy = xy.copy()
    xy[1900:, 20] = np.nan
    return xy, team


def test_counts_ambiguous_but_the_track_behaves_like_the_keeper():
    xy, team = _held_keeper_scene(2040)
    xy, team = _ambiguous_counts(xy, team)
    out, rep = tracks.clean(_trk(xy, team), FPS, pitch=PITCH)
    ti = rep["team_inherit"]
    assert ti["keeper_takeover_by_behaviour"] == 1 and rep["tracklets_keeper_takeover"] == 1, ti
    assert _near(out, 2500, 0, xy[2500, -1]) == 1 and _near(out, 2500, 1, xy[2500, -1]) == 0
    _, rep = tracks.clean(_trk(xy, team), FPS, pitch=PITCH, cfg=tracks.CleanConfig(keeper_behaviour=False))
    assert rep["tracklets_keeper_takeover"] == 0                                 # v4e: the counts alone refused


def test_attacker_who_touches_the_held_keeper_and_leaves_keeps_his_team():
    xy, team = _held_keeper_scene(2040)
    xy, team = _ambiguous_counts(xy, team)
    n = len(xy)
    xy[2040:, -1] = np.linspace([3.8, 33.3], [60.0, 40.0], n - 2040)              # runs out of the box at ~8 m/s
    out, rep = tracks.clean(_trk(xy, team), FPS, pitch=PITCH)
    assert rep["tracklets_keeper_takeover"] == 0 and rep["tracklets_team_changed"] == 0
    assert any("does not behave like the keeper" in m for m in
               [m for trail in tracks_trace(xy, team).values() for m in trail])


def tracks_trace(xy, team):
    tr = {}
    tracks.clean_with_flags(_trk(xy, team), FPS, pitch=PITCH, trace=tr)
    return tr["why"]


def test_clean_cfg_parses_text_fields():
    from mqf.external.evaluate import parse_clean_cfg
    cfg = parse_clean_cfg(["keeper_live_test=held_or_duplicate", "keeper_behaviour=0"])
    assert cfg.keeper_live_test == "held_or_duplicate" and cfg.keeper_behaviour is False



# ---------------------------------------------------------------- v4g: sides from the half start, still tracks
def test_box_rule_uses_the_sides_counted_at_the_half_start():
    """Mid-half the home team pushes into the other half for minutes, so the teams' medians swap sides; the
    keeper picked up as away in the HOME box must still go to home (sides counted at the half start)."""
    n = 6000
    xy, team = _teams(n)
    for k in range(1, 11):                                                        # from 60 s on: home outfield deep
        xy[1500:, k] = _sway(n - 1500, 70 + 2 * k, 6 + 6 * (k - 1), phase=k)      # in the away half, away dropped
    for k in range(12, 22):                                                       # back into the home half
        xy[1500:, k] = _sway(n - 1500, 25 + 2 * (k - 12), 6 + 6 * (k - 12), phase=k + 0.5)
    xy[3000:, 0] = np.nan                                                         # the home keeper lost ...
    new = np.full((n, 2), np.nan)
    new[3100:] = _sway(n - 3100, 3.5, 33.0)                                       # ... back alone in his box, as away
    xy = np.concatenate([xy, new[:, None]], axis=1)
    halves = ((1, 0, n - 1),)
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH, halves=halves)
    assert rep["home_defends_left_by_half"] == [True]
    assert _near(out, 4000, 0, xy[4000, -1]) == 1 and _near(out, 4000, 1, xy[4000, -1]) == 0
    _, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH, halves=halves,
                          cfg=tracks.CleanConfig(side_from_half_start=False))
    assert rep["team_inherit"]["keeper_box_pickups_changed"] == 0                # medians: the wrong side


def test_new_track_next_to_a_still_track_in_the_box_is_the_keeper_and_the_still_one_goes():
    """Sigmar's rule: in the box with nobody moving near it (only a track that does not move): the keeper of the
    side defending the box; the still track is hidden while it stays still."""
    n = 3000
    xy, team = _teams(n)
    xy[:, 11] = np.nan                                                            # away keeper not tracked as away ...
    ghost = np.full((n, 2), np.nan)
    ghost[1500:] = [104.0, 34.0]                                                  # ... a track that never moves on the line
    new = np.full((n, 2), np.nan)
    new[2000:] = _sway(n - 2000, 102.0, 33.0)                                     # the away keeper, picked up as home
    xy = np.concatenate([xy, ghost[:, None], new[:, None]], axis=1)
    out, rep, flags = tracks.clean_with_flags(_trk(xy, team + [0, 0]), FPS, pitch=PITCH, halves=((1, 0, n - 1),))
    ti = rep["team_inherit"]
    assert ti["keeper_box_still_hidden"] == 1, ti                                 # (the still track, new at its own
    assert _near(out, 2500, 1, xy[2500, -1]) == 1 and _near(out, 2500, 0, xy[2500, -1]) == 0   # start, was a box pickup
    assert _near(out, 2500, 0, [104.0, 34.0], r=0.5) == 0 and _near(out, 2500, 1, [104.0, 34.0], r=0.5) == 0   # too)



def test_box_rule_hides_the_keepers_held_copy_a_few_metres_away():
    """v4g: when the box rule decides, the keeper's held copy up to inherit_reach_m + jump_tol_m away is
    hidden too (the pickup rule used to do it; #33: held points hidden fell 1517 -> 490 on BAR-ATH)."""
    n = 3000
    xy, team = _teams(n)
    xy[2000:2078, 0] = xy[1999, 0]                                                # held copy at the keeper's last place
    xy[2078:, 0] = np.nan
    new = np.full((n, 2), np.nan)
    new[2040:] = _sway(n - 2040, 3.0 + 4.5, 34.0)                                 # the keeper picked up 4.5 m away, as away
    xy = np.concatenate([xy, new[:, None]], axis=1)
    out, rep = tracks.clean(_trk(xy, team + [1]), FPS, pitch=PITCH, halves=((1, 0, n - 1),))
    ti = rep["team_inherit"]
    assert ti["keeper_box_pickups_changed"] == 1 and ti["keeper_box_still_hidden"] == 1, ti
    assert _near(out, 2060, 0, xy[1999, 0], r=0.5) == 0                           # held copy hidden
    assert _near(out, 2060, 0, xy[2060, -1]) == 1                                 # the keeper, home
