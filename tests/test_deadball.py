"""mqf.deadball: ball inputs, the semi-Markov decoder (exact), the scorer, the ball-loss condition."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from mqf.deadball import decoder, evaluate, features, model

PITCH = (105.0, 68.0)
FPS = 25


# ------------------------------------------------------------------------------------ features

def test_restart_spot():
    x = np.array([52.5, 52.5, 30.0, 0.5, 3.0, 50.0, 52.5, np.nan])
    y = np.array([1.0, 34.0, 34.0, 20.0, 34.0, 34.0, 67.5, 10.0])
    #            touch centre open  goal-l goal-area open touch missing
    assert features.restart_spot(x, y, PITCH).tolist() == [True, True, False, True, True, False, True, False]


def test_ball_gap_inputs():
    n = 200
    b = np.tile([40.0, 30.0], (n, 1))
    b[49] = [40.0, 0.5]                     # last seen on the touchline ...
    b[50:100] = np.nan                      # ... then a 2 s gap
    f = features.ball(b, FPS, PITCH)
    assert f["b_gap_s"][50] == 2.0 and f["b_gap_s"][100] == 0.0
    assert f["b_since_s"][50] == pytest.approx(0.04) and f["b_until_s"][99] == pytest.approx(0.04)
    assert f["b_gap_start_spot"][50:100].all() and not f["b_gap_end_spot"][50:100].any()
    assert f["b_seen"][:50].all() and not f["b_seen"][50:100].any()


# ------------------------------------------------------------------------------------ decoder

def _segments(seq):
    out, s = [], 0
    for t in range(1, len(seq) + 1):
        if t == len(seq) or seq[t] != seq[s]:
            out.append((s, t, seq[s]))
            s = t
    return out


def _path_score(seq, ll, logp, logs):
    dmax = len(logp[0]) - 1
    segs = _segments(list(seq))
    v = 0.0
    for i, (a, b, s) in enumerate(segs):
        d = b - a
        v += ll[s, a:b].sum()
        if i == 0:
            v += logs[s][min(d, dmax)]
        elif i == len(segs) - 1:
            v += logs[s][d] if d <= dmax else -np.inf
        else:
            v += logp[s][d] if d <= dmax else -np.inf
    return v


@pytest.mark.parametrize("seed", range(30))
def test_viterbi_is_exact(seed):
    rng = np.random.default_rng(seed)
    m, dmax = 8, 5
    ll = rng.normal(0, 1.5, size=(2, m))
    logp, logs = [], []
    for _ in range(2):
        pmf = rng.dirichlet(np.ones(dmax))
        logp.append(np.r_[-np.inf, np.log(pmf)])
        logs.append(np.r_[0.0, np.log(np.cumsum(pmf[::-1])[::-1])])
    best = max(_path_score(s, ll, logp, logs) for s in itertools.product((0, 1), repeat=m))
    assert _path_score(decoder.viterbi(ll, logp, logs), ll, logp, logs) == pytest.approx(best)


def _tables():
    rng = np.random.default_rng(0)
    return (decoder.length_tables(rng.lognormal(np.log(40), 0.8, 2000)),
            decoder.length_tables(rng.lognormal(np.log(15), 0.6, 2000)))


def test_blip_absorbed_stoppage_whole_and_half_opening_mid_stoppage():
    n = FPS * 400
    p = np.full(n, 0.1)
    p[FPS * 10:FPS * 130] = 0.9                   # 2-minute stoppage right at the half's start
    p[FPS * 200:FPS * 200 + 10] = 0.9             # 0.4 s blip in open play
    p[FPS * 300:FPS * 320] = 0.85                 # 20 s stoppage ...
    p[FPS * 308:FPS * 310] = 0.3                  # ... with a doubtful 2 s hole
    dead = decoder.decode(p, ((1, FPS * 10, n - 1),), FPS, _tables(), w=0.2, prior=0.45)
    assert not dead[:FPS * 10].any() and dead[FPS * 10:FPS * 128].all()
    assert not dead[FPS * 195:FPS * 205].any()
    assert dead[FPS * 301:FPS * 319].all()
    assert len([s for s in _segments(list(dead[FPS * 140:].astype(int))) if s[2] == 1]) == 1


# ------------------------------------------------------------------------------------ scorer

def _gt():
    return np.array(["X"] * 100 + ["H"] * 500 + ["D"] * 250 + ["A"] * 600 + ["D"] * 50 + ["H"] * 400 + ["X"] * 100)


def test_perfect_scores_one():
    gt = _gt()
    r = evaluate.score(gt, gt == "D", FPS)
    for k in ("alive_dead_f1", "stopF1", "dead_precision", "dead_recall", "start_f1", "end_f1"):
        assert r[k] == pytest.approx(1.0), k
    assert r["stops_true"] == r["stops_pred"] == 2 and r["start_err_s"] == 0.0


def test_missed_and_late():
    gt = _gt()
    pred = gt == "D"
    pred[1450:1500] = False                       # the 2 s stoppage missed
    pred[600:625] = False                         # the 10 s one found 1 s late
    r = evaluate.score(gt, pred, FPS)
    assert r["stops_pred"] == 1 and r["found_0-5s"] == 0.0 and r["found_5-15s"] == 1.0
    assert r["start_err_s"] == pytest.approx(1.0) and 0 < r["stopF1"] < 1


def test_outside_the_graded_window_costs_nothing():
    gt = _gt()
    pred = gt == "D"
    pred[:100] = True
    assert evaluate.score(gt, pred, FPS)["alive_dead_f1"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------ condition

def test_hide_stoppage_ball():
    gt = _gt()
    b = np.ones((len(gt), 2))
    whole = model.hide_stoppage_ball(b, gt, 1.0, np.random.default_rng(0))
    assert np.isnan(whole[gt == "D"]).all() and not np.isnan(whole[gt != "D"]).any()
    part = model.hide_stoppage_ball(b, gt, 0.0, np.random.default_rng(0))
    assert not np.isnan(part[gt != "D"]).any() and 0 < np.isnan(part[gt == "D", 0]).sum() < (gt == "D").sum()


def test_run_lengths_drop_runs_cut_by_the_edges():
    gt = np.array(["D"] * 25 + ["H"] * 50 + ["D"] * 75 + ["A"] * 100 + ["D"] * 10)
    r = model.run_lengths(gt, ((1, 0, len(gt) - 1),), FPS)
    assert r["dead"] == [3.0] and sorted(r["alive"]) == [2.0, 4.0]


# ------------------------------------------------------------------------------------ by type

def test_restart_type_by_location():
    L, W = PITCH
    assert evaluate.restart_type(L / 2, W / 2, PITCH) == "kick-off"
    assert evaluate.restart_type(0.5, 0.5, PITCH) == "corner"
    assert evaluate.restart_type(5.5, W / 2 + 5, PITCH) == "goal kick"
    assert evaluate.restart_type(40.0, 0.3, PITCH) == "throw-in"
    assert evaluate.restart_type(70.0, 20.0, PITCH) == "free kick"
    assert evaluate.restart_type(np.nan, 20.0, PITCH) == "unknown"


def test_by_type_found_late_and_covered():
    gt = _gt()                                    # stoppages [600, 850) and [1450, 1500)
    ball = np.tile([70.0, 20.0], (len(gt), 1))    # restarts in open play -> free kicks
    ball[1500:] = [40.0, 0.3]                     # the second restarts on the touchline -> throw-in
    stops = evaluate.stoppage_types(gt, ball, PITCH, FPS)
    assert [t for _, _, t in stops] == ["free kick", "throw-in"]
    pred = gt == "D"
    pred[600:650] = False                         # the free kick found 2 s late
    pred[1450:1500] = False                       # the throw-in missed
    r = evaluate.by_type(stops, pred, FPS)
    assert r[0]["found"] and r[0]["late_s"] == pytest.approx(2.0) and r[0]["covered"] == pytest.approx(0.8)
    assert not r[1]["found"] and np.isnan(r[1]["late_s"]) and r[1]["covered"] == 0.0


def test_drift_ball_loss_starts_before_the_stoppage_and_adds_live_gaps():
    gt = np.array(["H"] * 3000 + ["D"] * 500 + ["A"] * 6000)
    b = np.ones((len(gt), 2))
    out = model.drift_ball_loss(b, gt, FPS, np.random.default_rng(3), p_whole=1.0, lead_p=1.0, live_per_min=4.0,
                                lag_p=0.0)
    assert np.isnan(out[3000:3500, 0]).all()                       # the whole stoppage
    assert np.isnan(out[2990:3000, 0]).all()                       # ... and the gap began in the play before it
    assert np.isnan(out[3600:, 0]).any()                           # long gaps in open play too
    none = model.drift_ball_loss(b, gt, FPS, np.random.default_rng(3), p_whole=1.0, lead_p=0.0, live_per_min=0.0,
                                 lag_p=0.0)
    assert not np.isnan(none[gt != "D", 0]).any()                  # all off: exactly the stoppage


# ------------------------------------------------------------------------------------ v3 inputs

def test_state_inputs_count_the_stopped_state():
    n = FPS * 60
    speed = np.full(n, 3.0)
    speed[FPS * 20:FPS * 40] = 0.8                 # everyone slows at 20 s, speeds up again at 40 s
    slow = np.where(speed < 1, 0.6, 0.1)
    seen = np.ones(n, bool)
    seen[FPS * 21:FPS * 40] = False                # the ball is lost during the stoppage
    s = features.state(speed, slow, seen, FPS)
    assert s["s_since_brake"][FPS * 30] == pytest.approx(10.0, abs=1.1)
    assert s["s_until_go"][FPS * 30] == pytest.approx(10.0, abs=1.1)
    assert s["s_speed_rel"][FPS * 30] < 0.5 < s["s_speed_rel"][FPS * 10]
    assert s["s_slow_nob_s"][FPS * 35] == pytest.approx(14.0, abs=0.1) and s["s_slow_nob_s"][FPS * 45] == 0.0


def test_setpiece_inputs_nan_without_a_ball():
    n = FPS * 10
    team = np.array([0] * 3 + [1] * 3)
    players = np.zeros((n, 6, 2))
    players[:, :, 1] = 30.0
    players[:, :, 0] = [50.5, 55.0, 60.0, 59.8, 60.0, 61.0]   # home taker at the ball; away kept 9.8-11 m away
    b = np.tile([50.0, 30.0], (n, 1))
    b[FPS * 8:] = np.nan
    f = features.setpiece(players, team, b, FPS)
    i = FPS * 5
    assert f["fk_still_s"][i] > 4.0 and f["fk_near"][i] == pytest.approx(0.5)
    assert f["fk_n9_min"][i] == 0 and f["fk_n9_max"][i] == 2 and f["fk_ring"][i] == 3   # 10.0, 9.8, 10.0 m
    assert np.isnan(f["fk_near"][FPS * 9]) and np.isnan(f["fk_n9_min"][FPS * 9])


# ------------------------------------------------------------------------------------ v4 inputs

def test_out_memory_holds_until_a_restart_kick():
    n = FPS * 40
    b = np.tile([30.0, 30.0], (n, 1)).astype(float)
    b[FPS * 5:FPS * 8] = [30.0, -2.0]              # ball seen off the pitch (beyond the touchline)
    b[FPS * 8:FPS * 20] = [30.0, 1.0]              # brought back on, held at the line
    t = np.arange(n) / FPS
    b[FPS * 20:, 0] = 30.0 + 12.0 * (t[FPS * 20:] - 20.0)    # the throw: 12 m/s from a still ball
    players = np.tile([[30.0, 10.0]], (n, 1, 1))
    m = features.memory(b, players, np.full(n, 0.6), FPS, PITCH)
    assert m["m_out_open_s"][FPS * 15] == pytest.approx(7.0, abs=0.1)      # still open 7 s after the last off frame
    assert m["m_kick_since"][FPS * 15] == 0.0
    assert m["m_kick_since"][FPS * 25] == 1.0 and m["m_out_open_s"][FPS * 25] == 0.0


def test_carried_while_everyone_is_slow():
    n = FPS * 10
    t = np.arange(n) / FPS
    b = np.stack([20.0 + 2.0 * t, np.full(n, 30.0)], axis=1)              # ball at walking pace
    players = np.stack([b + [0.5, 0.0], np.tile([60.0, 30.0], (n, 1))], axis=1)   # a player right next to it
    slow = features.memory(b, players, np.full(n, 0.7), FPS, PITCH)
    fast = features.memory(b, players, np.full(n, 0.1), FPS, PITCH)
    assert slow["c_carry_slow"][FPS * 5] == 1.0 and fast["c_carry_slow"][FPS * 5] == 0.0


def test_combine_lets_a_steady_players_signal_win():
    n = FPS * 20
    p_ball = np.full(n, 0.2)                     # the ball says "in play"
    p_pl = np.full(n, 0.95)                      # the players say "stopped", steadily
    assert evaluate.combine(p_ball, p_pl, 1.0, FPS)[FPS * 10] > 0.8
    assert evaluate.combine(p_ball, p_pl, 0.0, FPS)[FPS * 10] == pytest.approx(0.2, abs=1e-3)
    blip = np.full(n, 0.1); blip[FPS * 10:FPS * 10 + 5] = 0.99        # a 0.2 s players blip is averaged away
    assert evaluate.combine(p_ball, blip, 1.0, FPS)[FPS * 10] < 0.2


# ------------------------------------------------------------------------------------ v5 simulation

def test_carried_ball_is_visible_follows_a_player_and_ends_at_the_restart():
    n = FPS * 60
    gt = np.array(["H"] * (FPS * 20) + ["D"] * (FPS * 20) + ["A"] * (FPS * 20))
    t = np.arange(n) / FPS
    b = np.stack([10.0 + 3.0 * t, np.full(n, 30.0)], axis=1)          # play: the ball moves at 3 m/s
    b[FPS * 40:] = [70.0, 0.0]                                         # the restart: on the touchline
    players = np.zeros((n, 3, 2))
    players[:, 0] = np.stack([60.0 + 0.5 * t - 10.0, np.full(n, 25.0)], axis=1)   # walks at 0.5 m/s
    players[:, 1] = [90.0, 50.0]
    players[:, 2] = [20.0, 60.0]
    out = model.carried_ball(b, players, gt, FPS, np.random.default_rng(0), p_carry=1.0, pass_p=0.0)
    dead = slice(FPS * 20, FPS * 40)
    assert not np.isnan(out[dead]).any()                               # visible the whole stoppage
    assert not np.allclose(out[FPS * 25], out[FPS * 30])               # and moving
    mid = FPS * 30
    assert np.linalg.norm(out[mid] - players[mid, 0]) < 1.0            # carried by the nearest player
    assert np.allclose(out[FPS * 40 - 1], [70.0, 0.0])                 # put down where play restarts
    assert np.allclose(out[:FPS * 20], b[:FPS * 20]) and np.allclose(out[FPS * 40:], b[FPS * 40:])
    keep = model.carried_ball(b, players, gt, FPS, np.random.default_rng(0), p_carry=0.0)
    assert np.allclose(keep, b)


# ------------------------------------------------------------------------------------ v6 inputs

def test_bridge_fills_short_dips_only():
    f = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    assert features._bridge(f, 1).tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 1]


def _v6_scene(n_s=80):
    n = FPS * n_s
    t = np.arange(n) / FPS
    speed = np.full(n, 3.0); speed[FPS * 20:FPS * 50] = 0.8          # whistle at 20 s, restart at 50 s
    slow = np.where(speed < 1, 0.7, 0.1)
    slow[FPS * 35:FPS * 35 + 5] = 0.0                                  # a one-frame-ish dip in the stoppage
    b = np.stack([10.0 + 2.0 * t, np.full(n, 30.0)], axis=1)           # ball rolls on, straight, at 2 m/s ...
    b[FPS * 30:FPS * 32] = np.stack([70.0 + 0.0 * t[:FPS * 2], np.full(FPS * 2, 69.0)], axis=1)   # ... off the pitch
    b[FPS * 32:FPS * 48] = [70.0, 66.0]                                # brought back, held at the line
    b[FPS * 40:FPS * 40 + 10, 0] = 70.0 + np.linspace(0, 4, 10)        # a little kick: the ball being returned
    b[FPS * 50:, 0] = 70.0 - 12.0 * (t[FPS * 50:] - 50.0)              # the throw-in: 12 m/s from still
    b[FPS * 50:, 1] = 66.0
    players = np.tile([[30.0, 30.0]], (n, 1, 1))
    return speed, slow, b, players


def test_v6_unplayed_returns_restart_and_smoothing():
    speed, slow, b, players = _v6_scene()
    f = features.stopped_v6(b, players, speed, slow, FPS, PITCH)
    assert f["u_unplayed_s"][FPS * 25] > 3.0                           # rolled on untouched after the brake
    assert f["m2_open_s"][FPS * 45] > 10.0                             # the return kick did NOT close it
    assert f["m2_returns"][FPS * 45] >= 1
    assert f["m2_open_s"][FPS * 60] == 0.0                             # the real restart (team speeds up) did
    assert f["s2_team_stopped_s"][FPS * 45] > 20.0                     # the short dip did not reset it


def test_v6_memory_times_out_when_play_clearly_resumed():
    n = FPS * 60
    speed = np.full(n, 3.0)
    slow = np.full(n, 0.05)
    b = np.tile([50.0, 30.0], (n, 1)).astype(float)
    b[FPS * 5] = [50.0, -2.0]                                          # one frame off the pitch, then play on
    players = np.tile([[30.0, 30.0]], (n, 1, 1))
    f = features.stopped_v6(b, players, speed, slow, FPS, PITCH)
    assert f["m2_open_s"][FPS * 15] > 0.0 and f["m2_open_s"][FPS * 40] == 0.0


# ------------------------------------------------------------------------------------ v7 "CR7" inputs

def test_v7_memory_opens_only_on_a_real_out():
    n = FPS * 30
    speed, slow = np.full(n, 1.0), np.full(n, 0.6)
    t = np.arange(n) / FPS
    mid = np.stack([30.0 + 3.0 * t, np.full(n, 34.0)], axis=1)
    mid[FPS * 10:FPS * 20] = np.nan                                    # vanishes in midfield: NOT an out
    assert (features.stopped_v7(mid, speed, slow, FPS, PITCH)["m3_open_s"] == 0).all()
    line = np.stack([np.full(n, 50.0), 48.0 + 2.0 * t], axis=1)        # rolls toward the far touchline (y = 68)
    line[FPS * 10:FPS * 20] = np.nan                                   # vanishes at y = 68, moving outward
    f = features.stopped_v7(line, speed, slow, FPS, PITCH)
    assert f["m3_open_s"][FPS * 15] > 4.0


def test_v7_restart_closes_memory_and_switches_team_stopped_off():
    speed, slow, b, players = _v6_scene()
    f = features.stopped_v7(b, speed, slow, FPS, PITCH)
    assert f["m3_open_s"][FPS * 45] > 10.0                             # open through the stoppage
    assert f["m3_returns"][FPS * 45] >= 1                              # the return kick, team stopped
    assert f["s3_team_stopped_s"][FPS * 45] > 20.0
    i = FPS * 50 + 15                                                  # just after the throw-in
    assert f["m3_open_s"][i] == 0.0 and f["s3_team_stopped_s"][i] == 0.0
    assert f["r_since_restart_s"][i] < 1.0 and f["r_until_restart_s"][FPS * 48] < 3.0


def test_v7_memory_closes_once_play_clearly_looks_alive():
    n = FPS * 40
    speed, slow = np.full(n, 3.0), np.full(n, 0.05)                    # play is clearly alive throughout
    b = np.tile([50.0, 30.0], (n, 1)).astype(float)
    b[FPS * 5] = [50.0, -2.0]                                          # one frame seen off the pitch
    f = features.stopped_v7(b, speed, slow, FPS, PITCH)
    assert f["m3_open_s"][FPS * 6] > 0.0 and f["m3_open_s"][FPS * 14] > 0.0     # the 10 s timeout first
    assert f["m3_open_s"][FPS * 20] == 0.0                                          # then closed: play is alive


# ------------------------------------------------------------------------------------ v8 keeper inputs

def test_keeper_hold_after_a_save_including_while_the_ball_is_lost_in_his_hands():
    n = FPS * 20
    t = np.arange(n) / FPS
    players = np.zeros((n, 3, 2))
    players[:, 0] = [2.0, 34.0]                                        # the keeper, deepest, on his line
    players[:, 1] = [30.0, 30.0]; players[:, 2] = [40.0, 40.0]
    b = np.tile([2.5, 34.0], (n, 1)).astype(float)                     # in his hands ...
    b[:FPS * 5, 0] = 2.5 + 15.0 * (5.0 - t[:FPS * 5])                  # ... after a 15 m/s shot at goal
    b[FPS * 10:FPS * 14] = np.nan                                      # the tracker loses it in his hands
    k = features.keeper_v8(b, players, FPS, PITCH)
    assert k["k_hold"][FPS * 7] == 1.0 and k["k_save_hold"][FPS * 7] == 1.0
    assert k["k_hold"][FPS * 12] == 1.0                                # carried through the gap
    assert k["k_since_shot_s"][FPS * 7] < 5.0
    players[:, 1] = [0.5, 30.0]                                        # someone deeper than him: not the keeper rule
    assert features.keeper_v8(b, players, FPS, PITCH)["k_hold"][FPS * 7] == 0.0


# ------------------------------------------------------------------------------------ v8 start gate

def test_gate_moves_an_early_start_to_when_the_players_have_stopped():
    dead = np.zeros(100, bool); dead[20:60] = True                     # called from frame 20 ...
    gate = np.zeros(100, bool); gate[30:] = True                       # ... players only stopped at 30
    out = decoder.gate_starts(dead, gate)
    assert not out[20:30].any() and out[30:60].all()
    never = decoder.gate_starts(dead, np.zeros(100, bool))
    assert (never == dead).all()                                       # never gated: kept as it was ...
    assert not decoder.gate_starts(dead, np.zeros(100, bool), drop=True).any()   # ... or dropped


def test_start_gate_players_or_ball_out():
    X = pd.DataFrame({"m3_open_s": [0, 0, 2.0, 0], "b_off": [0, 1, 0, 0]})
    g = decoder.start_gate(X, np.array([0.1, 0.1, 0.1, 0.6]), 0.5)
    assert g.tolist() == [False, True, True, True]


def test_early_start_measure():
    gt = _gt()                                                        # stoppages start at 600 and 1450
    pred = gt == "D"
    assert evaluate.score(gt, pred, FPS)["early_s"] == 0.0
    pred[550:600] = True                                               # 2 s too early before the first one
    assert evaluate.score(gt, pred, FPS)["early_s"] == pytest.approx(1.0)   # 2 s over 2 stoppages
