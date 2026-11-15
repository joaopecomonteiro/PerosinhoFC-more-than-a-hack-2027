"""mqf.deadball: ball inputs, the semi-Markov decoder (exact), the scorer, the ball-loss condition."""
from __future__ import annotations

import itertools

import numpy as np
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
