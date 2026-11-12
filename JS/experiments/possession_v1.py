"""possession_v1.py -- More than a Hack 2027, Phase 1: who has the ball, frame by frame.

One script, run it from the root of the challenge repository:

    python possession_v1.py                    # writes evaluation/results/possession_v1.csv
    python possession_v1.py --plot             # also writes possession_v1_overview.png
    python evaluation/validate_submission.py evaluation/results/possession_v1.csv
    python evaluation/run_evaluation.py        # public 5 min: a CORRECTNESS check, not a tuning set

How it works (the whole idea in four steps)
-------------------------------------------
1. READ    the automatic tracking (ATD): 46 anonymous slots + the ball, per frame, in metres.
2. FEATURES per frame: distance of the nearest home / away player to the ball, ball speed,
            is the ball outside the pitch, is the ball still, are the players idle.
3. EVIDENCE turn the features into a log-score for each state H / A / D ("how well does this
            frame fit each state?"). Where the ball is missing the evidence is neutral, so the
            decoder simply keeps the previous state -- which is what a possession is.
4. DECODE   a 3-state HMM solved with Viterbi, once per half. Switching state costs points, so
            1-frame flicker cannot split a possession in two. That is the whole reason the
            ranking metric (spellF1) likes this more than the "majority vote" baseline.

Rules this respects: no hand labels, nothing tuned on the public 5 minutes, no Smart Tagging
yet (see NEXT STEPS at the bottom). Every constant below is a plain football-sense default,
not a fitted number -- change them one at a time and re-read the summary it prints.
"""

import argparse
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Constants. All of these are guesses from football sense. None was fitted to this match.
# ----------------------------------------------------------------------------------------------
FPS = 25
N_FRAMES = GAMES[GAME]["n_video_frames"]          # 199319
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]           # 105.3 x 68.0 m

# Ball: fill gaps of at most this many frames by linear interpolation (0.5 s).
BALL_GAP_FRAMES = 12

# Evidence for "who has it": logit = (dist_away - dist_home) / SCALE_M, clipped.
# The evidence fades to zero when the nearest player is far (ball in the air / loose).
SCALE_M = 2.0
LOGIT_CLIP = 3.0
CLOSE_M, FAR_M = 2.0, 8.0                         # full weight at <=2 m, none at >=8 m

# Evidence for "dead ball" (log-score added to D). The bias makes D lose unless evidence adds up.
D_BIAS = 2.0
W_OUT = 6.0                                       # ball outside the pitch
W_STILL = 3.0                                     # ball almost motionless for ~2 s
W_IDLE = 2.0                                      # all players nearly motionless (~3 s median)
W_FEW = 4.0                                       # fewer than 10 players tracked (not a game)
W_FAST = 3.0                                      # ball clearly moving fast -> definitely live
OUT_MARGIN_M = 0.3
STILL_SPEED_MS, STILL_WINDOW = 0.7, 50            # ball still: max speed over 2 s below 0.7 m/s
IDLE_SPEED_MS, IDLE_WINDOW = 0.6, 75              # players idle: median speed over 3 s below 0.6
FAST_SPEED_MS = 4.0
MIN_PLAYERS = 10

# Decoder: cost (in log-score points) of changing state. Staying put is free.
COST_HA = 18.0                                    # H <-> A  (hand-over of possession)
COST_LIVE_DEAD = 10.0                             # H/A <-> D (stoppage / restart)

LABELS = np.array(["H", "A", "D"])
HOME, AWAY = 0, 1


# ----------------------------------------------------------------------------------------------
# 1. READ
# ----------------------------------------------------------------------------------------------
def team_of_each_slot(meta_path):
    """0 for slots of the home team (first team in the metadata), 1 for away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def read_atd(raw_path, n_slots, chunk_lines=20000):
    """Raw ATD -> (players[N, S, 3], ball[N, 3]); x, y normalised 0-1, speed in m/s, NaN = missing.

    Line format:  <frame_id>:x,y,s;x,y,s;...:bx,by,bs   (frame_id counts from 1)
    Parsed in chunks with one numpy call per chunk, which is far faster than splitting by hand.
    """
    width = 1 + 3 * n_slots + 3
    players = np.full((N_FRAMES, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((N_FRAMES, 3), np.nan, dtype=np.float32)

    def flush(lines):
        text = ",".join(lines).replace(":", ",").replace(";", ",")
        flat = np.fromstring(text, sep=",", dtype=np.float64)
        if flat.size != width * len(lines):
            raise ValueError("unexpected ATD line width: got {} values for {} lines, expected {} "
                             "each (is the metadata xml the one for this txt?)".format(
                                 flat.size, len(lines), width))
        rows = flat.reshape(len(lines), width)
        frames = rows[:, 0].astype(np.int64) - 1          # video_frame = frame_id - 1
        keep = (frames >= 0) & (frames < N_FRAMES)
        players[frames[keep]] = rows[keep, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
        ball[frames[keep]] = rows[keep, -3:]

    buf = []
    with open(raw_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                buf.append(line)
            if len(buf) >= chunk_lines:
                flush(buf)
                buf = []
    if buf:
        flush(buf)
    return players, ball


# ----------------------------------------------------------------------------------------------
# 2. FEATURES
# ----------------------------------------------------------------------------------------------
def rolling(series, window, how):
    """Centred rolling stat on a numpy array (NaN-aware, needs half the window to be present)."""
    return getattr(pd.Series(series).rolling(window, center=True, min_periods=window // 2), how)() \
             .to_numpy()


def compute_features(players, ball, team_of_slot):
    px = players[:, :, 0].astype(np.float64) * PITCH_L
    py = players[:, :, 1].astype(np.float64) * PITCH_W
    pspeed = players[:, :, 2].astype(np.float64)

    bx = pd.Series(ball[:, 0].astype(np.float64) * PITCH_L)
    by = pd.Series(ball[:, 1].astype(np.float64) * PITCH_W)
    seen_raw = bx.notna().to_numpy()
    # Interpolate only short gaps; long gaps stay NaN on purpose (we do not invent a ball).
    bx = bx.interpolate(limit=BALL_GAP_FRAMES, limit_area="inside").to_numpy()
    by = by.interpolate(limit=BALL_GAP_FRAMES, limit_area="inside").to_numpy()
    ball_ok = ~np.isnan(bx)

    dist = np.hypot(px - bx[:, None], py - by[:, None])      # (N, S), NaN where anything missing
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home = dist[:, team_of_slot == HOME].min(axis=1)
    d_away = dist[:, team_of_slot == AWAY].min(axis=1)

    # Ball speed (m/s) from a +-4 frame central difference.
    k = 4
    vx = (np.roll(bx, -k) - np.roll(bx, k)) / (2 * k / FPS)
    vy = (np.roll(by, -k) - np.roll(by, k)) / (2 * k / FPS)
    ball_speed = np.hypot(vx, vy)
    ball_speed[:k] = ball_speed[-k:] = np.nan

    n_players = np.sum(~np.isnan(px), axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # frames with no players -> NaN, on purpose
        median_player_speed = np.nanmedian(np.where(n_players[:, None] >= 3, pspeed, np.nan), axis=1)

    out = ball_ok & ((bx < -OUT_MARGIN_M) | (bx > PITCH_L + OUT_MARGIN_M)
                     | (by < -OUT_MARGIN_M) | (by > PITCH_W + OUT_MARGIN_M))

    ball_still = rolling(ball_speed, STILL_WINDOW, "max") < STILL_SPEED_MS
    players_idle = rolling(median_player_speed, IDLE_WINDOW, "median") < IDLE_SPEED_MS

    return {
        "ball_seen_raw": seen_raw, "ball_ok": ball_ok, "d_home": d_home, "d_away": d_away,
        "ball_speed": ball_speed, "n_players": n_players, "out": out,
        "ball_still": np.nan_to_num(ball_still, nan=0).astype(bool),
        "players_idle": np.nan_to_num(players_idle, nan=0).astype(bool),
    }


# ----------------------------------------------------------------------------------------------
# 3. EVIDENCE  ->  log-score of each state per frame, shape (N, 3) for [H, A, D]
# ----------------------------------------------------------------------------------------------
def build_emissions(f):
    n = len(f["d_home"])
    emit = np.zeros((n, 3))

    # H vs A: only where the ball is located and both teams have someone tracked.
    usable = f["ball_ok"] & np.isfinite(f["d_home"]) & np.isfinite(f["d_away"])
    with np.errstate(invalid="ignore"):                      # inf - inf where a team is untracked
        logit = np.clip((f["d_away"] - f["d_home"]) / SCALE_M, -LOGIT_CLIP, LOGIT_CLIP)
        nearest = np.minimum(f["d_home"], f["d_away"])
        weight = np.clip((FAR_M - nearest) / (FAR_M - CLOSE_M), 0.0, 1.0)   # fades out when far
        ev = np.where(usable, logit * weight, 0.0)
    emit[:, HOME] = +0.5 * ev
    emit[:, AWAY] = -0.5 * ev

    # D: add up the "this does not look like live play" clues.
    d = np.zeros(n)
    d += W_OUT * f["out"]
    d += W_STILL * f["ball_still"] * f["ball_ok"]
    d += W_IDLE * f["players_idle"]
    d += W_FEW * (f["n_players"] < MIN_PLAYERS)
    fast = np.nan_to_num(f["ball_speed"], nan=0.0) > FAST_SPEED_MS
    d -= W_FAST * fast
    emit[:, 2] = d - D_BIAS
    return emit


# ----------------------------------------------------------------------------------------------
# 4. DECODE
# ----------------------------------------------------------------------------------------------
def viterbi(emit):
    """Best state sequence for one half. Maximises sum(emission) - sum(switch costs)."""
    cost = np.array([[0.0, COST_HA, COST_LIVE_DEAD],
                     [COST_HA, 0.0, COST_LIVE_DEAD],
                     [COST_LIVE_DEAD, COST_LIVE_DEAD, 0.0]])
    n = len(emit)
    back = np.empty((n, 3), dtype=np.int8)
    score = emit[0].copy()
    for t in range(1, n):
        cand = score[:, None] - cost              # cand[from, to]
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_match(emit):
    labels = np.full(N_FRAMES, "X", dtype="<U1")      # X outside the two graded halves: free
    for half, first, last in half_video_blocks(GAME):
        labels[first:last + 1] = LABELS[viterbi(emit[first:last + 1])]
        print("  half {}: frames {}-{} decoded".format(half, first, last))
    return labels


# ----------------------------------------------------------------------------------------------
# Output, summary, plot
# ----------------------------------------------------------------------------------------------
def write_csv(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)


def possessions(labels):
    """(team, start, end) for each maximal run of H or A. D and X break runs."""
    out, start = [], None
    for i in range(len(labels) + 1):
        cur = labels[i] if i < len(labels) else "D"
        prev = labels[i - 1] if i > 0 else "D"
        if i == len(labels) or cur != prev:
            if start is not None and prev in ("H", "A"):
                out.append((prev, start, i - 1))
            start = i if cur in ("H", "A") else None
    return out


def summary(labels, feats):
    graded = np.zeros(N_FRAMES, bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    g = labels[graded]
    n = graded.sum()
    print("\n=== summary (graded frames only) ===")
    for lab in "HAD":
        print("  {}: {:>7} frames  {:5.1f}%".format(lab, int((g == lab).sum()), 100.0 * (g == lab).mean()))
    spells = []
    for _, a, b in half_video_blocks(GAME):
        spells += possessions(labels[a:b + 1])
    durs = np.array([(e - s + 1) / FPS for _, s, e in spells])
    print("  possessions: {}  (mean {:.1f} s, median {:.1f} s)".format(
        len(spells), durs.mean() if len(durs) else 0, np.median(durs) if len(durs) else 0))
    print("  ball located on {:.1f}% of graded frames (before interpolation {:.1f}%)".format(
        100.0 * feats["ball_ok"][graded].mean(), 100.0 * feats["ball_seen_raw"][graded].mean()))
    print("  clue rates on graded frames: ball out {:.1f}%, ball still {:.1f}%, players idle {:.1f}%".format(
        100.0 * feats["out"][graded].mean(), 100.0 * feats["ball_still"][graded].mean(),
        100.0 * feats["players_idle"][graded].mean()))
    print("  sanity: a real match has a few hundred possessions of ~10-20 s and the ball is\n"
          "  dead roughly 25-45% of the time. If you see 2000 possessions, switch costs are too low;\n"
          "  if D is 0% or 80%, look at the clue rates above.")


def plot(labels, path, public_gt=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    code = {"H": 0, "A": 1, "D": 2, "X": 3}
    cmap = ListedColormap(["#c0392b", "#2c3e50", "#bdc3c7", "#ffffff"])
    rows = []
    for half, a, b in half_video_blocks(GAME):
        rows.append(("half {} (whole)".format(half), labels[a:b + 1]))
    if public_gt is not None:
        a = 11185
        b = a + 7499
        rows.append(("public 5 min: ours", labels[a:b + 1]))
        rows.append(("public 5 min: truth", public_gt[a:b + 1]))
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 1.0 * len(rows) + 1.2))
    axes = np.atleast_1d(axes)
    for ax, (title, lab) in zip(axes, rows):
        arr = np.array([code[x] for x in lab])[None, :]
        ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_yticks([])
        ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_xticks(np.arange(0, len(lab), 5 * 60 * FPS))
        ax.set_xticklabels(["{}'".format(int(t / (60 * FPS))) for t in np.arange(0, len(lab), 5 * 60 * FPS)]
                           if len(lab) > 10000 else ["{:d}s".format(int(t / FPS)) for t in
                                                     np.arange(0, len(lab), 5 * 60 * FPS)], fontsize=7)
    fig.legend(handles=[Patch(color="#c0392b", label="H (Barça)"), Patch(color="#2c3e50", label="A (Athletic)"),
                        Patch(color="#bdc3c7", label="D (dead ball)")], loc="upper right", ncol=3, fontsize=8)
    fig.suptitle("Ball possession, decoded by possession_v1", x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    print("plot -> {}".format(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(HERE, "evaluation", "results", "possession_v1.csv"))
    ap.add_argument("--data-dir", default=None, help="folder that contains tactical/ (default: ./data)")
    ap.add_argument("--plot", action="store_true", help="also write possession_v1_overview.png")
    args = ap.parse_args()

    paths = asset_paths(data_dir=args.data_dir)
    for key in ("atd_meta", "atd_raw"):
        if not os.path.isfile(paths[key]):
            sys.exit("missing {} -- unpack the data bundle into ./data (see DATA.md)".format(paths[key]))

    team_of_slot = team_of_each_slot(paths["atd_meta"])
    print("reading tracking ({} slots: {} home, {} away) ...".format(
        len(team_of_slot), int((team_of_slot == HOME).sum()), int((team_of_slot == AWAY).sum())))
    players, ball = read_atd(paths["atd_raw"], len(team_of_slot))

    print("features ...")
    feats = compute_features(players, ball, team_of_slot)
    print("decoding ...")
    labels = decode_match(build_emissions(feats))

    write_csv(args.out, labels)
    print("wrote {}".format(args.out))
    summary(labels, feats)

    if args.plot:
        gt_path = os.path.join(HERE, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
        gt = pd.read_csv(gt_path)["label"].to_numpy() if os.path.isfile(gt_path) else None
        plot(labels, os.path.join(HERE, "possession_v1_overview.png"), gt)

    print("\nnext: python evaluation/validate_submission.py {}".format(args.out))


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------------------------
# NEXT STEPS (in the order I would try them; accept one only if it wins by more than 0.04 spellF1
# on a *different* validation source, never on the public 5 minutes)
#   1. Minimum possession length / explicit durations (HSMM) instead of a flat switch cost.
#   2. Smart Tagging: SET PIECES windows as a prior for D, team tags as a weak prior for H/A.
#   3. Long ball gaps: carry a "who was last close" memory instead of staying neutral.
#   4. Learn the evidence weights on OTHER games (public tracking with labels), then apply here.
# ----------------------------------------------------------------------------------------------
