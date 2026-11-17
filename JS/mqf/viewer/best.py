"""The viewer's "Best so far" lane: our current best pipeline, run here on the tracking the page shows.

  tracking  the Fixed set (fixed.py): cleaned tracklets + the ball filled with the frozen setting --
            exactly what the pitch view draws in Fixed
  who       the owner TCN ensemble `ens_impff` (EXPERIMENTS #40): three TCNs (init seeds 0 1 2,
            feature set full) trained on IDSSE 5 + Metrica 2 + PFF 56 -- mean P(home) and mean
            P(boundary) of the three, each one averaged over both team orders (mqf.owner.models)
  dead      DEAD_NONE (default): there is no dead-ball model in this lane yet. Every frame of the
            halves counts as in play, so the lane is H / A only; the page leaves the truth's dead
            frames out of the disagreements for it. DEAD_V1: the learned P(dead) (models/dead_v1.joblib)
            inside the production 3-state HMM (dead_who=max) -- #40's layer, no Smart Tagging
  decoder   #40's "ens_impff + own boundary": H <-> A switch cost 18 x (1 - 0.5 P(boundary)); a 2-state
            Viterbi per half (none) or the 3-state HMM (dead_v1)

Computed on this machine and cached in cache/viewer (gitignored) by a hash of the Fixed tracking, the
model files, the options and the source of every module the numbers depend on. Nothing leaves the
machine; the startup log prints aggregates only (spells, shares), which are safe to paste.

On BAR-ATH the ground truth is the public 5 minutes: compare there to check the lane is *correct*,
never to choose or tune (RULES.md). Decisions stay on the other games (python -m mqf.owner.tcn_check).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..config import MODELS_DIR
from .game import VIEWER_CACHE, Lane, Log, TrackingSet

VERSION = 1
DEAD_NONE, DEAD_V1 = "none", "dead_v1"
DEAD_CHOICES = (DEAD_NONE, DEAD_V1)
ENSEMBLE = tuple(MODELS_DIR / f"owner_tcn_full_impff_s{k}.pt" for k in (0, 1, 2))   # #40's ens_impff
DEAD_MODEL = MODELS_DIR / "dead_v1.joblib"
ALPHA = 0.5                      # #40: boundary-aware switch cost, alpha 0.5
P_NONE = 255                     # page encoding of P: round(250 p), 255 = no value


@dataclass
class Best:
    codes: np.ndarray            # (n,) uint8 0 H, 1 A, 2 D, 3 no label
    p_home: np.ndarray           # (n,) float, NaN outside the halves
    p_boundary: np.ndarray       # (n,) float, NaN outside the halves
    dead: str
    summary: str


def _code_files():
    from .. import decode, dead, features, who
    from ..owner import combine, feats, models, rules
    return [Path(m.__file__) for m in (decode, dead, features, who, combine, feats, models, rules)] + [Path(__file__)]


def _key(ts: TrackingSet, fps, pitch, halves, model_paths, dead_mode, dead_path) -> str:
    h = hashlib.md5()
    for a in (ts.players, ts.team_of_slot, ts.ball, ts.ball_source if ts.ball_source is not None else np.zeros(0)):
        h.update(np.ascontiguousarray(a).tobytes())
    for f in _code_files():
        h.update(f.read_bytes())
    files = list(model_paths) + ([dead_path] if dead_mode == DEAD_V1 else [])
    stats = []
    for p in files:
        st = Path(p).stat()
        stats.append([Path(p).name, st.st_size, st.st_mtime_ns])
    h.update(json.dumps([VERSION, float(fps), list(pitch), [list(map(int, x)) for x in halves], dead_mode, ALPHA,
                         stats]).encode())
    return h.hexdigest()[:12]


def available(model_paths=ENSEMBLE, dead_mode: str = DEAD_NONE, dead_path: Path = DEAD_MODEL) -> str:
    """'' when the lane can run here, else why not (shown as a note)."""
    missing = [Path(p).name for p in model_paths if not Path(p).is_file()]
    if dead_mode == DEAD_V1 and not Path(dead_path).is_file():
        missing.append(Path(dead_path).name)
    if missing:
        return f"best lane skipped: missing {', '.join(missing)} in {MODELS_DIR}"
    try:
        import torch  # noqa: F401
    except ImportError:
        return "best lane skipped: torch is not installed here (python tasks.py install-torch)"
    return ""


def _inside(halves, n):
    m = np.zeros(n, dtype=bool)
    for _, a, b in halves:
        m[a:b + 1] = True
    return m


def evidence(ts: TrackingSet, fps, pitch, halves, model_paths=ENSEMBLE, device: str = "cpu"):
    """(P(home), P(boundary)) of the TCN ensemble on the Fixed tracking: as mqf.owner.experiment builds
    them (feature views for both team orders, each TCN swap-averaged, then the mean over the TCNs)."""
    from ..owner import feats
    from ..owner.models import OwnerModel
    n = len(ts.ball)
    players = np.asarray(ts.players)
    ball = np.asarray(ts.ball, dtype=np.float64)
    team0 = np.asarray(ts.team_of_slot)
    models = [OwnerModel.load(p) for p in model_paths]
    bad = [Path(p).name for p, m in zip(model_paths, models) if m.kind != "tcn"]
    if bad:
        raise ValueError(f"not TCNs with a boundary head: {bad}")
    fsets = sorted({m.feature_set for m in models})
    fset = "full_rules_heading" if len(fsets) > 1 else fsets[0]        # one table every model can pick from
    filled = None
    if ts.ball_source is not None:
        from ..ball import impute
        src = np.asarray(ts.ball_source)
        filled = ((src == impute.INTERP) | (src == impute.MODEL)) & ~np.isnan(ball[:, 0])
    views = []
    for k in (0, 1):
        team = team0 if k == 0 else 1 - team0
        X = feats.features(players, team, ball, fps, pitch, halves, fset, ball_filled=filled)
        S, sc = feats.seq_inputs(players, team, ball, fps, pitch, halves, X_full=X, columns="all")
        views.append({"X": X, "S": S, "seq_columns": sc})
    ps, bs = [], []
    for m in models:
        p, pb = m.p_home_and_boundary(views, halves, n, fps, device)
        ps.append(p)
        bs.append(pb)
    return np.mean(np.stack(ps), axis=0), np.mean(np.stack(bs), axis=0), filled


def decode(p_home, p_bnd, halves, n: int, dead_mode: str = DEAD_NONE, feat=None, params=None) -> np.ndarray:
    """Labels (n,) H/A/D/X. DEAD_NONE: every frame of the halves in play, 2-state Viterbi with the
    boundary-aware switch cost. DEAD_V1: the 3-state HMM on `feat` (features.build with p_dead)."""
    from .. import decode as dec
    from ..owner import combine
    params = params or replace(dec.Params(), dead_who="max")
    sw = combine.switch_costs(p_bnd, params.switch_cost, ALPHA)
    if dead_mode == DEAD_NONE:
        return combine.decode_given_alive(p_home, _inside(halves, n), halves, n, sw, params.w_who_model,
                                          params.who_clip)
    if feat is None:
        raise ValueError("dead_v1 needs the decoder features (features.build with p_dead)")
    return combine.decode_hmm3(feat, p_home, halves, n, params, sw)


def _summary(lab: np.ndarray, p_home, fps, dead_mode) -> str:
    from .. import decode as dec
    s = dec.spell_stats(lab, fps)
    g = lab[lab != "X"]
    share = {k: float(np.mean(g == k)) if len(g) else 0.0 for k in "HAD"}
    inp = ~np.isnan(p_home)
    txt = (f"{s['possessions']} possessions, mean {s['mean_s']:.1f} s, median {s['median_s']:.1f} s; "
           f"labels H {share['H']:.1%} / A {share['A']:.1%} / D {share['D']:.1%}; "
           f"P(home) >= 0.5 on {float(np.mean(p_home[inp] >= 0.5)) if inp.any() else 0:.1%} of frames")
    return txt + ("; no dead-ball model" if dead_mode == DEAD_NONE else "; dead from dead_v1")


def best_lane(ts: TrackingSet | None, fps, pitch, halves, n: int, dead_mode: str = DEAD_NONE,
              model_paths=ENSEMBLE, dead_path: Path = DEAD_MODEL, device: str = "auto", force: bool = False,
              log: Log = print) -> tuple[Lane | None, str]:
    """(the lane, '') or (None, why it was skipped). Never raises: the viewer must open anyway."""
    if dead_mode not in DEAD_CHOICES:
        return None, f"best lane skipped: --best-dead must be one of {DEAD_CHOICES}"
    if ts is None:
        return None, "best lane skipped: it runs on the Fixed tracking (drop --no-fixed)"
    why = available(model_paths, dead_mode, dead_path)
    if why:
        return None, why
    try:
        b = _run(ts, fps, pitch, halves, n, dead_mode, model_paths, dead_path, device, force, log)
    except Exception as e:                                  # never fatal
        return None, f"best lane skipped: {type(e).__name__}: {e}"
    src = ("TCN x3 + boundary · no dead-ball model" if dead_mode == DEAD_NONE else "TCN x3 + boundary · dead_v1")
    lane = Lane("Old best (TCN only)", "model", b.codes, src, p_home=b.p_home, p_boundary=b.p_boundary,
                no_dead=dead_mode == DEAD_NONE, note=b.summary)
    return lane, ""


def _run(ts, fps, pitch, halves, n, dead_mode, model_paths, dead_path, device, force, log) -> Best:
    cache = VIEWER_CACHE / f"best_{_key(ts, fps, pitch, halves, model_paths, dead_mode, dead_path)}.npz"
    if cache.exists() and not force:
        z = np.load(cache)
        b = Best(z["codes"], z["p_home"], z["p_boundary"], str(z["dead"]), str(z["summary"]))
        log("best lane (cached): " + b.summary)
        return b
    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    log(f"best lane: TCN ensemble ({len(model_paths)} models) on {dev}, dead-ball: {dead_mode}")
    p_home, p_bnd, filled = evidence(ts, fps, pitch, halves, model_paths, dev)
    feat = None
    if dead_mode == DEAD_V1:
        from .. import atd, dead, features
        p_dead = dead.predict(dead.load_model(dead_path), ts.players, ts.team_of_slot, fps, pitch, halves)
        speed = np.full(ts.players.shape[:2], np.nan, dtype=np.float32)   # unused: P(dead) replaces v0's speed rule
        trk = atd.Tracking(ts.players, speed, np.asarray(ts.ball, dtype=np.float32), ts.team_of_slot)
        feat = features.build(trk, pitch, fps, ball_filled=filled, p_dead=p_dead)
    lab = decode(p_home, p_bnd, halves, n, dead_mode, feat)
    from .game import LABELS
    codes = np.full(n, 3, dtype=np.uint8)
    for k, name in enumerate(LABELS[:3]):
        codes[lab == name] = k
    b = Best(codes, p_home, p_bnd, dead_mode, _summary(lab, p_home, fps, dead_mode))
    VIEWER_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, codes=b.codes, p_home=b.p_home, p_boundary=b.p_boundary, dead=np.array(b.dead),
                        summary=np.array(b.summary))
    log("best lane: " + b.summary)
    return b


def encode_p(p) -> bytes:
    """(n,) probability -> uint8 bytes for the page: round(250 p), P_NONE where unknown."""
    p = np.asarray(p, dtype=np.float64)
    out = np.where(np.isfinite(p), np.round(np.clip(p, 0.0, 1.0) * 250.0), P_NONE)
    return out.astype(np.uint8).tobytes()
