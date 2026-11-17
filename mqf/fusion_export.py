"""The fusion suite's inputs, written from the final_check runs (other games only), and the
production decoder rebuilt from them.

    python -m mqf.alive.final_check --jobs 3 --whos tcn --tag fusion16 \
        --fusion-export harness_results/fusion_preds          # writes <column>/<game>__s<seed>.npz

Per (column, val game, noise seed) the file holds exactly what FINAL's decoder read in that run --
P(dead) = round(mean(asf, v8), 1e-4), the owner-TCN P(home) and boundary, the emission features
(ball seen / imputed / out, distances, mean speed), decode.Params, the halves -- plus the separate
streams the fusion knobs need (asf, v8, the 3 TCN seeds, the asf boundary, optionally who_v1) and
`pred_final`, the labels final_check scored for FINAL. export() re-decodes FINAL and stops if its
spellF1 is not the one final_check just computed for the same run, so a file can only exist if it
reproduces the harness row.

prod(g) is the production decoder (alive.combine.decode_hmm -> decode.emissions + viterbi3) on those
arrays: `python run_fusion.py` uses it as --baseline-hook (via mqf.decode.fusion_baseline) and checks
prod(g) == pred_final on every file. submission_g() builds the same dict on BAR-ATH inside
make_submission (--fusion-decoder), so a fusion decoder reads the arrays FINAL reads.
"""
from __future__ import annotations

import hashlib
import json
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import atd, decode, features
from .fusion import FEAT_COLS, save_game

CODES = {"H": 0, "A": 1, "D": 2, "X": 3}


def to_codes(labels) -> np.ndarray:
    lab = np.asarray(labels)
    out = np.full(len(lab), 3, np.int8)
    for k, v in CODES.items():
        out[lab == k] = v
    return out


def _halves(g):
    return [(k + 1, int(a), int(b)) for k, (a, b) in enumerate(np.asarray(g["halves"]).reshape(-1, 2))]


def prod(g) -> np.ndarray:
    """FINAL's decoder on the exported arrays: codes 0..3 (X outside the halves)."""
    from .alive import combine as acombine
    from .owner import combine as ocombine
    params = decode.Params(**json.loads(str(g["params_json"])))
    feat = pd.DataFrame({c: np.asarray(g["feat_" + c]) for c in FEAT_COLS})
    feat["p_home_model"] = np.asarray(g["p_home"], dtype=np.float64)
    alpha = float(g.get("who_alpha", 0.0))
    sw_t = (ocombine.switch_costs(g["p_bnd"], params.switch_cost, alpha)
            if "p_bnd" in g and alpha > 0 else None)
    tags = None
    if "tag_vote" in g:
        tv = np.asarray(g["tag_vote"], dtype=np.float64)
        tags = types.SimpleNamespace(vote=np.stack([tv, np.zeros_like(tv)], 1))
    n = len(feat)
    lab = acombine.decode_hmm(feat, np.asarray(g["p_dead"], dtype=np.float64), _halves(g), n, params,
                              None, 0.0, False, sw_t, tags)
    return to_codes(lab)


def _inside(halves, n):
    m = np.zeros(n, dtype=bool)
    for _, a, b in halves:
        m[a:b + 1] = True
    return m


def _feat(spec, trk, ball_f, filled, p_home):
    """features.build exactly as experiment.run_game builds it for the decoder."""
    tracking = atd.Tracking(trk.players, trk.player_speed, np.asarray(ball_f, dtype=np.float32), trk.team_of_slot)
    return features.build(tracking, spec.pitch, spec.fps, ball_filled=filled, p_home=p_home)


def export(root, cond, spec, source, seed, gt, trk, ball_f, filled, p_home, pb_who, P, PB,
           params: decode.Params, who_alpha: float, final_method: str = "asf_v8", check_spell=None,
           p_home_seeds=None, p_home_v1=None) -> Path:
    """One fusion input file; returns its path. check_spell: final_check's spellF1 of the same run
    for (tcn, final_method, hmm) -- the export stops if FINAL re-decoded here scores differently."""
    from .alive import combine as acombine
    from .harness.evaluate import score
    from .owner import combine as ocombine
    if str(spec.name).upper().startswith("BAR"):
        raise SystemExit("fusion export is for other games only")
    if final_method not in P or P[final_method] is None:
        raise SystemExit(f"fusion export: method {final_method!r} not computed in this run")
    n, fps, halves = spec.n_frames, spec.fps, spec.halves
    feat = _feat(spec, trk, ball_f, filled, p_home)
    sw_t = None if pb_who is None else ocombine.switch_costs(pb_who, params.switch_cost, who_alpha)
    lab = acombine.decode_variant("hmm", feat, P[final_method], halves, n, fps, params, None,
                                  PB.get(final_method), None, 0.0, sw_t)
    if check_spell is not None:
        s = score(gt, lab, fps)["spellF1"]
        if s != check_spell:
            raise SystemExit(f"fusion export {spec.name} s{seed} {cond}: FINAL re-decoded scores {s:.6f}, "
                             f"final_check scored {check_spell:.6f} -- the export does not reproduce the run")
    ins = _inside(halves, n)
    p_dead = np.asarray(P[final_method], dtype=np.float64)
    nan_dead = int(np.isnan(p_dead[ins]).sum())
    nan_home = int(np.isnan(np.asarray(p_home, dtype=np.float64)[ins]).sum())
    path = Path(root) / cond / f"{spec.name}__s{seed}.npz"
    save_game(root, cond, spec.name, seed, to_codes(gt), fps, source=source, halves=halves,
              p_dead=p_dead, p_dead_asf=P.get("asf"), p_dead_v8=P.get("v8"), p_home=p_home,
              p_home_seeds=p_home_seeds, p_home_v1=p_home_v1, p_bnd=pb_who, p_bnd_dead=PB.get("asf"),
              ball_out=feat["ball_out"].to_numpy(), ball_filled=feat["ball_filled"].to_numpy(),
              feat={c: feat[c].to_numpy() for c in FEAT_COLS}, pred_final=to_codes(lab),
              params_json=json.dumps(asdict(params)), who_alpha=who_alpha,
              extra={"nan_dead_in": nan_dead, "nan_home_in": nan_home})
    return path


def member_p_home(spec, trk, ball_f, filled, who_paths, device, key_extra, cache_dir: Path):
    """(S, n) P(home) of each owner model on its own (rounded to 1e-4, as owner_p_home), cached."""
    from .alive.predict import owner_p_home
    tags = [f"{Path(p).name}:{Path(p).stat().st_size}" if Path(p).exists() else str(p) for p in who_paths]
    key = hashlib.sha1(json.dumps([spec.name, *key_extra, *tags], default=str).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"{spec.name}_{key}.npz"
    if path.exists():
        return np.load(path)["P"]
    out = []
    for p in who_paths:
        ph, _ = owner_p_home([p], trk.players, trk.team_of_slot, ball_f, filled, spec.fps, spec.pitch,
                             spec.halves, device)
        out.append(ph)
    P = np.stack(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, P=P)
    return P


def from_job(a, ctx, spec, source, seed, mode, trk, gt, ball_f, filled, p_home, pb_who, P, PB, rows,
             key_extra):
    """experiment._job's hook (--fusion-export): one file for this game x seed x ball mode."""
    from .alive import experiment
    cond = f"{a.fusion_cond or a.condition}_{mode}"
    final = a.fusion_final
    check = next((r["spellF1"] for r in rows if r["method"] == final and r["variant"] == "hmm"), None)
    if check is None:
        raise SystemExit(f"--fusion-export: no ({final}, hmm) row in this run (--ensemble / --deadball?)")
    who_name, who_paths = ctx["who"]
    seeds = None
    if not a.fusion_no_seeds and len(who_paths) > 1:
        seeds = member_p_home(spec, trk, ball_f, filled, who_paths, ctx["device"], key_extra,
                              experiment.EVIDENCE_CACHE / "fusion_members")
    v1 = None
    if a.fusion_who_v1:
        v1_paths = [experiment.MODELS_DIR / "who_v1.joblib"]
        extra_v1 = list(key_extra[:-1]) + ["who_v1"]           # final_check's who_v1 run -> cache hit
        _, _, v1, _ = experiment.evidence(spec, trk, v1_paths, ctx["ball_model"], ctx["ball_path"],
                                          ctx["device"], extra_v1, not a.no_cache)
    return export(a.fusion_export, cond, spec, source, seed, gt, trk, ball_f, filled, p_home, pb_who, P, PB,
                  ctx["params"], a.who_bnd_alpha, final, check, seeds, v1)


def submission_g(spec, feat, p_dead, p_home, pb_who, who_alpha, params: decode.Params, tags=None,
                 p_dead_asf=None, p_dead_v8=None, p_bnd_dead=None, p_home_seeds=None, p_home_v1=None) -> dict:
    """The fusion input dict for make_submission (BAR-ATH, on the team's machine): the same arrays
    FINAL's decoder reads. Nothing is written to disk."""
    g = {"y": np.zeros(spec.n_frames, np.int8), "fps": float(spec.fps),
         "halves": np.asarray([(a, b) for _, a, b in spec.halves], np.int64),
         "p_dead": np.asarray(p_dead, dtype=np.float64), "p_home": np.asarray(p_home, dtype=np.float64),
         "ball_out": feat["ball_out"].to_numpy(bool), "ball_filled": feat["ball_filled"].to_numpy(bool),
         "params_json": json.dumps(asdict(params)), "who_alpha": float(who_alpha)}
    for c in FEAT_COLS:
        g["feat_" + c] = feat[c].to_numpy()
    for k, v in (("p_bnd", pb_who), ("p_dead_asf", p_dead_asf), ("p_dead_v8", p_dead_v8),
                 ("p_bnd_dead", p_bnd_dead), ("p_home_seeds", p_home_seeds), ("p_home_v1", p_home_v1)):
        if v is not None:
            g[k] = np.asarray(v, dtype=np.float64)
    if tags is not None:
        g["tag_vote"] = tags.vote[:, 0] - tags.vote[:, 1]          # float32, as decode.emissions adds it
    return g
