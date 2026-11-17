"""The last stage on the timeline: ground truth, the submission, and the decoders it was chosen among.

The lanes all come from the same arrays the submission's decoder reads (mqf.fusion_export's input files):

  Ground truth               the game's labels -- val games: the full match; BAR-ATH: the public 5 min
  Submission · stoppage rules   make_submission.FINAL (since 17 Nov, #48): the stoppage decoder below with the
                             lengths chosen on the val games (make_submission.STOPPAGE). On BAR-ATH: the CSV's
                             labels. Compared with the truth when the page opens
  Best so far · no decoder   every frame on its own: the most likely of H / A / D under FINAL's evidence
                             (P(dead) = mean(asf, v8), the owner TCN's P(home), the same emission terms the
                             HMM reads), no temporal model -- what the decoder starts from
  Best so far · HMM decoder  make_submission.FINAL_HMM: the production 3-state HMM on those arrays (H <-> A cost
                             18 x (1 - 0.5 P(boundary)), 10 into / out of D) -- the submission until 16 Nov
  Best so far · runs >= 0.5 s   the same evidence as the two lanes above and ONE rule: every run (H, A or D)
                             lasts at least 0.5 s -- no switch costs, no boundary. Exact: the labelling with the
                             highest total evidence among those whose runs are all >= 0.5 s (semi-Markov Viterbi
                             with a hard minimum duration), per half
  Best so far · runs >= 0.5 s, stoppages >= 2 s   the same evidence and three length rules only: every run
                             >= 0.5 s, every dead-ball run >= 2 s, and a possession (H or A) with a dead ball on
                             BOTH sides >= 2 s. No switch costs, no boundary. Exact per half: a 7-state
                             semi-Markov Viterbi (dead; per team: between two dead balls / after a dead ball and
                             before the other team or the half's end / after the other team or the half's start)
  v8 dead ball · decoder     the production decoder with P(dead) = Italo's v8 alone instead of mean(asf, v8)
                             (final_check's "tcn x v8 P(dead) in our HMM")
  <decoder> (fusion)   optional, --fusion-best: a decoder / config of the fusion suite, e.g. the verdict's
                       PIPELINE pick (not adopted; for looking only)

The model lanes carry the TCN ensemble's P(home) (white line) and P(boundary) (ticks); the submission lane is
compared with the truth when the page opens. On BAR-ATH these are the only lanes (--all-lanes brings back
the old TCN-only lane and every results/*.csv).

    python -m mqf.viewer --val list                                  # val games / columns / seeds exported
    python -m mqf.viewer --val PFF-3822 --column atd_fuzzy --seed 0  # full ground truth, another game
    python -m mqf.make_submission --final --dump-arrays cache/viewer/final_arrays.npz --out cache/viewer/final.csv
    python -m mqf.viewer                                             # BAR-ATH: picks those arrays up

Val games: the pitch shows the tracking the models read in that column (mqf.alive.experiment.load_condition:
noise, ball mode, cleaner -- the ball as tracked, not filled). There is no video for them. Scores in the lane
notes are the official building blocks on that one file. On BAR-ATH no score is shown: the public 5 min
are for correctness only (RULES.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .game import Lane, Log, Video, ViewerGame, team_names

LAB = np.array(["H", "A", "D", "X"])
RAW, FIN = "Best so far · no decoder", "Best so far · HMM decoder"
SUB = "Submission · stoppage rules"
MIN, V8_DEC = "Best so far · runs ≥ 0.5 s", "v8 dead ball · decoder"
STOP = "Best so far · runs ≥ 0.5 s, stoppages ≥ 2 s"
MIN_RUN_S = 0.5
MIN_DEAD_S = 2.0                # a dead-ball stoppage
MIN_BETWEEN_S = 2.0             # a possession with a dead ball before AND after it


def load_best(path, row: str = "PIPELINE") -> dict:
    """{decoder, config, label} from a fusion _best.json (row: PIPELINE, hsmm_soft, ...) or a run_fusion
    _adopt.json."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    e = d if "decoder" in d and "config" in d else d[row]
    held = e.get("held_out_spellF1")
    lab = f"{e['decoder']} (fusion" + (f", held-out {held:.3f})" if held is not None else ")")
    return {"decoder": e["decoder"], "config": e["config"], "label": lab}


def _note(codes, fps, y=None) -> str:
    from .. import decode
    lab = LAB[np.asarray(codes, dtype=np.int64)]
    s = decode.spell_stats(lab, fps)
    g = lab[lab != "X"]
    share = {k: float(np.mean(g == k)) if len(g) else 0.0 for k in "HAD"}
    txt = (f"{s['possessions']} possessions, median {s['median_s']:.1f} s; H {share['H']:.1%} / A {share['A']:.1%} / "
           f"D {share['D']:.1%}")
    if y is not None:
        from .. import fusion
        sc = fusion.score(y, codes)
        txt += (f"; this file: spellF1 {sc['spellF1']:.3f}, macroF1 {sc['macroF1']:.3f}, "
                f"spells {sc['cnt_err']:+d} vs truth")
    return txt


def with_v8(g: dict) -> dict:
    """The same arrays with P(dead) = v8 alone (what FINAL decodes with mean(asf, v8))."""
    return {**g, "p_dead": np.asarray(g["p_dead_v8"], dtype=np.float64)}


def min_run_decode(g: dict, min_s: float = MIN_RUN_S, cfg: dict | None = None) -> np.ndarray:
    """Codes: per-frame evidence (FINAL's emission terms), no switch costs, every run at least min_s long --
    the exact best labelling under that one constraint, per half (mqf.fusion's "minrun")."""
    from .. import fusion
    return fusion.decode("minrun", g, {**(cfg or {}), "min_s": min_s})


def stoppage_path(E3, m: int, m_dead: int, m_between: int) -> np.ndarray:
    from .. import fusion
    return fusion.stoppage_path(E3, m, m_dead, m_between)


def stoppage_decode(g: dict, min_s: float = MIN_RUN_S, min_dead_s: float = MIN_DEAD_S,
                    min_between_s: float = MIN_BETWEEN_S, cfg: dict | None = None) -> np.ndarray:
    """Codes of the submission's decoder (mqf.fusion's "stoppage"): FINAL's evidence, no switch costs, and three
    length rules -- runs >= 0.5 s, dead balls >= 2 s, a possession between two dead balls >= 2 s."""
    from .. import fusion
    return fusion.decode("stoppage", g, {**(cfg or {}), "min_s": min_s, "min_dead_s": min_dead_s,
                                         "min_between_s": min_between_s})


def lanes(g: dict, best: dict | None = None, y=None) -> list[Lane]:
    """The model lanes of one fusion input dict: no decoder, FINAL, runs >= 0.5 s (same evidence), v8 + decoder,
    [fusion best]. y: score them (val games)."""
    from .. import fusion, fusion_export
    fps = float(g["fps"])
    raw = fusion.decode("argmax", g, fusion.CENTRE).astype(np.uint8)
    from ..make_submission import FINAL
    fin = fusion_export.prod(g).astype(np.uint8)                   # FINAL_HMM's decoder (== pred_final up to 16 Nov)
    dec, cfg = FINAL["options"]["fusion_decoder"], FINAL["options"]["fusion_config"]
    if "pred_final" in g and str(g.get("submission_decoder", "")) == dec:
        sub = np.asarray(g["pred_final"]).astype(np.uint8)          # the CSV's labels (make_submission --dump-arrays)
    else:
        sub = fusion.decode(dec, g, cfg).astype(np.uint8)
    ph, pb = g.get("p_home"), g.get("p_bnd")
    out = [Lane(SUB, "model", sub, "runs ≥ {min_s:g} s · dead balls ≥ {min_dead_s:g} s · a possession between two "
                                   "dead balls ≥ {min_between_s:g} s · lengths chosen on the val games".format(**cfg),
                p_home=ph, note=_note(sub, fps, y), compare=True),
           Lane(RAW, "model", raw, "per-frame argmax · TCN x3 + mean(asf, v8)", p_home=ph, note=_note(raw, fps, y)),
           Lane(FIN, "model", fin, "production HMM · 18 × (1 − 0.5 P(bnd)), 10 / 10", p_home=ph, p_boundary=pb,
                note=_note(fin, fps, y))]
    mr = min_run_decode(g).astype(np.uint8)
    out.append(Lane(MIN, "model", mr, f"same evidence · every run ≥ {MIN_RUN_S:g} s, nothing else", p_home=ph,
                    note=_note(mr, fps, y)))
    st = stoppage_decode(g).astype(np.uint8)
    out.append(Lane(STOP, "model", st, f"runs ≥ {MIN_RUN_S:g} s · dead balls ≥ {MIN_DEAD_S:g} s · a possession "
                                       f"between two dead balls ≥ {MIN_BETWEEN_S:g} s · nothing else",
                    p_home=ph, note=_note(st, fps, y)))
    if "p_dead_v8" in g:
        dec = fusion.decode("hmm3_bnd", with_v8(g), fusion.CENTRE).astype(np.uint8)
        out.append(Lane(V8_DEC, "model", dec, "P(dead) = v8 alone · same production HMM", p_home=ph, p_boundary=pb,
                        note=_note(dec, fps, y)))
    if best:
        codes = fusion.decode(best["decoder"], g, best["config"]).astype(np.uint8)
        out.append(Lane(best["label"], "model", codes, "fusion suite · not adopted", p_home=ph, p_boundary=pb,
                        note=_note(codes, fps, y)))
    return out


def arrays_lanes(path, n_frames: int, best: dict | None = None, log: Log = print) -> tuple[list[Lane], str]:
    """BAR-ATH: lanes from make_submission --dump-arrays. Never raises (the viewer must open anyway)."""
    from .. import fusion
    try:
        g = fusion.load_npz(path)
        if len(g["p_dead"]) != n_frames:
            return [], f"fusion lanes skipped: {Path(path).name} has {len(g['p_dead'])} frames, the match {n_frames}"
        out = lanes(g, best)
    except Exception as e:                                   # never fatal
        return [], f"fusion lanes skipped: {type(e).__name__}: {e}"
    for lane in out:
        log(f"{lane.name}: {lane.note}")                     # aggregates only
    return out, ""


def val_files(preds_dir) -> list[dict]:
    from .. import fusion
    return fusion.discover(preds_dir)


def val_list(preds_dir) -> str:
    files = val_files(preds_dir)
    by = {}
    for f in files:
        by.setdefault(f["game"], set()).add(f"{f['cond']} s{f['seed']}")
    return "\n".join(f"  {g}: " + ", ".join(sorted(v)) for g, v in sorted(by.items()))


def _tracking(game: str, column: str, seed: int, log: Log):
    """(spec, trk, gt) as final_check's run read them, or None (no bundle here)."""
    from ..alive import experiment, final_check
    from ..external import bundle
    from ..external.evaluate import parse_noise
    ds = bundle.find(names=[game])
    if not ds:
        return None
    cname, _, mode = column.rpartition("_")
    args = final_check.CONDITIONS.get(cname)
    if args is None:
        raise ValueError(f"column {column!r}: condition {cname!r} not in final_check.CONDITIONS")
    condition = args[args.index("--condition") + 1]
    noise = []
    if "--noise" in args:
        i = args.index("--noise") + 1
        while i < len(args) and not args[i].startswith("--"):
            noise.append(args[i])
            i += 1
    log(f"tracking: {ds[0].name}, {column}, noise seed {seed} (as final_check read it, cleaner on)")
    return experiment.load_condition(ds[0], condition, seed, mode, True, parse_noise(noise) if noise else None)


def val_game(preds_dir, game: str, column: str = "atd_fuzzy", seed: str = "0", best: dict | None = None,
             log: Log = print) -> ViewerGame:
    """A val game with its full ground truth and the fusion lanes, from harness_results/fusion_preds."""
    from .. import fusion
    from ..fusion_export import to_codes
    f = next((f for f in val_files(preds_dir) if f["game"] == game and f["cond"] == column and f["seed"] == str(seed)),
             None)
    if f is None:
        raise SystemExit(f"no {column}/{game}__s{seed}.npz in {preds_dir}; available:\n{val_list(preds_dir)}")
    g = fusion.load_npz(f["path"])
    n, fps = len(g["y"]), float(g["fps"])
    halves = tuple((k + 1, int(a), int(b)) for k, (a, b) in enumerate(np.asarray(g["halves"]).reshape(-1, 2)))
    notes = []
    loaded = None
    try:
        loaded = _tracking(game, column, int(seed), log)
    except Exception as e:                                   # the lanes still open without the pitch
        notes.append(f"tracking skipped: {type(e).__name__}: {e}")
    home, away, pitch = "Home", "Away", (105.0, 68.0)
    players = np.full((n, 1, 2), np.nan, dtype=np.float32)
    team, ball = np.zeros(1, dtype=int), np.full((n, 2), np.nan, dtype=np.float32)
    if loaded is not None:
        spec, trk, gt = loaded
        if spec.n_frames != n or not np.array_equal(to_codes(gt), g["y"]):
            notes.append("tracking skipped: the bundle's labels are not this file's ground truth")
        else:
            home, away = team_names(spec.atd_meta)
            pitch, players, team, ball = spec.pitch, trk.players, trk.team_of_slot, trk.ball
    elif not notes:
        notes.append(f"no bundle for {game} under external/bundles: lanes only, empty pitch")
    truth = Lane("Ground truth", "truth", np.asarray(g["y"], dtype=np.uint8), f"{g.get('source', '')} · full match")
    model = lanes(g, best, g["y"])
    for lane in model:
        log(f"{lane.name}: {lane.note}")
    video = Video("tactical", "Tactical", None, fps, n,
                  note=f"No video for the val games. The pitch shows the tracking the models read ({column}, "
                       f"noise seed {seed}).")
    return ViewerGame(key=f"{game}-{column}-s{seed}", title=f"{game} · {column} · seed {seed}", home=home, away=away,
                      fps=fps, n_frames=n, pitch=tuple(pitch), halves=halves, players=np.asarray(players),
                      team_of_slot=np.asarray(team), ball=np.asarray(ball), lanes=[truth, *model],
                      videos=[video], notes=notes)
