"""Another game, written in the tactical data definition of the competition (DATA.md) -- as is.

A bundle is what the pipeline reads for BAR-ATH, so `atd.load`, `make_submission.run`,
`mqf.profile`, the viewer and the official scorer work on it unchanged:

    external/bundles/<GAME>/
      <GAME>_tracking_meta.xml    FIFA/EPTS-style metadata: 2 <Team> (first = home), one <Player> per slot
      <GAME>_tracking_raw.txt     "<frame_id>:<x,y,s>;...;<x,y,s>:<bx,by,bs>", normalised 0-1, NaN missing,
                                  frame_id from 1 (video_frame = frame_id - 1), one record per frame
      <GAME>_possession_gt.csv    frame,label -- H/A/D inside the halves, X outside
      <GAME>_game.json            frames, fps, pitch, halves, teams, provenance (what games.py says for BAR-ATH)

No noise and no resampling: positions, ball, missing values, player identities (one slot per
player, not 46 anonymous tracklets), frame rate and pitch size are the source's own (PFF: 29.97 fps, stored as a float). The only
things not in a source are format requirements: coordinates normalised by the source's pitch, and
the per-slot speed field `s`, which the ATD carries and our features read -- when a source has no
speed it is the centred difference of the source's own positions over 0.4 s (NaN at gaps).

File names avoid the patterns reserved for the real data (*_atd.*, *_homography.csv,
*_smart_tagging.xml), so bundles of other games stay inspectable.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from ..config import EXTERNAL_DIR, GameSpec

BUNDLES_DIR = EXTERNAL_DIR / "bundles"
SPEED_WINDOW_S = 0.4
VERSION = 1


def speed_from_positions(xy: np.ndarray, fps: int, window_s: float = SPEED_WINDOW_S) -> np.ndarray:
    """|centred difference| over +/- window/2, m/s, per slot; NaN wherever an end point is missing."""
    w = max(1, int(round(window_s * fps / 2)))
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    if len(xy) > 2 * w:
        out[w:-w] = np.linalg.norm(xy[2 * w:] - xy[:-2 * w], axis=-1) * fps / (2 * w)
    out[np.isnan(xy[..., 0])] = np.nan
    return out


def _groups(xn: np.ndarray, yn: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(n,) strings "x,y,s" with 5 / 5 / 2 decimals; "NaN,NaN,NaN" where x is missing."""
    xs = np.char.mod("%.5f", np.nan_to_num(xn))
    ys = np.char.mod("%.5f", np.nan_to_num(yn))
    ss = np.where(np.isnan(s), "NaN", np.char.mod("%.2f", np.nan_to_num(s)))
    g = np.char.add(np.char.add(np.char.add(np.char.add(xs, ","), ys), ","), ss)
    return np.where(np.isnan(xn), "NaN,NaN,NaN", g)


def write_raw(path: Path, players: np.ndarray, speed: np.ndarray, ball: np.ndarray, pitch) -> None:
    L, W = pitch
    n, P, _ = players.shape
    body = None
    for k in range(P):
        g = _groups(players[:, k, 0] / L, players[:, k, 1] / W, speed[:, k])
        body = g if body is None else np.char.add(np.char.add(body, ";"), g)
    bg = _groups(ball[:, 0] / L, ball[:, 1] / W, np.full(n, np.nan))
    fid = np.arange(1, n + 1).astype(str)
    lines = np.char.add(np.char.add(np.char.add(np.char.add(fid, ":"), body), ":"), bg)
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines.tolist()))
        fh.write("\n")


def write_meta(path: Path, home: str, away: str, team_of_slot: np.ndarray, fps: int, pitch,
               slot_names=None) -> None:
    names = slot_names or [f"Track {i + 1}" for i in range(len(team_of_slot))]
    players = "\n".join(
        f'      <Player id="P{i + 1}" teamId="{"T1" if t == 0 else "T2"}">'
        f"<Name>{escape(str(names[i]))}</Name><ShirtNumber>{i + 1}</ShirtNumber></Player>"
        for i, t in enumerate(team_of_slot))
    path.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig><FrameRate>{fps}</FrameRate></GlobalConfig>
    <Sessions><Session><MatchParameters><FieldSize><Width>{pitch[0]:g}</Width><Height>{pitch[1]:g}</Height></FieldSize></MatchParameters></Session></Sessions>
    <Teams>
      <Team id="T1"><Name>{escape(home)}</Name></Team>
      <Team id="T2"><Name>{escape(away)}</Name></Team>
    </Teams>
    <Players>
{players}
    </Players>
  </Metadata>
</main>
""", encoding="utf-8")


def write_bundle(m: dict, root: Path = BUNDLES_DIR) -> Path:
    """m: players (n,P,2) m from the pitch corner, team_of_player (P,), ball (n,2), labels (n,),
    halves ((half, first, last), ...) inclusive, pitch (L, W), fps, name, home, away, source,
    labels_from, licence; optional speed (n,P) and slot_names."""
    name = m["name"]
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    players = np.asarray(m["players"], dtype=np.float64)
    ball = np.asarray(m["ball"], dtype=np.float64)
    labels = np.asarray(m["labels"]).astype("<U1")
    n = len(labels)
    assert players.shape[0] == ball.shape[0] == n, "players, ball and labels must cover the same frames"
    assert set(np.unique(labels)) <= set("HADX")
    speed = m.get("speed")
    speed = speed_from_positions(players, m["fps"]) if speed is None else np.asarray(speed, dtype=np.float32)
    team = np.asarray(m["team_of_player"], dtype=np.int8)

    files = {"atd_meta": f"{name}_tracking_meta.xml", "atd_raw": f"{name}_tracking_raw.txt",
             "gt": f"{name}_possession_gt.csv"}
    write_meta(out / files["atd_meta"], m["home"], m["away"], team, m["fps"], m["pitch"], m.get("slot_names"))
    write_raw(out / files["atd_raw"], players, speed, ball, m["pitch"])
    with open(out / files["gt"], "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.write("\n".join(f"{i},{lab}" for i, lab in enumerate(labels)))
        fh.write("\n")
    fps = float(m["fps"])
    info = {"name": name, "version": VERSION, "source": m["source"], "fps": int(fps) if fps.is_integer() else fps,
            "n_frames": int(n),
            "pitch": [float(v) for v in m["pitch"]], "halves": [[int(v) for v in h] for h in m["halves"]],
            "home": m["home"], "away": m["away"], "n_slots": int(len(team)),
            "labels_from": m["labels_from"], "licence": m.get("licence", ""),
            "speed_field": m.get("speed_from", "centred difference of the source positions, 0.4 s"
                                 if m.get("speed") is None else "source"),
            "files": files, "qa": qa(m)}
    (out / f"{name}_game.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return out


def spec(bundle_dir: Path) -> GameSpec:
    bundle_dir = Path(bundle_dir)
    info = json.loads(next(bundle_dir.glob("*_game.json")).read_text(encoding="utf-8"))
    f = info["files"]
    return GameSpec(name=info["name"], n_frames=info["n_frames"], fps=info["fps"], pitch=tuple(info["pitch"]),
                    halves=tuple(tuple(h) for h in info["halves"]), atd_meta=bundle_dir / f["atd_meta"],
                    atd_raw=bundle_dir / f["atd_raw"], smart_tagging=None, gt_public=bundle_dir / f["gt"])


def find(source: str | None = None, names=None, root: Path = BUNDLES_DIR) -> list[Path]:
    """Bundle folders, optionally of one source ("idsse" / "metrica" / "skillcorner") or named games
    (full name "SKC-1874553" or the id "1874553")."""
    out = []
    for d in sorted(Path(root).glob("*/")):
        info = list(d.glob("*_game.json"))
        if not info:
            continue
        meta = json.loads(info[0].read_text(encoding="utf-8"))
        if source and meta.get("source") != source:
            continue
        if names is not None and d.name not in names and d.name.split("-", 1)[-1] not in names:
            continue
        out.append(d)
    return out


def qa(m: dict) -> dict:
    """Label and tracking aggregates of one converted game (other games only -- safe to print)."""
    labels, fps = np.asarray(m["labels"]), m["fps"]
    graded = np.zeros(len(labels), dtype=bool)
    for _, a, b in m["halves"]:
        graded[a:b + 1] = True
    lab = labels[graded]
    spells = []
    for _, a, b in m["halves"]:
        seg = labels[a:b + 1]
        edges = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        for s, e in zip(np.r_[0, edges], np.r_[edges, len(seg)]):
            if seg[s] in ("H", "A"):
                spells.append((e - s) / fps)
    spells = np.array(spells) if spells else np.zeros(1)
    ball_seen = ~np.isnan(np.asarray(m["ball"])[:, 0])
    alive = np.isin(labels, ["H", "A"]) & graded
    dead = (labels == "D") & graded
    n_obj = (~np.isnan(np.asarray(m["players"])[graded, :, 0])).sum(axis=1)
    return {
        "graded_frames": int(graded.sum()), "graded_minutes": round(graded.sum() / fps / 60, 1),
        "share": {k: round(float(np.mean(lab == k)), 3) for k in "HADX"},
        "possessions": int(len(spells)), "possessions_per_90min": round(len(spells) / (graded.sum() / fps / 60) * 90, 1),
        "mean_s": round(float(spells.mean()), 2), "median_s": round(float(np.median(spells)), 2),
        "share_under_3s": round(float(np.mean(spells < 3)), 3),
        "ball_seen": {"alive": round(float(ball_seen[alive].mean()), 3) if alive.any() else None,
                      "dead": round(float(ball_seen[dead].mean()), 3) if dead.any() else None},
        "players_tracked_mean": round(float(n_obj.mean()), 1) if len(n_obj) else None,
    }
