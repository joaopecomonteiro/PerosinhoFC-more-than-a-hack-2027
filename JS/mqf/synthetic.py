"""A synthetic match in the EXACT file formats of the real bundle, with known ground truth.

Why: the competition terms forbid sending the real data (or extracts) to any cloud-hosted LLM.
This generator lets the whole pipeline be built, tested and debugged -- by you or by an AI
assistant -- without the real files ever leaving your machine. It is NOT training data for the
real match and its numbers mean nothing about the leaderboard.

Mimics the documented noise: ~40% ball missing (more in duels), 46 track slots for 22 players
with fragmentation, players dropping out, Smart Tagging with jittered/wrong/missing phases and
'N/A' teams on some set pieces.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GameSpec
from .harness import noise

HOME_NAME, AWAY_NAME = "FC Sintetic", "Athletic Sintetico"
TAG_NAMES = ("Sintetic", "Atletic Sintetico")      # deliberately not identical, like the real file
FORMATION = np.array([[5, 34], [20, 10], [18, 26], [18, 42], [20, 58], [35, 20], [33, 34],
                      [35, 48], [50, 12], [52, 34], [50, 56]], dtype=float)


def _segments(rng, first, last, fps, lead_dead_s=None):
    """[(start, end_exclusive, label)] covering [first, last]; label in 'HAD'.

    lead_dead_s: length of the opening stoppage. The real 2nd half is graded ~2 min before its
    kickoff (all D); the generator reproduces that shape so stoppage detection is exercised.
    """
    segs, t = [], first
    d = int((lead_dead_s if lead_dead_s is not None else rng.uniform(2, 6)) * fps)
    segs.append((t, t + d, "D")); t += d
    team = rng.choice(["H", "A"])
    while t <= last:
        d = int(np.clip(rng.lognormal(np.log(10), 0.8), 1.5, 70) * fps)
        segs.append((t, min(t + d, last + 1), team)); t += d
        if t > last:
            break
        if rng.random() < 0.3:
            d = int(rng.uniform(3, 25) * fps)
            segs.append((t, min(t + d, last + 1), "D")); t += d
            team = rng.choice(["H", "A"])
        else:
            team = "A" if team == "H" else "H"
    return segs


def _ar_noise(rng, n, k, scale, rho=0.995):
    e = rng.normal(0, scale * np.sqrt(1 - rho ** 2), size=(n, k, 2))
    out = np.empty_like(e)
    out[0] = rng.normal(0, scale, size=(k, 2))
    for t in range(1, n):
        out[t] = rho * out[t - 1] + e[t]
    return out


def clean_match(n_frames: int = 45_000, fps: int = 25, seed: int = 7,
                halves=((1, 1_000, 21_999), (2, 23_500, 44_499)), pitch=(105.3, 68.0)) -> dict:
    """A clean match: 22 identified players, full ball, per-frame truth. Input to the noise model."""
    rng = np.random.default_rng(seed)
    L, W = pitch
    gt = np.full(n_frames, "X", dtype="<U1")
    ball = np.full((n_frames, 2), np.nan)
    pos = np.full((n_frames, 22, 2), np.nan)
    all_segs = []

    for half, first, last in halves:
        segs = _segments(rng, first, last, fps, lead_dead_s=rng.uniform(100, 130) if half == 2 else None)
        all_segs += [(half, *s) for s in segs]
        n = last - first + 1
        flip = 1 if half == 1 else -1          # home attacks right in the 1st half
        noise = _ar_noise(rng, n, 22, 3.0)
        b = np.zeros((n, 2)); b[0] = (L / 2, W / 2)
        carrier = rng.integers(0, 11)
        for s, e, lab in segs:
            gt[s:e] = lab
            i0, i1 = s - first, e - first
            if lab == "D":
                spot = b[i0 - 1] if i0 > 0 else b[0]
                if rng.random() < 0.5:   # out of play over a line
                    spot = np.array([rng.choice([-1.5, L + 1.5]), rng.uniform(0, W)]) \
                        if rng.random() < 0.3 else np.array([rng.uniform(0, L), rng.choice([-1.5, W + 1.5])])
                b[i0:i1] = spot
                continue
            team_off = 0 if lab == "H" else 11
            t = i0
            while t < i1:
                hold = int(rng.uniform(1.0, 3.0) * fps)
                nxt = rng.integers(0, 11)
                fl = int(rng.uniform(0.4, 1.0) * fps)
                anchor = _anchor(team_off + carrier, b[t - 1] if t > 0 else b[0], flip, L, W)
                tgt = _anchor(team_off + nxt, anchor, flip, L, W)
                b[t:min(t + hold, i1)] = anchor + rng.normal(0, 0.3, 2)
                tt = min(t + hold, i1)
                m = min(fl, i1 - tt)
                if m > 0:
                    a = np.linspace(0, 1, m)[:, None]
                    b[tt:tt + m] = anchor * (1 - a) + tgt * a
                t, carrier = tt + m, nxt
        # players: formation shifted with the ball, plus smooth noise; carrier pinned near ball
        for k in range(22):
            base = FORMATION[k % 11].copy()
            if (k >= 11) ^ (flip == -1):
                base[0] = L - base[0]; base[1] = W - base[1]
            pos[first:last + 1, k] = base + 0.35 * (b - [L / 2, W / 2]) + noise[:, k]
        live = np.isin(gt[first:last + 1], ["H", "A"])
        for t in np.flatnonzero(live):
            team_off = 0 if gt[first + t] == "H" else 11
            d = np.linalg.norm(pos[first + t, team_off:team_off + 11] - b[t], axis=1)
            j = team_off + int(np.argmin(d))
            pos[first + t, j] = b[t] + rng.normal(0, 0.6, 2)
        dead = ~live
        pos[first:last + 1][dead] = pos[first:last + 1][dead] * 0.98 + 0.02 * pos[first:last + 1][dead].mean(0)
        ball[first:last + 1] = b

    # smooth stoppage movement (players slow down when the ball is dead)
    for half, first, last in halves:
        sl = slice(first, last + 1)
        dead = gt[sl] == "D"
        p = pos[sl]
        for t in np.flatnonzero(dead)[1:]:
            p[t] = p[t - 1] + 0.15 * (p[t] - p[t - 1])
        # Physical smoothness: the AR drift and the carrier pinning above are rough at frame level
        # (~10 m/s of apparent speed). A 1 s centred mean brings mean speed to football values
        # (~2-3 m/s in play, < 1 m/s at stoppages) so stillness-based stoppage cues are testable.
        flat = pd.DataFrame(p.reshape(len(p), -1))
        pos[sl] = flat.rolling(fps, center=True, min_periods=1).mean().to_numpy().reshape(p.shape)

    return {"players": pos, "team_of_player": np.repeat(np.array([0, 1], dtype=np.int8), 11),
            "ball": ball, "labels": gt, "halves": tuple(halves), "pitch": tuple(pitch), "fps": fps,
            "segments": all_segs, "seed": seed}


def generate(out_dir: Path, n_frames: int = 45_000, fps: int = 25, seed: int = 7,
             halves=((1, 1_000, 21_999), (2, 23_500, 44_499)),
             pitch=(105.3, 68.0), force: bool = False) -> GameSpec:
    """Synthetic match written in the EXACT file formats of the real bundle, plus its truth."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    spec = GameSpec(name="SYN", n_frames=n_frames, fps=fps, pitch=pitch, halves=tuple(halves),
                    atd_meta=out_dir / "SYN_tracking_meta.xml", atd_raw=out_dir / "SYN_tracking_raw.txt",
                    smart_tagging=out_dir / "SYN_tags.xml",
                    gt_public=out_dir / "SYN_possession_gt.csv")
    # regenerate whenever the generator, the noise model or the arguments change (stale files
    # would silently test an old version)
    stamp = out_dir / "SYN_stamp.txt"
    key = hashlib.md5(Path(__file__).read_bytes() + Path(noise.__file__).read_bytes()
                      + repr((n_frames, fps, seed, tuple(halves), tuple(pitch))).encode()).hexdigest()
    if (spec.atd_raw.exists() and spec.gt_public.exists() and stamp.exists()
            and stamp.read_text() == key and not force):
        return spec
    m = clean_match(n_frames, fps, seed, halves, pitch)
    trk = noise.corrupt(m["players"], m["team_of_player"], m["ball"], m["labels"] != "X", fps,
                        noise.NoiseConfig(seed=seed))
    rng = np.random.default_rng(seed + 1)
    _write_meta(spec.atd_meta)
    _write_raw(spec.atd_raw, trk.players, trk.player_speed, trk.ball, pitch)
    _write_tags(spec.smart_tagging, rng, m["segments"], fps)
    with open(spec.gt_public, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{lab}\n" for i, lab in enumerate(m["labels"]))
    stamp.write_text(key)
    return spec


def _anchor(k, ball, flip, L, W):
    base = FORMATION[k % 11].copy()
    if (k >= 11) ^ (flip == -1):
        base[0] = L - base[0]; base[1] = W - base[1]
    return np.clip(base + 0.35 * (ball - [L / 2, W / 2]), [1, 1], [L - 1, W - 1])


def _write_meta(path):
    players = "\n".join(
        f'    <Player id="P{i + 1}" teamId="{"T1" if i < 23 else "T2"}">'
        f"<Name>Track {i + 1}</Name><ShirtNumber>{i + 1}</ShirtNumber></Player>"
        for i in range(46))
    path.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <Teams>
      <Team id="T1"><Name>{HOME_NAME}</Name></Team>
      <Team id="T2"><Name>{AWAY_NAME}</Name></Team>
    </Teams>
    <Players>
{players}
    </Players>
  </Metadata>
</main>
""")


def _fmt(x, y, s):
    if np.isnan(x):
        return "NaN,NaN,NaN"
    return f"{x:.5f},{y:.5f},{'NaN' if np.isnan(s) else f'{s:.2f}'}"


def _write_raw(path, xy, sp, ball, pitch):
    L, W = pitch
    with open(path, "w") as fh:
        for t in range(len(xy)):
            groups = ";".join(_fmt(xy[t, k, 0] / L, xy[t, k, 1] / W, sp[t, k]) for k in range(xy.shape[1]))
            bx, by = ball[t]
            fh.write(f"{t + 1}:{groups}:{_fmt(bx / L, by / W, 0.0)}\n")


def _write_tags(path, rng, segs, fps):
    rows, iid = [], 0
    for half, s, e, lab in segs:
        hn = "1st Half" if half == 1 else "2nd Half"
        if lab == "D":
            iid += 1
            team = "N/A" if rng.random() < 0.35 else rng.choice(TAG_NAMES)
            rows.append((iid, "SET PIECES", (e - 2 * fps) / fps, (e + 3 * fps) / fps,
                         {"Team": team, "Half": hn, "Type": "Throw-in"}))
            continue
        if (e - s) < 5 * fps or rng.random() < 0.2:
            continue
        side = 0 if lab == "H" else 1
        if rng.random() < 0.1:
            side = 1 - side
        iid += 1
        jit = rng.normal(0, 2 * fps, 2)
        rows.append((iid, rng.choice(["BUILD UP", "PROGRESSION", "CREATING CHANCES"]),
                     (s + jit[0]) / fps, (e + jit[1]) / fps, {"Team": TAG_NAMES[side], "Half": hn}))
    body = []
    for iid, code, st, en, tags in rows:
        labels = "".join(f"<label><group>{g}</group><text>{v}</text></label>" for g, v in tags.items())
        body.append(f"<instance><ID>{iid}</ID><start>{max(st, 0):.2f}</start><end>{max(en, 0):.2f}</end>"
                    f"<code>{code}</code>{labels}</instance>")
    path.write_text('<?xml version="1.0" encoding="utf-8"?>\n<file><ALL_INSTANCES>\n'
                    + "\n".join(body) + "\n</ALL_INSTANCES></file>\n")
