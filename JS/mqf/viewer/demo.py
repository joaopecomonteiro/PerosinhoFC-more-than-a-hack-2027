"""Synthetic videos for the viewer: try every view, and the broadcast sync, without the real data.

    python -m mqf.viewer.demo                    # tracking + homographies + videos (needs ffmpeg)
    python -m mqf.viewer.demo --no-video         # tracking + homographies only
    python -m mqf.viewer --synthetic             # then open it

From the synthetic match (mqf/synthetic.py) it writes, into synthetic_data/ (gitignored):
  * a tactical video: a fixed wide camera, rendered from the clean positions, with its homography;
  * a broadcast feed on its own timeline: a camera that follows the ball, a studio intro, a
    recording cut that shifts the offset by 40 frames, halftime, replays during stoppages and
    close-ups -- with its own tracking (only what the camera sees) and homography, in the same file
    formats as the real bundle, plus the true broadcast -> match map (`SYN_broadcast_truth.npz`)
    so the sync can be tested.

Every frame carries its index as an 18-bit strip in the top-left corner (white = 1), which lets a
test read back exactly which frame the browser is showing. File names avoid the patterns that
.claude/settings.json reserves for the real data.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..config import SYNTHETIC_DIR, GameSpec

HOME_RGB, AWAY_RGB, BALL_RGB = (61, 126, 255), (240, 72, 72), (255, 255, 255)
GRASS, STANDS, LINE = (22, 74, 40), (28, 30, 32), (226, 232, 226)
N_SLOTS_B = 51
BITS = 18
KIND_STUDIO, KIND_LIVE, KIND_CUTAWAY, KIND_REPLAY = 0, 1, 2, 3


def paths(out_dir: Path = SYNTHETIC_DIR) -> dict:
    d = Path(out_dir)
    return {"tactical_video": d / "SYN_tactical.mp4", "tactical_hom": d / "SYN_tactical_hom.csv",
            "broadcast_video": d / "SYN_broadcast.mp4", "broadcast_hom": d / "SYN_broadcast_hom.csv",
            "broadcast_raw": d / "SYN_broadcast_tracking_raw.txt",
            "broadcast_meta": d / "SYN_broadcast_tracking_meta.xml",
            "broadcast_truth": d / "SYN_broadcast_truth.npz"}


def broadcast_spec(out_dir: Path, spec: GameSpec) -> GameSpec | None:
    f = paths(out_dir)
    if not (f["broadcast_raw"].is_file() and f["broadcast_truth"].is_file()):
        return None
    nb = int(np.load(f["broadcast_truth"])["n_frames"])
    return GameSpec(name="SYN-broadcast", n_frames=nb, fps=spec.fps, pitch=spec.pitch, halves=(),
                    atd_meta=f["broadcast_meta"], atd_raw=f["broadcast_raw"])


# ---------------------------------------------------------------------------------- cameras

def homography_from_quad(src_norm, dst_px) -> np.ndarray:
    """3x3 (row-major, h22 = 1) mapping 4 normalised pitch points onto 4 pixels (DLT)."""
    A, rhs = [], []
    for (x, y), (u, v) in zip(src_norm, dst_px):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); rhs.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); rhs.append(v)
    h = np.linalg.solve(np.array(A, float), np.array(rhs, float))
    return np.r_[h, 1.0].reshape(3, 3)


def tactical_homography(size) -> np.ndarray:
    W, H = size
    return homography_from_quad([(0, 0), (1, 0), (1, 1), (0, 1)],
                                [(0.17 * W, 0.20 * H), (0.83 * W, 0.20 * H), (0.97 * W, 0.94 * H), (0.03 * W, 0.94 * H)])


def broadcast_homography(cx, w_near, w_far, pitch, size) -> np.ndarray:
    """A camera on the near touchline (y = W) aimed at x = cx: the far touchline at the top of the
    picture, the near one at the bottom, a wider stretch of the far side than the near side."""
    L, W = pitch
    Wi, Hi = size
    return homography_from_quad([((cx - w_far / 2) / L, 0), ((cx + w_far / 2) / L, 0),
                                 ((cx + w_near / 2) / L, 1), ((cx - w_near / 2) / L, 1)],
                                [(0, 0.10 * Hi), (Wi, 0.10 * Hi), (Wi, Hi), (0, Hi)])


def project(H: np.ndarray, xy_m: np.ndarray, pitch) -> np.ndarray:
    L, W = pitch
    p = np.c_[xy_m[:, 0] / L, xy_m[:, 1] / W, np.ones(len(xy_m))] @ H.T
    return p[:, :2] / p[:, 2:3]


# ---------------------------------------------------------------------------------- broadcast feed

def make_broadcast(match: dict, size=(960, 540), seed: int = 11) -> dict:
    """Broadcast timeline built from the clean match, with its true link to the match timeline."""
    rng = np.random.default_rng(seed)
    fps, pitch = match["fps"], match["pitch"]
    L, W = pitch
    n_t = len(match["labels"])
    (h1, f1, l1), (h2, f2, l2) = match["halves"][:2]

    # pieces of the recording: (kind, length, first match frame)
    cut = f1 + (l1 - f1) // 2
    pieces = [(KIND_STUDIO, 8 * fps, -1),
              (KIND_LIVE, cut - (f1 - 12 * fps), f1 - 12 * fps),
              (KIND_LIVE, (l1 + 6 * fps) - (cut + 40), cut + 40),      # recording cut: 40 frames lost
              (KIND_STUDIO, 16 * fps, -1),
              (KIND_LIVE, (l2 + 8 * fps) - (f2 - 10 * fps), f2 - 10 * fps),
              (KIND_STUDIO, 6 * fps, -1)]
    to_match, kind = [], []
    for k, n, t0 in pieces:
        to_match.append(np.arange(t0, t0 + n) if k == KIND_LIVE else np.full(n, -1))
        kind.append(np.full(n, k))
    to_match = np.concatenate(to_match).astype(np.int64)
    kind = np.concatenate(kind).astype(np.int8)
    to_match[(to_match < 0) | (to_match >= n_t)] = -1
    nb = len(to_match)
    content = to_match.copy()                       # match frame whose picture is on screen

    # replays and close-ups inside stoppages, a few short close-ups in play
    b_of_t = np.full(n_t, -1, dtype=np.int64)
    live_b = np.flatnonzero(to_match >= 0)
    b_of_t[to_match[live_b]] = live_b
    for half, s, e, lab in match["segments"]:
        if lab != "D" or (e - s) < 8 * fps or b_of_t[s] < 0 or b_of_t[min(e, n_t - 1)] < 0:
            continue
        b0 = b_of_t[s] + fps
        n = min(6 * fps, (e - s) - 2 * fps)
        if rng.random() < 0.65:
            kind[b0:b0 + n] = KIND_REPLAY
            content[b0:b0 + n] = np.arange(s - 6 * fps, s - 6 * fps + n)
        else:
            kind[b0:b0 + 3 * fps] = KIND_CUTAWAY
            content[b0:b0 + 3 * fps] = -1
    for b0 in rng.choice(live_b, size=max(1, len(live_b) // (fps * 240)), replace=False):
        sl = slice(b0, b0 + int(1.5 * fps))
        live = kind[sl] == KIND_LIVE
        kind[sl][live] = KIND_CUTAWAY
        content[sl][live] = -1
    content[(kind == KIND_STUDIO) | (kind == KIND_CUTAWAY)] = -1

    # camera: follows the (smoothed) ball of the frame on screen; tighter on replays
    ball = match["ball"]
    bx = np.where(np.isfinite(ball[:, 0]), ball[:, 0], L / 2)
    cam = np.empty(n_t)
    cam[0] = bx[0]
    a = 1.0 / (1.5 * fps)
    for t in range(1, n_t):
        cam[t] = cam[t - 1] + a * (bx[t] - cam[t - 1])
    hom = np.full((nb, 9), np.nan)
    players = np.full((nb, N_SLOTS_B, 2), np.nan, dtype=np.float32)
    ball_b = np.full((nb, 2), np.nan, dtype=np.float32)
    team = np.r_[np.zeros(25, np.int8), np.ones(N_SLOTS_B - 25, np.int8)]
    slot_of_player = np.r_[np.arange(11), 25 + np.arange(11)]
    jit = rng.normal(0, 0.3, (nb, 22, 2)).astype(np.float32)
    for b in np.flatnonzero(content >= 0):
        c = content[b]
        replay = kind[b] == KIND_REPLAY
        wn, wf = (22.0, 34.0) if replay else (34.0, 52.0)
        cx = float(np.clip(cam[c], wf / 2 - 2, L - wf / 2 + 2))
        Hm = broadcast_homography(cx, wn, wf, pitch, size)
        hom[b] = Hm.reshape(-1)
        pos = match["players"][c]
        if np.isfinite(pos[:, 0]).any():
            uv = project(Hm, np.nan_to_num(pos), pitch)
            vis = np.isfinite(pos[:, 0]) & (uv[:, 0] >= 0) & (uv[:, 0] <= size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] <= size[1])
            players[b, slot_of_player[vis]] = pos[vis] + jit[b, vis]
        if np.isfinite(ball[c, 0]) and rng.random() < 0.6:
            uv = project(Hm, ball[c][None], pitch)[0]
            if 0 <= uv[0] <= size[0] and 0 <= uv[1] <= size[1]:
                ball_b[b] = ball[c] + rng.normal(0, 0.3, 2)
    return {"to_match": to_match, "content": content, "kind": kind, "hom": hom, "players": players,
            "ball": ball_b, "team_of_slot": team, "size": size, "n_frames": nb}


def write_broadcast(out_dir: Path, bc: dict, pitch) -> None:
    from ..synthetic import AWAY_NAME, HOME_NAME, _write_raw
    f = paths(out_dir)
    pl = "\n".join(f'    <Player id="B{i + 1}" teamId="{"T1" if bc["team_of_slot"][i] == 0 else "T2"}">'
                   f"<Name>Track {i + 1}</Name><ShirtNumber>{i + 1}</ShirtNumber></Player>"
                   for i in range(N_SLOTS_B))
    f["broadcast_meta"].write_text(
        f'<?xml version="1.0" encoding="utf-8"?>\n<main>\n  <Metadata>\n    <Teams>\n'
        f'      <Team id="T1"><Name>{HOME_NAME}</Name></Team>\n      <Team id="T2"><Name>{AWAY_NAME}</Name></Team>\n'
        f"    </Teams>\n    <Players>\n{pl}\n    </Players>\n  </Metadata>\n</main>\n")
    _write_raw(f["broadcast_raw"], bc["players"], np.full(bc["players"].shape[:2], np.nan), bc["ball"], pitch)
    write_homography(f["broadcast_hom"], bc["hom"])
    np.savez_compressed(f["broadcast_truth"], to_match=bc["to_match"], content=bc["content"],
                        kind=bc["kind"], n_frames=bc["n_frames"], size=np.array(bc["size"]))


def write_homography(path: Path, hom: np.ndarray) -> None:
    ok = np.isfinite(hom).all(axis=1)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        for i, row in enumerate(hom):
            if ok[i]:
                fh.write(f"{i}," + ",".join(f"{v:.7g}" for v in row) + ",0.25,1\n")
            else:
                fh.write(f"{i}," + ",".join(["NaN"] * 9) + ",NaN,0\n")


# ---------------------------------------------------------------------------------- rendering

def _pitch_lines(pitch):
    """Polylines of the markings, metres."""
    L, W = pitch
    lines = [[(0, 0), (L, 0), (L, W), (0, W), (0, 0)], [(L / 2, 0), (L / 2, W)]]
    a = np.linspace(0, 2 * np.pi, 48)
    lines.append(list(zip(L / 2 + 9.15 * np.cos(a), W / 2 + 9.15 * np.sin(a))))
    for x0, s in ((0, 1), (L, -1)):
        lines.append([(x0, W / 2 - 20.16), (x0 + s * 16.5, W / 2 - 20.16), (x0 + s * 16.5, W / 2 + 20.16), (x0, W / 2 + 20.16)])
        lines.append([(x0, W / 2 - 9.16), (x0 + s * 5.5, W / 2 - 9.16), (x0 + s * 5.5, W / 2 + 9.16), (x0, W / 2 + 9.16)])
    return [np.array(l, float) for l in lines]


def _barcode(draw, value: int, cell: int):
    draw.rectangle([0, 0, cell * (BITS + 2), cell], fill=(0, 0, 0))
    for k in range(BITS):
        if (value >> (BITS - 1 - k)) & 1:
            draw.rectangle([cell * (k + 1), 0, cell * (k + 2) - 1, cell - 1], fill=(255, 255, 255))


def _scene(draw, Hm, pitch, size, players, team, ball, lines, background=True):
    if background:
        corners = project(Hm, np.array([[0, 0], [pitch[0], 0], [pitch[0], pitch[1]], [0, pitch[1]]]), pitch)
        draw.polygon([tuple(p) for p in corners], fill=GRASS)
    for ln in lines:
        uv = project(Hm, ln, pitch)
        draw.line([tuple(p) for p in uv], fill=LINE, width=max(1, size[0] // 640))
    ok = np.isfinite(players[:, 0])
    if ok.any():
        uv = project(Hm, players[ok], pitch)
        for (u, v), tm in zip(uv, team[ok]):
            r = (3 + 6 * v / size[1]) * size[0] / 960
            draw.ellipse([u - r, v - 2 * r, u + r, v], fill=HOME_RGB if tm == 0 else AWAY_RGB)
    if np.isfinite(ball[0]):
        u, v = project(Hm, ball[None], pitch)[0]
        r = 2.5 * size[0] / 960
        draw.ellipse([u - r, v - r, u + r, v + r], fill=BALL_RGB)


class _Encoder:
    def __init__(self, path: Path, size, fps):
        self.proc = subprocess.Popen(
            [shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "26", "-pix_fmt", "yuv420p", "-g", str(fps), "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, img):
        self.proc.stdin.write(img.tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait():
            raise RuntimeError("ffmpeg failed")


def render_tactical(path: Path, match: dict, Hm: np.ndarray, size, log=print) -> None:
    from PIL import Image, ImageDraw
    pitch, fps = match["pitch"], match["fps"]
    lines = _pitch_lines(pitch)
    base = Image.new("RGB", size, STANDS)
    corners = project(Hm, np.array([[0, 0], [pitch[0], 0], [pitch[0], pitch[1]], [0, pitch[1]]]), pitch)
    ImageDraw.Draw(base).polygon([tuple(p) for p in corners], fill=GRASS)
    _scene(ImageDraw.Draw(base), Hm, pitch, size, np.zeros((0, 2)), np.zeros(0), np.array([np.nan, np.nan]), lines, False)
    team = match["team_of_player"]
    cell = max(4, size[0] // 120)
    enc = _Encoder(path, size, fps)
    n = len(match["labels"])
    t0 = time.time()
    for t in range(n):
        img = base.copy()
        d = ImageDraw.Draw(img)
        _scene(d, Hm, pitch, size, match["players"][t], team, match["ball"][t], [], background=False)
        _barcode(d, t, cell)
        enc.write(img)
        if t and t % 10000 == 0:
            log(f"  tactical {t}/{n} ({time.time() - t0:.0f} s)")
    enc.close()


def render_broadcast(path: Path, match: dict, bc: dict, log=print) -> None:
    from PIL import Image, ImageDraw
    pitch, fps, size = match["pitch"], match["fps"], bc["size"]
    lines = _pitch_lines(pitch)
    team = match["team_of_player"]
    cell = max(4, size[0] // 120)
    studio = Image.new("RGB", size, (18, 18, 20))
    ImageDraw.Draw(studio).rectangle([size[0] * 0.3, size[1] * 0.35, size[0] * 0.7, size[1] * 0.65], fill=(52, 52, 58))
    closeup = Image.new("RGB", size, GRASS)
    ImageDraw.Draw(closeup).ellipse([size[0] * 0.35, size[1] * 0.1, size[0] * 0.65, size[1] * 1.2], fill=HOME_RGB)
    enc = _Encoder(path, size, fps)
    t0 = time.time()
    for b in range(bc["n_frames"]):
        k = bc["kind"][b]
        if k == KIND_STUDIO:
            img = studio.copy()
        elif k == KIND_CUTAWAY or bc["content"][b] < 0:
            img = closeup.copy()
        else:
            c = bc["content"][b]
            img = Image.new("RGB", size, STANDS)
            d = ImageDraw.Draw(img)
            _scene(d, bc["hom"][b].reshape(3, 3), pitch, size, match["players"][c], team, match["ball"][c], lines)
            if k == KIND_REPLAY:
                d.rectangle([size[0] - 7 * cell, cell, size[0] - cell, 2 * cell], fill=(240, 240, 240))
        _barcode(ImageDraw.Draw(img), b, cell)
        enc.write(img)
        if b and b % 10000 == 0:
            log(f"  broadcast {b}/{bc['n_frames']} ({time.time() - t0:.0f} s)")
    enc.close()


def build(out_dir: Path = SYNTHETIC_DIR, video: bool = True, size=(960, 540), log=print) -> dict:
    from .. import synthetic
    spec = synthetic.generate(out_dir)
    m = synthetic.clean_match(spec.n_frames, spec.fps, 7, spec.halves, spec.pitch)  # generate()'s defaults
    f = paths(out_dir)
    Ht = tactical_homography(size)
    write_homography(f["tactical_hom"], np.tile(Ht.reshape(1, 9), (spec.n_frames, 1)))
    bc = make_broadcast(m, size)
    write_broadcast(out_dir, bc, spec.pitch)
    log(f"broadcast feed: {bc['n_frames']} frames, tracking + homography written")
    if video:
        if not shutil.which("ffmpeg"):
            log("ffmpeg not found: skipped the videos (Windows: winget install Gyan.FFmpeg)")
        else:
            log("rendering tactical video")
            render_tactical(f["tactical_video"], m, Ht, size, log)
            log("rendering broadcast video")
            render_broadcast(f["broadcast_video"], m, bc, log)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--size", default="960x540", help="video size, e.g. 960x540")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    build(SYNTHETIC_DIR, video=not args.no_video, size=(w, h))
    print("done -> python -m mqf.viewer --synthetic")


if __name__ == "__main__":
    main()
