"""Paths and per-game constants.

The official challenge repo is cloned (pinned) into vendor/mtah2027 by `python bootstrap.py`.
Everything about the real match comes from its games.py, and validation/scoring from its
evaluation/ scripts -- we never copy their numbers or reimplement their metric.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = Path(os.environ.get("MTAH_OFFICIAL_DIR", ROOT / "vendor" / "mtah2027"))
DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data"))          # gated match bundle
CACHE_DIR = Path(os.environ.get("MQF_CACHE_DIR", ROOT / "cache"))         # arrays derived from it
RESULTS_DIR = ROOT / "results"                                            # our real-match submissions
REPORTS_DIR = ROOT / "reports"                                            # official scorer output on them
SYNTHETIC_DIR = ROOT / "synthetic_data"                                   # safe to inspect
EXTERNAL_DIR = Path(os.environ.get("MQF_EXTERNAL_DIR", ROOT / "external"))  # other games (IDSSE...)
HARNESS_DIR = ROOT / "harness_results"                                    # aggregates on other games

OFFICIAL_COMMIT = "f8f2629ff1e0985f81d7e59d8413678514ee1530"   # bump deliberately after reading upstream changes

LABELS = ("H", "A", "D")          # state order used everywhere: 0=H, 1=A, 2=D
H, A, D = 0, 1, 2


def official():
    """Import the official games.py and possession_metrics.py from vendor/."""
    if not (OFFICIAL_DIR / "games.py").exists():
        raise SystemExit(f"Official repo not found at {OFFICIAL_DIR}. Run:  python bootstrap.py")
    for p in (OFFICIAL_DIR, OFFICIAL_DIR / "evaluation"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import games  # noqa: E402
    import possession_metrics  # noqa: E402
    return games, possession_metrics


@dataclass(frozen=True)
class GameSpec:
    """Everything the pipeline needs to know about one recording."""
    name: str
    n_frames: int                                   # rows in the submission
    fps: int
    pitch: tuple                                    # metres (length, width)
    halves: tuple                                   # ((half, first_frame, last_frame), ...) inclusive
    atd_meta: Path = field(default=Path())
    atd_raw: Path = field(default=Path())
    smart_tagging: Path | None = None
    gt_public: Path | None = None


def competition_game(data_dir: Path | None = None, source: str = "tactical") -> GameSpec:
    """The competition match. Tactical (default) = the submission's timeline: row i = video frame i.

    Halves are the published graded blocks (games.half_video_blocks). Nothing inside them -- e.g.
    the 2nd-half kickoff frame -- is hard-coded: stoppages must be found by the algorithm
    (RULES.md forbids rules or constants tied to specific frames of the provided game).

    source="broadcast" is the TV recording on its OWN timeline (not synchronised, no graded
    blocks, no Smart Tagging): `halves` is empty and only profiling uses it.
    """
    games, _ = official()
    g = games.GAMES[games.GAME]
    paths = games.asset_paths(games.GAME, str(data_dir or DATA_DIR), source=source)
    tactical = source == "tactical"
    return GameSpec(
        name=games.GAME if tactical else f"{games.GAME}-{source}",
        n_frames=games.source_frames(source, games.GAME),
        fps=g["fps"],
        pitch=tuple(g["pitch"]),
        halves=tuple(tuple(b) for b in games.half_video_blocks(games.GAME)) if tactical else (),
        atd_meta=Path(paths["atd_meta"]),
        atd_raw=Path(paths["atd_raw"]),
        smart_tagging=Path(paths["smart_tagging"]) if "smart_tagging" in paths else None,
        gt_public=(OFFICIAL_DIR / "evaluation" / "gt" / f"{games.GAME}_possession_gt_public.csv"
                   if tactical else None),
    )


def homography_path(data_dir: Path | None = None, source: str = "tactical") -> Path:
    games, _ = official()
    return Path(games.asset_paths(games.GAME, str(data_dir or DATA_DIR), source=source)["homography"])
