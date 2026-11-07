"""Write the submission file and run the OFFICIAL validator / scorer (from vendor/)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import OFFICIAL_DIR, official


def write(path: Path, labels: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{lab}\n" for i, lab in enumerate(labels))
    return path


def _official(script: str, *args: str) -> int:
    official()  # clear error if vendor/ is missing
    exe = OFFICIAL_DIR / "evaluation" / script
    return subprocess.call([sys.executable, str(exe), *args])


def validate(path: Path) -> int:
    return _official("validate_submission.py", str(Path(path).resolve()))


def score(results_dir: Path, gt: str, out_dir: Path, bootstrap: int = 0, detail: bool = False) -> int:
    """Official scoreboard for every *.csv in results_dir. gt = 'public' or a path."""
    args = ["--results", str(Path(results_dir).resolve()), "--gt", gt,
            "--out", str(Path(out_dir).resolve()), "--bootstrap", str(bootstrap)]
    if not detail:
        args.append("--no-detail")
    return _official("run_evaluation.py", *args)
