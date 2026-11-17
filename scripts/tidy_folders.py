"""One-off (17 Nov 2026): sort harness_results/ by pipeline stage and models/old/ by model family. Dry run by default.

    python scripts/tidy_folders.py            # the plan, nothing touched
    python scripts/tidy_folders.py --apply    # do it (git mv for tracked files, a plain move otherwise)

Run from the repository root. harness_results/fusion_preds/ (the decoder suite's gitignored inputs) stays where
tasks.py and the viewer look for it. New runs still write to the top of harness_results/: file them by stage.

    harness_results/0_baseline     v0 on the bundles (#4-#5)
                    1_tracking     tracklet cleaner (#21, #23, #39)
                    2_ball         missing-ball imputation (#10-#18), train/ = training curves
                    3_dead_ball    dead_v1, mqf.alive, i8 (#42, #46), train/ = training curves (+ v8)
                    4_possession   who_v1 / who_v2, owner suite, TCN and heading checks (#34-#41), train/, who_lab/
                    5_final_check  FINAL vs FINAL_V1 and the fusion export runs (#43, #45, #47)
                    6_decoder      HMM cost sweep (#36), fusion suite (#47), stoppage decoder (#48)
    models/old/ball  dead_ball  possession
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HARNESS = [   # (pattern on the name, destination folder) -- first match wins
    (r"^fusion_preds$", None),
    (r"^README\.md$", None),
    (r"^(bundles_v0.*|idsse_dev)\.json$", "0_baseline"),
    (r"^(teamfix_.*|bundles_v4_full|bundles_whoclean_.*)\.json$", "1_tracking"),
    (r"^ball_train_", "2_ball/train"),
    (r"^ball_v", "2_ball"),
    (r"^(alive_train_|deadball_train_)", "3_dead_ball/train"),
    (r"^(dead_v1\.json|bundles_dead_|alive_alive16|alive_sens_|alive_i8_|i8_check_)", "3_dead_ball"),
    (r"^owner_train_", "4_possession/train"),
    (r"^who$", "4_possession/who_lab"),
    (r"^bundles_who_v1_costs_", "6_decoder"),
    (r"^(who_v\d|bundles_who_|owner_)", "4_possession"),
    (r"^(final_check_|alive_final16_|alive_fusion16_)", "5_final_check"),
    (r"^fusion_", "6_decoder"),
]
OLD_MODELS = [(r"^ball_", "ball"), (r"^alive_", "dead_ball"), (r"^(owner_|who_)", "possession")]


def git(*args, check=True):
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def tracked(p: Path) -> bool:
    return bool(git("ls-files", "--", str(p), check=False).stdout.strip())


def move(src: Path, dst: Path, root: Path, apply: bool) -> None:
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst.relative_to(root).as_posix()}")
    how = "git mv" if tracked(src) else "move  "
    print(f"  {how} {src.relative_to(root).as_posix():<52} -> {dst.relative_to(root).as_posix()}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        git("mv", str(src), str(dst)) if how == "git mv" else shutil.move(str(src), str(dst))


def sort_dir(folder: Path, rules, root: Path, apply: bool) -> int:
    n, left = 0, []
    for p in sorted(folder.iterdir()):
        dest = next((d for pat, d in rules if re.search(pat, p.name)), "?")
        if dest is None:
            continue
        if dest == "?":
            if not (p.is_dir() and re.match(r"^\d_|^(ball|dead_ball|possession)$", p.name)):
                left.append(p.name)
            continue
        move(p, folder / dest / p.name, root, apply)
        n += 1
    if left:
        print(f"  (left in place, no rule: {', '.join(left)})")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    root = Path.cwd()
    if not ((root / ".git").exists() and (root / "harness_results").is_dir() and (root / "mqf").is_dir()):
        raise SystemExit("run from the repository root (after scripts/restructure_repo.py)")
    print(("APPLYING" if a.apply else "DRY RUN (add --apply)") + "\n\nharness_results/ by stage")
    n = sort_dir(root / "harness_results", HARNESS, root, a.apply)
    if (root / "models" / "old").is_dir():
        print("\nmodels/old/ by family")
        n += sort_dir(root / "models" / "old", OLD_MODELS, root, a.apply)
    print(f"\n{n} move(s). " + ("Done: git status; git commit -m 'Tidy harness_results/ and models/old/'"
                               if a.apply else "Nothing changed. Re-run with --apply."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
