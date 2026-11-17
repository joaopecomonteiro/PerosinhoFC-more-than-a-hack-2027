"""One-off (17 Nov 2026): make the repository presentable. Dry run by default -- it prints every move.

    python JS/scripts/restructure_repo.py            # the plan, nothing touched
    python JS/scripts/restructure_repo.py --apply    # do it

Run it from the repository root (the folder with .git and JS/), with a Python that is NOT JS/.venv (Windows
cannot move a venv that is running: use conda's / the system python, or `py -3.13`), after:
  1. the submission CSV is built and validated  (python tasks.py final-check, then submit)
  2. every change is committed                  (the moves then form one clean commit)

Layout after it:
    README.md  Perosinho_FC.md  CLAUDE.md  LICENSE  Makefile    (ours; LICENSE is the organisers')
    bootstrap.py  tasks.py  run_fusion.py  pytest.ini  requirements*.txt  splits.json
    mqf/            the pipeline package          tests/        every test (pytest)
    experiments/    one-off lab scripts           docs/         CONTEXT, EXPERIMENTS, TECHNICAL, ...
    models/         the submitted models (+ FINAL_V1's)     models/old/   every other model we trained
    harness_results/  aggregates on other games   challenge/    the organisers' files as forked (README, RULES,
                                                                games.py, evaluation/, starter/)
    data/  external_data/  cache/  results/  reports/  vendor/  external/  synthetic_data/   (gitignored)

Tracked files move with `git mv` (history follows them); ignored folders (cache/, results/, ... -- never opened,
only renamed) with a plain move. Nothing is deleted except Python caches and folders left empty in JS/. Then:
    python -m pytest -q                       # from the root
    python tasks.py reproduce --team Perosinho_FC   # byte-identical: the move changed nothing
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# the organisers' files at the top of the fork -> challenge/ (our code imports the pinned vendor/ copy instead)
OFFICIAL = ["README.md", "CHALLENGE.md", "CITATION.cff", "DATA.md", "EVALUATION.md", "FAQ.md", "OVERVIEW.md",
            "RULES.md", "SUBMISSION.md", "games.py", "requirements.txt", "requirements-starter.txt", "evaluation",
            "starter"]
# inside JS/: moved to the root as they are, except these renames (paths relative to JS/ -> to the root)
RENAMES = {"docs/README.md": "README.md", "docs/CLAUDE.md": "CLAUDE.md",
           "who/who_lab.py": "experiments/who_lab.py", "who/who_who_noise.json": "harness_results/who/who_noise.json",
           "who/who_who2.json": "harness_results/who/who2.json"}
SKIP_IN_JS = {".gitignore", "who", "__pycache__", ".pytest_cache"}       # .gitignore: merged into the root one
VENVS = (".venv", ".venv-harness")
# models make_submission.FINAL / FINAL_HMM / FINAL_V1 and the tests read; everything else -> models/old/
MODELS_KEEP = {f"owner_tcn_full_impff_s{k}.pt" for k in range(3)} | {f"alive_asf_full_impff_s{k}.pt" for k in range(3)} \
    | {"deadball_v8.joblib", "ball_B_off_v5.pt", "ball_B_off_v5.json", "dead_v1.joblib", "who_v1.joblib", "README.md"}
OTHER = {"Claude outputs/EXTERNAL_DATA_PLAN.md": "docs/EXTERNAL_DATA_PLAN.md",
         "Claude outputs/settings.json": ".claude/settings.json"}

GITIGNORE = """\
# This repository is PUBLIC. Nothing derived from the match data is ever committed (CLAUDE.md, rule 5):
# when in doubt, add it here first and take it out later.

# -- the organisers' private ground truth (challenge/.gitignore's rules, kept) --------------------------
*_possession_gt.csv
*_possession_gt_meta.json
*_possession_gt_reference.csv
possession_by_frame.pkl
challenge/evaluation/results/*.csv
!challenge/evaluation/results/baseline_*.csv
challenge/evaluation/reports/
challenge/evaluation/scoreboard.csv

# -- gated data: data/ is committed empty with its README; other games as downloaded --------------------
/data/*
!/data/README.md
/external_data/

# -- derived from the real match: arrays, submissions, official scorer output, profile aggregates ---------
/cache/
/results/
/reports/
*.npz
*.mp4
*.gzip
*.pl3container
*_atd.*
*_homography.csv
*_smart_tagging.xml
*.pdf

# -- generated / downloaded (re-created by bootstrap.py and tasks.py) -------------------------------------
/vendor/
/external/
/synthetic_data/
organizers/

# harness_results/ (aggregates on OTHER games) is committed on purpose; its *.npz inputs are not (above).

# -- editors and build artifacts --------------------------------------------------------------------------
.venv*/
venv/
env/
__pycache__/
*.py[cod]
*.nbi
*.nbc
.pytest_cache/
.ipynb_checkpoints/
.idea/
.vscode/
.DS_Store
"""

SETTINGS_FIX = {"/JS/cache/": "/cache/", "/JS/results/": "/results/", "/JS/reports/": "/reports/",
                "/JS/vendor/": "/vendor/", "/evaluation/": "/challenge/evaluation/"}


def git(*args, check=True):
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def tracked(p: Path) -> bool:
    return bool(git("ls-files", "--", str(p), check=False).stdout.strip())


class Plan:
    """The moves; a dry run keeps track of what it would have moved, so its collision checks match --apply."""

    def __init__(self, apply: bool, root: Path):
        self.apply, self.root, self.n = apply, root, 0
        self.moved = {}                                    # dst -> src (dry run: where a path really is now)

    def _real(self, p: Path) -> Path:
        if self.apply:
            return p
        for dst, src in self.moved.items():
            if p == dst or dst in p.parents:
                return self._real(src / p.relative_to(dst))
        return p

    def _exists(self, p: Path) -> bool:
        r = self._real(p)
        gone = any(src == r or src in r.parents for src in self.moved.values()) if not self.apply else False
        return r.exists() and (r != p or not gone)

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    def move(self, src: Path, dst: Path, why: str = ""):
        if not self._exists(src):
            return
        if self._exists(dst):
            raise SystemExit(f"refusing to overwrite {self._rel(dst)} (moving {self._rel(src)})")
        how = "git mv" if tracked(self._real(src)) else "move  "
        self.n += 1
        print(f"  {how} {self._rel(src):<40} -> {self._rel(dst)}{'   # ' + why if why else ''}")
        if not self.apply:
            self.moved[dst] = src
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if how == "git mv":
            git("mv", str(src), str(dst))          # a folder: renamed on disk, untracked files inside go along
        else:
            shutil.move(str(src), str(dst))

    def write(self, path: Path, text: str, why: str):
        self.n += 1
        print(f"  write  {self._rel(path):<40}    # {why}")
        if self.apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            git("add", str(path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="do it (default: print the plan)")
    ap.add_argument("--team", default="Perosinho_FC")
    a = ap.parse_args(argv)
    root = Path.cwd()
    js = root / "JS"
    if not (root / ".git").exists() or not js.is_dir():
        raise SystemExit("run from the repository root (the folder with .git and JS/)")
    dirty = [l for l in git("status", "--porcelain", "--untracked-files=no").stdout.splitlines() if l.strip()]
    if dirty and a.apply:
        raise SystemExit(f"{len(dirty)} tracked file(s) changed -- commit first, so the moves are one clean commit")
    if not (js / "results" / f"{a.team}.csv").is_file() and not (root / "results" / f"{a.team}.csv").is_file():
        print(f"WARNING: no JS/results/{a.team}.csv yet -- build and validate the submission first "
              f"(python tasks.py final-check --team {a.team}).")
    in_venv = any(Path(sys.prefix).resolve().is_relative_to((js / v).resolve()) for v in VENVS)
    p = Plan(a.apply, root)
    print(("APPLYING" if a.apply else "DRY RUN (add --apply)") + f" in {root}\n\n1. the organisers' files -> challenge/")
    for name in OFFICIAL:
        p.move(root / name, root / "challenge" / name)
    print("\n2. JS/* -> the root")
    for src in sorted(js.iterdir()):
        if src.name in SKIP_IN_JS:
            continue
        if src.name in VENVS:
            continue                                   # last (step 7): Windows cannot move a venv that is in use
        if src.name == "docs":
            for d in sorted(src.iterdir()):
                rel = f"docs/{d.name}"
                p.move(d, root / RENAMES.get(rel, rel))
            continue
        p.move(src, root / src.name)
    for rel, dst in RENAMES.items():
        if rel.startswith("who/"):
            p.move(js / rel, root / dst)
    print("\n3. models: the submitted ones stay, the rest -> models/old/")
    models = p._real(root / "models")
    for m in sorted(models.glob("*")) if models.is_dir() else []:
        if m.is_file() and m.name not in MODELS_KEEP:
            p.move(root / "models" / m.name, root / "models" / "old" / m.name)
    print("\n4. loose notes")
    for rel, dst in OTHER.items():
        if not (root / dst).exists():
            p.move(root / rel, root / dst)
    print("\n5. the root .gitignore (ours + the organisers' private-GT rules); JS/.gitignore retired")
    if (js / ".gitignore").exists():
        print("  git rm JS/.gitignore")
        if a.apply:
            git("rm", "-q", str(js / ".gitignore")) if tracked(js / ".gitignore") else (js / ".gitignore").unlink()
    p.write(root / ".gitignore", GITIGNORE, "merged")
    sp = root / ".claude" / "settings.json"
    if a.apply and sp.is_file():
        t = sp.read_text(encoding="utf-8")
        for old, new in SETTINGS_FIX.items():
            t = t.replace(old, new)
        p.write(sp, t, "deny paths without JS/, evaluation/ -> challenge/evaluation/")
    else:
        print("  write  .claude/settings.json   # deny paths without JS/ (after the move)")
    print("\n6. the venvs (last: a venv still in use only leaves itself behind)")
    for v in VENVS:
        src = js / v
        if not src.exists():
            continue
        if in_venv:
            print(f"  skip   JS/{v}   # this python runs from it: move it by hand later, or recreate (python tasks.py venv)")
            continue
        try:
            p.move(src, root / v, "gitignored; use `python -m pip` / `python -m pytest` afterwards")
        except OSError as e:
            print(f"  FAILED JS/{v}: {e}\n         close every program using it (VS Code, terminals), then move it "
                  f"by hand: move JS\\{v} {v}   -- or recreate it at the root: python tasks.py venv + setup")
    print("\n7. Python caches and empty folders left in JS/")
    if a.apply:
        for c in [*js.rglob("__pycache__"), *js.rglob(".pytest_cache")]:
            if c.is_dir():
                shutil.rmtree(c)
        for d in sorted((d for d in [js, *js.rglob("*")] if d.is_dir()), key=lambda d: -len(d.parts)):
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        left = [x.name for x in js.iterdir()] if js.exists() else []
        print(f"  JS/ {'removed' if not js.exists() else 'still holds: ' + ', '.join(left)}")
        if (root / "Claude outputs").is_dir() and not any((root / "Claude outputs").iterdir()):
            (root / "Claude outputs").rmdir()
    print(f"\n{p.n} step(s). " + ("Done. Now: git status; python -m pytest -q; python tasks.py reproduce --team "
                                   f"{a.team}; git commit -m 'Restructure: JS/ to the root'" if a.apply else
                                   "Nothing changed. Re-run with --apply."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
