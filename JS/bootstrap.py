"""Fetch the official challenge repo, pinned, into vendor/mtah2027 (read-only for us).

    python bootstrap.py                   # clone + check out the pinned commit (idempotent)
    python bootstrap.py --check-updates   # organiser commits newer than the pin, and what they touch
    python bootstrap.py --status          # pin, vendor HEAD, local edits, drift of the fork's copies

The official repo is the authority: we import its games.py and run its validator and scorer from
vendor/ -- never edit vendor/, never copy its numbers. Bump OFFICIAL_COMMIT in mqf/config.py only
after reading what --check-updates lists (the rules were clarified mid-competition).

Works in both layouts: our code at the top of its own repo, or in a subfolder (JS/) of a fork of
the official repo. In the fork the official files also sit one level up; --status reports whether
those copies still match the pin. vendor/ is what the pipeline uses either way, so a merge from
upstream into the fork can never silently change the scorer under us.

Standard library + git only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "https://github.com/metrica-sports/more-than-a-hack-2027.git"
VENDOR = Path(os.environ.get("MTAH_OFFICIAL_DIR", ROOT / "vendor" / "mtah2027"))
# files whose drift would change what we import or how we are scored
WATCHED = ("games.py", "evaluation/possession_metrics.py", "evaluation/run_evaluation.py",
           "evaluation/validate_submission.py", "evaluation/gt/BAR-ATH_possession_gt_public.csv")
RULE_FILES = re.compile(r"(\.md$|^games\.py$|^evaluation/)")


def pinned_commit() -> str:
    text = (ROOT / "mqf" / "config.py").read_text(encoding="utf-8")
    m = re.search(r'^OFFICIAL_COMMIT\s*=\s*"([0-9a-f]{40})"', text, re.M)
    if not m:
        raise SystemExit("OFFICIAL_COMMIT not found in mqf/config.py")
    return m.group(1)


def git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    if shutil.which("git") is None:
        raise SystemExit("git is not on PATH")
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def has_commit(sha: str) -> bool:
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=VENDOR, capture_output=True)
    return r.returncode == 0


def ensure(pin: str) -> None:
    if not (VENDOR / ".git").exists():
        if VENDOR.exists() and any(VENDOR.iterdir()):
            raise SystemExit(f"{VENDOR} exists but is not a git checkout -- move it away and rerun")
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {URL} -> {VENDOR}")
        git("clone", "--quiet", URL, str(VENDOR))
    if not has_commit(pin):
        git("fetch", "--quiet", "origin", cwd=VENDOR)
    if not has_commit(pin):
        raise SystemExit(f"pinned commit {pin[:12]} not found upstream")
    if git("rev-parse", "HEAD", cwd=VENDOR) != pin:
        git("checkout", "--quiet", "--detach", pin, cwd=VENDOR)
    head = git("rev-parse", "HEAD", cwd=VENDOR)
    if head != pin or not (VENDOR / "games.py").is_file():
        raise SystemExit(f"vendor checkout is at {head[:12]}, expected {pin[:12]}")
    dirty = git("status", "--porcelain", cwd=VENDOR)
    if dirty:
        print("WARNING: vendor/ has local edits (never edit vendor/). Undo with:\n"
              f"  git -C {VENDOR} checkout -- .   and   git -C {VENDOR} clean -fd")
    print(f"official repo at pinned commit {pin[:12]}  ({VENDOR})")


def check_updates(pin: str) -> int:
    ensure(pin)
    git("fetch", "--quiet", "origin", cwd=VENDOR)
    upstream = git("rev-parse", "--abbrev-ref", "origin/HEAD", cwd=VENDOR, check=False) or "origin/main"
    log = git("log", "--no-merges", "--reverse", "--format=%h %ad %s", "--date=short",
              f"{pin}..{upstream}", cwd=VENDOR)
    if not log:
        print(f"up to date: no organiser commits after {pin[:12]} on {upstream}")
        return 0
    print(f"{len(log.splitlines())} organiser commit(s) after the pin on {upstream}:\n")
    for line in log.splitlines():
        sha = line.split()[0]
        files = git("show", "--name-only", "--format=", sha, cwd=VENDOR).splitlines()
        flag = "  <- rules / metric / data contract" if any(RULE_FILES.search(f) for f in files) else ""
        print(f"  {line}{flag}")
        for f in files:
            print(f"        {f}")
    print(f"\nRead them (git -C {VENDOR} show <sha>), then bump OFFICIAL_COMMIT in mqf/config.py "
          f"to the new sha and rerun python bootstrap.py.")
    return 0


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    # line endings normalised, so a Windows checkout (autocrlf) compares equal to a Linux one
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def status(pin: str) -> int:
    print(f"pin          {pin}")
    if not (VENDOR / ".git").exists():
        print("vendor       missing -- run python bootstrap.py")
        return 1
    head = git("rev-parse", "HEAD", cwd=VENDOR)
    print(f"vendor HEAD  {head}  {'OK' if head == pin else 'MISMATCH -- run python bootstrap.py'}")
    print(f"vendor edits {'none' if not git('status', '--porcelain', cwd=VENDOR) else 'YES (never edit vendor/)'}")
    fork = ROOT.parent
    if (fork / "games.py").is_file() and (fork / "evaluation").is_dir():
        print(f"\nfork copies in {fork} vs the pin (the pipeline always uses vendor/):")
        for rel in WATCHED:
            a, b = _sha(fork / rel), _sha(VENDOR / rel)
            state = "missing in fork" if a is None else "same" if a == b else "DIFFERENT"
            print(f"  {rel:<55} {state}")
    return 0 if head == pin else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-updates", action="store_true", help="list organiser commits newer than the pin")
    ap.add_argument("--status", action="store_true", help="pin, vendor HEAD, local edits, fork drift")
    args = ap.parse_args()
    pin = pinned_commit()
    if args.check_updates:
        return check_updates(pin)
    if args.status:
        return status(pin)
    ensure(pin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
