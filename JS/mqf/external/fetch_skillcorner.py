"""Replace the Git LFS pointer files of the SkillCorner open data with the real files.

    python -m mqf.external.fetch_skillcorner                 # every *_tracking_extrapolated.jsonl pointer
    python -m mqf.external.fetch_skillcorner --matches 1874553 1886347
    python -m mqf.external.fetch_skillcorner --dry-run       # list what would be downloaded
    python -m mqf.external.fetch_skillcorner --all           # any LFS pointer under the folder

Why this exists: a copy of github.com/SkillCorner/opendata made without git (ZIP download, copied
folder) keeps only the 133-byte pointers, and `git lfs pull` cannot fetch them because the folder
is not a git checkout of that repository. This script does what git-lfs does, with the standard
library only: it reads each pointer (oid + size), asks the repository's LFS batch API for a
download URL, streams the file, checks size and SHA-256 against the pointer, and only then
replaces the pointer (atomically). Rerunning skips files that are already real.

If the folder IS a git clone, `git -C external_data/skill_corner lfs pull` does the same.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..config import EXTERNAL_RAW_DIR

REPO = "SkillCorner/opendata"
POINTER_HEAD = b"version https://git-lfs.github.com/spec/v1"
LFS_MEDIA = "application/vnd.git-lfs+json"
CHUNK = 1 << 20


@dataclass(frozen=True)
class Pointer:
    path: Path
    oid: str
    size: int


def parse_pointer(path: Path) -> Pointer | None:
    """The pointer in `path`, or None if it is a real file."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1024:
        return None
    raw = path.read_bytes()
    if not raw.startswith(POINTER_HEAD):
        return None
    fields = dict(line.split(" ", 1) for line in raw.decode("ascii").splitlines() if " " in line)
    algo, _, oid = fields.get("oid", "").partition(":")
    if algo != "sha256" or len(oid) != 64 or not fields.get("size", "").isdigit():
        raise ValueError(f"{path}: malformed LFS pointer")
    return Pointer(path, oid, int(fields["size"]))


def find_pointers(root: Path, matches=None, any_file: bool = False) -> list[Pointer]:
    root = Path(root)
    pattern = "**/*" if any_file else "data/matches/*/*_tracking_extrapolated.jsonl"
    out = []
    for p in sorted(root.glob(pattern)):
        if matches and p.parent.name not in set(map(str, matches)):
            continue
        ptr = parse_pointer(p)
        if ptr is not None:
            out.append(ptr)
    return out


def _post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Accept": LFS_MEDIA, "Content-Type": LFS_MEDIA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _open(url: str, headers: dict):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def download_action(repo: str, ptr: Pointer) -> dict:
    """{'href': ..., 'header': {...}} from the LFS batch API, or SystemExit with its reason."""
    url = f"https://github.com/{repo}.git/info/lfs/objects/batch"
    try:
        resp = _post_json(url, {"operation": "download", "transfers": ["basic"],
                                "objects": [{"oid": ptr.oid, "size": ptr.size}]})
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise SystemExit(f"LFS batch API refused ({e.code}): {body}\n"
                         "Over its data quota -> GitHub serves no LFS files for this repo right now; "
                         "try later. Otherwise the same files come with a fresh clone:\n"
                         "  git lfs install\n"
                         "  git clone https://github.com/SkillCorner/opendata.git external_data/skill_corner") from e
    obj = resp["objects"][0]
    if "error" in obj:
        raise SystemExit(f"{ptr.path.name}: LFS error {obj['error'].get('code')}: {obj['error'].get('message')}")
    return obj["actions"]["download"]


def fetch(ptr: Pointer, repo: str = REPO, log=print) -> None:
    action = download_action(repo, ptr)
    part = ptr.path.with_name(ptr.path.name + ".part")
    digest, n = hashlib.sha256(), 0
    try:
        with _open(action["href"], action.get("header", {})) as r, open(part, "wb") as fh:
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                digest.update(chunk)
                n += len(chunk)
        if n != ptr.size or digest.hexdigest() != ptr.oid:
            raise SystemExit(f"{ptr.path.name}: got {n} bytes / sha256 {digest.hexdigest()[:12]}, "
                             f"expected {ptr.size} / {ptr.oid[:12]} -- pointer left in place")
        os.replace(part, ptr.path)
    finally:
        if part.exists():
            part.unlink()
    log(f"  {ptr.path.parent.name}/{ptr.path.name}  {n / 1e6:.1f} MB  sha256 ok")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=EXTERNAL_RAW_DIR / "skill_corner",
                    help="local copy of SkillCorner/opendata (default: external_data/skill_corner)")
    ap.add_argument("--matches", nargs="*", help="match ids (folder names under data/matches/)")
    ap.add_argument("--all", action="store_true", help="any LFS pointer, not just the tracking files")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.dir.is_dir():
        raise SystemExit(f"no folder {args.dir}")
    ptrs = find_pointers(args.dir, args.matches, args.all)
    if not ptrs:
        print(f"nothing to do: no LFS pointers left under {args.dir}")
        return 0
    total = sum(p.size for p in ptrs) / 1e9
    print(f"{len(ptrs)} LFS pointer(s) under {args.dir}, {total:.2f} GB to download")
    if args.dry_run:
        for p in ptrs:
            print(f"  {p.path.relative_to(args.dir)}  {p.size / 1e6:.1f} MB")
        return 0
    for i, p in enumerate(ptrs, 1):
        print(f"[{i}/{len(ptrs)}]", end="")
        fetch(p, args.repo)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
