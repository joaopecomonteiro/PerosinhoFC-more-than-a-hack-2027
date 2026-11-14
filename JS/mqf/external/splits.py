"""Train / validation split of the other games, by GAME, per source -- fixed and committed.

    python -m mqf.external.splits            # print the split (and rewrite splits.json if the rule changed)

Whole games only, so no frame of a validation game ever trains anything. The draw is a seeded
shuffle inside strata, written once to splits.json (committed) so every run, person and model sees
the same games. `source` stays a column everywhere; which sources a task uses is decided per task
(`train_ok` is the default: SkillCorner is 10 fps and is not used for training).

Use these games only when a task needs a lot of data -- the competition's tactical and broadcast
recordings come first (CLAUDE.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import ROOT

SPLITS_FILE = ROOT / "splits.json"
SEED = 2027
IDSSE_LEAGUE = {"J03WMX": "Bundesliga", "J03WN1": "Bundesliga", "J03WPY": "2. Bundesliga",
                "J03WOH": "2. Bundesliga", "J03WQQ": "2. Bundesliga", "J03WOY": "2. Bundesliga",
                "J03WR9": "2. Bundesliga"}
SKILLCORNER_IDS = ("1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846", "1986691",
                   "1996435", "1996436", "2006229", "2006363", "2007448", "2007721", "2010085", "2011166",
                   "2013725", "2015213", "2016236", "2017461")
PFF_GROUP = tuple(str(g) for g in range(3812, 3860))          # 48 group-stage games
PFF_KNOCKOUT = tuple(str(g) for g in range(10502, 10518))     # 16 knock-out games
SOURCES = {
    # source: (games -> stratum, validation games per stratum, train_ok, fps, why)
    "idsse": ({f"IDSSE-{m}": lg for m, lg in IDSSE_LEAGUE.items()}, {"Bundesliga": 1, "2. Bundesliga": 1},
              True, 25, "stratified by league: 1 Bundesliga + 1 2. Bundesliga game held out"),
    "metrica": ({f"METRICA-{g}": "all" for g in (1, 2, 3)}, {"all": 1}, True, 25, "1 of 3 held out"),
    "skillcorner": ({f"SKC-{m}": "all" for m in SKILLCORNER_IDS}, {"all": 5}, False, 10,
                    "5 of 20 held out; train_ok false: 10 fps, not used for training (Sigmar, 10 Nov 2026)"),
    # added last so the draws above stay what they were
    "pff": ({**{f"PFF-{g}": "group" for g in PFF_GROUP}, **{f"PFF-{g}": "knockout" for g in PFF_KNOCKOUT}},
            {"group": 6, "knockout": 2}, True, 29.97,
            "World Cup 2022, stratified by stage: 6 group + 2 knock-out games held out"),
}


def make() -> dict:
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "unit": "game", "rule": "seeded shuffle within strata, per source", "sources": {}}
    for source, (games, n_val, train_ok, fps, why) in SOURCES.items():
        val = []
        for stratum in sorted(set(games.values())):
            members = sorted(g for g, s in games.items() if s == stratum)
            pick = rng.permutation(len(members))[:n_val[stratum]]
            val += [members[i] for i in sorted(pick)]
        train = sorted(g for g in games if g not in val)
        out["sources"][source] = {"fps": fps, "train_ok": train_ok, "note": why,
                                  "train": train, "val": sorted(val)}
    return out


def load(path: Path = SPLITS_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def games(part: str, sources=None, train_ok_only: bool = False, path: Path = SPLITS_FILE) -> list[str]:
    """Game names of one part ('train' | 'val'), optionally of some sources / only train_ok sources."""
    s = load(path)["sources"]
    names = []
    for src, info in s.items():
        if sources and src not in sources:
            continue
        if train_ok_only and not info["train_ok"]:
            continue
        names += info[part]
    return names


def main() -> int:
    split = make()
    text = json.dumps(split, indent=2) + "\n"
    if not SPLITS_FILE.exists() or SPLITS_FILE.read_text(encoding="utf-8") != text:
        SPLITS_FILE.write_text(text, encoding="utf-8")
        print(f"wrote {SPLITS_FILE}")
    for src, info in split["sources"].items():
        print(f"{src:<12} {info['fps']} fps  train_ok={info['train_ok']}  "
              f"train {len(info['train'])}  val {len(info['val'])}: {', '.join(info['val'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
