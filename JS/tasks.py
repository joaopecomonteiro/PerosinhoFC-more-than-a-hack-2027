"""Task runner: the Makefile's targets, also on Windows (no make needed).

    python tasks.py help
    python tasks.py setup                          # install-dev + bootstrap + test
    python tasks.py submit --team Perosinho_FC     # real match -> results/<team>.csv, validated
    python tasks.py harness -- --noise jitter_corr_s=1.0     # args after -- go to the command

Runs from anywhere (it switches to this folder). Targets marked [REAL DATA] read the match bundle:
run them yourself and share only the aggregate numbers they print (CLAUDE.md, rule 1).
"""
from __future__ import annotations

import argparse
import filecmp
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
TASKS: dict = {}


def task(name: str, real: bool = False):
    def deco(fn):
        TASKS[name] = (fn, (fn.__doc__ or "").strip().splitlines()[0], real)
        return fn
    return deco


def run(*cmd, env: dict | None = None, check: bool = True) -> int:
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.call(cmd, cwd=ROOT, env={**os.environ, **(env or {})})
    if check and r:
        raise SystemExit(r)
    return r


def cfg():
    sys.path.insert(0, str(ROOT))
    from mqf import config
    return config


def official_dir() -> Path:
    c = cfg()
    if not (c.OFFICIAL_DIR / "games.py").is_file():
        raise SystemExit("official repo missing: python tasks.py bootstrap")
    return c.OFFICIAL_DIR


# ------------------------------------------------------------------------------------ setup

@task("setup")
def setup(a, extra):
    """install-dev + bootstrap + test: everything needed on a fresh clone."""
    install_dev(a, [])
    bootstrap(a, [])
    test(a, [])


@task("bootstrap")
def bootstrap(a, extra):
    """Clone the official repo at the pinned commit into vendor/mtah2027."""
    run(PY, "bootstrap.py", *extra)


@task("check-updates")
def check_updates(a, extra):
    """Organiser commits newer than our pin (read them before bumping OFFICIAL_COMMIT)."""
    run(PY, "bootstrap.py", "--check-updates")


@task("status")
def status(a, extra):
    """Pin, vendor HEAD, local edits, drift of the fork's own copies of the official files."""
    run(PY, "bootstrap.py", "--status")


def pip_install(py, req: str) -> None:
    """Install a requirements file into the interpreter `py`: uv when it is on PATH (fast, and
    works in uv-made venvs, which have no pip), otherwise that interpreter's own pip."""
    if shutil.which("uv"):
        run("uv", "pip", "install", "--python", py, "-r", req)
    elif subprocess.call([str(py), "-m", "pip", "--version"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL) == 0:
        run(py, "-m", "pip", "install", "-r", req)
    else:
        raise SystemExit(f"{py} has no pip and uv is not on PATH")


@task("venv")
def venv(a, extra):
    """Create .venv (pipeline, Python 3.13) with uv if available, else python -m venv."""
    _make_venv(ROOT / ".venv")
    act = r".venv\Scripts\Activate.ps1" if os.name == "nt" else "source .venv/bin/activate"
    print(f"\nactivate it with:  {act}   then:  python tasks.py setup")


def _make_venv(path: Path, python: str = "3.13") -> Path:
    if not path.exists():
        if shutil.which("uv"):
            run("uv", "venv", path, "--python", python)          # uv fetches 3.13 if it is missing
        else:
            if sys.version_info >= (3, 14) or sys.version_info < (3, 12):
                raise SystemExit("needs Python 3.12/3.13: run this with such an interpreter, or install uv")
            run(PY, "-m", "venv", path)
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


@task("install")
def install(a, extra):
    """Install the pinned submission requirements into the current interpreter."""
    pip_install(PY, "requirements.txt")


@task("install-dev")
def install_dev(a, extra):
    """Install pinned requirements + pytest + Pillow into the current interpreter."""
    pip_install(PY, "requirements-dev.txt")


@task("venv-harness")
def venv_harness(a, extra):
    """Create .venv-harness (floodlight, pandas < 3, Python 3.13) for IDSSE."""
    py = _make_venv(ROOT / ".venv-harness")
    pip_install(py, "requirements-harness.txt")
    act = r".venv-harness\Scripts\Activate.ps1" if os.name == "nt" else "source .venv-harness/bin/activate"
    print(f"\nactivate it with:  {act}")


@task("test")
def test(a, extra):
    """Run the test suite (synthetic match, official scorer and validator, harness)."""
    run(PY, "-m", "pytest", "-q", *extra)


@task("clean")
def clean(a, extra):
    """Delete regenerable files (synthetic_data/, caches of the tests). Never touches cache/ or results/."""
    for d in [ROOT / "synthetic_data", ROOT / ".pytest_cache", *ROOT.rglob("__pycache__")]:
        if d.exists() and "vendor" not in d.parts:
            shutil.rmtree(d)
            print(f"removed {d.relative_to(ROOT)}")


# ------------------------------------------------------------------------------- no real data

@task("synthetic")
def synthetic(a, extra):
    """Run the whole pipeline on the synthetic match (no data needed)."""
    run(PY, "-m", "mqf.make_submission", "--synthetic", *extra)


@task("harness")
def harness(a, extra):
    """Validation harness on synthetic games (3 matches x 3 noise seeds)."""
    run(PY, "-m", "mqf.harness.run", "--source", "synthetic", "--seeds", "0", "1", "2", "--tag", a.tag, *extra)


def _need_floodlight():
    if importlib.util.find_spec("floodlight") is None:
        act = r".venv-harness\Scripts\Activate.ps1" if os.name == "nt" else "source .venv-harness/bin/activate"
        raise SystemExit(f"floodlight not installed here: python tasks.py venv-harness, then {act}")


@task("idsse-cache")
def idsse_cache(a, extra):
    """Parse the 7 IDSSE matches from external_data/idsse once (cached) and print a summary each."""
    _need_floodlight()
    run(PY, "-m", "mqf.harness.idsse", *extra)


@task("harness-idsse")
def harness_idsse(a, extra):
    """Validation harness on the 7 IDSSE games x 3 noise seeds (harness venv)."""
    _need_floodlight()
    run(PY, "-m", "mqf.harness.run", "--source", "idsse", "--matches", "all", "--seeds", "0", "1", "2",
        "--tag", a.tag, *extra)


@task("fetch-skillcorner")
def fetch_skillcorner(a, extra):
    """Replace the SkillCorner Git LFS pointers with the real tracking files (~1.8 GB)."""
    run(PY, "-m", "mqf.external.fetch_skillcorner", *extra)


@task("bundles")
def bundles(a, extra):
    """Write IDSSE / Metrica / SkillCorner as tactical-format bundles (no noise, native fps) + QA lines."""
    run(PY, "-m", "mqf.external.build", *extra)


@task("eval-bundles")
def eval_bundles(a, extra):
    """Our pipeline on the bundles, official metric, per source (harness_results/bundles_<tag>.json)."""
    run(PY, "-m", "mqf.external.evaluate", "--tag", a.tag, *extra)


def _need_torch():
    if importlib.util.find_spec("torch") is None:
        raise SystemExit("torch not installed: pip install -r requirements-model.txt (GPU: torch for your CUDA first)")


@task("install-torch")
def install_torch(a, extra):
    """CUDA build of torch into THIS interpreter (default cu132, torch 2.15.0), then requirements-model.txt."""
    spec = "torch" if a.torch == "any" else f"torch=={a.torch}"
    index = f"https://download.pytorch.org/whl/{a.cuda}"
    if shutil.which("uv"):                       # uv venvs have no pip, and a bare `pip` may be another env's
        run("uv", "pip", "install", "--python", PY, spec, "--index-url", index)
    else:
        run(PY, "-m", "pip", "install", spec, "--index-url", index)
    pip_install(PY, "requirements-model.txt")
    torch_info(a, [])


@task("torch-info")
def torch_info(a, extra):
    """Which torch this environment has, and whether it sees the GPU."""
    _need_torch()
    run(PY, "-c", "import torch, sys; print('python', sys.executable); print('torch', torch.__version__, "
                  "'| CUDA build', torch.version.cuda, '| GPU available', torch.cuda.is_available()); "
                  "torch.cuda.is_available() and print('GPU', torch.cuda.get_device_name(0), "
                  "'| bf16', torch.cuda.is_bf16_supported())")


@task("train-ball")
def train_ball(a, extra):
    """Train a ball model on IDSSE + Metrica bundles (e.g. -- --arch A --player-mask off)."""
    _need_torch()
    run(PY, "-m", "mqf.ball.train", "--tag", a.tag, *extra)


@task("train-ball-all")
def train_ball_all(a, extra):
    """The 4 runs of the plan: A / B x player mask off / on."""
    _need_torch()
    for arch in ("A", "B"):
        for pm in ("off", "on"):
            run(PY, "-m", "mqf.ball.train", "--arch", arch, "--player-mask", pm, "--tag", a.tag, *extra)


@task("eval-ball")
def eval_ball(a, extra):
    """Ball protocol on the validation games: buckets, team-nearest agreement, downstream spellF1."""
    run(PY, "-m", "mqf.ball.evaluate", "--tag", a.tag, *extra)


@task("viewer-synthetic")
def viewer_synthetic(a, extra):
    """Synthetic tactical + broadcast feeds, then the viewer on them."""
    run(PY, "-m", "mqf.viewer.demo", *(extra or ["--no-video"]))
    run(PY, "-m", "mqf.viewer", "--synthetic")


# ------------------------------------------------------------------------------- real data

@task("baseline", real=True)
def baseline(a, extra):
    """Rebuild the 4 official baselines from the ATD and byte-compare them with the organisers' files."""
    c, off = cfg(), official_dir()
    out = c.RESULTS_DIR / "baselines"
    run(PY, off / "starter" / "baseline.py", "--out", out, env={"MTAH_DATA_DIR": str(c.DATA_DIR)})
    ref = off / "evaluation" / "results"
    for f in sorted(ref.glob("baseline_*.csv")):
        mine = out / f.name
        same = mine.is_file() and filecmp.cmp(mine, f, shallow=False)
        print(f"  {f.name:<28} {'identical to the official file' if same else 'DIFFERENT'}")


@task("submit", real=True)
def submit(a, extra):
    """Build results/<team>.csv from the real match and run the official validator on it."""
    run(PY, "-m", "mqf.make_submission", "--team", a.team, *extra)


@task("check", real=True)
def check(a, extra):
    """Official scorer on results/*.csv vs the public 5 minutes -- correctness only, never tuning."""
    c, off = cfg(), official_dir()
    run(PY, off / "evaluation" / "run_evaluation.py", "--results", c.RESULTS_DIR, "--gt", "public",
        "--out", c.REPORTS_DIR / "check", "--bootstrap", "0", "--no-detail", *extra)


@task("reproduce", real=True)
def reproduce(a, extra):
    """Regenerate results/<team>.csv into cache/reproduce/ and byte-compare."""
    run(PY, "-m", "mqf.make_submission", "--team", a.team, "--reproduce")


@task("profile", real=True)
def profile(a, extra):
    """Aggregate-only profile of the tactical + broadcast ATD (the block you may paste)."""
    run(PY, "-m", "mqf.profile", *extra)


@task("impute-ball", real=True)
def impute_ball(a, extra):
    """Fill BAR-ATH's missing ball (e.g. -- --model models/ball_A_off.pt --method hybrid --tau 1); aggregates only."""
    run(PY, "-m", "mqf.ball.impute", *extra)


@task("compare", real=True)
def compare(a, extra):
    """How much two submissions differ (share of frames changed, possessions) -- never which is better."""
    run(PY, "-m", "mqf.compare", *extra)


@task("viewer", real=True)
def viewer(a, extra):
    """BAR-ATH in the local viewer (keep screenshots out of AI chats)."""
    run(PY, "-m", "mqf.viewer", *extra)


# ------------------------------------------------------------------------------------ main

def show_help():
    print(__doc__)
    w = max(map(len, TASKS))
    for name, (_, desc, real) in TASKS.items():
        print(f"  {name:<{w}}  {'[REAL DATA] ' if real else ''}{desc}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("task", nargs="?", default="help")
    ap.add_argument("--team", default=os.environ.get("TEAM", "Perosinho_FC"))
    ap.add_argument("--tag", default="dev", help="harness_results/<source>_<tag>.json")
    ap.add_argument("--cuda", default="cu132", help="install-torch: PyTorch wheel index (cu132, cu134, cpu ...)")
    ap.add_argument("--torch", default="2.15.0", help="install-torch: version, or 'any' for the newest on that index")
    a, rest = ap.parse_known_args(argv)
    if a.task in ("help", "-h", "--help") or a.task not in TASKS:
        if a.task not in ("help", "-h", "--help"):
            print(f"unknown task {a.task!r}\n")
        show_help()
        return 0 if a.task in ("help", "-h", "--help") else 2
    TASKS[a.task][0](a, rest + extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
