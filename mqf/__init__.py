"""Més que un Frame -- frame-level possession for More than a Hack 2027 (Phase 1).

Pipeline: atd -> features (+ smart_tagging) -> decode (HMM / Viterbi per half) -> submission.
Entry points:
    python -m mqf.make_submission --team <team_name>     # real match
    python -m mqf.harness.run --source synthetic|idsse   # validation on other games
"""
