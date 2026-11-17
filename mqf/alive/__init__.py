"""Is the ball in play? The dead-ball (alive layer) experiments of 16 Nov 2026 -- built like mqf/owner.

Every method here answers one question, P(dead) per frame (and, for the sequence models, P(an
alive <-> dead change near this frame)). WHO has the ball comes from a fixed owner model (who_v1 by
default, as in make_submission.FINAL), so the alive layer is compared on its own.

    feats.py        the feature table: dead_v1's 36 columns + collective speed / acceleration, restart
                    burst and stillness clocks, set-piece shapes (Laws 8, 13-17), the ball where TRACKED
                    (never its absence, never a filled ball), rule columns
    rules.py        Law-based stoppage rules (ball out, ball still on a restart spot, throw-in / corner /
                    goal-kick / kickoff shapes, collective stillness, long breaks) -> a hard dead mask, a
                    soft P(dead), restart / stop events for boundary snapping
    data.py         training rows (5 Hz) per game x view, cached under external/cache/alive; the ball's
                    missingness is made label-independent in every view (see the module doc)
    models.py       GBM, MLP, TCN (ASRF-style, boundary head), ASFormer-lite -- one predict() for all
    train.py        python -m mqf.alive.train gbm|mlp|tcn|asf ...
    combine.py      decoders (production 3-state HMM, boundary-aware dead costs, minimum stoppage length,
                    half-start prior, boundary snapping), rule x model hybrids, the Smart Tagging prior
                    (and tag-like windows simulated from labels), stoppage-level diagnostics
    experiment.py   python -m mqf.alive.experiment ...   (official metric, paired vs dead_v1)
    sensitivity.py  python -m mqf.alive.sensitivity      (what each kind of alive-layer error costs)
    suite.py        python -m mqf.alive.suite             (everything, in order, --resume)

Other games only (splits.json). Nothing here reads the competition match; make_submission reads it
through predict_p_dead() on the team's machine.
"""
