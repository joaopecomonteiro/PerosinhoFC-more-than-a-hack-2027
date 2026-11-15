"""Who owns the ball while it is in play -- the experiments of 15 Nov 2026 (EXPERIMENTS #38).

Every method here answers one question, P(home owns the ball | ball alive), per frame. Where the ball
is alive comes from somewhere else (an "alive layer": the true mask, the official smooth50
baseline, or the learned P(dead) of mqf.dead), so the team part can be compared on its own.

    rules.py       hand rules: control (close, free, moving with the ball), passes, touches ignored
    feats.py       the feature table (who.py's 59 columns + relative velocity, +-3 s windows,
                   team shape, optional rule columns) and the 5 Hz sequences for the GRU
    data.py        training rows / sequences per game, cached under external/cache/owner
    models.py      GBM (HistGradientBoosting), MLP, TCN + boundary head, BiGRU -- one predict() for all
    train.py       python -m mqf.owner.train gbm|mlp|tcn|gru ...
    combine.py     hard vs probabilistic evidence, rule x model hybrids, spell veto, decoders
    experiment.py  python -m mqf.owner.experiment ...  (official metric, paired vs who_v1)
    suite.py       python -m mqf.owner.suite            (everything, in order)
    pathcrf_bridge.py  PathCRF (KDD 2026) trained on our IDSSE split as an outside reference
    tcn_check.py   python -m mqf.owner.tcn_check        (TCN seeds x sources under the submission's conditions)

Other games only (splits.json). Nothing here reads the competition match.
"""
