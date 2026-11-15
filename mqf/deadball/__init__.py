"""Is the ball in play? Dead-ball detection only -- no teams, no possession.

    python -m mqf.deadball.model train --tag v1        # learn P(dead) on other games -> models/deadball_v1.joblib
    python -m mqf.deadball.evaluate --tag v1           # score it on other games (official building blocks)
    python -m mqf.deadball.run --tag v1                # our match -> results/deadball/v1.csv (D / X)
    python -m mqf.viewer --deadball                    # gray where the ball is dead, nothing else

features.py  players (mqf.dead) + ball + slow-down / speed-up edges, per frame
model.py     gradient-boosted P(dead), trained on other games under randomised ball loss
decoder.py   two-state semi-Markov decoder with real stoppage / play lengths -> whole stoppages
evaluate.py  alive/dead scores on other games: frame F1, whole stoppages, start / restart timing
run.py       the competition match, aggregates only
"""
