"""Ball imputation on ALIVE frames: interpolation, a bidirectional transformer trained on other
games with masks, and the hybrid (interpolate gaps <= tau, model beyond).

    python -m mqf.ball.train --arch A --player-mask off --tag v1       # on the GPU machine
    python -m mqf.ball.evaluate --models models/ball_A_off.pt ... --tag v1
    python -m mqf.ball.impute --model models/ball_A_off.pt --method hybrid --tau 1   # BAR-ATH aggregates

interp.py   linear / cubic Hermite per gap          data.py      bundle games -> windows + masks
models.py   A SetTemporal, B Axial (torch)          train.py     training CLI
impute.py   BallFill, sliding-window inference      evaluate.py  the protocol (buckets, team-nearest, downstream)
Training data: other games only (IDSSE, Metrica; splits.json). Nothing here reads the real match
except impute.py, which the team runs locally and which prints aggregates only.
"""
