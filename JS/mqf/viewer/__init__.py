"""Match viewer: watch a game (tactical video, broadcast video or a 2D pitch) with the ground
truth and our model's possession labels running underneath, frame by frame.

    python -m mqf.viewer                          # BAR-ATH: public GT + every results/*.csv
    python -m mqf.viewer --pred results/a.csv --pred results/b.csv
    python -m mqf.viewer --synthetic              # synthetic match, full GT (no data needed)
    python -m mqf.viewer.demo                     # render synthetic videos to try every view

Everything runs on this machine: a small server on 127.0.0.1 streams the local files to your
browser. Nothing is uploaded, and nothing it writes leaves cache/ (gitignored).

Rules that apply to what you see here (CLAUDE.md):
  * The real match never goes to a cloud AI tool -- no screenshots or clips of this viewer on
    BAR-ATH in a chat. Share aggregates only.
  * On BAR-ATH the ground truth covers the public 5 minutes. Use it to check that the pipeline is
    *correct* (frames aligned, labels meaning what we think). Fixing the model case by case against
    those 17 possessions, or picking between two versions by eye there, is tuning on the public
    window -- RULES.md forbids it. Iterate on games with full ground truth (synthetic, IDSSE, PFF...).

Modules
  game.py    the data contract (`ViewerGame`) and the adapters that build it: BAR-ATH, synthetic.
             A new source of games (PFF, IDSSE) is one more adapter returning a ViewerGame.
  sync.py    broadcast <-> tactical frame alignment, computed from the two tracking files.
  server.py  localhost HTTP server: page, binary arrays, videos with Range requests.
  demo.py    synthetic broadcast feed + rendered videos, for trying the viewer without real data.
"""
