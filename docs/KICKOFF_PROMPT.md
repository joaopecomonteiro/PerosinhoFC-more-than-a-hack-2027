# Kickoff prompt (first message in a new Claude session on this repo)

Read CLAUDE.md and docs/CONTEXT.md, then:

1. In ≤ 10 lines: the task, how spellF1 matches spells (what splitting vs merging costs), how X
   is treated, and the rules that could disqualify us.
2. Review `mqf/` (atd → features → smart_tagging → decode → submission) and `mqf/harness/`
   (noise model, IDSSE adapter, official-metric scoring). Tell me which failure modes the harness
   doesn't exercise yet and how to add them.
3. Help me get the harness running on IDSSE (`pip install -r requirements-harness.txt`,
   `python -m mqf.harness.idsse J03WMX`, then `python -m mqf.harness.run --source idsse`). IDSSE
   is other games, so you may read it. Then propose the first 3 experiments (stoppage detector,
   duration-aware decoding, control model for switches) as hypothesis → change → expected effect
   on spellF1 / macroF1 / length buckets.
4. The exact commands I run locally on the real data today (bootstrap, `make baseline`,
   `make submit`, `make check`) and which aggregate numbers to paste back. Never ask for raw rows,
   arrays or frames of the competition match.

Team name for the submission file: <team_name>
