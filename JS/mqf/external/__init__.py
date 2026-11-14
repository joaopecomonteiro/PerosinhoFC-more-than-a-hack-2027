"""Other games (IDSSE, Metrica sample data, SkillCorner open data) in the tactical data definition.

    python -m mqf.external.fetch_skillcorner   # SkillCorner LFS pointers -> real files
    python -m mqf.external.build               # every source -> external/bundles/<GAME>/ (no noise, native fps)
    python -m mqf.external.evaluate --tag v0   # pipeline + official metric on the bundles

Raw downloads live in config.EXTERNAL_RAW_DIR (external_data/, gitignored); anything derived from
them in config.EXTERNAL_DIR (external/, gitignored).
"""
