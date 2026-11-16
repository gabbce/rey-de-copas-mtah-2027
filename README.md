# Rey de Copas — More than a Hack 2027

Team **Rey de Copas** (Mateo Toniolo, Gabriel Canavesio, Santiago Vidoz).

This repository labels every frame of the competition match as home possession, away possession, or dead ball. A gradient-boosted tree, trained on public DFL tracking, is adapted to the match by expectation-maximisation. A duration-aware decoder then turns that evidence into whole possessions, and a second stage trained on DFL can hand a segment it judges wrong to its neighbour.

The default artifact applies expectation-maximisation tables that were fit on BAR-ATH's unlabelled tracking and homography. That fit does not use ground truth, the public window, hand labels, or video.

The match data is not in this repository. It is never sent to a hosted AI service.

Our code is MIT, copyright 2026 Rey de Copas (see `LICENSE`). The DFL tracking used for training is not included. It is CC BY 4.0 and stays under its own terms: credit the Deutsche Fußball Liga (DFL) and cite Bassek et al. 2025.

## Requirements

- Python 3.11, installed and run with [uv](https://docs.astral.sh/uv/)
- The competition match bundle, in `data/` (gitignored) or pointed to by `MTAH_DATA_DIR`
- The seven public DFL / IDSSE matches, under `external/dfl/` (gitignored)

The DFL matches are Bassek et al., *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data, 2025, CC BY 4.0, figshare [10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177). Place each match's `matchinformation` and `positions_raw_observed` XML in `external/dfl/`.

## Reproduce the submission

```bash
git clone --recursive https://github.com/mateotoniolo/rey-de-copas-mtah-2027
cd rey-de-copas-mtah-2027
uv sync --locked
uv run python fresh/predict.py
uv run python fresh/predict.py --refit
```

Both commands write `fresh/out/fresh-B-setp-dur-cor-gbm.csv`. Expected SHA-256:

```
0fa36ddbb2707f2c032a67cb742ab7dcb2dda427bfb3ec251f4f6146f9f16999
```

`--refit` learns the saved model again from the external matches and the tracked decoder search, then writes the same file. `fresh/out/` is gitignored.

Check the CSV with the pinned official validator:

```bash
python vendor/mtah-2027/evaluation/validate_submission.py fresh/out/fresh-B-setp-dur-cor-gbm.csv
```

## Compliance

Training, tuning, and selection used external data only. No hand labels were used. The public five-minute window and the match video were not used for training, tuning, or selection.

## Layout

| path | what |
| --- | --- |
| `fresh/` | the pipeline and the saved model |
| `src/possession/` | tracking readers, the integer decoder, and the submission writer |
| `vendor/mtah-2027/` | the official scorer and validator, pinned |
