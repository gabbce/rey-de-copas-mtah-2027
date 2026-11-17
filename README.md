# Rey de Copas — More than a Hack 2027

Team **Rey de Copas** (Mateo Toniolo, Gabriel Canavesio, Santiago Vidoz).

The submission is variant A. A gradient-boosted tree is trained only on the seven external DFL matches, after a fixed degradation (version 4): 25-frame smoothing, 5 cm jitter, the ball hidden on half of frames, and log-normal gaps with an 8 s median. A duration-aware Viterbi decoder turns that evidence into possessions, and a segment corrector trained on DFL can hand a segment it judges wrong to its neighbour. There is no expectation-maximisation and no adaptation of tables to the provided match.

Variant A's decoder parameters were searched on DFL evidence under the version-1 degradation, whose constants were set from summary statistics of the provided tracking; the tree itself uses the fixed version-4 degradation.

`uv run python fresh/predict.py --variant B` is an alternative and is not submitted. It uses expectation-maximisation tables fit on BAR-ATH's unlabelled tracking and homography. That fit does not use ground truth, the public window, hand labels, or video. It writes `fresh/out/fresh-B-setp-dur-cor-gbm.csv`. The SHA-256 of that earlier file, which is not the submission, is `0fa36ddbb2707f2c032a67cb742ab7dcb2dda427bfb3ec251f4f6146f9f16999`.

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

Both commands write `fresh/out/fresh-A-setp-dur-cor-gbm.csv`. Expected SHA-256:

```
112082e9a30598bf261966db8337b71a806a0fb3848bcdbce94f74395ebb4e06
```

`--refit` learns the saved model again from the external matches and the tracked decoder search, then writes the same file. `fresh/out/` is gitignored.

Check the CSV with the pinned official validator:

```bash
python vendor/mtah-2027/evaluation/validate_submission.py fresh/out/fresh-A-setp-dur-cor-gbm.csv
```

## Compliance

No hand labels were used. The public five-minute window and the match video were not used for training, tuning, or selection. Variant A does not run expectation-maximisation on the provided match. The decoder it loads was searched on DFL under the version-1 degradation described above.

## Layout

| path | what |
| --- | --- |
| `fresh/` | the pipeline and the saved model |
| `src/possession/` | tracking readers, the integer decoder, and the submission writer |
| `vendor/mtah-2027/` | the official scorer and validator, pinned |
