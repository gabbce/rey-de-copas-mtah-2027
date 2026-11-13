"""The submitted variant's settings, from the tracked fresh/config.json.

An environment variable may override a setting for experiments (FRESH_SET, FRESH_DUR, FRESH_CORRECT, FRESH_GAIN, FRESH_CHANGE,
FRESH_CLASSIFIER, FRESH_MERGE, FRESH_RANK, FRESH_DUR_MERGE, FRESH_DEGRADE,
FRESH_CORR_MERGE, FRESH_CORR_MIN, FRESH_TEST_DEGRADE, FRESH_SMOOTH, FRESH_DUR_SCALE,
FRESH_DECODER_SEED), but only
with a loud warning on stderr, so a stray variable cannot silently change the submission.
"""

import json
import os
import sys
from pathlib import Path

CONFIG = json.loads((Path(__file__).resolve().parent / "config.json").read_text())


def _setting(key, env, parse):
    value = CONFIG[key]
    if env in os.environ:
        override = parse(os.environ[env])
        if override != value:
            print(f"WARNING: {env}={os.environ[env]} overrides fresh/config.json {key}={value!r}; "
                  "the output is NOT the official submission", file=sys.stderr)
        value = override
    return value


SET = _setting("feature_set", "FRESH_SET", str)
DUR = _setting("duration_decoder", "FRESH_DUR", lambda v: v == "1")
GAIN = _setting("context_change_cost", "FRESH_GAIN", lambda v: v == "1")
CHANGE = _setting("handover_evidence", "FRESH_CHANGE", lambda v: v == "1")
CORRECT = _setting("segment_corrector", "FRESH_CORRECT", lambda v: v == "1")
CLASSIFIER = _setting("classifier", "FRESH_CLASSIFIER", str)
# DFL truth: spells shorter than this many seconds stay with the last team (public.merge_short).
# Chosen on DFL only, by the rule pre-registered in docs/merge-threshold.md.
MERGE = _setting("merge_seconds", "FRESH_MERGE", float)
# How decoder candidates are ranked on DFL: "merged" (truth merged at MERGE), "raw" or "mean" of both.
RANK = _setting("decoder_rank", "FRESH_RANK", str)
# DFL training degradation: 1 (state-free gaps) or 2 (docs/r2-rule.md E2, fresh/public.py).
DEGRADE = _setting("degradation", "FRESH_DEGRADE", int)
# Truth the segment corrector is trained and thresholded on (seconds merged; null = MERGE).
CORR_MERGE = _setting("corrector_merge_seconds", "FRESH_CORR_MERGE",
                      lambda v: None if v in ("", "none") else float(v))
# Evaluation only (docs/r2-rule.md E2-common): the held-out DFL match's inputs degraded with this
# version while everything fitted keeps DEGRADE (null = DEGRADE).
TEST_DEGRADE = _setting("test_degradation", "FRESH_TEST_DEGRADE",
                        lambda v: None if v in ("", "none") else int(v))
# E7 (docs/r2-rule.md): Metrica's sample games added to the evidence model's training matches.
METRICA = _setting("metrica_training", "FRESH_METRICA", lambda v: str(v).lower() in ("1", "true"))
# Truth the duration prior is fitted on (seconds merged; null = MERGE).
DUR_MERGE = _setting("duration_merge_seconds", "FRESH_DUR_MERGE",
                     lambda v: None if v in ("", "none") else float(v))
# E9-smooth (docs/r2-rule.md): positions averaged over this many frames (centred) before any
# player velocity is taken; 0 = raw one-frame differences.
SMOOTH = _setting("velocity_smooth_frames", "FRESH_SMOOTH", int)
# E11b (docs/r2-rule.md): multiply fitted duration lengths by this scale (mu <- mu + log(scale));
# 1.0 = unchanged prior.
DUR_SCALE = _setting("duration_length_scale", "FRESH_DUR_SCALE", float)

# E12 (docs/r2-rule.md): LOO threshold grid floor for the segment corrector; 0.5 = full current grid.
CORR_MIN = _setting("corrector_min_threshold", "FRESH_CORR_MIN", float)

# E10 (docs/r2-rule.md): which decoder search file to load (run_decoder.py --seed); 0 = official.
DECODER_SEED = _setting("decoder_search_seed", "FRESH_DECODER_SEED", int)
