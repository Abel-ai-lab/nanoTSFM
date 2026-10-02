---
name: rules
description: Rules - the score, the compute budget, what data and changes are allowed, and how to read the diagnostics.
meta:
  type: knowledge
---

# Rules

Set the record on GIFT-Eval with at most one hour of training on one A100 80GB-class GPU. The
baseline scored 0.670 after 2 minutes, about 114th of 131 models on the leaderboard; the
[README](../README.md#world-record) has the current record.

## Score

- **GIFT-Eval:** CRPS relative to the leaderboard's Seasonal Naive results, on 97 tasks from 23
  datasets the model never trains on, averaged geometrically over tasks. Lower is better; 1 matches
  Seasonal Naive. Like the leaderboard, we compute CRPS as the weighted quantile loss over the nine
  quantile levels. Relative MASE is reported beside it.
- **Repeated runs:** runs with different seeds vary by about $\pm 0.007$. Report the mean and
  spread over three repeated runs for every configuration you compare, and treat smaller differences
  as noise.
- **Evidence:** a submission reports every run of its final configuration, at least three repeated
  runs at one commit.
- **Verified score:** maintainers retrain the final configuration three times with new random
  seeds, and the mean of those runs is the submission's score. It sets a record when that mean
  beats the record's by the [record rule](submission.md#the-record-rule): at least 0.013.
- `./run.sh eval <run>` computes the score in about 5 minutes on 8 CPU cores. The benchmark is
  public; your report explains how you chose your final model.

## Diagnostics

GEP-Val and GEP-Test forecast held-out series from the training corpus (see
[data](data.md#gep-val-and-gep-test)). They take 20 seconds and never count toward the score.

A change that improves them but not GIFT-Eval fits the corpus without forecasting new data better.
In the pilot, longer training moved GEP-Test from 0.618 to 0.586 and GIFT-Eval from 0.666 to 0.693,
while better source sampling and more data improved both. Making in-distribution gains carry over
is the challenge.

## Budget

- At most 3,600 seconds of training per run, as `run.json` records. Setup, data loading,
  validation, checkpointing and evaluation are off the clock.
- All model training happens on this clock, including any model that selects or weights data.
- Final runs use an A100 80GB. Develop on any GPU; final runs on other hardware may be re-timed.
- Maintainers verify every submission by retraining it from its code on an A100 80GB, three
  times with new seeds; each retrain must finish within the cap.

## Data

- Allowed: any part of [GIFT-Eval Pretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
  (revision 6830b62), in any selection, mixture or preprocessing, including the
  [GEP slices](data.md).
- Not allowed: GIFT-Eval test data, other datasets, synthetic series, pretrained weights, and other
  models' forecasts. Training on GEP-Val or GEP-Test series is allowed but voids those diagnostics.
- No code but the evaluator may read GIFT-Eval data, at any step: building data, training or
  forecasting.
- **Explain data changes.** If your data differs from the record's you build on, in selection,
  mixing or preprocessing, the report says what changed, how you chose it, and why. A submission
  that does not is closed without review.

## Fixed

- `src/nanotsfm/evaluation.py`, `configs/gift-full.json`, `scripts/` and `.github/`: maintainers
  verify every submission with the official versions of these files, and a check fails any
  submission that changes them.
- Your code forecasts; official code scores. `evaluation.py` writes the GIFT-Eval forecasts, and
  `scripts/score.py` scores them in its own process, which loads none of `src/`.
- The forecast interface in the [README](../README.md), which the evaluator calls.

Everything else is yours to change.

## Open source

Your fork must be public, and the runs' commit must be pushed. Submissions arrive as pull requests;
see [submission](submission.md).

## Credit

- **Records:** the [record page](https://abel-ai-lab.github.io/nanoTSFM/) and the README's badge
  show the latest record and its contributors, and update when the record merges.
- **Notable submissions:** a submission that misses the record or breaks a rule but tests a useful
  idea can be listed with its verified score under Notable submissions in the README, so the idea
  is not lost.

## Rulings

Maintainers decide edge cases and record each ruling here, with its date and pull request, so the
same question gets the same answer. None yet.
