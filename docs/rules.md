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
- **Seeds:** runs vary by about $\pm 0.007$. Report the mean and spread over three seeds for every
  configuration you compare, and treat smaller differences as noise.
- **Evidence:** a submission reports every run of its final configuration, at least three seeds at
  one commit. It sets a record when its mean beats the record's by the
  [record rule](submission.md#the-record-rule): at least 0.013 with three runs each.
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
- Final runs use an A100 80GB. Develop on any GPU; final runs on other hardware may be re-timed.
- Maintainers verify every submission by retraining each run from its code on an A100 80GB: each
  retrain must finish within the cap and score within 0.01 of its report.

## Data

- Allowed: any part of [GIFT-Eval Pretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
  (revision 6830b62), in any selection, mixture or preprocessing, including the
  [GEP slices](data.md), and synthetic series your own code generates.
- Not allowed: GIFT-Eval test data, other real datasets, pretrained weights, and other models'
  forecasts. Training on GEP-Val or GEP-Test series is allowed but voids those diagnostics.
- Your code must not read GIFT-Eval data while training or forecasting.

## Fixed

- `src/nanotsfm/evaluation.py`, `configs/gift-full.json`, `scripts/submission.py` and `.github/`:
  maintainers verify every submission with the official versions of these files.
- The forecast interface in the [README](../README.md), which the evaluator calls.

Everything else is yours to change.

## Open source

Your fork must be public, and the runs' commit must be pushed. Submissions arrive as pull requests;
see [submission](submission.md).

## Credit

- **Records:** each new record is announced in a short post that credits its contributors.
- **Notable submissions:** a submission that misses the record or breaks a rule but tests a useful
  idea can be listed with its verified score under Notable submissions in the README, so the idea
  is not lost.

## Rulings

Maintainers decide edge cases and record each ruling here, with its date and pull request, so the
same question gets the same answer. None yet.
