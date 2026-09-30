---
name: todo
description: Open work on nanoTSFM itself - data, evaluation, tooling and records.
meta:
  type: knowledge
---

# To do

Open work, most useful first.

## Launch

- [ ] Protect `main` with a ruleset: pull requests only, no force-push.
- [ ] Announce the internship's partner and terms on the [course](course.md) page.
- [ ] Test the Runpod credit code end to end, and give participants a one-step way to run nanoTSFM
  on a pod: a Runpod template that clones the repository and runs `./run.sh setup`, or a short
  "Run on Runpod" guide.

## Rules

Open questions that [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) answers for
its own record.

- [ ] Keep notable submissions that break a rule or miss the record in a separate list, so good ideas
  are not lost.
- [ ] Record rulings on edge cases in one place, a FAQ in the [rules](rules.md).
- [ ] Announce each record in a short post that credits its contributors.

## Data

- [ ] Ship an example synthetic-series generator: coverage gaps (yearly, 10-second, 10- and
  15-minute data) drove most of the GIFT-Eval loss from longer training.
- [ ] Detect near-duplicate series stored under two sources (for example related PEMS datasets),
  which the name-based split cannot see.

## Evaluation

- [ ] Find a cheap signal that predicts GIFT-Eval. Neither held-out series (correlation $-0.15$ over
  48 runs) nor 16 held-out sources tracked it in the pilot.
- [ ] Speed up GIFT-Eval: three `electricity/15T` tasks take about 4 minutes each and set the floor
  for any number of workers.

## Later

- [ ] Host the checkpoints that verification produces, one folder per team in a model repository
  such as `abel-lab/nanoTSFM-models`, so anyone can load a verified model.

## Background

The numbers in these pages come from about 50 pilot runs on A100 80GB GPUs in September 2026:
5,000–40,000 steps, six data slices (GEP-S, GEP-M and GEP-L among them), contexts of 512–2,048, 3.3M and 12.7M
parameters and three sampling schemes, each scored on GIFT-Eval, GEP-Val and GEP-Test.
