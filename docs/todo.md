---
name: todo
description: Open work on nanoTSFM itself - data, evaluation, tooling and records.
meta:
  type: knowledge
---

# To do

Open work, most useful first.

## Launch

- [ ] Make the repository public.
- [ ] Protect `main` with a ruleset (pull requests, no force-push), and let the leaderboard push its
  figure: an org GitHub App with Contents write access, on the ruleset's bypass list, with its
  client ID in the `LEADERBOARD_CLIENT_ID` variable and its private key in the
  `LEADERBOARD_APP_KEY` secret.
- [ ] Make an example submission from a personal fork with one real change, to walk the whole flow:
  pull request, guard, `verify` and the leaderboard.
- [ ] Fill in the [course](course.md) page's *TBA* items: the submission deadline and the internship
  partner and terms.
- [ ] Test the Runpod credit code end to end, and give participants a one-step way to run nanoTSFM
  on a pod: a Runpod template that clones the repository and runs `./run.sh setup`, or a short
  "Run on Runpod" guide.

## Data

- [ ] Ship an example synthetic-series generator: coverage gaps (yearly, 10-second, 10- and
  15-minute data) drove most of the GIFT-Eval loss from longer training.
- [ ] Detect near-duplicate series stored under two sources (for example related PEMS datasets),
  which the name-based split cannot see.

## Evaluation

- [ ] Find a cheap signal that predicts GIFT-Eval. Neither held-out series (correlation −0.15 over
  48 runs) nor 16 held-out sources tracked it in the pilot.
- [ ] Speed up GIFT-Eval: three `electricity/15T` tasks take about 4 minutes each and set the floor
  for any number of workers.

## Later

- [ ] Host the checkpoints that verification produces, one folder per team in a model repository
  such as `abel-lab/nanoTSFM-models`, so anyone can load a verified model.

## Background

The numbers in these pages come from about 50 pilot runs on A100 80GB GPUs in September 2026:
5,000–40,000 steps, six data slices (three now published), contexts of 512–2,048, 3.3M and 12.7M
parameters and three sampling schemes, each scored on GIFT-Eval, GEP-Val and GEP-Test.
