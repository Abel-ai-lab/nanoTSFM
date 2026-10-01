---
name: submission
description: How to submit a result, and how maintainers verify it.
meta:
  type: knowledge
---

# Submission

nanoTSFM keeps a world record, as [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
does. A submission is a pull request that brings your code and a record folder. If its verified runs
beat the record, it is merged and `main` becomes the new record; otherwise it is closed with its
verified score.

Each record's `result.json` names the commit its runs trained at, and the pull request keeps that
commit, so any record can be reproduced and you may start from any of them.
[The record page](https://abel-ai-lab.github.io/nanoTSFM/) shows every record with its runs, and
[`records/`](../records/) has the reports.

## What a submission holds

- **Your code**, as commits on a branch of your fork. Merge `main` before your final runs, so that
  after the merge `main` trains exactly what your runs trained; the guard checks this.
- **Three or more runs** of your final configuration, with different seeds and at one commit.
  Report every run of that configuration, not the best ones.
- **A record folder**, `records/<date>_<name>/`:
  - `README.md`, your report. Its front matter gives the team, a short description of the change,
    one or two members with their GitHub handles, and any AI assistance. If your data differs from
    the record's, the report says what changed, how you chose it, and why; without that, the
    submission is closed without review ([rules](rules.md#data)).
  - `result.json`, written by `./run.sh submit`: the commit, configuration, data and versions, and
    for each run its checkpoint's SHA-256, training time, GIFT-Eval, GEP-Val and GEP-Test scores
    and training log, with the mean and spread over runs.

## Steps

1. Fork nanoTSFM, clone your fork and work on a branch. Copy
   [`records/template/`](../records/template/README.md) to `records/<YYYY-MM-DD>_<name>/`, with
   hyphens for spaces, and fill in its README.
2. Commit and push your code, then train and score three repeated runs of the final configuration:

   ```shell
   ./run.sh train final-s7 configs/my.yaml 7   # the last argument is the seed
   ./run.sh train final-s1 configs/my.yaml 1
   ./run.sh train final-s2 configs/my.yaml 2
   ./run.sh eval final-s7                      # and final-s1, final-s2
   ./run.sh submit <YYYY-MM-DD>_<name> final-s7 final-s1 final-s2
   ```

   `submit` writes `result.json`, checks each checkpoint and compares the runs with the record.
3. Commit the folder, push, and open a pull request from your branch to nanoTSFM's `main`. Its
   template asks for what changed and why, your results beside the record, an ablation, and how to
   reproduce.

Report other runs and ablations in the README. If you train on data your own code builds, give the
command in the README; verification runs it. To try again, push new runs to the same branch or open
a new pull request; keep one submission per team open at a time.

## The record rule

A submission sets a record when the mean GIFT-Eval CRPS of its three verified runs is below the
record's by at least $2.33 \times 0.007 \times \sqrt{1/3 + 1/3} = 0.013$, where 0.007 is the
baseline's spread between runs: a one-sided test at $p < 0.01$.

## Checks

- On every pull request from a fork, a guard checks that fixed files are untouched, no file is
  over 1 MB, the record folder is well formed, and after the merge the training code (`src/`,
  `configs/`, `pyproject.toml` and `uv.lock`) is identical to the runs' commit. It also says
  whether the runs claim the record.
- `./run.sh submit` also checks your local checkpoints: their SHA-256, steps and training time match
  `result.json`, and each returns finite, ordered quantiles through the forecast interface.

## Review

Maintainers read the code and the report, then retrain your final configuration with the
official fixed files, so your code trains and the official code scores:

```shell
gh repo clone Abel-ai-lab/nanoTSFM review && cd review && gh pr checkout <number>
git checkout origin/main -- src/nanotsfm/evaluation.py configs/gift-full.json scripts/submission.py
uv run --extra gift python -m scripts.submission verify records/<folder> --output verify
```

`verify` draws three new random seeds, retrains the configuration once with each, requires every
retrain to finish within the cap, and compares the mean of the three with the record. Each retrain
runs in its own process, as does each scoring. The seeds are kept in the output folder, and runs
already verified there are skipped, so `--runs run-1` can split the work across jobs; three
full-hour runs take about 3.5 A100-hours. `verify` also says whether your reported runs are
within run-to-run noise of the retrains.

`verify` writes `verified.json` into the record folder. The maintainers commit it to the pull
request, with the README's record row from `./run.sh table` if the submission sets a record, then
merge it, squashed or not; the record page updates itself. Otherwise the pull request is closed
with its verified score.
