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
    one or two members with their GitHub handles, and any AI assistance.
  - `result.json`, written by `./run.sh submit`: the commit, configuration, data and versions, and
    for each run its checkpoint's SHA-256, training time, GIFT-Eval, GEP-Val and GEP-Test scores
    and training log, with the mean and spread over runs.
- **A row in the README's record table**, if your runs beat the record.

## Steps

1. Fork nanoTSFM, clone your fork and work on a branch. Copy
   [`records/template/`](../records/template/README.md) to `records/<YYYY-MM-DD>_<name>/`, with
   hyphens for spaces, and fill in its README.
2. Commit and push your code, then train and score three seeds of the final configuration:

   ```shell
   ./run.sh train final-s7 configs/my.yaml 7   # the last argument is the seed
   ./run.sh train final-s1 configs/my.yaml 1
   ./run.sh train final-s2 configs/my.yaml 2
   ./run.sh eval final-s7                      # and final-s1, final-s2
   ./run.sh submit <YYYY-MM-DD>_<name> final-s7 final-s1 final-s2
   ```

   `submit` writes `result.json`, checks each checkpoint and compares the runs with the record.
3. If they beat it, add your row to the README's table: `./run.sh table` prints it.
4. Commit the folder, push, and open a pull request from your branch to nanoTSFM's `main`. Its
   template asks for what changed and why, your results beside the record, an ablation, and how to
   reproduce.

Report other seeds and ablations in the README. If you train on data your own code builds, give the
command in the README; verification runs it. To try again, push new runs to the same branch or open
a new pull request; keep one submission per team open at a time.

## The record rule

A submission sets a record when its mean GIFT-Eval CRPS is below the record's by at least
$2.33 \times 0.007 \times \sqrt{1/n + 1/m}$, where $n$ and $m$ are the two run counts and 0.007 is
the baseline's seed spread: a one-sided test at $p < 0.01$. With three runs each the gap must be at
least 0.013; more runs lower it.

## Checks

- On every pull request from a fork, a guard checks that fixed files are untouched, no file is
  over 1 MB, the record folder is well formed, and after the merge the training code (`src/`,
  `configs/`, `pyproject.toml` and `uv.lock`) is identical to the runs' commit. It also says
  whether the runs claim the record.
- `./run.sh submit` also checks your local checkpoints: their SHA-256, steps and training time match
  `result.json`, and each returns finite, ordered quantiles through the forecast interface.

## Review

Maintainers read the code and the report, then retrain every run from the pull request with the
official fixed files, so your code trains and the official code scores:

```shell
gh repo clone Abel-ai-lab/nanoTSFM review && cd review && gh pr checkout <number>
git checkout origin/main -- src/nanotsfm/evaluation.py configs/gift-full.json scripts/submission.py
uv run --extra gift python -m scripts.submission verify records/<folder> --output verify
```

`verify` retrains each run, requires it to finish within the cap and to score within 0.01 of its
report, and compares the verified mean with the record. It skips runs already verified in its
output folder, so `--runs` can split the work across jobs; three full-hour runs take about 3.5
A100-hours. A record is merged, squashed or not, and the record page updates itself.
