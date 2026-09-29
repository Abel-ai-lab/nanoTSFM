---
name: submission
description: How to package, check and hand in a submission.
meta:
  type: knowledge
---

# Submission

A submission is one run in a folder of three files, `submissions/<team>/`:

- `README.md`, your report. Its front matter names the team, one or two members with their emails,
  the repository and any AI assistance.
- `result.json`, written by `./run.sh submit` from the run: commit, configuration, data, versions,
  training time, the checkpoint's SHA-256, GIFT-Eval, GEP-Val and GEP-Test scores, and the
  training log.
- `changes.diff`, your code changes against the nanoTSFM commit you started from, so the record
  stands on its own.

## Status

- **Reported:** your pull request. Its `result.json` holds the score you measured; it is a
  reference for reviewers and does not reach the leaderboard.
- **Verified:** a maintainer retrains your run from your commit and configuration on an A100 80GB
  and scores it. If the retrained score is within 0.01 of the reported one, the maintainer records
  the verified score, and only that score, on the [leaderboard](../README.md#leaderboard).

## Steps

1. Copy [`submissions/template/`](../submissions/template/README.md) to `submissions/<team>/`, with
   hyphens for spaces, and fill in its README.
2. Commit and push your code, train the final run, score it, then package and check it:

   ```shell
   ./run.sh eval final
   ./run.sh submit final <team>
   ```

   `submit` compares your commit with nanoTSFM's `main` to write `changes.diff`; add
   `--base <commit>` to name the starting commit yourself.
3. Open a pull request as [contributing](../CONTRIBUTING.md) describes.

Report other seeds and ablations in the README; only the submitted run is packaged. If you train on
data your own code builds, give the command in the README; verification runs it.

## Checks

- The folder holds exactly the three files, and the front matter is filled in.
- The run was committed, trained within the one-hour cap, left `evaluation.py` as it was at the
  base commit and did not use the toy data.
- `./run.sh submit` also checks your local checkpoint: its SHA-256, step and training time match
  `result.json`, and it returns finite, ordered quantiles through the forecast interface.

## Review

Maintainers verify a submission inside the team's repository at its commit, with the official fixed
files, so the team's code trains and the official code scores. On an A100 80GB this takes the run's
training time plus about 4 minutes:

```shell
git clone <repository> review && cd review && git checkout <commit>
git fetch https://github.com/Abel-ai-lab/nanoTSFM main
git checkout FETCH_HEAD -- src/nanotsfm/evaluation.py configs/gift-full.json scripts/submission.py
git fetch https://github.com/Abel-ai-lab/nanoTSFM pull/<number>/head
git checkout FETCH_HEAD -- submissions/<team>
uv run --extra gift python -m scripts.submission verify submissions/<team> --output verify
```

`verify` retrains the run from its recorded configuration, scores it on GIFT-Eval, and prints the
row for `docs/leaderboard.csv` when the retrained score is within 0.01 of the reported one.
Reviewers also read `changes.diff` and the report, then merge the pull request and add the row;
the leaderboard figure updates from that file.
