# Contributing

## Submitting a result

1. Fork this repository and work in your fork. It must be public; your code stays there.
2. Score one run on GIFT-Eval and package it (see [submission](docs/submission.md)):

   ```shell
   ./run.sh eval final
   ./run.sh submit final <team>
   ```

3. Open a pull request that adds only `submissions/<team>/`. Your score there is reported.
4. A check confirms the folder is well formed. A maintainer then retrains your run from your code
   on an A100 and scores it; only this verified score reaches the leaderboard.

To update a result, change your folder in a new pull request.

## Keep pull requests small

- One submission per pull request, and nothing else in it.
- Never commit runs, checkpoints, data, caches or notebook outputs. Git ignores them; don't
  force-add them. Files over 1 MB are rejected.

## Changing nanoTSFM itself

Open an issue first and keep the pull request focused. These files are fixed and change only
through such an issue: `src/nanotsfm/evaluation.py`, `configs/gift-full.json` and
`scripts/submission.py`.
