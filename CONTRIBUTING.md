# Contributing

## Attempting the record

1. Fork this repository and work on a branch of your fork, which must be public.
2. Train three or more seeds of your final configuration on top of the current record, score them
   on GIFT-Eval and package them (see [submission](docs/submission.md)):

   ```shell
   ./run.sh submit <YYYY-MM-DD>_<name> final-s7 final-s1 final-s2
   ```

3. Open a pull request from your branch with your code and `records/<YYYY-MM-DD>_<name>/`, and fill
   in its template.
4. A check confirms the folder is well formed and the code matches the runs. A maintainer reads
   the code, retrains every run on an A100 and merges the pull request if the verified runs beat
   the record; otherwise it is closed with its verified score.

## Keep pull requests small

- One attempt per pull request, with only the code it needs.
- Never commit runs, checkpoints, data, caches or notebook outputs. Git ignores them; don't
  force-add them. Files over 1 MB are rejected.

## Changing nanoTSFM itself

Open an issue first and keep the pull request focused. These files are fixed and change only
through such an issue: `src/nanotsfm/evaluation.py`, `configs/gift-full.json`,
`scripts/submission.py` and `.github/`.
