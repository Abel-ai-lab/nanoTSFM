# Contributing

## Submitting a result

1. Fork this repository and work on a branch of your fork, which must be public.
2. Merge `main`, then train three or more seeds of your final configuration, score them on
   GIFT-Eval and package them (see [submission](docs/submission.md)):

   ```shell
   ./run.sh submit <YYYY-MM-DD>_<name> final-s7 final-s1 final-s2
   ```

3. Open a pull request from your branch with your code and `records/<YYYY-MM-DD>_<name>/`, and fill
   in its template.
4. A check confirms the folder is well formed and the code matches the runs. A maintainer reads
   the code, retrains every run on an A100 and merges the pull request if the verified runs beat
   the record; otherwise it is closed with its verified score.

## Keep pull requests small

- One submission per pull request, with only the code it needs.
- Never commit runs, checkpoints, data, caches or notebook outputs. Git ignores them; don't
  force-add them. Files over 1 MB are rejected.

## Changing nanoTSFM itself

Open an issue first, then a focused pull request that fills in the template's Change section. A
maintainer reviews it.

- **Training stays fixed.** A change to nanoTSFM leaves `src/`, `configs/`, `pyproject.toml` and
  `uv.lock` alone, so `main` keeps training the current record and open submissions stay valid. A
  check fails any pull request that changes these files without a record folder.
- **When training must change** (a bug fix, a dependency update), a maintainer re-runs the current
  record's seeds at the new code and adds the `changes training` label. The change ships in a new
  minor version. Open submissions then merge `main` and train again.
- **The score's files**, `src/nanotsfm/evaluation.py` and `configs/gift-full.json`, change only in
  a new major version. Maintainers own `scripts/submission.py` and `.github/`.

## Versions

Releases are tagged `vMAJOR.MINOR.PATCH`, and the version is in `pyproject.toml` and `CITATION.cff`.

- **Major:** the score or the rules change, so results are not comparable across majors.
- **Minor:** training code or dependencies change; the current record is re-verified.
- **Patch:** documentation and tools only.
