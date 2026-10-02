<!-- Submitting a result? Fill in Submission and delete Change. Otherwise, the reverse.
Title a submission "<team>: <mean GIFT-Eval CRPS>, <what changed>", for example
"my-team: 0.633, weight every series equally". -->

## Submission

- Record folder: `records/<YYYY-MM-DD>_<name>/`
- Runs' commit, from `result.json`:
- GIFT-Eval relative CRPS, $\text{mean} \pm \text{sd}$ over $n$ runs, from `./run.sh submit`:

**What changed and why**, in two or three sentences; the report in your README has the details.

**Results**, $\text{mean} \pm \text{sd}$ over every run of each configuration:

| | GIFT-Eval CRPS | GEP-Val | Training |
| --- | ---: | ---: | ---: |
| Record you built on | | | |
| This submission | | | |

**Ablation:** how much of the gain each change brings.

**Reproduce:** `./run.sh train <run> <config> <seed>` for each run, and any data command.

- [ ] Every run of the final configuration is in `result.json`, trained at one pushed commit
      after merging `main`.
- [ ] `./run.sh submit` passed and says whether the runs beat the record.
- [ ] If my data differs from the record's, the report says what changed, how I chose it, and why.
- [ ] My fork is public.
- [ ] No runs, checkpoints, data or notebook outputs are committed.

## Change

What changes and why, the issue it addresses, and how you checked it.
