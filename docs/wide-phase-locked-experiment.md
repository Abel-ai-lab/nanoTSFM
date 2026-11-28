# Wider phase sampling with locked history

This configuration ports the archived seed-7 wide-phase recipe to the standard nanoTSFM
runner. It starts from random weights and trains one 23,350,176-parameter model on GEP-L.
The 74,972 updates switch from 512 to 1,024 input points after update 42,972. The batch size,
optimizer schedule and 3,600-second cap follow the existing experiment.

During training, one quarter of windows select a phase stride. The sampling list has five
entries each for strides 2 and 4, and one entry for stride 3 and each stride from 5 through 12.
At inference, the published CDF and locked-history rules use the existing observed-history
period detector. The same checkpoint serves all phases and views. See
[period-locked history and CDF pooling](locked-history-experiment.md) for the prediction formulas.

## Archived result

All rows use the same wide-phase checkpoint and the fixed 97-task GIFT-Eval scorer.

| History used by CDF pooling | Relative CRPS | Relative MASE |
| --- | ---: | ---: |
| Original history | 0.519380232 | 0.758161646 |
| Extended history for the first period-locked block | **0.518656237** | **0.757495044** |

The A100 80 GB PCIe run completed in 3,012.86 measured training seconds. The PyTorch allocation
peak was 3.36 GiB. The extended-history evaluation reused signed same-checkpoint predictions on
unaffected rows. Its recorded runtime includes selective recomputation.

- Archived training config SHA-256 `ead24cc72c34ea1f82393d5523ad822dc708ca58afddb76555d0d357a4ee4202`
- Seed-7 checkpoint SHA-256 `94184eceea890997f834d4a8060dc241ce7f14abc4584dc491382c39de36f435`
- Locked-history score report SHA-256 `70007dbbf4217671384d5f3ef17152e158759fd6bcb3ad047a6a6e6d736030a8`

The portable config retains the archived training fields and enables locked history in the model
config. The reported score comes from the archived single-seed run. A fresh run of this port
and a same-commit three-seed leaderboard submission are pending.

## Run on a project RunPod volume

```bash
export TORCH=cuda
project_volume=/workspace
export RUNS="$project_volume/runs"
export HF_HOME="$project_volume/hf-cache"
./run.sh setup
./run.sh train wide-phase-s7 configs/wide-phase-locked-74972.json 7

upstream="$project_volume/gift-eval"
git clone https://github.com/SalesforceAIResearch/gift-eval.git "$upstream"
git -C "$upstream" checkout 9a014e9e8ea130ba39c100c60d5dcbab7db57ac9
uv run --locked --extra gift --extra cuda python -m nanotsfm.evaluation gift \
  --checkpoint "$RUNS/wide-phase-s7/checkpoint.pt" --upstream "$upstream" \
  --tasks configs/gift-full.json --output "$RUNS/wide-phase-s7/gift-forecasts" \
  --device cuda --workers 1
uv run --locked --extra gift --extra cuda python -m scripts.score \
  --forecasts "$RUNS/wide-phase-s7/gift-forecasts" --upstream "$upstream" \
  --tasks configs/gift-full.json --output "$RUNS/wide-phase-s7/gift.json" --workers 6
```

Both evaluation commands use the fixed upstream code revision above and the GIFT data revision
`30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`.
