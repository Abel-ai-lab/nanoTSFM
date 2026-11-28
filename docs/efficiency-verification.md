# Efficiency verification for the wide-phase model

The reference is the archived GEP-L seed-7 run from the
[wide-phase experiment](wide-phase-locked-experiment.md). Its locked-history predictions score
**0.5186562372783312 relative CRPS** and **0.7574950436648171 relative MASE** on the fixed 97
GIFT-Eval tasks.

| Change | Measured resource effect | Quality evidence |
| --- | --- | --- |
| Skip unused phases in the last rolling block | 358.20 to 355.28 seconds for the nine affected A100 tasks | 3,096,828 recomputed quantiles were bitwise identical. The remaining 88 tasks reused the original signed forecasts. The fixed 97-task scores above were reproduced. |
| Export a forecast-only checkpoint | 280,350,923 to 93,443,115 bytes, a 66.67% reduction | All 114 model tensors and their dtypes and strides matched. Production forecasts matched bitwise for 25,632 values on six GEP origins across 48, 480 and 896 steps. |
| Use CUDA Graph training | 15.501 to 8.540 training seconds in a same-seed, 192-update A100 pair with real GEP-L batches | Final losses, model tensors, AdamW state and CPU/CUDA RNG state matched exactly. Peak allocated training memory was 3.364 and 3.405 GiB respectively. |

The model currently has **23,350,176 parameters**. The forecast export drops AdamW and
training RNG state. Keep the training checkpoint for resuming a run. The phase change
keeps the same tensor geometry for every phase it executes and checks history availability
for every phase. Earlier rolling blocks still compute every phase for median feedback.

The training speed measurement covers the short paired run and both training lengths.
An independently seeded full-budget graph run completed and scored
0.523015030 on GIFT-Eval. A full-budget same-seed timing and score comparison is pending.

## Reproduction

Use the fixed scorer and data revision documented in the wide-phase experiment above.
For the optional CUDA Graph training run on A100, select the graph configuration.

```bash
./run.sh train wide-phase-graph-s7 configs/wide-phase-locked-graph-74972.json 7
```

The forecast checkpoint accepts the usual evaluation and forecast entry points. Export
from the sealed training checkpoint, with both files on the project's persistent volume.

```bash
uv run --locked --extra cuda python tools/export_forecast_checkpoint.py \
  /workspace/runs/wide-phase-s7/checkpoint.pt /workspace/runs/wide-phase-s7/forecast-only.pt \
  --checkpoint-sha256 94184eceea890997f834d4a8060dc241ce7f14abc4584dc491382c39de36f435
```

The digest in this example is for the archived seed-7 checkpoint. For a fresh training run,
use the digest of its own checkpoint.

- Locked-history reference score report SHA-256
  `70007dbbf4217671384d5f3ef17152e158759fd6bcb3ad047a6a6e6d736030a8`
- Nine-task phase-skip and full-score verification manifest SHA-256
  `ec5082c82b120b8b739c0567201f705e5e1a386f395331ac7a5678900d8d9c1d`
- Forecast-only checkpoint SHA-256
  `d083636e0b3311cd832a95e6149bf275c365b699170c7e76f618d69f9f964b66`

The smaller 8-layer model has 16,261,536 parameters and a complete score of
0.521731717. It exceeds the reference CRPS. A 19,805,856-parameter 10-layer run is
queued for full verification. This branch reports a single-seed efficiency experiment.
