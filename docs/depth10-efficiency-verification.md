# Complete GIFT-Eval verification of the wide-phase 10-layer model

The 10-layer model was trained from random initialization with seed 7 on GEP-L.
It keeps the 384-dimensional state, eight heads, phase augmentation, and locked-history
CDF decoder. The 74,972 updates use batch size 256 and switch from 512 to 1,024
training points after step 42,972. CUDA Graph training is enabled in the reproduced configuration.

## Full-run comparison

Both models were scored on the same 97 GIFT-Eval tasks with matching windows and Seasonal
Naive denominators. The [23.35M reference](wide-phase-locked-experiment.md) is the archived
wide-phase seed-7 run.

| Metric | 12-layer reference | 10-layer candidate | Change |
| --- | ---: | ---: | ---: |
| Relative CRPS | 0.518656237278 | **0.518239700996** | −0.080% |
| Relative MASE | 0.757495043665 | **0.756230873446** | −0.167% |
| Parameters | 23,350,176 | **19,805,856** | −15.18% |
| Formal training time, seconds | 3,012.86 | **2,263.04** | −24.89% |
| Peak allocated training memory, GiB | 3.364 | **2.913** | −13.4% |
| Resumable checkpoint, bytes | 280,350,923 | **237,799,307** | −15.18% |
| Forecast-only checkpoint, bytes | 93,443,115 | **79,260,013** | −15.18% |
| Same-card peak CUDA reserved memory, MiB | 142 | **120** | −15.49% |

The training-time comparison is between the actual full runs. The reference used an A100 80 GB
PCIe and the candidate used an A100-SXM4-80GB, so hardware and graph execution both affect
this timing. The candidate training process took 2,299.76 seconds including its other work.
The candidate generated fresh predictions for all 97 tasks in 2,891.49 seconds and the
independent scorer took 317.63 seconds.

The CUDA allocator comparison used the same SXM4 card and eight fixed GEP windows at each
of 48, 480, and 896 steps. The maximum extra memory allocated during forecasting relative
to the model-and-input footprint was 13,329,920 bytes for each model. Total peak reserved memory
was lower for the 10-layer run. The forecast-only exporter preserved the values, dtypes,
and strides of all 98 tensors of the 10-layer checkpoint.

| Forecast length | Tasks | 12-layer relative CRPS | 10-layer relative CRPS | Change |
| --- | ---: | ---: | ---: | ---: |
| 1–48 | 53 | 0.530771 | 0.528939 | −0.345% |
| 49–512 | 21 | 0.474545 | 0.475417 | +0.184% |
| 513+ | 23 | 0.533358 | 0.534908 | +0.291% |

The complete benchmark led to a lower overall CRPS and MASE in this seed.
The small score margin and mixed CRPS across horizons warrant independent seeds.
This branch records a verified single-seed experiment.

## Reproduce

Set up the runtime, GIFT-Eval checkout, and persistent volume as described in the
[wide-phase experiment](wide-phase-locked-experiment.md). Use GIFT-Eval revision
`9a014e9e8ea130ba39c100c60d5dcbab7db57ac9` and data revision
`30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`. Set `UPSTREAM`, `GIFT_DATA_ROOT`,
and `RUNS` to the resulting absolute paths.

```bash
export TORCH=cuda
./run.sh setup
./run.sh train wide-phase-depth10-s7 configs/wide-phase-depth10-locked-74972.json 7
run="$RUNS/wide-phase-depth10-s7"
sha=$(sha256sum "$run/checkpoint.pt" | cut -d ' ' -f 1)
uv run --locked --extra gift --extra cuda python tools/forecast_locked_checkpoint.py \
  --checkpoint "$run/checkpoint.pt" --checkpoint-sha256 "$sha" \
  --upstream "$UPSTREAM" --tasks configs/gift-full.json --data-root "$GIFT_DATA_ROOT" \
  --output "$run/gift-forecasts" --hardware-receipt "$run/forecast-hardware.json"
uv run --locked --extra gift --extra cuda python -m scripts.score \
  --forecasts "$run/gift-forecasts" --upstream "$UPSTREAM" --tasks configs/gift-full.json \
  --data-root "$GIFT_DATA_ROOT" --output "$run/gift.json" --workers 6
uv run --locked --extra cuda python tools/export_forecast_checkpoint.py \
  "$run/checkpoint.pt" "$run/forecast-only.pt" --checkpoint-sha256 "$sha"
```

The versioned training, data sampling, and prediction files match the tested code.
The fixed evaluator, score implementation and task manifest use the pinned copies.

| Archived evidence | SHA-256 |
| --- | --- |
| Training configuration | `46a24f1f9870814734956ecfb1b2513fd76c04d2f3dfa977261a2d45e382d71c` |
| Resumable checkpoint | `07e52a29b29f90c9091867850a0c7f890e19c5a59591de11481a372c3e745a8f` |
| Forecast-only checkpoint | `cab686f18efce666cee549160a8869dea1052fc60181f9619aeff3394644e0cd` |
| Official score report | `93b59c463a178cd6ada544bcf13e6fe2ea4d03609f445185ca80fa6c265a27e0` |
| Complete run verification manifest | `07acdc361361defb50aaf8979e6007497fc6d062a0167417bce2d6916dd13770` |
| Paired efficiency acceptance manifest | `69516c002a2cea554845aa804984987f481acb04ba19d6b478b656d8b53d2b85` |
