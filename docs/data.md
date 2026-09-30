---
name: data
description: The pretraining corpus, the split, and the GEP datasets on Hugging Face.
meta:
  type: knowledge
---

# Data

All data comes from one pinned corpus, GIFT-Eval Pretrain. A hash of each series' name puts it in
train (80%), validation (10%) or test (10%). The baseline trains on GEP-M; GEP-Val and GEP-Test
are in-distribution diagnostics. GIFT-Eval, the score, shares no series with any of it.

## Loading

[abel-lab/nanoTSFM-pretrain](https://huggingface.co/datasets/abel-lab/nanoTSFM-pretrain) has one
config per dataset:

```python
from datasets import load_dataset

train = load_dataset("abel-lab/nanoTSFM-pretrain", "GEP-M", split="train")
```

Rows hold `item_id` (`<source>/<item_id>`), `source`, `freq`, `source_row` and `target`, a
`[variate][time]` float32 list with NaN for missing values. In the code, `nanotsfm.data.load(name,
split)` loads at the pinned revision, and `arrays(dataset)` returns the targets as numpy views.

A training config's `data:` names a config such as `GEP-L`, a local folder of `train-*.parquet`
and `validation-*.parquet` files in the same layout, or `toy`, the built-in sine waves behind
`./run.sh toy`. `./run.sh data` downloads GEP-M, GEP-Val and GEP-Test (730 MB) into the Hugging
Face cache, which `HF_HOME` moves (on Sol, use `/scratch`).

## Source corpus

[GIFT-Eval Pretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain) (revision 6830b62)
was released with GIFT-Eval and shares no series with it: about 230B points in 975 GB across 87
dataset families, 88% of the bytes in two climate simulations (cmip6, era5). The GEP datasets use
8.6 GB of it: every source under 250 MB, three mid-size sources, and the first shard of each of the
eleven large ones (five for GEP-L). That covers every GIFT-Eval frequency except 10-second data.

## Split

The hash of `<source>/<item_id>` fixes each series' split in every dataset, so no slice trains on a
series that another holds out. A multivariate series stays whole.

## Training slices

Each slice keeps up to N series per source (seed 7) and the latest points of each. Training samples
a source uniformly, then a series, then a crop.

| Slice | Series per source | Points per series | Series | Points | Download |
| --- | --- | --- | --- | --- | --- |
| GEP-S (small) | $\leq$ 50 | $\leq$ 4,096 | 3.4k | 33M | 94 MB |
| **GEP-M** (medium, default) | $\leq$ 1,000 | $\leq$ 8,192 | 32.9k | 252M | 615 MB |
| GEP-L (large) | $\leq$ 50,000, 5 shards of large sources | $\leq$ 8,192 | 500k | 2.1B | 6.2 GB |

GEP-M is the smallest slice that gives the 2-minute baseline its full score. More data pays off
with longer training: at 20,000 steps, GIFT-Eval was 0.693 on GEP-M and 0.661 on GEP-L.
Training speed does not depend on the slice. The first load of GEP-L takes 6.5 minutes (download
plus conversion to an Arrow cache) and 15 GB of disk; later loads take 2 seconds and 2 GB of RAM.

## GEP-Val and GEP-Test

Each holds up to 100 held-out series per source (latest 8,192 points). The `GEP-tasks` config
defines their tasks the way GIFT-Eval does:

- One task per source and term. Short horizons follow GIFT-Eval's table by frequency (48 steps for
  hourly and minute data, 30 for daily, 12 for monthly); medium and long terms are $10\times$ and $15\times$ that,
  for sub-daily sources only.
- Each task forecasts the last k horizons of every long-enough series, with k = ceil(10% of the
  shortest series ÷ horizon), between 1 and 20. Tasks with fewer than 10 forecasts are dropped.
- Scores are CRPS and MASE relative to Seasonal Naive per task, then a geometric mean over tasks.

| Set | Series | Sources | Tasks (short, medium, long) | Forecast windows | Download |
| --- | --- | --- | --- | --- | --- |
| GEP-Val | 3,279 | 70 | 129 (70, 30, 29) | 29.2k | 56 MB |
| GEP-Test | 3,306 | 70 | 131 (70, 31, 30) | 30.0k | 62 MB |

Fifteen sources have no series in a given set, and two more have too few forecasts for a task.

## Known gaps

- The split cannot see the same series stored under two sources, such as related PEMS datasets.
- Large sources contribute only their first shards, which may not be random samples.

## Rebuilding

`build/build.py` in the dataset repository rebuilds every GEP config row for row; its card has the
commands.
