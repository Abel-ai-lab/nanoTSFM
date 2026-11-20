---
team: anannay-sharma
description: source_power 1.5, sign-flip and downsampling augmentation, flip-symmetric nonnegative forecasts
members:
  - name: Anannay Sharma
    github: Anannay16
  - name: Josh Wright
    github: JoshWright22
ai_disclosure: Claude (Anthropic) helped find research papers relevant to the problem, helped implement the code changes, assisted in running the experiments, and helped refine this report. Every number here was checked against the run logs and score files.
---

# anannay-sharma

## Hypothesis

Built on record 2 (Shu-Wan, `configs/series.yaml`: GEP-M, 5,000 steps, `source_power: 1`). The
current record is record 3 (sahil-saahir, GEP-L at 20,000 steps, verified 0.6121), so a record
needs a verified mean of 0.5991 or lower.

Record 2 trains for about 81 seconds of a 3,600-second budget and still loses most to Seasonal
Naive on data the corpus covers thinly. We expected three cheap changes, none of which adds
training time, to each lower GIFT-Eval, and to stack:

1. **Tilt sampling further toward large sources** (`source_power` 1 → 1.5). Record 2 found each
   step up in power helped (0 → 0.5 → 1) and stopped at the code's cap of 1.
2. **Augment real series** with negation and coarser views of the same series. The forecaster
   should be indifferent to a series' sign, and averaging over 2–4 steps shows it frequencies the
   corpus has few of.
3. **Build two invariances into `forecast()`.** Forecasting `-x` should give `-f(x)` with the
   quantiles mirrored, and a series that has never been negative should get no negative quantiles.

The ablation adds one change at a time, 3 seeds per rung.

## Method

- **Model and objective:** unchanged from record 2. The same 3.3M-parameter patch transformer
  (3,326,240 parameters): width 256, 3 causal time layers plus one variate layer, 4 heads, patch 32,
  context 1,024, prediction 512. Masked next-patch quantile loss over 9 quantiles. The forecast
  options add no parameters.
- **Data (changed): what, how we chose it, and why.** Same corpus (GEP-M, 26,154 series of GIFT-Eval
  Pretrain). Two things differ from record 2. Both are applied to real GEP-M series as windows are
  drawn, and nothing reads GIFT-Eval data:
  - *Sampling:* a source is drawn with weight (series count)^1.5 instead of ^1. Variety beats
    repetition: large sources hold many distinct series, while up-weighting small ones repeats the
    same few. We chose 1.5 from a sweep of 1, 1.5, 2 and 3 (pilots below). The cap on
    `source_power` is raised from 1 to 2.
  - *Augmentation:* with probability 0.5 a window is negated (`sign_flip`). With probability 0.3 it
    is averaged over k = 2, 3 or 4 consecutive steps (`downsample`), a coarser view of the same
    series. NaNs are skipped in the average, and a factor is dropped if the series would become too
    short. Both probabilities were set once and not tuned. TiRex, Reverso and TinyCast all report
    that augmentation like this helps, and Reverso and TinyCast use exactly these two. Before relying
    on them we asked the maintainer (Shu Wan), who confirmed that transformations of the training
    data are allowed.
- **Forecasting (`model.py`, `forecast()`), interface unchanged:**
  - `flip_symmetric`: the forecast is `(f(x) − reverse(f(−x))) / 2`, where reverse flips the order
    of the quantiles. Negating the input then exactly negates the forecast. It doubles forecast
    time but not training. TimesFM 2.5 does this by default (flip invariance); Reverso and TinyCast
    report gains from it.
  - `clamp_positive`: if every observed value of a series' context is ≥ 0, its quantiles are clipped
    at 0. When the truth is ≥ 0, raising a negative quantile to 0 can only lower the pinball loss.
    TimesFM 2.5 also does this by default (`infer_is_positive`).
  - We asked the maintainer whether test-time changes are allowed: yes, provided the report gives
    each part's gain (see the ablation). Ensembles of several models were ruled out, and we do not
    use them.
- **Training:** unchanged otherwise. 5,000 steps, batch 256, AdamW at a peak learning rate of 1e-3,
  seed-controlled data order. Config: `configs/final.yaml`. Every new option defaults to off, so
  older configs train exactly as before.
- **Seeds and hardware:** final runs used seeds 7, 1 and 2 at commit `cb9e828`, on one RunPod A100
  SXM 80GB. Training took TODO_TRAIN_S seconds per run, well under the cap. Pilots and the ablation
  ran on an H100 SXM.
- **Choosing the final model:** by mean GIFT-Eval over 3 seeds, compared against our own 3-seed
  reproduction of record 2. GIFT-Eval is public and we selected on it, so part of each step's gain
  may be selection. Few choices were made this way: the power value, and whether to keep each rung.
  We report GEP-Val and GEP-Test beside it.

## Results

GIFT-Eval: geometric relative CRPS against Seasonal Naive (lower is better), mean ± sd over seeds.

### Final runs (`result.json`)

| Run | Seed | GIFT-Eval CRPS | GEP-Val CRPS | GEP-Test CRPS | Train s | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final-s7 | 7 | TODO | TODO | TODO | TODO | 5,000 |
| final-s1 | 1 | TODO | TODO | TODO | TODO | 5,000 |
| final-s2 | 2 | TODO | TODO | TODO | TODO | 5,000 |
| **Mean** | | **TODO ± TODO** | TODO | TODO | | |

`./run.sh submit` reports a gap of TODO to record 3 (0.6121), against a required margin of 0.013.

### Ablation: one change at a time (3 seeds each, seeds 7, 1, 2)

| Configuration | GIFT-Eval CRPS | Step | GEP-Val CRPS |
| --- | ---: | ---: | ---: |
| Record 2 (our reproduction) | 0.6340 ± TODO | | TODO |
| + `source_power` 1.5 | 0.6250 ± TODO | −0.009 | TODO |
| + `sign_flip` 0.5, `downsample` 0.3 | 0.5923 ± TODO | −0.033 | TODO |
| + `flip_symmetric` | 0.5852 ± TODO | −0.007 | (unchanged weights) |
| **+ `clamp_positive` (this submission)** | **0.5815 ± 0.0009** | −0.004 | (unchanged weights) |

The two forecast options do not change the weights, so they were scored on the same checkpoints as
the rung above them. Without augmentation they help as well: `source_power` 1.5 with flip scores
0.6099 ± 0.0007, and with flip and clamp 0.6049 ± 0.0011.

### Other pilots (1 seed each unless noted, on GEP-M at 5,000 steps)

The pilot configs and the code for options we did not keep (`freq_power`, `mask_span_max`, EMA,
weight decay) are not in this pull request. They are at
[`ffe802c`](https://github.com/Anannay16/nanoTSFM/tree/ffe802c/configs) on our fork, with the
scripts that ran them in `exp/`.

| Change | GIFT-Eval CRPS |
| --- | ---: |
| `source_power` 2 / 3 | TODO / TODO |
| frequency-group balancing `freq_power` 0.5 / 1 | TODO / TODO |
| masked spans up to 16 patches | TODO |
| context 2,048 | TODO |
| patch 16 | TODO |
| EMA of weights | TODO |
| weight decay 0.1 | TODO |

## Discussion

The results support the hypothesis: each change lowered GIFT-Eval, and they stack to 0.5815, which is
0.052 below our reproduction of record 2 and TODO below record 3's verified score.

- **Augmentation is most of the gain (−0.033).** TODO: the groups it helped most (for example
  minute-level data, which downsampling fills in).
- **Sampling (−0.009)** agrees with record 2's trend. Above 1.5 it got worse again, as the largest
  sources start to crowd out the rest. Record 3 found 1.5 hurt on GEP-L, whose sources are capped at
  50,000 series, so the best power depends on the slice.
- **The forecast options (−0.007, −0.004)** cost no training. Clamping helps most on count-like
  series (solar, sales, car parts, hospital) and cannot hurt when their truth is nonnegative.

**What did not help:** frequency-group balancing. Up-weighting rare frequencies in GEP-M means
repeating a handful of sources (the secondly group is 2 series), which overfits, the same finding as
record 3. TODO: the other negative pilots.

**Limitations and next steps:** most pilots have one seed. Our changes are independent of record 3's
(GEP-L with 20,000 steps), so the obvious next step is to combine them, retuning `source_power` for
GEP-L. Secondly data is still worse than Seasonal Naive.

## Contributions and attribution

- **Anannay Sharma:** TODO (the contributions paragraph).
- **Josh Wright:** TODO (his part in this submission).
- **Built on** record 2 by Shu-Wan and the nanoTSFM baseline; record 3 by sahil-saahir for context.
- **Borrowed ideas:** flip invariance and the positivity clamp from
  [TimesFM 2.5](https://huggingface.co/docs/transformers/model_doc/timesfm2_5); sign-flip and
  downsampling augmentation from [Reverso](https://arxiv.org/abs/2602.17634) and
  [TinyCast](https://arxiv.org/abs/2608.15767); augmentation ablations in
  [TiRex](https://arxiv.org/abs/2505.23719); frequency balancing from
  [Toto 2.0](https://arxiv.org/abs/2605.20119) and TinyCast.
- **AI assistance:** Claude (Anthropic) helped find relevant research papers, helped implement the code
  changes, assisted in running the experiments, and helped refine this report. All numbers were
  checked against the run outputs.
