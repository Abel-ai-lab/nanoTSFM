---
team: anannay-josh
description: source_power 1.5, sign-flip and downsampling augmentation, flip-symmetric nonnegative forecasts
members:
  - name: Anannay Sharma
    github: Anannay16
  - name: Josh Wright
    github: JoshWright22
ai_disclosure: Claude (Anthropic) helped find research papers relevant to the problem, helped implement the code changes, assisted in running the experiments, and helped refine this report. Every number here was checked against the run logs and score files.
---

# anannay-josh

## Hypothesis

Built on record 2 (Shu-Wan, `configs/series.yaml`: GEP-M, 5,000 steps, `source_power: 1`). The
current record is record 3 (sahil-saahir, GEP-L at 20,000 steps, verified 0.6121), so a record
needs a verified mean of about 0.599 or lower (a margin of 0.0133).

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
    same few. We chose 1.5 from a sweep of 1, 1.5, 2 and 3 (pilots below; power 2 tied on GIFT-Eval but was worse on GEP-Val). The cap on
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
  SXM 80GB. Training took 94 to 97 seconds per run, far under the
  3,600-second cap. Pilots and the ablation ran on an H100 SXM.
- **Choosing the final model:** by mean GIFT-Eval over 3 seeds, compared against our own 3-seed
  reproduction of record 2. GIFT-Eval is public and we selected on it, so part of each step's gain
  may be selection. Few choices were made this way: the power value, and whether to keep each rung.
  We report GEP-Val beside it.

## Results

GIFT-Eval: geometric relative CRPS against Seasonal Naive (lower is better), mean ± sd over seeds.

### Final runs (`result.json`)

| Run | Seed | GIFT-Eval CRPS | GIFT-Eval MASE | GEP-Val CRPS | Train s | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final-s7 | 7 | 0.5818 | 0.8511 | 0.6084 | 94 | 5,000 |
| final-s1 | 1 | 0.5798 | 0.8464 | 0.6122 | 97 | 5,000 |
| final-s2 | 2 | 0.5802 | 0.8488 | 0.6081 | 96 | 5,000 |
| **Mean** | | **0.5806 ± 0.0011** | 0.8488 ± 0.0023 | 0.6096 ± 0.0023 | | |

The mean is 0.032 below record 3's verified 0.6121, against a required margin of 0.013. The final
runs match the pilots on the H100 (0.5815 ± 0.0009). GEP-Test was not run.

**By horizon and frequency** (geometric mean of relative CRPS over the tasks in each group, mean of
3 seeds; record 2's numbers are from its report):

| Group | Record 2 | This submission |
| --- | ---: | ---: |
| short | 0.618 | 0.574 |
| medium | 0.665 | 0.592 |
| long | 0.644 | 0.585 |
| second | 1.689 | 1.277 |
| minute | 0.650 | 0.573 |
| hour | 0.528 | 0.505 |
| day | 0.486 | 0.458 |
| week | 0.702 | 0.692 |
| month / quarter / year | 0.864 (together) | 0.774 / 0.809 / 0.842 |

### Ablation: one change at a time (3 seeds each, seeds 7, 1, 2, on an H100)

| Configuration | GIFT-Eval CRPS | Step | GEP-Val CRPS |
| --- | ---: | ---: | ---: |
| Record 2 (our reproduction, `source_power` 1) | 0.6340 ± 0.0066 | | 0.6140 |
| + `source_power` 1.5 | 0.6250 ± 0.0026 | −0.009 | 0.6171 |
| + `sign_flip` 0.5, `downsample` 0.3 | 0.5923 ± 0.0024 | −0.033 | 0.6179 |
| + `flip_symmetric` | 0.5852 ± 0.0014 | −0.007 | (same weights) |
| **+ `clamp_positive` (this submission)** | **0.5815 ± 0.0009** | −0.004 | (same weights) |

The two forecast options change no weights, so they were scored on the checkpoints of the rung
above them, alone and together:

| Forecast options | on `source_power` 1.5 | on + augmentation |
| --- | ---: | ---: |
| none | 0.6250 ± 0.0026 | 0.5923 ± 0.0024 |
| `clamp_positive` only | | 0.5883 ± 0.0019 |
| `flip_symmetric` only | 0.6099 ± 0.0007 | 0.5852 ± 0.0014 |
| both | 0.6049 ± 0.0011 | 0.5815 ± 0.0009 |

Each option helps on its own, and together they help more than either. On record 2's seed-7
checkpoint (0.6401) they scored 0.6363 (clamp), 0.6226 (flip) and 0.6176 (both).

**A caveat on the sampling step.** With augmentation on, `source_power` 1 scored 0.5924 on seed 7
against 0.5942 for 1.5 (one seed each). So the −0.009 from `source_power` 1.5 is clear without
augmentation, but we cannot show it survives once augmentation is on. We kept 1.5 because it was
chosen, and confirmed with 3 seeds, before the augmentation runs.

### Other pilots (seed 7, GEP-M, 5,000 steps, `source_power` 1 unless noted)

The pilot configs and the code for options we did not keep (`freq_power`, `mask_span_max`, EMA,
weight decay) are not in this pull request. They are at
[`ffe802c`](https://github.com/Anannay16/nanoTSFM/tree/ffe802c/configs) on our fork, with the
scripts that ran them in `exp/`. Record 2's seed 7 scored 0.6401.

| Change | GIFT-Eval CRPS | GEP-Val CRPS |
| --- | ---: | ---: |
| `source_power` 1.5 / 2 / 3 | 0.6228 / 0.6237 / 0.6319 | 0.6154 / 0.6270 / 0.6403 |
| frequency-group balancing `freq_power` 0.5 / 1 | 0.6402 / 0.6502 | 0.6284 / 0.6384 |
| masked spans up to 16 patches | 0.6339 | 0.6139 |
| context 2,048 | 0.6340 | 0.6005 |
| patch 16 | 0.6305 | 0.6099 |
| EMA of weights (0.999) | 0.6335 | 0.6089 |
| weight decay 0.1 | 0.6332 | 0.6185 |
| augmentation (`source_power` 1) | 0.5924 | 0.6185 |
| 3-model ensemble (ruled out by the maintainer; not used) | 0.6170 | |

## Discussion

The results mostly support the hypothesis. Augmentation and both forecast options clearly lowered
GIFT-Eval; the sampling change helped on its own but not visibly once augmentation was on. Together
they reach 0.5806 in the final runs, 0.053 below our reproduction of record 2 and 0.032 below
record 3's verified score (0.6121).

- **Augmentation is most of the gain (−0.033).** Against record 2, the final model improves at every
  horizon and frequency. The largest gains are on secondly data (1.689 to 1.277, still worse than
  Seasonal Naive) and minute data (0.650 to 0.573), and medium and long horizons gain more than
  short. This fits downsampling, which shows the model coarser views of fine-grained series,
  though we did not break the ablation down by group to confirm it.
- **Sampling (−0.009 without augmentation)** follows record 2's trend from 0 to 1. Power 2 tied
  1.5 on seed 7 (0.6237 against 0.6228) but did worse on GEP-Val, and power 3 was worse on both,
  so 1.5 was kept. As noted above, its gain is not visible once augmentation is on. Record 3 found
  1.5 hurt on GEP-L, so the best power depends on the slice.
- **The forecast options (−0.007, −0.004)** cost no training time, only a second forward pass for
  the flip. Clamping can only lower the loss on series whose future stays nonnegative.

**What did not help:** frequency-group balancing (0.640 and 0.650). Up-weighting rare frequencies
in GEP-M means repeating a handful of sources (the secondly group is 2 series), which overfits; GEP-Val
got worse too. Masked spans of up to 16, patch 16, EMA and weight decay were all within noise of record 2's
spread (± 0.0066). Context 2,048 improved GEP-Val most (0.6005) but not GIFT-Eval (0.6340): it fits
the corpus without transferring.

**Limitations and next steps:** most pilots have one seed, and record 2's reproduction is noisy
(± 0.0066), so small pilot differences are not meaningful. Our changes are independent of record 3's
(GEP-L with 20,000 steps), so the obvious next step is to combine them, retuning `source_power` for
GEP-L. Secondly data is still worse than Seasonal Naive.

## Contributions and attribution

- **Anannay Sharma:** surveyed recent small forecasting models for ideas that fit the one-hour
  budget, designed and ran every pilot, ablation and final run on RunPod, checked the rules with the
  maintainer (augmentation, test-time changes and ensembles), analysed the results, and wrote this
  report.
- **Josh Wright:** discussed ideas and ran his own experiments on other techniques, which he is
  developing into the team's separate submission.
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
