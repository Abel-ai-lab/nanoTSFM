---
team: shira-manas
description: Record 4 with record 3's data and longer training - GEP-L, source_power 1, 40,000 steps
members:
  - name: Shira Wisniewski
    github: swisnie2
  - name: Manas Tripathi
    github: manas-tripathi27
ai_disclosure: Claude (Anthropic) assisted with experiment design, code changes, analysis scripts, and drafting this report. The team ran all experiments and checked every number, config and line of text.
---

# shira-manas

**Score: 0.5633 ± 0.0026 GIFT-Eval CRPS**, 0.018 below record 4 (0.5811).

We took record 4 and trained it the way record 3 trained: on the bigger GEP-L dataset and for longer.
The whole change is three lines in one config file, [`configs/combo-40k.yaml`](../../configs/combo-40k.yaml).
No code changed.

## Hypothesis

The two most recent records improved the model in different ways:

- **Record 3** gave the model **more data and more training**: the GEP-L dataset (about 400k series
  instead of 26k) and 20,000 training steps instead of 5,000. Neither half worked alone. More steps
  on the small dataset made the model memorize it, and the big dataset at 5,000 steps didn't have
  time to pay off.
- **Record 4** kept the small dataset but **made the model see each series in more ways**. It flips
  training windows upside down and smooths them into coarser versions. At forecast time it also
  averages each forecast with the forecast of the flipped series, and never predicts negative values
  for a series that has never been negative.

Since these are different kinds of improvements, we expected them to **add up**. Our goal was to beat
record 4 by more than the required 0.013.

We also made a second, more specific guess. Record 3 found that training past 20,000 steps didn't
help. We thought record 4's augmentation would change that: the model sees a slightly different
version of each window every time, so it should take longer to start memorizing. That means
**40,000 steps should beat 20,000 steps** once augmentation is on.

To test this, our ablation adds one change at a time on top of record 4: first the big dataset with
20,000 steps, then 40,000 steps. Each step has 3 seeds.

## Method

**What stayed the same as record 4:** the model (3,326,240 parameters: width 256, 3 time layers plus
1 variate layer, 4 heads, patches of 32, 1,024 steps of context, 512 steps of forecast), the training
objective (quantile loss over 9 quantiles), the augmentation (`sign_flip: 0.5`, `downsample: 0.3`),
and the two forecast tricks (`flip_symmetric`, `clamp_positive`).

**What we changed:**

| Setting | Record 4 | Ours | Why |
| --- | --- | --- | --- |
| `data` | GEP-M (26,154 series) | **GEP-L** (399,356 series) | Record 3 showed longer training only works with more series |
| `steps` | 5,000 | **40,000** | Our guess that augmentation lets the model use more training (tested below) |
| `source_power` | 1.5 | **1** | See below |

**Why `source_power` goes back to 1:** this setting decides how often big data sources are picked
compared with small ones. On the small dataset, every source is capped at 1,000 series, and 1.5
works best; both our own sweep (below) and record 4 found that. GEP-L lets a source hold up to
50,000 series, so 1.5 would spend almost all of training on a few huge sources. Record 3 tested
1.5 on GEP-L, and it hurt (0.623 against 0.607), so we used 1. We did not re-tune it ourselves.

**Data rules:** both datasets are official slices of GIFT-Eval Pretrain, built by the repository's
own code. We use no other data, and nothing reads GIFT-Eval's test data.

**Training:** batch 256 and AdamW at a peak learning rate of 1e-3, with 5% warmup and a cosine decay
that stretches to the longer run. Each final run took **574 to 773 seconds**, well under the
one-hour limit. The seed-7 run was slower; it was the first run in a fresh folder, and we did not
find the cause.

**Runs:** seeds 7, 1 and 2, at commit `d58bbe4` (record 4's code plus our one config file), on one
RunPod **A100 80GB PCIe**.

**How we picked the final setup:** we first tried 20,000 and 40,000 steps with one seed each, and
kept 40,000 because it scored better on GIFT-Eval. That is the only choice we made by looking at
the benchmark. We then ran the 3 final seeds on a clean branch that holds only the final config.

## Results

All scores are GIFT-Eval CRPS relative to Seasonal Naive (lower is better), as mean ± spread over 3
seeds.

### Final runs

| Run | Seed | GIFT-Eval CRPS | MASE | GEP-Val CRPS | Training (s) | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final40k-s7 | 7 | 0.5608 | 0.8168 | 0.5955 | 773 | 40,000 |
| final40k-s1 | 1 | 0.5631 | 0.8226 | 0.5957 | 590 | 40,000 |
| final40k-s2 | 2 | 0.5659 | 0.8251 | 0.5969 | 574 | 40,000 |
| **Mean** | | **0.5633 ± 0.0026** | **0.8215 ± 0.0043** | 0.5960 ± 0.0008 | | |

The gap to record 4 is **0.0178**; the rule needs 0.0133. The seed-7 run gave exactly the same score
(0.5608) as our earlier one-seed try, so the result is reproducible. GEP-Test was not run.

### Ablation: adding one change at a time

3 seeds each (7, 1 and 2), all on the same A100 80GB PCIe.

| Setup | Data | Steps | GIFT-Eval CRPS | Change | GEP-Val CRPS | Training (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Record 4 (verified by the maintainers) | GEP-M | 5,000 | 0.5811 ± 0.0010 | | | |
| Record 4 (our rerun) | GEP-M | 5,000 | 0.5806 ± 0.0011 | | 0.6096 | 82–101 |
| + big dataset, power 1, 20,000 steps | GEP-L | 20,000 | 0.5697 ± 0.0008 | −0.011 | 0.6017 | 385–389 |
| **+ 40,000 steps (our submission)** | GEP-L | 40,000 | **0.5633 ± 0.0026** | **−0.006** | 0.5960 | 574–773 |

Our rerun of record 4 matches its reported runs seed for seed (0.5818, 0.5798 and 0.5802 on seeds 7,
1 and 2, the same to four decimals), so the comparison is fair. Each step helps on every seed: all three 20,000-step runs (0.5689 to 0.5706) beat all three
record 4 runs (0.5798 to 0.5818), and all three 40,000-step runs (0.5608 to 0.5659) beat all three
20,000-step runs. Welch's t is 13.8 for the first step and 4.1 for the second.

The record 4 rerun used record 4's own `configs/final.yaml` (at
[`fba15aa`](https://github.com/swisnie2/nanoTSFM/tree/fba15aa/configs)), and the 20,000-step runs
used `configs/combo.yaml` (at [`70f5ddd`](https://github.com/swisnie2/nanoTSFM/tree/70f5ddd/configs)).
Only the final configuration is in this pull request.

### Where it helps and hurts

Geometric mean of relative CRPS over each group's tasks, as the mean of 3 seeds:

| Group | Tasks | Record 4 (our rerun) | This submission | Change |
| --- | ---: | ---: | ---: | ---: |
| All | 97 | 0.581 | 0.563 | −0.017 |
| Short | 55 | 0.574 | 0.564 | −0.011 |
| Medium | 21 | 0.592 | 0.559 | −0.033 |
| Long | 21 | 0.585 | 0.566 | −0.020 |
| Secondly | 6 | 1.277 | 1.053 | −0.224 |
| Minutely | 30 | 0.573 | 0.562 | −0.011 |
| Hourly | 31 | 0.505 | 0.496 | −0.009 |
| Daily | 15 | 0.458 | 0.446 | −0.012 |
| Weekly | 8 | 0.692 | 0.693 | +0.000 |
| Monthly, quarterly, yearly | 7 | 0.788 | 0.763 | −0.025 |

Our submission is better than the record 4 rerun on 65 of the 97 tasks. Datasets without a frequency
in their name are grouped by their known frequency (for example, `bizitobs_application` is 10-second
data and `m4_yearly` is yearly).

## Development process

We worked in three rounds.

**Round 1: `source_power` above 1.** Starting from record 2, we tested powers 1, 1.5 and 2 with 3
seeds each (configs at [`6f23cc5`](https://github.com/swisnie2/nanoTSFM/tree/6f23cc5/configs)). 1.5
scored best (0.6232 ± 0.0025 against 0.6317 ± 0.0013), and 2 overshot (0.6272). Record 4 later
found the same result independently. *What we learned:* favoring big sources helps fast data and
long forecasts but hurts slow data, and GEP-Val can point the opposite way from GIFT-Eval.

**Round 2: frequency balancing (dropped).** We wrote an option to give slow data a fair share of
training. Before we ran it, records 3 and 4 published results showing that this idea hurts, so we
skipped it. *What we learned:* reading the other teams' reports saved us time and compute.

**Round 3: combining records 3 and 4.** One-seed pilots at 20,000 steps (0.5696,
[`70f5ddd`](https://github.com/swisnie2/nanoTSFM/tree/70f5ddd/configs)) and 40,000 steps (0.5608,
[`4412b20`](https://github.com/swisnie2/nanoTSFM/tree/4412b20/configs)) both beat record 4, and
40,000 was better. That became our final configuration.

## Discussion

**Did it work?** Yes. The combination scores 0.5633, which is 0.018 better than record 4's verified
score and 0.017 better than our own rerun of it. So the two records' improvements mostly add up.
The bigger dataset with 20,000 steps gives about two thirds of the gain (−0.011), and doubling the
training to 40,000 steps gives the rest (−0.006).

**Our second guess also held:** longer training helped once augmentation was on. Our explanation is
that the flipped and smoothed windows act like extra data, so the model can train longer before it
starts memorizing.

**The in-corpus score agreed this time.** GEP-Val went down at each step too (0.610, then 0.602,
then 0.596). In round 1 the two scores pointed in opposite directions; here they move together,
which is what we'd expect from a change that really helps forecasting.

**Where it helps and hurts:**
- **Medium and long forecasts gain the most** (−0.033 and −0.020), more than short ones (−0.011).
  Longer training on more series seems to help the model most when it has to look further ahead.
- **Secondly data improves a lot** (1.277 to 1.053), but is still worse than Seasonal Naive (above
  1). The biggest single gains are on these 10-second `bizitobs` tasks (−16% to −26%) and on
  `us_births/D` short (−27%).
- **Weekly data does not improve** (0.692 to 0.693). Record 3 saw weekly data get worse on GEP-L.
  GEP-L keeps only the latest 8,192 points of each series, so long low-frequency histories gain
  little from it. The biggest single losses are short-term tasks: `jena_weather/D` (+14%),
  `electricity/W` (+14%) and `m4_hourly` (+12%).
- Every other frequency group improves.

**Limitations:**
- Only 3 seeds per setup and one type of GPU.
- We didn't re-tune `source_power` or the augmentation settings for the big dataset; both were
  chosen on the small one.
- GIFT-Eval chose between 20,000 and 40,000 steps.
- We didn't go past 40,000 steps, even though the one-hour limit allows about 4 times more.

**What we'd try next:** tune `source_power` and the augmentation on GEP-L; train even longer within
the hour (record 3 reached 66,000 steps); test the QK-norm idea that looked promising in record 3's
pilots (0.601 on one seed); and look at weekly data, which the big dataset made worse in record 3.

## Contributions and attribution

- **Shira Wisniewski and Manas Tripathi** worked together on every part of the project and
  contributed equally. Together we learned the codebase and the benchmark, planned and compared the
  experiments in all three rounds, ran every training and evaluation job, checked every number and
  every line of code and configuration, packaged the submission, and wrote and reviewed this report.
- **Built on:** record 4 by Anannay Sharma and Josh Wright (augmentation, the flip-symmetric and
  clamped forecasts, and `configs/final.yaml`); record 3 by Sahil Pai and Saahir Khan (GEP-L with
  longer training, and their pilots on `source_power` and training length); record 2 by Shu Wan
  (`source_power`); and the nanoTSFM baseline by the Abel AI Lab.
- **AI assistance:** Claude (Anthropic) assisted with experiment design, code changes, analysis
  scripts, and drafting this report. We ran every experiment ourselves and checked every number here
  against the run outputs.

## References

1. Abel AI Lab. *nanoTSFM: hill-climbing GIFT-Eval with one A100 and one hour.*
   https://github.com/Abel-ai-lab/nanoTSFM
2. T. Aksu et al. *GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation.*
   arXiv:2410.10393, 2024. The benchmark and the GIFT-Eval Pretrain data.
3. Datadog. *Toto 2.0.* arXiv:2605.20119. The model nanoTSFM is based on.
4. Google Research. *TimesFM 2.5*: flip invariance and `infer_is_positive`.
   https://huggingface.co/docs/transformers/model_doc/timesfm2_5. The source of record 4's
   forecast tricks.
5. *Reverso*, arXiv:2602.17634, and *TinyCast*, arXiv:2608.15767. Sign-flip and downsampling
   augmentation, as cited by record 4.
6. nanoTSFM records 2, 3 and 4 (Shu-Wan, sahil-saahir, anannay-josh), in the `records/` folder of [1].
