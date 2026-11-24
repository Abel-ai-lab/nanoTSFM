---
team: evan-gage
description: Record 3 (GEP-L, 20,000 steps) with profile-aware sampling, profile_power 0.5
members:
  - name: Evan Hochhalter
    github: ehochhal
  - name: Gage Stenhouse
    github: GageStu
ai_disclosure: Claude (Anthropic) helped draft documentation and this report and with minor code (the subset test script and merging main). ChatGPT helped with the code and report. The team ran every experiment and checked every number.
---

# evan-gage

**nanoTSFM: Pretraining a Time-Series Foundation Model from Scratch**

**Score: 0.6064 ± 0.0050 GIFT-Eval CRPS** over seeds 7, 1 and 2. That is 0.005 below our three
runs of record 3's configuration (0.6115 ± 0.0065), inside run-to-run noise, and not a record
(the current record is 0.5605).

## Hypothesis

We built on **record 3** (sahil-saahir: GEP-L, 20,000 steps, `source_power` 1; verified
0.6121 ± 0.0036). Our research question: can statistical characteristics of individual series
explain differences in generalization, and can they be used to choose better training data?

Record 3 samples a source with weight proportional to its series count, then a series uniformly
inside that source. Inside a large source, most series look alike, so rare kinds of series (very
skewed, heavy-tailed, noisy or strongly autocorrelated) are seen rarely. We describe every training
series with four statistics, group series with similar statistics into **profiles**, and add a
profile level to the sampler so rare profiles are drawn more often. If GIFT-Eval rewards models
that have seen a wider variety of series behaviour, flattening the profile distribution should
lower GIFT-Eval CRPS at no extra compute.

The ablation is `profile_power`: 1 reproduces record 3's sampling exactly, 0 weights every profile
in a source equally, and 0.5 is in between. A second experiment (the subset test below) asks the
same question in a different way: if we keep only 25% of the series, does picking them by profile
beat picking them at random?

## Method

**Model and objective.** Unchanged from record 3: the nanoTSFM model (context 1,024, prediction
512, patch 32, width 256, 3 layers, 4 heads; about 3.3M parameters), its quantile objective, batch
256, learning rate 0.001, 20,000 steps, GEP-L, `source_power` 1, `max_seconds` 3,600. The config is
[`configs/stat.yaml`](../../configs/stat.yaml), identical to record 3's apart from its name.

**Statistics.** For every GEP-L training series,
[`statistical-approach/precompute_statistics.py`](../../statistical-approach/precompute_statistics.py)
computes relative interquartile range (IQR divided by the median's magnitude), skewness, excess
kurtosis and lag-1 autocorrelation, and writes one row per series to a parquet file.

**Profiles.** Each feature is cut into 5 global quantile bins over the series with all four
features finite, giving up to 5^4 = 625 profiles. Series with a missing or non-finite feature
(for example a constant series) share one extra profile, so no series is dropped.

**Sampling.** In `Windows` in [`src/nanotsfm/data.py`](../../src/nanotsfm/data.py), a training
window is drawn in four steps:

1. a source, with weight (series in the source) ^ `source_power`, as before;
2. a profile inside that source, with weight (series in the profile) ^ `profile_power`;
3. a series uniformly inside that profile;
4. a random window of that series, as before.

With `profile_power` 1, steps 2 and 3 together pick a series uniformly, which is exactly record 3.
Profile sampling is switched on by `STATISTICAL_SAMPLING = True` and `PROFILE_POWER = 0.5` at the top
of `data.py` and applies only when `source_power` is 1, so the validation windows (`source_power`
0) are drawn exactly as before. We set these in code rather than in the config because `train.py`
accepts only a fixed set of training fields.

**Data.** Our data differs from record 3's only in how often each series is drawn: the same GEP-L
training series, the same windows and no new data. We chose `profile_power` 0.5 from the pilots
below: it was the best value on GIFT-Eval over three seeds, and 0 and 0.65 were worse on single
seeds.

**Building the data.** The statistics file is not committed (parquet is ignored and it is over
1 MB). Build it before training:

```shell
./run.sh data
uv run --locked python statistical-approach/precompute_statistics.py --dataset GEP-L --split train --horizon 1
```

Run it from the repository root. It writes `statistical-approach/series_stats_GEP-L_train.parquet`,
which is where training reads it; training stops with an error if the file is missing.

**Runs and model selection.** We compared configurations on GIFT-Eval CRPS, with GEP-Val as a
secondary check, always over the same seeds (7, 1, 2). Final runs: seeds 7, 1 and 2 at one commit
after merging `main`. Every run stopped at 20,000 steps, within the 3,600-second cap; training
seconds for the final runs are in the table below and in `result.json`.

## Results

GIFT-Eval relative CRPS, lower is better. Mean ± sd over every seed we ran.

**Final configuration and ablation** (GEP-L, 20,000 steps, `source_power` 1):

| Configuration | Seeds | GIFT-Eval CRPS | GEP-Val CRPS | Training (s) | Steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Record 3, as verified | 3 new seeds | 0.6121 ± 0.0036 | n/a | n/a | 20,000 |
| Record 3 setup, our runs (`profile_power` 1) | 7, 1, 2 | 0.6115 ± 0.0065 | 0.6082 | n/a | 20,000 |
| **This submission (`profile_power` 0.5)** | 7, 1, 2 | **0.6064 ± 0.0050** | [[GEP-Val from submit]] | 410 (407 to 412) | 20,000 |
| `profile_power` 0.65 | 7 | 0.6105 | 0.6152 | n/a | 20,000 |
| `profile_power` 0 | 7 | 0.6156 | 0.6372 | n/a | 20,000 |

Per seed (7 / 1 / 2): `profile_power` 1 scored 0.6187 / 0.6099 / 0.6060 and `profile_power` 0.5
scored 0.6007 / 0.6085 / 0.6101. The 0.5 runs were better on seeds 7 and 1 and worse on seed 2.

**Pilots on GEP-M** (5,000 steps, seed 7, `source_power` 1):

| `profile_power` | GIFT-Eval CRPS | GIFT-Eval MASE | GEP-Val CRPS |
| --- | ---: | ---: | ---: |
| 1 (record 2 setup) | 0.6284 | 0.8997 | 0.6166 |
| 0.5 | 0.6316 | 0.9080 | 0.6124 |
| 0 | 0.6267 | 0.9019 | 0.6260 |

All three are within the ± 0.007 seed noise of each other. With 5,000 steps on 26k series, each
series is already seen about 49 times, so reweighting changes little; on GEP-L with 20,000 steps
each series is seen far less often, which is why we moved the final experiments there.

**Where GEP-Val and GIFT-Eval disagree.** In a GEP-M pilot of `source_power` (seed 7, plain
sampler), lowering it improved GEP-Val but hurt GIFT-Eval:

| `source_power` | GIFT-Eval CRPS | GEP-Val CRPS |
| --- | ---: | ---: |
| 1 | 0.6284 | 0.6166 |
| 0.5 | 0.6330 | 0.5878 |
| 0.25 | 0.6520 | 0.5849 |
| 0 | 0.6616 | 0.5874 |

GEP-Val is drawn from the same corpus with every source weighted equally, so upweighting small
sources fits it better without transferring to GIFT-Eval. The same pattern shows on GEP-L: at
`profile_power` 0 GEP-Val got worse (0.6372) as did GIFT-Eval, and `profile_power` 0.5 left GEP-Val
unchanged while GIFT-Eval improved slightly. We therefore chose on GIFT-Eval.

**Subset test** (GEP-M, 5,000 steps, plain sampler). From each source we kept 25% of its series,
either at random or "smart" (one series from every profile present in the source first, then
random fill), so the source mix is identical and only the choice of series differs:

| Training data | Seed 7 | Seed 1 | Seed 2 | Mean ± sd | Mean MASE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random 25% | 0.6423 | 0.6480 | 0.6521 | 0.6475 ± 0.0049 | 0.9298 |
| Smart 25% | 0.6363 | 0.6494 | 0.6425 | 0.6427 ± 0.0066 | 0.9184 |
| Full GEP-M | 0.6331 | 0.6317 | 0.6304 | 0.6317 ± 0.0014 | 0.9032 |

Smart subsets kept about 90% of the 3,613 (source, profile) buckets, against about 54% for random
ones. They scored 0.0048 lower on average, better on 2 of 3 seeds, but both subsets were clearly
worse than full GEP-M, so the data is not redundant at this budget.

**Extra exploratory tests that influenced this experiment**
Extra exploratory runs not apart of the submission that led to final submission experiment.
| Run | What changed | GIFT-Eval CRPS | Notes / observations |
|---:|---|---:|---|
| 1 | `profile_power = 0.5`, `series.yaml`(GEP-M 5K steps) | 0.6316 | Initial profile-aware sampling experiment. |
| 2 | `profile_power = 1.0`, `series.yaml`(GEP-M 5k steps) | 0.6284 | Slightly better than Run 1. |
| 3 | `profile_power = 0.0`, `series.yaml`(GEP-M 5k steps) | 0.6267 | Best of the initial three profile-power settings. |
| 4 | `profile_power = 0.5`, `stat.yaml`, GEP-L, 20k steps | 0.6007 | Initial GEP-L experiment; strongest result at this stage. |
| 5 | `profile_power = 1.0`, `stat.yaml`, GEP-L, 20k steps, seed 7 | 0.6187 | Worse than Run 4; stronger emphasis on common profiles did not improve GIFT-Eval. |
| 6 | `profile_power = 0.0`, `stat.yaml`, GEP-L, 20k steps | 0.6156 | Better than Run 5 but worse than Run 4. |
| 7 | `profile_power = 0.5`, `source_power = 0.5`, GEP-L, 20k steps | 0.6330 | Hybrid source/profile weighting performed worse than Run 4. |
| 8 | `profile_power = 0.5`, `source_power = 0.25`, GEP-L, 20k steps | 0.6520 | Further reducing source power degraded GIFT-Eval performance. |
| 9 | `profile_power = 1.0`, `source_power = 0.0`, GEP-L, 20k steps | 0.6616 | Worst result among the source-power experiments. |
| 10 | `profile_power = 1.0`, `source_power = 1.0`, GEP-L, 20k steps, removed kurtosis | 0.6325 | Removing the feature heavily hurt performance indicating wrong selection of statistical features can harm generalized performance |
| 11 | `profile_power = 1.0`, `source_power = 1.0`, GEP-L, 20k steps, removed IQR | 0.6075 | not as much harm as removed kurtosis |
| 12 | `profile_power = 1.0`, `source_power = 1.0`, GEP-L, 20k steps, removed skewness | 0.6041 | not as much harm as removed kurtosis |
| 13 | `profile_power = 1.0`, `source_power = 1.0`, GEP-L, 20k steps, removed skewness and IQR | 0.6085 | shows how kurtosis + autocorrelation might be most important |
| 14 | `profile_power = 0.5`, seed 1, GEP-L, 20k steps | 0.6085 |  |
| 15 | `profile_power = 1.0`, seed 2, GEP-L, 20k steps | 0.6060 |  |
| 16 | `profile_power = 1.0`, seed 1, GEP-L, 20k steps | 0.6099 |  |
| 17 | `profile_power = 0.65`, seed 7, GEP-L, 20k steps | 0.6105 | did not outperform the best 0.5 results. |


## Discussion

**Did the results support the hypothesis?** Weakly. `profile_power` 0.5 improved GIFT-Eval by
0.005 over record 3's sampler, and the subset test points the same way (coverage-aware subsets beat
random ones by 0.005). Both gains are smaller than the ± 0.007 seed noise and far below the 0.013
margin a record needs, so we read them as two consistent hints rather than a proven effect. Going
all the way to equal profile weights (`profile_power` 0) hurt both GIFT-Eval and GEP-Val: very rare
profiles are often odd or degenerate series, and drawing them as often as common ones wastes
training steps.

Results show that this statistical sampling technique is weakly supported for the 4 statistics that produces similar results to previous record. Reasoning for this that relative IQR and Skewness are static statistics of the series and likely don't contain much valuable information for nanoTSFM. Exchanging statistics for different ones like spectral entropy and seasonality strength might prove better results.

**Where it helps or hurts.** We compared only the overall GIFT-Eval scores; we did not break the
results down by frequency or horizon, which would show whether the gain comes from the short,
irregular series that rare profiles tend to contain.

**Limitations.** 
- The profile representation depends on four statistical features and five percentile bins per feature. Other features or binning strategies may capture temporal behavior more effectively.
- Rare profiles may represent useful forecasting patterns, redundant behavior, or statistical noise. The current sampling strategy does not distinguish between these cases.
- Series with missing or non-finite statistical features are grouped into a separate profile, which may combine otherwise dissimilar series.
- The current profile power is manually configured in `data.py` rather than exposed as a YAML parameter. This is to minimize changes to `train.py` and its yaml validation
- The current results do not establish which frequencies, forecast horizons, or individual task categories benefit from the sampling change.

**Next.** 
Potential follow-up experiments include:
- Investigate other statistical features, such as permutation entropy, spectral entropy, seasonality strength and alternative autocorrelation measures. Some of these features should focus more on learning what is useful and probably stay less than 4 total features to reduce number of profiles
- Investigate whether profile weighting can be improved by identifying redundant or noisy rare profiles rather than treating all rare profiles equally.
- Investigate more stable profiles: binning within each source, fewer bins per feature, or merging groups below a minimum size, to stop single outliers from being upweighted.
- Integrate with new records: This experiment tried to focus on isolated results of this sampling technique, therefore there might be different or compounding results by integrating with other improvements

## Contributions and attribution

- **Evan Hochhalter:** research question, statistical feature analysis, `precompute_statistics.py`,
  the profile-aware sampler in `data.py`, the `profile_power` and `source_power` pilots, and the
  final runs.
- **Gage Stenhouse:** experiment review and design feedback, the random vs smart subset test (code,
  runs and analysis), merging `main` for the final runs, and the report.

Pilot code, figures and the full log of runs are on the
[`statistical-data-sampling` branch](https://github.com/ehochhal/nanoTSFM/tree/statistical-data-sampling/statistical-approach)
of our fork; the subset test code is on the
[`subset-test` branch](https://github.com/ehochhal/nanoTSFM/tree/5ff8c91/subset-test) (commit 5ff8c91).

Built on nanoTSFM and record 3 (sahil-saahir). Prior work on weighting and selecting training data:
Arivazhagan et al. (2019) on temperature-based sampling across languages, Sorscher et al. (2022) on
data pruning, and Wang et al. (2006) on characteristic-based clustering of time series.

AI assistance: Claude (Anthropic) helped draft documentation and this report and with minor code
(the subset test script and merging `main`). ChatGPT assisted with the code and report. The team ran every experiment and
checked every number.
