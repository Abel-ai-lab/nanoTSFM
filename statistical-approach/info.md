### contextual info and notes:
big main question: Can statistical characterization of individual time series explain differences in generalization across training datasets, and can those characteristics be used to construct more effective training subsets?

what the experiment code should answer: Can statistical characteristics of individual time series be exploited to construct more effective training subsets and improve forecasting generalization under a fixed computational budget?

idea is that data selection will be very important for training in under an hour.

Within-source diversity: differences between individual series belonging to the same dataset.

uv run python .\statistical-approach\analyze_series.py --dataset GEP-M --split train

uv run python -c "import pandas as pd; df = pd.read_parquet('statistical-approach/data/series_stats_GEP-M_train.parquet'); print('Shape:', df.shape); print('\nMissing values:'); print(df.isna().mean()); print('\nFeature summary:'); print(df[['iqr', 'skewness', 'kurtosis', 'autocorrelation']].describe().T)"

uv run .\statistical-approach\parquet_analysis.py

GEP-M datasets from the sources are statistically diverse:
| Source | Median Relative IQR | Median Skewness | Median Excess Kurtosis | Median Autocorrelation |
|---|---:|---:|---:|---:|
| `godaddy` | 0.074 | 0.148 | -0.803 | 0.907 |
| `rideshare_with_missing` | 0.160 | 0.495 | 1.858 | 0.216 |
| `PEMS07` | 0.684 | -0.452 | -0.981 | 0.974 |
| `favorita_sales` | 1.000 | 1.891 | 5.680 | 0.198 |
| `wiki-rolling_nips` | 0.446 | 4.756 | 37.764 | 0.678 |

The GEP-M training corpus contains 26,154 time series distributed across 87 sources. Statistical features were extracted to characterize relative dispersion, distributional asymmetry, tail heaviness, and temporal dependence. Source-level summaries revealed substantial differences in these characteristics. For example, the median autocorrelation ranged from 0.198 in favorita_sales to 0.974 in PEMS07, while median relative IQR ranged from 0.074 in godaddy to 1.000 in favorita_sales among the examples examined. These findings demonstrate that the training corpus contains substantial between-source statistical heterogeneity.

continue to look at within-source diversity: determining how much individual time series differ from one another inside the same source. (hypothesis: uniformly sampling across the different series might not be optimal and exploiting a statistic might improve training time or accuracy)

### context/proposal for LLM context
Research Proposal: Statistical Diversity-Aware Data Sampling for nanoTSFM
1. Research Motivation
nanoTSFM is a time-series foundation model trained on a subset of the GIFT-Eval Pretrain corpus. The objective is to achieve the best possible GIFT-Eval forecasting performance within a maximum training budget of one hour on a single A100 80GB GPU.
Existing experiments suggest that training-data composition and training duration interact in their effect on generalization. In particular, a recent experiment found that increasing training from 5,000 to 20,000 steps worsened GIFT-Eval performance on GEP-M but improved performance on GEP-L.
This raises an important question: does the additional training data improve generalization simply because it provides more examples, or because it introduces a broader representation of statistical and temporal behaviors?
This project proposes using mathematical statistics to characterize individual time series, investigate differences in statistical composition across existing training datasets, and develop sampling strategies that prioritize useful statistical coverage.
2. Research Question
Can statistical characterization of individual time series explain differences in generalization across training datasets, and can those characteristics be used to construct more effective training subsets?
Hypothesis
Differences in the statistical composition of training datasets contribute to differences in generalization. Sampling strategies informed by the statistical characteristics of individual series may improve GIFT-Eval forecasting performance by providing more effective coverage of temporal and distributional behaviors.
3. Proposed Methodology
Stage 1: Cross-Dataset Statistical Characterization
Analyze GEP-S, GEP-M, and GEP-L using consistent statistical feature calculations.
Investigate:
	• How statistical feature distributions change with dataset size.
	• Whether additional series introduce new statistical behaviors.
	• Whether particular statistical characteristics are concentrated within specific sources.
	• Whether GEP-L provides meaningfully different statistical coverage compared with GEP-M.
Initial candidate features:
	• Interquartile range (IQR): robust dispersion.
	• Skewness: distribution asymmetry.
	• Kurtosis: tail behavior.
	• Autocorrelation: temporal dependence.
	• Spectral entropy: frequency-domain complexity.
Additional metadata:
	• Source/dataset family.
	• Sampling frequency.
	• Number of variates.
	• Series length.
	• Missing-value proportion.
Stage 2: Statistical Sampling Strategy
Use the findings from the dataset characterization to investigate statistical stratification as an initial implementation.
Series will be grouped into statistical categories using their descriptive features, potentially using quantile-based bins or clustering.
Sampling probabilities will be adjusted to investigate whether improved representation of statistical characteristics produces better generalization.
A possible extension is hybrid sampling:
[
P(i)=P(S_i)P(G_k\mid S_i)
]
where source-level sampling is retained while statistical groups are sampled conditionally within each source.
The objective is not necessarily to create a perfectly balanced dataset, but to determine whether statistical coverage can improve training-data selection.
Stage 3: Experimental Evaluation
Initial training experiments will use GEP-M with a fixed model architecture and training configuration.
Proposed configurations:
	1. Uniform source sampling (baseline).
	2. Statistical diversity-aware sampling.
	3. Hybrid source and statistical sampling.

Use equivalent training steps and settings when comparing sampling strategies. Run three independent seeds for final configurations where computationally feasible.
Stage 4: Dataset Size Investigation
Investigate whether statistical sampling improvements transfer across GEP-S, GEP-M, and GEP-L.
This will help determine whether statistical sampling is particularly useful for smaller training datasets or continues to provide benefits as the training corpus grows.
4. Evaluation
Primary metric:
	• GIFT-Eval relative CRPS (lower is better).
Secondary diagnostics:
	• GEP-Val relative CRPS.
	• GEP-Test relative CRPS.
GEP-Val and GEP-Test are existing held-out diagnostic datasets from the training corpus. They will not be replaced with custom evaluation datasets.
GIFT-Eval remains the external benchmark and primary measure of generalization.
The analysis will consider whether differences in statistical composition correspond to differences in generalization, and whether improvements in training-corpus diagnostics translate into improved performance on previously unseen forecasting tasks.
5. Expected Contribution
The intended contribution is an experimentally evaluated investigation of how statistical composition influences time-series foundation model generalization, together with a statistical sampling strategy that extends existing source-count-based sampling.
Rather than considering only the number and source distribution of training series, this approach investigates whether their individual distributional and temporal characteristics can guide more effective training-data selection.
A successful result would demonstrate that statistical characterization can improve training-data selection and potentially retain the generalization benefits of larger datasets with smaller, more carefully selected training subsets.
6. Initial Implementation Scope
	1. Inspect the existing data-loading and sampling implementation.
	2. Reproduce or establish the existing sampling baselines.
	3. Calculate statistical features for GEP-S, GEP-M, and GEP-L.
	4. Examine statistical distributions globally and across sources.
	5. Identify potentially underrepresented statistical characteristics.
	6. Implement statistical stratification on GEP-M.
	7. Compare against existing source-level sampling strategies.
	8. Run repeated experiments and analyze results.
The initial priority is to understand the relationship between statistical composition and generalization, then test one reproducible sampling improvement. Implementing every proposed statistical technique or testing every dataset/configuration combination is not required.

