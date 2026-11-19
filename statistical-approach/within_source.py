
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nanotsfm.data import arrays, load

dataset = load("GEP-M", "train")

df = pd.read_parquet(
    "statistical-approach/data/series_stats_GEP-M_train.parquet"
)

df = df[df["eligible_for_training"]].copy()

features = [
    "relative_iqr",
    "skewness",
    "kurtosis",
    "autocorrelation",
]

# Select the 15 largest sources for easier visualization.
source_counts = df.groupby("source").size()
selected_sources = source_counts.nlargest(15).index

plot_df = df[df["source"].isin(selected_sources)]

# Order sources by number of series.
source_order = source_counts.loc[selected_sources].sort_values(
    ascending=False
).index.tolist()

for feature in features:
    data = [
        plot_df.loc[plot_df["source"] == source, feature].dropna()
        for source in source_order
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(
        data,
        tick_labels=source_order,
        showfliers=False,
    )

    ax.set_title(f"Within-Source Distribution: {feature}")
    ax.set_xlabel("Source")
    ax.set_ylabel(feature)
    ax.tick_params(axis="x", rotation=60)

    plt.tight_layout()
    plt.savefig(
        f"statistical-approach/data/within_source_{feature}.png",
        dpi=200,
    )
    plt.show()

df = pd.read_parquet(
    "statistical-approach/data/series_stats_GEP-M_train.parquet"
)

features = ["skewness", "kurtosis"]

for feature in features:
    print(f"\n{'=' * 70}")
    print(f"TOP 15 SERIES BY {feature.upper()}")
    print("=" * 70)

    columns = [
        "item_id",
        "source",
        "series_length",
        "num_variates",
        "missing_proportion",
        feature,
    ]

    print(
        df.nlargest(15, feature)[columns].to_string(index=False)
    )

    print(f"\nBOTTOM 15 SERIES BY {feature.upper()}")

    print(
        df.nsmallest(15, feature)[columns].to_string(index=False)
    )





# ============================================================
# AZURE VM SERIES ANALYSIS
# ============================================================

target_ids = [
    "azure_vm_traces_2017/74018",
    "azure_vm_traces_2017/97425",
]

# Load the original dataset containing the time series.
# Assumes `dataset` and `arrays()` are already initialized.

# Match item IDs to their original dataset rows.
id_to_index = {
    str(row["item_id"]): i
    for i, row in enumerate(dataset)
}

# ------------------------------------------------------------
# 1. INDIVIDUAL SERIES
# ------------------------------------------------------------

for item_id in target_ids:
    idx = id_to_index[item_id]
    values = arrays(dataset.select([idx]))[0][0]

    values = values[np.isfinite(values)]

    print(f"\n{'=' * 70}")
    print(f"SERIES: {item_id}")
    print("=" * 70)

    print(f"Length: {len(values)}")
    print(f"Min: {np.min(values)}")
    print(f"Median: {np.median(values)}")
    print(f"Mean: {np.mean(values)}")
    print(f"Std: {np.std(values)}")
    print(f"95th percentile: {np.percentile(values, 95)}")
    print(f"99th percentile: {np.percentile(values, 99)}")
    print(f"Max: {np.max(values)}")

    plt.figure(figsize=(12, 4))
    plt.plot(values)
    plt.title(item_id)
    plt.xlabel("Time index")
    plt.ylabel("Value")
    plt.tight_layout()

    plt.savefig(
        f"statistical-approach/data/{item_id.split('/')[-1]}.png",
        dpi=200,
    )

    plt.show()


# ------------------------------------------------------------
# AVERAGE AZURE VM SERIES
# ------------------------------------------------------------

azure_ids = set(
    df.loc[
        df["source"] == "azure_vm_traces_2017",
        "item_id",
    ].astype(str)
)

azure_indices = [
    id_to_index[item_id]
    for item_id in azure_ids
    if item_id in id_to_index
]

azure_series = []

for idx in azure_indices:
    values = arrays(dataset.select([idx]))[0][0]

    values = np.where(
        np.isfinite(values),
        values,
        np.nan,
    )

    azure_series.append(values)

# Pad shorter series with NaN instead of truncating longer ones.
max_length = max(len(series) for series in azure_series)

azure_matrix = np.full(
    (len(azure_series), max_length),
    np.nan,
)

for i, series in enumerate(azure_series):
    azure_matrix[i, :len(series)] = series

# Calculate mean and median at each time index.
average_azure = np.nanmean(azure_matrix, axis=0)
median_azure = np.nanmedian(azure_matrix, axis=0)

# Number of contributing series at each time index.
observations_per_timestep = np.sum(
    np.isfinite(azure_matrix),
    axis=0,
)

print("Average series length:", len(average_azure))
print("Maximum contributing series:", observations_per_timestep.max())
print("Minimum contributing series:", observations_per_timestep.min())

# Plot average and median.
plt.figure(figsize=(12, 5))

plt.plot(average_azure, label="Mean", alpha=0.8)
plt.plot(median_azure, label="Median", alpha=0.8)

plt.title("Azure VM Aggregate Utilization")
plt.xlabel("Time index")
plt.ylabel("Average utilization")
plt.legend()

plt.tight_layout()
plt.savefig(
    "statistical-approach/data/average_azure_vm.png",
    dpi=200,
)
plt.show()


# ============================================================
# EXPERIMENT 1: QUANTIFY STATISTICAL REDUNDANCY
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Features used to characterize each series.
features = [
    "relative_iqr",
    "skewness",
    "kurtosis",
    "autocorrelation",
]

# Use eligible GEP-M series with valid feature values.
analysis_df = df[
    df["eligible_for_training"]
].copy()

# Remove invalid feature values.
analysis_df[features] = analysis_df[features].replace(
    [np.inf, -np.inf],
    np.nan,
)

analysis_df = analysis_df.dropna(subset=features).copy()

print("\n" + "=" * 70)
print("STATISTICAL REDUNDANCY ANALYSIS")
print("=" * 70)

print(f"Eligible series: {len(analysis_df)}")


# ------------------------------------------------------------
# 1. CREATE STATISTICAL PROFILES
# ------------------------------------------------------------

# Divide each feature into five quantile-based bins.
# Rank transformation handles ties and extreme values.

num_bins = 5

for feature in features:
    ranks = analysis_df[feature].rank(pct=True)

    analysis_df[f"{feature}_bin"] = np.minimum(
        (ranks * num_bins).astype(int),
        num_bins - 1,
    )

bin_columns = [
    f"{feature}_bin"
    for feature in features
]

# Each series receives a joint statistical profile.
# Example: (2, 4, 3, 1)
analysis_df["statistical_profile"] = list(
    analysis_df[bin_columns].itertuples(
        index=False,
        name=None,
    )
)

# Count series belonging to each statistical profile.
profile_counts = (
    analysis_df["statistical_profile"]
    .value_counts()
)

total_profiles = len(profile_counts)

print(f"Distinct statistical profiles: {total_profiles}")
print(f"Possible profiles: {num_bins ** len(features)}")

print(
    f"Profile occupancy: "
    f"{total_profiles / num_bins ** len(features):.2%}"
)


# ------------------------------------------------------------
# 2. IDENTIFY THE MOST COMMON PROFILES
# ------------------------------------------------------------

print("\nTOP 20 MOST COMMON STATISTICAL PROFILES")

print(
    profile_counts.head(20).to_string()
)

print("\nLEAST COMMON STATISTICAL PROFILES")

print(
    profile_counts.tail(20).to_string()
)


# ------------------------------------------------------------
# 3. MEASURE STATISTICAL COVERAGE
# ------------------------------------------------------------

full_profiles = set(
    analysis_df["statistical_profile"]
)

full_profile_count = len(full_profiles)

# Shannon entropy of the complete profile distribution.
full_probabilities = (
    analysis_df["statistical_profile"]
    .value_counts(normalize=True)
)

full_entropy = -np.sum(
    full_probabilities * np.log(full_probabilities)
)

sample_percentages = [10, 25, 50, 75]
repetitions = 30

results = []

for percentage in sample_percentages:

    sample_size = int(
        len(analysis_df) * percentage / 100
    )

    for seed in range(repetitions):

        sample = analysis_df.sample(
            n=sample_size,
            random_state=seed,
        )

        sample_profiles = set(
            sample["statistical_profile"]
        )

        # Percentage of full statistical profiles retained.
        coverage = (
            len(sample_profiles) / full_profile_count
        )

        # Distributional diversity within the sample.
        sample_probabilities = (
            sample["statistical_profile"]
            .value_counts(normalize=True)
        )

        sample_entropy = -np.sum(
            sample_probabilities
            * np.log(sample_probabilities)
        )

        entropy_retention = (
            sample_entropy / full_entropy
            if full_entropy > 0 else np.nan
        )

        results.append({
            "sample_percentage": percentage,
            "sample_size": sample_size,
            "seed": seed,
            "profile_coverage": coverage,
            "entropy_retention": entropy_retention,
            "profiles_retained": len(sample_profiles),
        })


results_df = pd.DataFrame(results)

# Summarize results across repetitions.
summary = (
    results_df
    .groupby("sample_percentage")
    .agg(
        sample_size=("sample_size", "first"),
        mean_coverage=("profile_coverage", "mean"),
        std_coverage=("profile_coverage", "std"),
        mean_entropy_retention=("entropy_retention", "mean"),
        std_entropy_retention=("entropy_retention", "std"),
        mean_profiles=("profiles_retained", "mean"),
    )
)

print("\n" + "=" * 70)
print("RANDOM SAMPLING: STATISTICAL COVERAGE")
print("=" * 70)

print(summary.to_string())


# ------------------------------------------------------------
# 4. VISUALIZE COVERAGE
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Unique profile coverage.
axes[0].errorbar(
    summary.index,
    summary["mean_coverage"] * 100,
    yerr=summary["std_coverage"] * 100,
    marker="o",
    capsize=4,
)

axes[0].set_title("Statistical Profile Coverage")
axes[0].set_xlabel("Percentage of Training Series Sampled")
axes[0].set_ylabel("Full Profile Coverage (%)")
axes[0].grid(alpha=0.3)

# Entropy retention.
axes[1].errorbar(
    summary.index,
    summary["mean_entropy_retention"] * 100,
    yerr=summary["std_entropy_retention"] * 100,
    marker="o",
    capsize=4,
)

axes[1].set_title("Statistical Diversity Retention")
axes[1].set_xlabel("Percentage of Training Series Sampled")
axes[1].set_ylabel("Entropy Retention (%)")
axes[1].grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "statistical-approach/data/statistical_redundancy.png",
    dpi=200,
)

plt.show()


# ============================================================
# EXPERIMENT 2: UNIFORM VS COVERAGE-AWARE SAMPLING
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sampling configurations.
sample_percentages = [10, 25, 50, 75]
repetitions = 30

# Original profile distribution.
full_profile_counts = (
    analysis_df["statistical_profile"]
    .value_counts()
)

all_profiles = full_profile_counts.index.tolist()

full_distribution = (
    full_profile_counts / len(analysis_df)
).reindex(all_profiles).to_numpy()


# Jensen-Shannon divergence.
def js_divergence(p, q):
    midpoint = (p + q) / 2

    def kl_divergence(a, b):
        mask = a > 0
        return np.sum(
            a[mask] * np.log(a[mask] / b[mask])
        )

    return (
        kl_divergence(p, midpoint)
        + kl_divergence(q, midpoint)
    ) / 2


# Group row indices by statistical profile.
profile_groups = {
    profile: group.index.to_numpy()
    for profile, group in analysis_df.groupby(
        "statistical_profile"
    )
}


results = []

for percentage in sample_percentages:

    sample_size = int(
        len(analysis_df) * percentage / 100
    )

    for seed in range(repetitions):

        rng = np.random.default_rng(seed)

        # ----------------------------------------------------
        # A. UNIFORM RANDOM SAMPLING
        # ----------------------------------------------------

        uniform_indices = rng.choice(
            analysis_df.index.to_numpy(),
            size=sample_size,
            replace=False,
        )

        uniform_sample = analysis_df.loc[uniform_indices]

        # ----------------------------------------------------
        # B. COVERAGE-AWARE SAMPLING
        # ----------------------------------------------------

        # Select one series from every observed profile.
        selected_indices = [
            rng.choice(indices)
            for indices in profile_groups.values()
        ]

        selected_indices = np.array(selected_indices)

        # Fill remaining capacity using random sampling.
        remaining_indices = np.setdiff1d(
            analysis_df.index.to_numpy(),
            selected_indices,
        )

        additional_count = sample_size - len(selected_indices)

        additional_indices = rng.choice(
            remaining_indices,
            size=additional_count,
            replace=False,
        )

        stratified_indices = np.concatenate([
            selected_indices,
            additional_indices,
        ])

        stratified_sample = analysis_df.loc[
            stratified_indices
        ]

        # ----------------------------------------------------
        # C. EVALUATE BOTH APPROACHES
        # ----------------------------------------------------

        for method, sample in [
            ("Uniform", uniform_sample),
            ("Coverage-aware", stratified_sample),
        ]:

            sample_counts = (
                sample["statistical_profile"]
                .value_counts()
            )

            sample_distribution = (
                sample_counts / len(sample)
            ).reindex(
                all_profiles,
                fill_value=0,
            ).to_numpy()

            coverage = (
                len(sample_counts) / len(all_profiles)
            )

            divergence = js_divergence(
                full_distribution,
                sample_distribution,
            )

            results.append({
                "percentage": percentage,
                "seed": seed,
                "method": method,
                "sample_size": len(sample),
                "coverage": coverage,
                "js_divergence": divergence,
            })


# ------------------------------------------------------------
# 3. SUMMARIZE RESULTS
# ------------------------------------------------------------

comparison_df = pd.DataFrame(results)

summary = (
    comparison_df
    .groupby(["percentage", "method"])
    .agg(
        sample_size=("sample_size", "first"),
        mean_coverage=("coverage", "mean"),
        std_coverage=("coverage", "std"),
        mean_js=("js_divergence", "mean"),
        std_js=("js_divergence", "std"),
    )
)

print("\n" + "=" * 75)
print("UNIFORM VS COVERAGE-AWARE SAMPLING")
print("=" * 75)

print(summary.to_string())


# ------------------------------------------------------------
# 4. VISUALIZE COMPARISON
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for method in ["Uniform", "Coverage-aware"]:

    method_results = (
        comparison_df[
            comparison_df["method"] == method
        ]
        .groupby("percentage")
    )

    means = method_results["coverage"].mean() * 100
    stds = method_results["coverage"].std() * 100

    axes[0].errorbar(
        means.index,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        label=method,
    )

    means = method_results["js_divergence"].mean()
    stds = method_results["js_divergence"].std()

    axes[1].errorbar(
        means.index,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        label=method,
    )

axes[0].set_title("Statistical Profile Coverage")
axes[0].set_xlabel("Percentage of Training Series Sampled")
axes[0].set_ylabel("Profile Coverage (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_title("Profile Distribution Divergence")
axes[1].set_xlabel("Percentage of Training Series Sampled")
axes[1].set_ylabel("Jensen-Shannon Divergence")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "statistical-approach/data/sampling_comparison.png",
    dpi=200,
)

plt.show()

# ============================================================
# EXPERIMENT 3: WITHIN-SOURCE PROFILE DISTRIBUTION
# ============================================================

source_profiles = (
    analysis_df
    .groupby(["source", "statistical_profile"])
    .size()
    .reset_index(name="series_count")
)

# Calculate profile distribution statistics for each source.
source_summary = (
    source_profiles
    .groupby("source")
    .agg(
        total_series=("series_count", "sum"),
        distinct_profiles=("statistical_profile", "count"),
        rare_profiles=("series_count", lambda x: (x == 1).sum()),
        profiles_with_5_or_less=("series_count", lambda x: (x <= 5).sum()),
        largest_profile=("series_count", "max"),
    )
)

# Proportion of each source belonging to its largest profile.
source_summary["largest_profile_percentage"] = (
    source_summary["largest_profile"]
    / source_summary["total_series"]
    * 100
)

source_summary["rare_profile_percentage"] = (
    source_summary["rare_profiles"]
    / source_summary["distinct_profiles"]
    * 100
)

# Calculate within-source Shannon entropy.
entropy = (
    source_profiles
    .assign(
        probability=lambda x: (
            x["series_count"]
            / x.groupby("source")["series_count"].transform("sum")
        )
    )
    .assign(
        entropy_term=lambda x: (
            -x["probability"] * np.log(x["probability"])
        )
    )
    .groupby("source")["entropy_term"]
    .sum()
)

source_summary["profile_entropy"] = entropy

# Normalize entropy to account for different profile counts.
source_summary["normalized_entropy"] = (
    source_summary["profile_entropy"]
    / np.log(source_summary["distinct_profiles"]).replace(0, np.nan)
)

print("\n" + "=" * 75)
print("WITHIN-SOURCE STATISTICAL PROFILE DISTRIBUTION")
print("=" * 75)

print(
    source_summary
    .sort_values("total_series", ascending=False)
    .to_string()
)

print("\nTOP 15 SOURCES BY PROFILE CONCENTRATION")

print(
    source_summary
    .nlargest(15, "largest_profile_percentage")
    .to_string()
)

print("\nTOP 15 SOURCES BY NORMALIZED PROFILE ENTROPY")

print(
    source_summary
    .nlargest(15, "normalized_entropy")
    .to_string()
)

# Save results.
source_summary.to_csv(
    "statistical-approach/data/source_profile_summary.csv"
)


# ============================================================
# EXPERIMENT 4: WITHIN-SOURCE SAMPLING COMPARISON
# ============================================================

from collections import defaultdict

sample_percentages = [10, 25, 50, 75]
repetitions = 30

rng = np.random.default_rng(42)

results = []


# ------------------------------------------------------------
# 1. GROUP SERIES BY SOURCE AND PROFILE
# ------------------------------------------------------------

source_groups = {}

for source, source_df in analysis_df.groupby("source"):

    source_df = source_df.reset_index(drop=True)

    profile_groups = {
        profile: group.index.to_numpy()
        for profile, group in source_df.groupby(
            "statistical_profile"
        )
    }

    source_groups[source] = {
        "data": source_df,
        "profiles": profile_groups,
    }


# ------------------------------------------------------------
# 2. SAMPLING SIMULATION
# ------------------------------------------------------------

for source, source_info in source_groups.items():

    source_df = source_info["data"]
    profile_groups = source_info["profiles"]

    original_size = len(source_df)
    original_profile_counts = (
        source_df["statistical_profile"].value_counts()
    )

    original_distribution = (
        original_profile_counts
        / original_size
    )

    original_entropy = -np.sum(
        original_distribution
        * np.log(original_distribution)
    )

    original_profile_count = len(original_profile_counts)

    for percentage in sample_percentages:

        sample_size = max(
            1,
            int(original_size * percentage / 100)
        )

        for seed in range(repetitions):

            rng = np.random.default_rng(
                42 + seed
            )

            # ------------------------------------------------
            # A. UNIFORM RANDOM SAMPLING
            # ------------------------------------------------

            uniform_indices = rng.choice(
                original_size,
                size=sample_size,
                replace=False,
            )

            uniform_sample = source_df.iloc[uniform_indices]

            # ------------------------------------------------
            # B. COVERAGE-AWARE SAMPLING
            # ------------------------------------------------

            # Randomly select one series from each profile.
            profile_positions = list(profile_groups.values())
            rng.shuffle(profile_positions)

            selected_indices = [
                rng.choice(positions)
                for positions in profile_positions[:sample_size]
            ]

            selected_indices = np.array(
                selected_indices,
                dtype=int,
            )

            # Fill remaining capacity uniformly.
            remaining_indices = np.setdiff1d(
                np.arange(original_size),
                selected_indices,
            )

            additional_count = (
                sample_size - len(selected_indices)
            )

            if additional_count > 0:
                additional_indices = rng.choice(
                    remaining_indices,
                    size=additional_count,
                    replace=False,
                )

                selected_indices = np.concatenate([
                    selected_indices,
                    additional_indices,
                ])

            coverage_sample = source_df.iloc[
                selected_indices
            ]

            # ------------------------------------------------
            # C. EVALUATE BOTH METHODS
            # ------------------------------------------------

            for method, sample in [
                ("Uniform", uniform_sample),
                ("Coverage-aware", coverage_sample),
            ]:

                sample_counts = (
                    sample["statistical_profile"]
                    .value_counts()
                )

                sample_distribution = (
                    sample_counts
                    .reindex(
                        original_profile_counts.index,
                        fill_value=0,
                    )
                    / len(sample)
                )

                # Profile coverage.
                retained_profiles = len(sample_counts)

                profile_coverage = (
                    retained_profiles
                    / original_profile_count
                )

                # Jensen-Shannon divergence.
                p = original_distribution.to_numpy()
                q = sample_distribution.to_numpy()

                midpoint = (p + q) / 2

                def kl_divergence(a, b):
                    mask = a > 0

                    return np.sum(
                        a[mask]
                        * np.log(a[mask] / b[mask])
                    )

                js = (
                    kl_divergence(p, midpoint)
                    + kl_divergence(q, midpoint)
                ) / 2

                # Entropy retention.
                sample_entropy = -np.sum(
                    sample_distribution[sample_distribution > 0]
                    * np.log(
                        sample_distribution[sample_distribution > 0]
                    )
                )

                entropy_retention = (
                    sample_entropy / original_entropy
                    if original_entropy > 0
                    else np.nan
                )

                results.append({
                    "source": source,
                    "retention_percentage": percentage,
                    "method": method,
                    "seed": seed,
                    "original_series": original_size,
                    "sample_size": len(sample),
                    "original_profiles": original_profile_count,
                    "retained_profiles": retained_profiles,
                    "profile_coverage": profile_coverage,
                    "js_divergence": js,
                    "entropy_retention": entropy_retention,
                })


# ------------------------------------------------------------
# 3. SAVE DETAILED RESULTS
# ------------------------------------------------------------

within_source_results = pd.DataFrame(results)

within_source_results.to_csv(
    "statistical-approach/data/within_source_sampling_results.csv",
    index=False,
)


# ------------------------------------------------------------
# 4. SUMMARIZE RESULTS BY SOURCE
# ------------------------------------------------------------

source_sampling_summary = (
    within_source_results
    .groupby([
        "source",
        "retention_percentage",
        "method",
    ])
    .agg(
        mean_coverage=("profile_coverage", "mean"),
        std_coverage=("profile_coverage", "std"),
        mean_js=("js_divergence", "mean"),
        std_js=("js_divergence", "std"),
        mean_entropy_retention=("entropy_retention", "mean"),
        std_entropy_retention=("entropy_retention", "std"),
        mean_retained_profiles=("retained_profiles", "mean"),
        sample_size=("sample_size", "first"),
    )
    .reset_index()
)

source_sampling_summary.to_csv(
    "statistical-approach/data/within_source_sampling_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# 5. OVERALL COMPARISON ACROSS SOURCES
# ------------------------------------------------------------

overall_sampling_summary = (
    within_source_results
    .groupby([
        "retention_percentage",
        "method",
        "seed",
    ])
    .agg(
        mean_profile_coverage=("profile_coverage", "mean"),
        mean_js_divergence=("js_divergence", "mean"),
        mean_entropy_retention=("entropy_retention", "mean"),
        total_retained_series=("sample_size", "sum"),
    )
    .reset_index()
)

overall_sampling_summary = (
    overall_sampling_summary
    .groupby([
        "retention_percentage",
        "method",
    ])
    .agg(
        mean_coverage=("mean_profile_coverage", "mean"),
        std_coverage=("mean_profile_coverage", "std"),
        mean_js=("mean_js_divergence", "mean"),
        std_js=("mean_js_divergence", "std"),
        mean_entropy_retention=("mean_entropy_retention", "mean"),
        std_entropy_retention=("mean_entropy_retention", "std"),
        mean_retained_series=("total_retained_series", "mean"),
    )
    .reset_index()
)

overall_sampling_summary.to_csv(
    "statistical-approach/data/within_source_overall_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# 6. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 85)
print("WITHIN-SOURCE SAMPLING: OVERALL COMPARISON")
print("=" * 85)

print(
    overall_sampling_summary.to_string(index=False)
)

print("\n" + "=" * 85)
print("SOURCES WITH THE LOWEST UNIFORM PROFILE COVERAGE")
print("=" * 85)

uniform_results = source_sampling_summary[
    source_sampling_summary["method"] == "Uniform"
]

print(
    uniform_results[
        uniform_results["retention_percentage"] == 25
    ]
    .sort_values("mean_coverage")
    .head(15)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 7. VISUALIZE OVERALL RESULTS
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

metrics = [
    ("mean_coverage", "Profile Coverage"),
    ("mean_js", "JS Divergence"),
    ("mean_entropy_retention", "Entropy Retention"),
]

for ax, (metric, title) in zip(axes, metrics):

    for method in ["Uniform", "Coverage-aware"]:

        method_df = overall_sampling_summary[
            overall_sampling_summary["method"] == method
        ]

        ax.errorbar(
            method_df["retention_percentage"],
            method_df[metric],
            yerr=method_df[
                "std_" + metric.replace("mean_", "")
            ] if "std_" + metric.replace("mean_", "") in method_df.columns
            else None,
            marker="o",
            capsize=4,
            label=method,
        )

    ax.set_title(title)
    ax.set_xlabel("Series Retained (%)")
    ax.grid(alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Coverage")
axes[1].set_ylabel("JS Divergence (nats)")
axes[2].set_ylabel("Entropy Retention")

plt.tight_layout()

plt.savefig(
    "statistical-approach/data/within_source_sampling_comparison.png",
    dpi=200,
)

plt.show()

print("\nWithin-source sampling experiment complete.")

