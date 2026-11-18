
"""Analyze statistical features extracted from GEP-M time series."""

import pandas as pd
import matplotlib.pyplot as plt


PARQUET_PATH = (
    "statistical-approach/data/series_stats_GEP-M_train.parquet"
)

df = pd.read_parquet(PARQUET_PATH)

FEATURES = [
    "iqr",
    "relative_iqr",
    "skewness",
    "kurtosis",
    "autocorrelation",
]


# ---------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------

print("\n=== DATASET OVERVIEW ===")
print("Shape:", df.shape)
print("Unique sources:", df["source"].nunique())
print("Unique series:", df["item_id"].nunique())

print("\nNumber of series per source:")
print(df["source"].value_counts().head(20))


# ---------------------------------------------------------
# 2. Statistical feature summaries
# ---------------------------------------------------------

print("\n=== FEATURE SUMMARY ===")

print(
    df[FEATURES].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999,
        ]
    ).T
)


# ---------------------------------------------------------
# 3. Missing statistical features
# ---------------------------------------------------------

print("\n=== MISSING FEATURE VALUES ===")

print("Missing counts:")
print(df[FEATURES].isna().sum())

print("\nMissing percentages:")
print((df[FEATURES].isna().mean() * 100).round(3))


# ---------------------------------------------------------
# 4. Raw IQR diagnostics
# ---------------------------------------------------------

print("\n=== RAW IQR DIAGNOSTICS ===")

print("\nIQR percentiles:")
print(
    df["iqr"].describe(
        percentiles=[0.90, 0.95, 0.99, 0.999]
    )
)

print("\nTop 15 IQR values:")
print(
    df.nlargest(15, "iqr")[
        [
            "item_id",
            "source",
            "freq",
            "num_variates",
            "series_length",
            "missing_proportion",
            "iqr",
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# 5. Relative IQR diagnostics
# ---------------------------------------------------------

print("\n=== RELATIVE IQR DIAGNOSTICS ===")

print("\nRelative IQR percentiles:")
print(
    df["relative_iqr"].describe(
        percentiles=[0.90, 0.95, 0.99, 0.999]
    )
)

print("\nTop 15 relative IQR values:")
print(
    df.nlargest(15, "relative_iqr")[
        [
            "item_id",
            "source",
            "freq",
            "num_variates",
            "series_length",
            "missing_proportion",
            "relative_iqr",
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# 6. Feature correlations
# ---------------------------------------------------------

print("\n=== SPEARMAN FEATURE CORRELATIONS ===")

correlations = df[FEATURES].corr(method="spearman")

print(correlations.round(3))


# ---------------------------------------------------------
# 7. Source-level feature summaries
# ---------------------------------------------------------

print("\n=== FEATURE DISTRIBUTIONS BY SOURCE ===")

source_summary = (
    df.groupby("source")
    .agg(
        series_count=("item_id", "count"),
        median_iqr=("iqr", "median"),
        median_relative_iqr=("relative_iqr", "median"),
        median_skewness=("skewness", "median"),
        median_kurtosis=("kurtosis", "median"),
        median_autocorrelation=("autocorrelation", "median"),
    )
    .sort_values("series_count", ascending=False)
)

print("\nTop 20 sources by number of series:")
print(source_summary.head(20).to_string())

print("\nTop 15 sources by median raw IQR:")
print(
    source_summary.sort_values(
        "median_iqr", ascending=False
    ).head(15).to_string()
)


# ---------------------------------------------------------
# 8. Variate count distribution
# ---------------------------------------------------------

print("\n=== NUMBER OF VARIATES ===")

print(df["num_variates"].value_counts().sort_index())

print(
    "\nPercentage of univariate items:",
    round((df["num_variates"] == 1).mean() * 100, 2),
    "%",
)


# ---------------------------------------------------------
# 9. Variate-level feature variability
# ---------------------------------------------------------

print("\n=== VARIATION BETWEEN VARIATES ===")

variate_std_columns = [
    f"{feature}_variate_std" for feature in FEATURES
]

print(
    df[variate_std_columns]
    .describe()
    .T
)


# ---------------------------------------------------------
# 10. Feature histograms
# ---------------------------------------------------------

print("\n=== GENERATING FEATURE HISTOGRAMS ===")

for feature in FEATURES:
    values = df[feature].dropna()

    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=50)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of series")
    plt.tight_layout()

    filename = f"statistical-approach/data/{feature}_distribution.png"
    plt.savefig(filename)
    plt.close()

    print(f"Saved: {filename}")


# ---------------------------------------------------------
# 11. Correlation heatmap
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.imshow(correlations, aspect="auto")
plt.colorbar(label="Spearman correlation")

plt.xticks(
    range(len(FEATURES)),
    FEATURES,
    rotation=45,
    ha="right",
)
plt.yticks(range(len(FEATURES)), FEATURES)

plt.title("Statistical Feature Correlations")
plt.tight_layout()

correlation_path = (
    "statistical-approach/data/feature_correlations.png"
)
plt.savefig(correlation_path)
plt.close()

print(f"\nSaved: {correlation_path}")

print("\n=== ANALYSIS COMPLETE ===")
