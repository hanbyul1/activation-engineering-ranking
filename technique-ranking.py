import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).parent
ACTIVATION_DIR = BASE_DIR / "activation_analysis_outputs"

METRICS_FILE = (
ACTIVATION_DIR
/ "activation_metrics_with_subspace.json"
)

INDICATOR_FILE = (
BASE_DIR
/ "CC_regenerated_validated"
/ "cc_indicators.csv"
)

OUTPUT_DIR = (
ACTIVATION_DIR
/ "representational_effectiveness"
)

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Load Activation Metrics
# ============================================================

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

print("Loaded activation records:", len(records))

# ============================================================
# Load Indicators
# ============================================================

indicator_df = pd.read_csv(INDICATOR_FILE)

print("Loaded indicator rows:", len(indicator_df))

print("\nComplexity label distribution:")
print(indicator_df["Complexity_Label"].value_counts(dropna=False))

indicator_lookup = {
    str(row["pair_id"]): row.to_dict()
    for _, row in indicator_df.iterrows()
}

for record in records:
    pair_id = str(record.get("pair_id"))

    if pair_id in indicator_lookup:
        for key, value in indicator_lookup[pair_id].items():
            if key not in record:
                record[key] = value

print("Indicators merged into activation records.")

# ============================================================
# Technique Definitions
# ============================================================

techniques = {
	
	"Cosine Similarity":
	"Cosine_Similarity",
	
	"Layer-wise Activation Distance":
	"Layerwise_Distance",
	
	"Attention Entropy Analysis":
	"Attention_Entropy_Delta",
	
	"Subspace-oriented Representational Analysis":
	"Subspace_Similarity",
	
	"CKA Similarity":
	"CKA_Similarity",
}

# ============================================================
# Indicator Definitions
# ============================================================

PEARSON_INDICATORS = [
    "Delta_CC",
    "Relative_CC_Reduction",
    "Delta_Branch_Count",
    "Delta_Nesting_Depth",
    "Decision_Density_Change",
    "Structural_Decomposition_Ratio",
]

SPEARMAN_INDICATORS = [
    "Delta_CC",
    "Relative_CC_Reduction",
    "Delta_Branch_Count",
    "Delta_Nesting_Depth",
    "Decision_Density_Change",
    "Structural_Decomposition_Ratio",
    "Complexity_Rank",
]

# ============================================================
# Helper Functions
# ============================================================

def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def get_indicator(record, key):
    value = record.get(key)

    if value is not None:
        if key == "Complexity_Label":
            return value
        return safe_float(value)

    if key == "Decision_Density_Change":
        before = safe_float(record.get("Decision_Density_Before"))
        after = safe_float(record.get("Decision_Density_After"))

        if before is not None and after is not None:
            return before - after

    return None


def restructuring_signal(

    technique_name,

    values,

):

    values = np.array(

        values,

        dtype=float,

    )

    if technique_name in [

        "Cosine Similarity",

        "CKA Similarity",

    ]:

        return 1.0 - values

    if technique_name == (

        "Attention Entropy Analysis"

    ):

        return np.abs(values)

    if technique_name == (

        "Subspace-oriented Representational Analysis"

    ):

        return values

    return values


def safe_pearson(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    if len(x) < 2 or len(y) < 2:
        return 0.0

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0

    r, _ = pearsonr(x, y)

    return 0.0 if np.isnan(r) else float(r)


def safe_spearman(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    if len(x) < 2 or len(y) < 2:
        return 0.0

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0

    r, _ = spearmanr(x, y)

    return 0.0 if np.isnan(r) else float(r)


def get_num_layers(records, metric_key):
    for record in records:
        values = record.get(metric_key)

        if values is not None:
            return len(values)

    return 0


def collect_layer_values(records, metric_key, indicator_key, layer_idx, technique_name):
    metric_values = []
    indicator_values = []

    for record in records:
        metric = record.get(metric_key)
        indicator = get_indicator(record, indicator_key)

        if metric is None or indicator is None:
            continue

        try:
            metric_values.append(metric[layer_idx])
            indicator_values.append(indicator)
        except Exception:
            continue

    metric_values = restructuring_signal(
        technique_name,
        np.array(metric_values, dtype=float),
    )

    indicator_values = np.array(indicator_values, dtype=float)

    return metric_values, indicator_values


def mean_layer_signal(records, metric_key, technique_name):
    num_layers = get_num_layers(records, metric_key)
    layer_means = []

    for layer_idx in range(num_layers):
        values = []

        for record in records:
            metric = record.get(metric_key)

            if metric is None:
                continue

            try:
                values.append(metric[layer_idx])
            except Exception:
                continue

        if len(values) == 0:
            layer_means.append(0.0)
        else:
            values = restructuring_signal(
                technique_name,
                np.array(values, dtype=float),
            )

            layer_means.append(float(np.mean(values)))

    return np.array(layer_means, dtype=float)


def sample_signal_strength(record, metric_key, technique_name):
    metric = record.get(metric_key)

    if metric is None:
        return None

    try:
        signal = restructuring_signal(
            technique_name,
            np.array(metric, dtype=float),
        )

        return float(np.percentile(np.abs(signal), 90))

    except Exception:
        return None


def pairwise_discrimination_score(records, metric_key, technique_name):
    y_true = []
    y_score = []
    restructuring_values = []

    for record in records:
        relative_cc = get_indicator(record, "Relative_CC_Reduction")

        if relative_cc is not None:
            restructuring_values.append(relative_cc)

    if len(restructuring_values) < 2:
        return 0.5

    threshold = np.median(restructuring_values)

    for record in records:
        metric = record.get(metric_key)
        relative_cc = get_indicator(record, "Relative_CC_Reduction")

        if metric is None or relative_cc is None:
            continue

        try:
            signal = restructuring_signal(
                technique_name,
                np.array(metric, dtype=float),
            )

            signal_strength = float(np.percentile(np.abs(signal), 90))

        except Exception:
            continue

        label = 1 if relative_cc >= threshold else 0

        y_true.append(label)
        y_score.append(signal_strength)

    if len(y_true) < 2:
        return 0.5

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        auc = roc_auc_score(y_true, y_score)
        return float(max(auc, 1.0 - auc))
    except Exception:
        return 0.5


def layer_stability_score(layer_correlations):
    layer_correlations = np.array(layer_correlations, dtype=float)

    if len(layer_correlations) < 2:
        return 0.0

    mean_corr = float(np.mean(np.abs(layer_correlations)))
    std_corr = float(np.std(layer_correlations))

    return float(mean_corr / (1.0 + std_corr))


def cross_sample_variance_score(records, metric_key, technique_name):
    grouped_scores = {}

    for record in records:
        label = record.get("Complexity_Label")
        score = sample_signal_strength(record, metric_key, technique_name)

        if label is None or score is None:
            continue

        grouped_scores.setdefault(str(label), []).append(score)

    group_scores = []

    for values in grouped_scores.values():
        if len(values) < 2:
            continue

        values = np.array(values, dtype=float)
        mean_signal = float(np.mean(np.abs(values)))
        variance = float(np.var(values))

        group_scores.append(mean_signal / (1.0 + variance))

    if len(group_scores) == 0:
        return 0.0

    return float(np.mean(group_scores))


def rank_column(df, score_col, rank_col, ascending=False):
    df[rank_col] = df[score_col].rank(
        method="min",
        ascending=ascending,
    )

    return df

# ============================================================
# Main Analysis
# ============================================================

summary_rows = []
layerwise_rows = []

for technique_name, metric_key in techniques.items():

    print(f"\nAnalyzing: {technique_name}")

    num_layers = get_num_layers(records, metric_key)

    if num_layers == 0:
        print(f"No valid metric values found for {technique_name}")
        continue

    # ========================================================
    # Pearson Alignment
    # ========================================================

    pearson_scores = []
    pearson_layer_scores = []

    for layer_idx in range(num_layers):
        layer_indicator_scores = []

        for indicator_key in PEARSON_INDICATORS:
            x, y = collect_layer_values(
                records,
                metric_key,
                indicator_key,
                layer_idx,
                technique_name,
            )

            r = abs(safe_pearson(x, y))

            layer_indicator_scores.append(r)
            pearson_scores.append(r)

        pearson_layer_scores.append(
            float(np.mean(layer_indicator_scores))
            if layer_indicator_scores else 0.0
        )

    S_CC_Pearson = float(np.mean(pearson_scores)) if pearson_scores else 0.0

    # ========================================================
    # Spearman Alignment
    # ========================================================

    spearman_scores = []
    spearman_layer_scores = []

    for layer_idx in range(num_layers):
        layer_indicator_scores = []

        for indicator_key in SPEARMAN_INDICATORS:
            x, y = collect_layer_values(
                records,
                metric_key,
                indicator_key,
                layer_idx,
                technique_name,
            )

            r = abs(safe_spearman(x, y))

            layer_indicator_scores.append(r)
            spearman_scores.append(r)

        spearman_layer_scores.append(
            float(np.mean(layer_indicator_scores))
            if layer_indicator_scores else 0.0
        )

    S_CC_Spearman = float(np.mean(spearman_scores)) if spearman_scores else 0.0

    # ========================================================
    # Pairwise Discrimination
    # ========================================================

    S_CC_Pairwise = pairwise_discrimination_score(
        records,
        metric_key,
        technique_name,
    )

    # ========================================================
    # Layer-wise Stability
    # ========================================================

    S_CC_Layer = layer_stability_score(
        spearman_layer_scores,
    )

    # ========================================================
    # Cross-sample Variance
    # ========================================================

    S_CC_Variance = cross_sample_variance_score(
        records,
        metric_key,
        technique_name,
    )

    # ========================================================
    # Layer-wise Output
    # ========================================================

    layer_means = mean_layer_signal(
        records,
        metric_key,
        technique_name,
    )

    for layer_idx in range(num_layers):
        layerwise_rows.append({
            "Technique": technique_name,
            "Layer": layer_idx,
            "Layer_Mean": float(layer_means[layer_idx]),
            "S_CC_Pearson_Layer": float(pearson_layer_scores[layer_idx]),
            "S_CC_Spearman_Layer": float(spearman_layer_scores[layer_idx]),
        })

    summary_rows.append({
        "Technique": technique_name,
        "Technique_Type": "Layer-wise scalar",
        "S_CC_Pearson": S_CC_Pearson,
        "S_CC_Spearman": S_CC_Spearman,
        "S_CC_Pairwise": S_CC_Pairwise,
        "S_CC_Layer": S_CC_Layer,
        "S_CC_Variance": S_CC_Variance,
        "Num_Layers": num_layers,
    })

# ============================================================
# Create DataFrames
# ============================================================

summary_df = pd.DataFrame(summary_rows)
layerwise_df = pd.DataFrame(layerwise_rows)

# ============================================================
# Per-measure Ranking
# ============================================================

summary_df = rank_column(summary_df, "S_CC_Pearson", "Rank_Pearson", ascending=False)
summary_df = rank_column(summary_df, "S_CC_Spearman", "Rank_Spearman", ascending=False)
summary_df = rank_column(summary_df, "S_CC_Pairwise", "Rank_Pairwise", ascending=False)
summary_df = rank_column(summary_df, "S_CC_Layer", "Rank_Layer", ascending=False)
summary_df = rank_column(summary_df, "S_CC_Variance", "Rank_Variance", ascending=False)

# ============================================================
# Aggregate Rank
# ============================================================

rank_columns = [
    "Rank_Pearson",
    "Rank_Spearman",
    "Rank_Pairwise",
    "Rank_Layer",
    "Rank_Variance",
]

# ============================================================
# Aggregate Score from Normalized Effectiveness Measures
# ============================================================

score_columns = [
    "S_CC_Pearson",
    "S_CC_Spearman",
    "S_CC_Pairwise",
    "S_CC_Layer",
    "S_CC_Variance",
]

normalized_columns = []

for col in score_columns:

    norm_col = col + "_Norm"

    min_v = summary_df[col].min()
    max_v = summary_df[col].max()

    if abs(max_v - min_v) < 1e-12:
        summary_df[norm_col] = 0.0
    else:
        summary_df[norm_col] = (
            summary_df[col] - min_v
        ) / (
            max_v - min_v
        )

    normalized_columns.append(norm_col)

summary_df["S_CC_Aggregate"] = (
    summary_df[normalized_columns]
    .mean(axis=1)
)

summary_df = summary_df.sort_values(
    by="S_CC_Aggregate",
    ascending=False,
)

summary_df["Rank"] = np.arange(len(summary_df)) + 1

# ============================================================
# Save Outputs
# ============================================================

summary_csv = OUTPUT_DIR / "representational_effectiveness_summary.csv"
summary_json = OUTPUT_DIR / "representational_effectiveness_summary.json"
layerwise_csv = OUTPUT_DIR / "layerwise_effectiveness.csv"

summary_df.to_csv(summary_csv, index=False)
layerwise_df.to_csv(layerwise_csv, index=False)

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(
        summary_df.to_dict(orient="records"),
        f,
        indent=2,
    )

# ============================================================
# Print Results
# ============================================================

print("\n================================================")
print("Representational Effectiveness Ranking")
print("================================================")

print(
    summary_df[
        [
            "Rank",
            "Technique",
            "S_CC_Aggregate",
            "S_CC_Pearson",
            "S_CC_Spearman",
            "S_CC_Pairwise",
            "S_CC_Layer",
            "S_CC_Variance",
            "Rank_Pearson",
            "Rank_Spearman",
            "Rank_Pairwise",
            "Rank_Layer",
            "Rank_Variance",
        ]
    ]
)

print("\nSaved files:")
print("-", summary_csv)
print("-", summary_json)
print("-", layerwise_csv)

print("\nRepresentational effectiveness analysis complete.")