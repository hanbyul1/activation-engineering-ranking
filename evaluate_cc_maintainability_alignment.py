# ============================================================
# evaluate_cc_maintainability_alignment.py
#
# Evaluate maintainability alignment of activation-space
# restructuring behaviors discovered by
# detect_cc_maintainability_patterns.py.
#
# This script performs ONLY maintainability interpretation:
#   - Pearson alignment
#   - Spearman alignment
#   - Pairwise discrimination
#   - Layer-wise stability
#   - Cross-sample variance consistency
#   - Technique-level ranking
#   - Pattern-level alignment scoring
#
# Author: Dae-Kyoo Kim
# ============================================================

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

ACTIVATION_FILE = (
    BASE_DIR
    / "activation_analysis_outputs"
    / "activation_metrics_with_subspace.json"
)

INDICATOR_FILE = (
    BASE_DIR
    / "CC_regenerated_validated"
    / "cc_indicators.csv"
)

PATTERN_DIR = (
    BASE_DIR
    / "activation_analysis_outputs"
    / "cc_maintainability_patterns"
)

PATTERN_FILE = (
    PATTERN_DIR
    / "cc_maintainability_patterns_summary.csv"
)

LAYERWISE_FILE = (
    PATTERN_DIR
    / "cc_maintainability_layerwise_summary.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "activation_analysis_outputs"
    / "cc_maintainability_alignment"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TECHNIQUE_SUMMARY_CSV = (
    OUTPUT_DIR
    / "cc_maintainability_technique_alignment_summary.csv"
)

TECHNIQUE_SUMMARY_JSON = (
    OUTPUT_DIR
    / "cc_maintainability_technique_alignment_summary.json"
)

PATTERN_ALIGNMENT_CSV = (
    OUTPUT_DIR
    / "cc_maintainability_pattern_alignment_summary.csv"
)

PATTERN_ALIGNMENT_JSON = (
    OUTPUT_DIR
    / "cc_maintainability_pattern_alignment_summary.json"
)


# ============================================================
# Technique Definitions
# ============================================================

TECHNIQUES = {
    "Cosine Similarity": "Cosine_Similarity",

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
# Load Data
# ============================================================

print("\nLoading activation metrics...")

with open(
    ACTIVATION_FILE,
    "r",
    encoding="utf-8",
) as f:

    records = json.load(f)

print("Loaded activation records:", len(records))

print("\nLoading CC-oriented maintainability indicators...")

indicator_df = pd.read_csv(
    INDICATOR_FILE
)

print("Loaded indicator rows:", len(indicator_df))

indicator_lookup = {

    str(row["pair_id"]): row.to_dict()

    for _, row in indicator_df.iterrows()
}

for record in records:

    pair_id = str(
        record.get("pair_id")
    )

    if pair_id in indicator_lookup:

        for key, value in indicator_lookup[pair_id].items():

            if key not in record:
                record[key] = value

print("Indicators merged into activation records.")

print("\nLoading discovered activation patterns...")

pattern_df = pd.read_csv(
    PATTERN_FILE
)

layerwise_df = pd.read_csv(
    LAYERWISE_FILE
)

print("Pattern rows:", len(pattern_df))
print("Layerwise rows:", len(layerwise_df))


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

        before = safe_float(
            record.get("Decision_Density_Before")
        )

        after = safe_float(
            record.get("Decision_Density_After")
        )

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

    if technique_name == "Attention Entropy Analysis":

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


def min_max_normalize(series):

    values = series.astype(float)

    min_v = values.min()
    max_v = values.max()

    if abs(max_v - min_v) < 1e-12:
        return np.zeros(len(values))

    return (
        values - min_v
    ) / (
        max_v - min_v
    )


def rank_column(
    df,
    score_col,
    rank_col,
    ascending=False,
):

    df[rank_col] = df[score_col].rank(
        method="min",
        ascending=ascending,
    )

    return df


def get_num_layers(
    records,
    metric_key,
):

    for record in records:

        values = record.get(metric_key)

        if values is not None:
            return len(values)

    return 0


# ============================================================
# Layer Collection
# ============================================================

def collect_layer_values(
    records,
    metric_key,
    indicator_key,
    layer_idx,
    technique_name,
):

    metric_values = []
    indicator_values = []

    for record in records:

        metric = record.get(metric_key)

        indicator = get_indicator(
            record,
            indicator_key,
        )

        if metric is None or indicator is None:
            continue

        try:

            value = metric[layer_idx]

            if value is None:
                continue

            metric_values.append(value)
            indicator_values.append(indicator)

        except Exception:

            continue

    metric_values = restructuring_signal(
        technique_name,
        np.array(metric_values, dtype=float),
    )

    indicator_values = np.array(
        indicator_values,
        dtype=float,
    )

    return metric_values, indicator_values


def collect_pattern_values(
    records,
    metric_key,
    technique_name,
    start_layer,
    end_layer,
    indicator_key,
):

    metric_values = []
    indicator_values = []

    for record in records:

        metric = record.get(metric_key)

        indicator = get_indicator(
            record,
            indicator_key,
        )

        if metric is None or indicator is None:
            continue

        try:

            segment = np.array(
                metric[start_layer:end_layer + 1],
                dtype=float,
            )

            signal = restructuring_signal(
                technique_name,
                segment,
            )

            strength = float(
                np.percentile(
                    np.abs(signal),
                    90,
                )
            )

            metric_values.append(strength)
            indicator_values.append(indicator)

        except Exception:

            continue

    return (
        np.array(metric_values, dtype=float),
        np.array(indicator_values, dtype=float),
    )


def sample_signal_strength(
    record,
    metric_key,
    technique_name,
):

    metric = record.get(metric_key)

    if metric is None:
        return None

    try:

        signal = restructuring_signal(
            technique_name,
            np.array(metric, dtype=float),
        )

        return float(
            np.percentile(
                np.abs(signal),
                90,
            )
        )

    except Exception:

        return None


def pattern_signal_strength(
    record,
    metric_key,
    technique_name,
    start_layer,
    end_layer,
):

    metric = record.get(metric_key)

    if metric is None:
        return None

    try:

        segment = np.array(
            metric[start_layer:end_layer + 1],
            dtype=float,
        )

        signal = restructuring_signal(
            technique_name,
            segment,
        )

        return float(
            np.percentile(
                np.abs(signal),
                90,
            )
        )

    except Exception:

        return None


# ============================================================
# Evaluation Measures
# ============================================================

def pairwise_discrimination_score(
    records,
    metric_key,
    technique_name,
    start_layer=None,
    end_layer=None,
):

    y_true = []
    y_score = []

    restructuring_values = []

    for record in records:

        relative_cc = get_indicator(
            record,
            "Relative_CC_Reduction",
        )

        if relative_cc is not None:
            restructuring_values.append(relative_cc)

    if len(restructuring_values) < 2:
        return 0.5

    low_threshold = np.percentile(
        restructuring_values,
        30,
    )

    high_threshold = np.percentile(
        restructuring_values,
        70,
    )

    for record in records:

        relative_cc = get_indicator(
            record,
            "Relative_CC_Reduction",
        )

        if relative_cc is None:
            continue

        if start_layer is None or end_layer is None:

            signal_strength = sample_signal_strength(
                record,
                metric_key,
                technique_name,
            )

        else:

            signal_strength = pattern_signal_strength(
                record,
                metric_key,
                technique_name,
                start_layer,
                end_layer,
            )

        if signal_strength is None:
            continue

        if relative_cc <= low_threshold:

            label = 0

        elif relative_cc >= high_threshold:

            label = 1

        else:

            continue

        y_true.append(label)
        y_score.append(signal_strength)

    if len(y_true) < 2:
        return 0.5

    y_true = np.array(
        y_true,
        dtype=int,
    )

    y_score = np.array(
        y_score,
        dtype=float,
    )

    if len(np.unique(y_true)) < 2:
        return 0.5

    try:

        auc = roc_auc_score(
            y_true,
            y_score,
        )

        return float(
            max(
                auc,
                1.0 - auc,
            )
        )

    except Exception:

        return 0.5


def layer_stability_score(
    layer_correlations,
):

    layer_correlations = np.array(
        layer_correlations,
        dtype=float,
    )

    if len(layer_correlations) < 2:
        return 0.0

    variance = float(
        np.var(layer_correlations)
    )

    return float(
        1.0 / (1.0 + variance)
    )


def cross_sample_variance_score(
    records,
    metric_key,
    technique_name,
    start_layer=None,
    end_layer=None,
):

    grouped_scores = {}

    for record in records:

        label = record.get(
            "Complexity_Label"
        )

        if label is None:
            continue

        if start_layer is None or end_layer is None:

            score = sample_signal_strength(
                record,
                metric_key,
                technique_name,
            )

        else:

            score = pattern_signal_strength(
                record,
                metric_key,
                technique_name,
                start_layer,
                end_layer,
            )

        if score is None:
            continue

        grouped_scores.setdefault(
            str(label),
            [],
        ).append(score)

    group_scores = []

    for values in grouped_scores.values():

        if len(values) < 2:
            continue

        values = np.array(
            values,
            dtype=float,
        )

        mean_signal = float(
            np.mean(
                np.abs(values)
            )
        )

        variance = float(
            np.var(values)
        )

        group_scores.append(
            mean_signal / (1.0 + variance)
        )

    if len(group_scores) == 0:
        return 0.0

    return float(
        np.mean(group_scores)
    )


# ============================================================
# Technique-Level Alignment
# ============================================================

summary_rows = []
layerwise_alignment_rows = []

for technique_name, metric_key in TECHNIQUES.items():

    print(
        f"\nEvaluating technique alignment: {technique_name}"
    )

    num_layers = get_num_layers(
        records,
        metric_key,
    )

    if num_layers == 0:

        print(
            f"No valid metric values found for "
            f"{technique_name}"
        )

        continue

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

            r = abs(
                safe_pearson(x, y)
            )

            layer_indicator_scores.append(r)
            pearson_scores.append(r)

        pearson_layer_scores.append(
            float(
                np.mean(layer_indicator_scores)
            )
            if layer_indicator_scores
            else 0.0
        )

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

            r = abs(
                safe_spearman(x, y)
            )

            layer_indicator_scores.append(r)
            spearman_scores.append(r)

        spearman_layer_scores.append(
            float(
                np.mean(layer_indicator_scores)
            )
            if layer_indicator_scores
            else 0.0
        )

    S_CC_Pearson = (
        float(np.mean(pearson_scores))
        if pearson_scores
        else 0.0
    )

    S_CC_Spearman = (
        float(np.mean(spearman_scores))
        if spearman_scores
        else 0.0
    )

    S_CC_Pairwise = pairwise_discrimination_score(
        records,
        metric_key,
        technique_name,
    )

    S_CC_Layer = layer_stability_score(
        spearman_layer_scores,
    )

    S_CC_Variance = cross_sample_variance_score(
        records,
        metric_key,
        technique_name,
    )

    for layer_idx in range(num_layers):

        layerwise_alignment_rows.append({

            "Technique":
                technique_name,

            "Layer":
                layer_idx,

            "S_CC_Pearson_Layer":
                float(
                    pearson_layer_scores[layer_idx]
                ),

            "S_CC_Spearman_Layer":
                float(
                    spearman_layer_scores[layer_idx]
                ),
        })

    summary_rows.append({

        "Technique":
            technique_name,

        "Metric_Key":
            metric_key,

        "Technique_Type":
            "Layer-wise scalar",

        "S_CC_Pearson":
            S_CC_Pearson,

        "S_CC_Spearman":
            S_CC_Spearman,

        "S_CC_Pairwise":
            S_CC_Pairwise,

        "S_CC_Layer":
            S_CC_Layer,

        "S_CC_Variance":
            S_CC_Variance,

        "Num_Layers":
            num_layers,
    })


technique_df = pd.DataFrame(
    summary_rows
)

layerwise_alignment_df = pd.DataFrame(
    layerwise_alignment_rows
)


# ============================================================
# Technique-Level Ranking
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

    technique_df[norm_col] = min_max_normalize(
        technique_df[col]
    )

    normalized_columns.append(norm_col)

technique_df["S_CC_Aggregate"] = (
    technique_df[
        normalized_columns
    ].mean(axis=1)
)

technique_df = technique_df.sort_values(
    by="S_CC_Aggregate",
    ascending=False,
).reset_index(drop=True)

technique_df["Rank"] = (
    np.arange(len(technique_df)) + 1
)

technique_df = rank_column(
    technique_df,
    "S_CC_Pearson",
    "Rank_Pearson",
    ascending=False,
)

technique_df = rank_column(
    technique_df,
    "S_CC_Spearman",
    "Rank_Spearman",
    ascending=False,
)

technique_df = rank_column(
    technique_df,
    "S_CC_Pairwise",
    "Rank_Pairwise",
    ascending=False,
)

technique_df = rank_column(
    technique_df,
    "S_CC_Layer",
    "Rank_Layer",
    ascending=False,
)

technique_df = rank_column(
    technique_df,
    "S_CC_Variance",
    "Rank_Variance",
    ascending=False,
)


# ============================================================
# Pattern-Level Alignment
# ============================================================

pattern_rows = []

for _, pattern in pattern_df.iterrows():

    technique_name = pattern["Technique"]

    metric_key = TECHNIQUES[
        technique_name
    ]

    start_layer = int(
        pattern["Layer_Start"]
    )

    end_layer = int(
        pattern["Layer_End"]
    )

    pearson_scores = []

    for indicator_key in PEARSON_INDICATORS:

        x, y = collect_pattern_values(
            records,
            metric_key,
            technique_name,
            start_layer,
            end_layer,
            indicator_key,
        )

        pearson_scores.append(
            abs(
                safe_pearson(x, y)
            )
        )

    spearman_scores = []

    for indicator_key in SPEARMAN_INDICATORS:

        x, y = collect_pattern_values(
            records,
            metric_key,
            technique_name,
            start_layer,
            end_layer,
            indicator_key,
        )

        spearman_scores.append(
            abs(
                safe_spearman(x, y)
            )
        )

    pattern_pairwise = pairwise_discrimination_score(
        records,
        metric_key,
        technique_name,
        start_layer,
        end_layer,
    )

    segment_alignment_df = layerwise_alignment_df[
        (
            layerwise_alignment_df["Technique"]
            == technique_name
        )
        &
        (
            layerwise_alignment_df["Layer"]
            >= start_layer
        )
        &
        (
            layerwise_alignment_df["Layer"]
            <= end_layer
        )
    ]

    if len(segment_alignment_df) >= 2:

        combined_layer_alignment = (
            segment_alignment_df[
                "S_CC_Pearson_Layer"
            ].astype(float).to_numpy()
            +
            segment_alignment_df[
                "S_CC_Spearman_Layer"
            ].astype(float).to_numpy()
        ) / 2.0

        pattern_layer_stability = (
            1.0
            /
            (
                1.0
                +
                float(
                    np.var(
                        combined_layer_alignment
                    )
                )
            )
        )

    else:

        pattern_layer_stability = 0.0

    pattern_variance = cross_sample_variance_score(
        records,
        metric_key,
        technique_name,
        start_layer,
        end_layer,
    )

    pattern_rows.append({

        **pattern.to_dict(),

        "Pattern_Pearson_Alignment":
            float(np.mean(pearson_scores))
            if pearson_scores
            else 0.0,

        "Pattern_Spearman_Alignment":
            float(np.mean(spearman_scores))
            if spearman_scores
            else 0.0,

        "Pattern_Pairwise_Discrimination":
            pattern_pairwise,

        "Pattern_Layer_Stability":
            pattern_layer_stability,

        "Pattern_Variance_Consistency":
            pattern_variance,
    })


pattern_alignment_df = pd.DataFrame(
    pattern_rows
)


# ============================================================
# Pattern-Level Aggregate Alignment
# ============================================================

pattern_score_columns = [
    "Pattern_Pearson_Alignment",
    "Pattern_Spearman_Alignment",
    "Pattern_Pairwise_Discrimination",
    "Pattern_Layer_Stability",
    "Pattern_Variance_Consistency",
]

pattern_normalized_columns = []

for col in pattern_score_columns:

    norm_col = col + "_Norm"

    pattern_alignment_df[norm_col] = min_max_normalize(
        pattern_alignment_df[col]
    )

    pattern_normalized_columns.append(norm_col)

pattern_alignment_df[
    "Pattern_Combined_Alignment"
] = (
    pattern_alignment_df[
        pattern_normalized_columns
    ].mean(axis=1)
)

pattern_alignment_df = pattern_alignment_df.sort_values(
    by=[
        "Pattern_Combined_Alignment",
        "Mean_Restructuring_Signal",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(drop=True)

pattern_alignment_df["Pattern_Alignment_Rank"] = (
    np.arange(len(pattern_alignment_df)) + 1
)


# ============================================================
# Save Outputs
# ============================================================

technique_df.to_csv(
    TECHNIQUE_SUMMARY_CSV,
    index=False,
)

with open(
    TECHNIQUE_SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        technique_df.to_dict(orient="records"),
        f,
        indent=2,
    )

pattern_alignment_df.to_csv(
    PATTERN_ALIGNMENT_CSV,
    index=False,
)

with open(
    PATTERN_ALIGNMENT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        pattern_alignment_df.to_dict(orient="records"),
        f,
        indent=2,
    )


# ============================================================
# Print Results
# ============================================================

print("\n================================================")
print("CC Maintainability Alignment Evaluation Complete")
print("================================================")

print("\nTechnique-level alignment summary:")

print(
    technique_df[
        [
            "Rank",
            "Technique",
            "S_CC_Aggregate",
            "S_CC_Pearson",
            "S_CC_Spearman",
            "S_CC_Pairwise",
            "S_CC_Layer",
            "S_CC_Variance",
        ]
    ]
)

print("\nTop maintainability-aligned patterns:")

if not pattern_alignment_df.empty:

    print(
        pattern_alignment_df[
            [
                "Pattern_Alignment_Rank",
                "Pattern_ID",
                "Technique",
                "Discovered_Pattern",
                "Layer_Start",
                "Layer_End",
                "Layer_Count",
                "Pattern_Combined_Alignment",
                "Pattern_Pearson_Alignment",
                "Pattern_Spearman_Alignment",
                "Pattern_Pairwise_Discrimination",
                "Pattern_Layer_Stability",
                "Pattern_Variance_Consistency",
            ]
        ].head(15)
    )

print("\nOutput files:")
print("-", TECHNIQUE_SUMMARY_CSV)
print("-", TECHNIQUE_SUMMARY_JSON)
print("-", PATTERN_ALIGNMENT_CSV)
print("-", PATTERN_ALIGNMENT_JSON)

print("\nMaintainability alignment evaluation complete.")
print("================================================")