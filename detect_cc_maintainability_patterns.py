# ============================================================
# detect_cc_maintainability_patterns.py
#
# Discover activation-space restructuring patterns from
# structurally detected representational changes.
#
# This script performs ONLY activation-magnitude-based pattern
# discovery. It does NOT compute maintainability alignment,
# correlations, pairwise discrimination, or technique ranking.
#
# Author: Dae-Kyoo Kim
# ============================================================

import json
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).parent

ACTIVATION_FILE = (
    BASE_DIR
    / "activation_analysis_outputs"
    / "activation_metrics_detected_changes.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "activation_analysis_outputs"
    / "cc_maintainability_patterns"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PATTERN_JSON = (
    OUTPUT_DIR
    / "cc_maintainability_patterns.json"
)

PATTERN_SUMMARY_CSV = (
    OUTPUT_DIR
    / "cc_maintainability_patterns_summary.csv"
)

LAYERWISE_SUMMARY_CSV = (
    OUTPUT_DIR
    / "cc_maintainability_layerwise_summary.csv"
)


# ============================================================
# Pattern Discovery Parameters
# ============================================================

MIN_PATTERN_LAYERS = 2
MERGE_OVERLAP_RATIO = 0.80

HIGH_SIGNAL_PERCENTILE = 75
SPIKE_Z_THRESHOLD = 1.5

STABLE_WINDOW_MIN_LAYERS = 5
STABLE_CV_THRESHOLD = 0.03


# ============================================================
# Technique Definitions
# ============================================================

TECHNIQUES = {
    "Cosine Similarity": "cosine_similarity",

    "Layer-wise Activation Distance":
        "activation_distance",

    "Attention Entropy Analysis":
        "attention_entropy",

    "Subspace-oriented Representational Analysis":
        "subspace_analysis",

    "CKA Similarity":
        "cka_similarity",
}


TECHNIQUE_BEHAVIOR = {
    "Cosine Similarity":
        "directional representational shift",

    "Layer-wise Activation Distance":
        "activation redistribution magnitude",

    "Attention Entropy Analysis":
        "attention redistribution behavior",

    "Subspace-oriented Representational Analysis":
        "latent subspace relocation behavior",

    "CKA Similarity":
        "relational preservation or reorganization behavior",
}


# ============================================================
# Load Structurally Detected Activation Changes
# ============================================================

print(
    "\nLoading structurally detected activation changes..."
)

with open(
    ACTIVATION_FILE,
    "r",
    encoding="utf-8",
) as f:
    records = json.load(f)

print(
    "Loaded detected-change records:",
    len(records),
)


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


def extract_layer_index(layer_name):

    return int(
        str(layer_name).replace("layer_", "")
    )


def get_sorted_layers(record):

    layers = record.get(
        "layers",
        {},
    )

    return sorted(
        layers.items(),
        key=lambda item: extract_layer_index(item[0]),
    )


def get_metric_value(
    layer_data,
    technique_key,
):

    metric_data = layer_data.get(
        technique_key,
        {},
    )

    if not isinstance(metric_data, dict):
        return safe_float(metric_data)

    if technique_key == "cosine_similarity":

        similarity = safe_float(
            metric_data.get(
                "cosine_similarity",
                metric_data.get(
                    "matrix_cosine_similarity",
                    None,
                ),
            )
        )

        if similarity is None:
            return safe_float(
                metric_data.get(
                    "restructuring_magnitude",
                    None,
                )
            )

        return 1.0 - similarity

    if technique_key == "activation_distance":

        return safe_float(
            metric_data.get(
                "mean_token_distance",
                metric_data.get(
                    "activation_distance",
                    None,
                ),
            )
        )

    if technique_key == "attention_entropy":

        value = safe_float(
            metric_data.get(
                "entropy_change",
                None,
            )
        )

        return abs(value) if value is not None else None

    if technique_key == "subspace_analysis":

        value = safe_float(
            metric_data.get(
                "restructuring_magnitude",
                None,
            )
        )

        if value is not None:
            return value

        similarity = safe_float(
            metric_data.get(
                "subspace_similarity",
                None,
            )
        )

        return similarity

    if technique_key == "cka_similarity":

        similarity = safe_float(
            metric_data.get(
                "cka_similarity",
                None,
            )
        )

        if similarity is None:
            return safe_float(
                metric_data.get(
                    "restructuring_magnitude",
                    None,
                )
            )

        return 1.0 - similarity

    return None


def get_num_layers(
    records,
):

    for record in records:

        layers = record.get(
            "layers",
            {},
        )

        if layers:
            return len(layers)

    return 0


def mean_layer_signal(
    records,
    technique_key,
):

    num_layers = get_num_layers(
        records
    )

    layer_values = {
        layer_idx: []
        for layer_idx in range(num_layers)
    }

    for record in records:

        sorted_layers = get_sorted_layers(
            record
        )

        for layer_name, layer_data in sorted_layers:

            layer_idx = extract_layer_index(
                layer_name
            )

            value = get_metric_value(
                layer_data,
                technique_key,
            )

            if value is None:
                continue

            layer_values.setdefault(
                layer_idx,
                [],
            ).append(abs(value))

    layer_means = []

    for layer_idx in range(num_layers):

        values = layer_values.get(
            layer_idx,
            [],
        )

        if values:
            layer_means.append(
                float(
                    np.mean(
                        np.array(
                            values,
                            dtype=float,
                        )
                    )
                )
            )

        else:
            layer_means.append(0.0)

    return np.array(
        layer_means,
        dtype=float,
    )


def contiguous_segments(
    layer_indices,
):

    if not layer_indices:
        return []

    layer_indices = sorted(layer_indices)

    segments = []

    start = layer_indices[0]
    previous = layer_indices[0]

    for value in layer_indices[1:]:

        if value == previous + 1:
            previous = value

        else:
            segments.append(
                (start, previous)
            )

            start = value
            previous = value

    segments.append(
        (start, previous)
    )

    return segments


def layer_region(
    start_layer,
    end_layer,
    num_layers,
):

    midpoint = (
        start_layer + end_layer
    ) / 2.0

    ratio = midpoint / max(
        num_layers - 1,
        1,
    )

    if ratio < 0.33:
        return "early-layer"

    if ratio < 0.66:
        return "mid-layer"

    return "deep-layer"


def describe_pattern(
    technique_name,
    behavior_type,
    region,
    start_layer,
    end_layer,
):

    base = TECHNIQUE_BEHAVIOR.get(
        technique_name,
        "representational restructuring behavior",
    )

    layer_range = f"layers {start_layer}--{end_layer}"

    if behavior_type == "high_signal":

        return (
            f"{region} {layer_range} "
            f"high-magnitude {base}"
        )

    if behavior_type == "signal_spike":

        return (
            f"{region} {layer_range} "
            f"localized spike in {base}"
        )

    if behavior_type == "stable_band":

        return (
            f"{region} {layer_range} "
            f"stable restructuring band of {base}"
        )

    return (
        f"{region} {layer_range} {base}"
    )


def segment_overlap_ratio(
    seg_a,
    seg_b,
):

    a_start, a_end = seg_a
    b_start, b_end = seg_b

    intersection = max(
        0,
        min(a_end, b_end)
        - max(a_start, b_start)
        + 1,
    )

    union = (
        max(a_end, b_end)
        - min(a_start, b_start)
        + 1
    )

    if union == 0:
        return 0.0

    return intersection / union


def merge_overlapping_candidates(
    candidates,
):

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda item: (
            item["Technique"],
            item["Layer_Start"],
            item["Layer_End"],
            -item["Preliminary_Strength"],
        ),
    )

    merged = []

    for candidate in candidates:

        should_merge = False

        for existing in merged:

            if (
                existing["Technique"]
                != candidate["Technique"]
            ):
                continue

            overlap = segment_overlap_ratio(
                (
                    existing["Layer_Start"],
                    existing["Layer_End"],
                ),
                (
                    candidate["Layer_Start"],
                    candidate["Layer_End"],
                ),
            )

            if overlap >= MERGE_OVERLAP_RATIO:

                behavior_types = set(
                    existing["Behavior_Type"].split("+")
                )

                behavior_types.add(
                    candidate["Behavior_Type"]
                )

                existing["Behavior_Type"] = "+".join(
                    sorted(behavior_types)
                )

                existing["Layer_Start"] = min(
                    existing["Layer_Start"],
                    candidate["Layer_Start"],
                )

                existing["Layer_End"] = max(
                    existing["Layer_End"],
                    candidate["Layer_End"],
                )

                existing["Preliminary_Strength"] = max(
                    existing["Preliminary_Strength"],
                    candidate["Preliminary_Strength"],
                )

                should_merge = True
                break

        if not should_merge:
            merged.append(
                candidate.copy()
            )

    return merged


def stable_band_segments(
    signal,
    layers,
):

    segments = []

    n = len(signal)

    for start_idx in range(n):

        for end_idx in range(
            start_idx
            + STABLE_WINDOW_MIN_LAYERS
            - 1,
            n,
        ):

            window = signal[
                start_idx:end_idx + 1
            ]

            mean_abs = np.mean(
                np.abs(window)
            )

            if mean_abs < 1e-12:
                continue

            cv = (
                np.std(window)
                / mean_abs
            )

            if cv <= STABLE_CV_THRESHOLD:

                segments.append(
                    (
                        int(layers[start_idx]),
                        int(layers[end_idx]),
                    )
                )

    maximal = []

    for seg in segments:

        contained = False

        for other in segments:

            if seg == other:
                continue

            if (
                seg[0] >= other[0]
                and seg[1] <= other[1]
            ):
                contained = True
                break

        if not contained:
            maximal.append(seg)

    return maximal


# ============================================================
# Discover Activation Patterns from Detected Changes
# ============================================================

def discover_activation_patterns(
    layerwise_df,
):

    raw_candidates = []

    for technique_name in (
        layerwise_df["Technique"].unique()
    ):

        subset = (
            layerwise_df[
                layerwise_df["Technique"]
                == technique_name
            ]
            .copy()
            .sort_values("Layer")
        )

        if subset.empty:
            continue

        layers = subset[
            "Layer"
        ].astype(int).to_numpy()

        signal = subset[
            "Layer_Mean_Restructuring_Signal"
        ].astype(float).to_numpy()

        signal_threshold = np.percentile(
            signal,
            HIGH_SIGNAL_PERCENTILE,
        )

        signal_mean = float(
            np.mean(signal)
        )

        signal_std = float(
            np.std(signal)
        )

        spike_threshold = (
            signal_mean
            + (
                SPIKE_Z_THRESHOLD
                * signal_std
            )
        )

        behavior_candidates = [
            {
                "behavior_type":
                    "high_signal",

                "layers":
                    layers[
                        signal >= signal_threshold
                    ].tolist(),

                "strength":
                    float(signal_threshold),
            },

            {
                "behavior_type":
                    "signal_spike",

                "layers":
                    layers[
                        signal >= spike_threshold
                    ].tolist(),

                "strength":
                    float(spike_threshold),
            },
        ]

        for start, end in stable_band_segments(
            signal,
            layers,
        ):

            segment_mask = (
                (layers >= start)
                &
                (layers <= end)
            )

            raw_candidates.append({

                "Technique":
                    technique_name,

                "Behavior_Type":
                    "stable_band",

                "Layer_Start":
                    int(start),

                "Layer_End":
                    int(end),

                "Preliminary_Strength":
                    float(
                        np.mean(
                            signal[segment_mask]
                        )
                    ),
            })

        for item in behavior_candidates:

            for start, end in contiguous_segments(
                item["layers"]
            ):

                if (
                    end - start + 1
                ) < MIN_PATTERN_LAYERS:
                    continue

                raw_candidates.append({

                    "Technique":
                        technique_name,

                    "Behavior_Type":
                        item["behavior_type"],

                    "Layer_Start":
                        int(start),

                    "Layer_End":
                        int(end),

                    "Preliminary_Strength":
                        float(item["strength"]),
                })

    merged_candidates = merge_overlapping_candidates(
        raw_candidates
    )

    discovered = []

    for candidate in merged_candidates:

        technique_name = candidate[
            "Technique"
        ]

        start = candidate[
            "Layer_Start"
        ]

        end = candidate[
            "Layer_End"
        ]

        subset = (
            layerwise_df[
                layerwise_df["Technique"]
                == technique_name
            ]
            .copy()
            .sort_values("Layer")
        )

        segment_df = subset[
            (
                subset["Layer"] >= start
            )
            &
            (
                subset["Layer"] <= end
            )
        ]

        if segment_df.empty:
            continue

        num_layers = len(subset)

        region = layer_region(
            start,
            end,
            num_layers,
        )

        pattern_name = describe_pattern(
            technique_name,
            candidate["Behavior_Type"],
            region,
            start,
            end,
        )

        discovered.append({

            "Technique":
                technique_name,

            "Discovered_Pattern":
                pattern_name,

            "Behavior_Type":
                candidate["Behavior_Type"],

            "Layer_Start":
                int(start),

            "Layer_End":
                int(end),

            "Layer_Count":
                int(end - start + 1),

            "Layer_Region":
                region,

            "Mean_Restructuring_Signal":
                float(
                    segment_df[
                        "Layer_Mean_Restructuring_Signal"
                    ].mean()
                ),

            "Peak_Restructuring_Signal":
                float(
                    segment_df[
                        "Layer_Mean_Restructuring_Signal"
                    ].max()
                ),

            "Min_Restructuring_Signal":
                float(
                    segment_df[
                        "Layer_Mean_Restructuring_Signal"
                    ].min()
                ),

            "Std_Restructuring_Signal":
                float(
                    segment_df[
                        "Layer_Mean_Restructuring_Signal"
                    ].std(ddof=0)
                ),
        })

    pattern_df = pd.DataFrame(
        discovered
    )

    if pattern_df.empty:
        return pattern_df

    pattern_df = pattern_df.sort_values(
        by=[
            "Technique",
            "Layer_Start",
            "Layer_End",
            "Mean_Restructuring_Signal",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    pattern_df["Pattern_ID"] = [
        f"PAT-{i:03d}"
        for i in range(1, len(pattern_df) + 1)
    ]

    columns = [
        "Pattern_ID",
        "Technique",
        "Discovered_Pattern",
        "Behavior_Type",
        "Layer_Start",
        "Layer_End",
        "Layer_Count",
        "Layer_Region",
        "Mean_Restructuring_Signal",
        "Peak_Restructuring_Signal",
        "Min_Restructuring_Signal",
        "Std_Restructuring_Signal",
    ]

    return pattern_df[columns]


# ============================================================
# Main Analysis
# ============================================================

layerwise_rows = []

for technique_name, technique_key in TECHNIQUES.items():

    print(
        f"\nAnalyzing detected-change magnitude: {technique_name}"
    )

    layer_means = mean_layer_signal(
        records,
        technique_key,
    )

    if len(layer_means) == 0:

        print(
            f"No valid metric values found for "
            f"{technique_name}"
        )

        continue

    for layer_idx, value in enumerate(layer_means):

        layerwise_rows.append({

            "Technique":
                technique_name,

            "Layer":
                layer_idx,

            "Layer_Mean_Restructuring_Signal":
                float(value),
        })


layerwise_df = pd.DataFrame(
    layerwise_rows
)

pattern_df = discover_activation_patterns(
    layerwise_df
)


# ============================================================
# Save Outputs
# ============================================================

layerwise_df.to_csv(
    LAYERWISE_SUMMARY_CSV,
    index=False,
)

pattern_df.to_csv(
    PATTERN_SUMMARY_CSV,
    index=False,
)

with open(
    PATTERN_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        pattern_df.to_dict(
            orient="records"
        ),
        f,
        indent=2,
    )


# ============================================================
# Print Results
# ============================================================

print("\n================================================")
print("Activation Pattern Detection Complete")
print("================================================")

print(
    "\nLayer-level rows:",
    len(layerwise_df),
)

print(
    "Discovered activation patterns:",
    len(pattern_df),
)

print("\nTop activation-magnitude patterns:")

if not pattern_df.empty:

    print(
        pattern_df[
            [
                "Pattern_ID",
                "Technique",
                "Discovered_Pattern",
                "Layer_Start",
                "Layer_End",
                "Layer_Count",
                "Mean_Restructuring_Signal",
                "Peak_Restructuring_Signal",
            ]
        ].head(15)
    )

print("\nOutput files:")
print("-", PATTERN_JSON)
print("-", PATTERN_SUMMARY_CSV)
print("-", LAYERWISE_SUMMARY_CSV)

print("\nActivation pattern discovery complete.")
print("================================================")