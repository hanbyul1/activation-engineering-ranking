# detect_representational_changes.py

import os
import json
import numpy as np
import pandas as pd

from pathlib import Path

from scipy.stats import entropy

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

INPUT_DIR = Path(
    "activation_analysis_outputs/token_level_activations"
)

OUTPUT_PATH = (
    "activation_analysis_outputs/"
    "activation_metrics_detected_changes.json"
)

SUMMARY_OUTPUT_PATH = (
    "activation_analysis_outputs/"
    "activation_metrics_detected_changes_summary.csv"
)


# ============================================================
# Utility Functions
# ============================================================

def save_json(data, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )


def to_array(value):

    return np.asarray(
        value,
        dtype=np.float32
    )


def l2_distance(a, b):

    return np.linalg.norm(a - b)


def flatten_matrix(matrix):

    return matrix.reshape(-1)


def safe_cosine(a, b):

    if np.linalg.norm(a) == 0:
        return 0.0

    if np.linalg.norm(b) == 0:
        return 0.0

    return float(
        cosine_similarity(
            a.reshape(1, -1),
            b.reshape(1, -1)
        )[0][0]
    )


def align_matrices(
    before_matrix,
    after_matrix
):

    min_rows = min(
        before_matrix.shape[0],
        after_matrix.shape[0]
    )

    min_cols = min(
        before_matrix.shape[1],
        after_matrix.shape[1]
    )

    before_aligned = before_matrix[
        :min_rows,
        :min_cols
    ]

    after_aligned = after_matrix[
        :min_rows,
        :min_cols
    ]

    return (
        before_aligned,
        after_aligned
    )


# ============================================================
# Technique 1:
# Matrix Cosine Similarity
# ============================================================

def detect_cosine_change(
    before_matrix,
    after_matrix
):

    before_matrix, after_matrix = (
        align_matrices(
            before_matrix,
            after_matrix
        )
    )

    before_flat = flatten_matrix(
        before_matrix
    )

    after_flat = flatten_matrix(
        after_matrix
    )

    similarity = safe_cosine(
        before_flat,
        after_flat
    )

    restructuring_magnitude = (
        1.0 - similarity
    )

    return {

        "cosine_similarity":
            similarity,

        "restructuring_magnitude":
            float(
                restructuring_magnitude
            ),
    }


# ============================================================
# Technique 2:
# Token Activation Distance
# ============================================================

def detect_activation_distance(
    before_matrix,
    after_matrix
):

    before_matrix, after_matrix = (
        align_matrices(
            before_matrix,
            after_matrix
        )
    )

    token_distances = np.linalg.norm(
        before_matrix - after_matrix,
        axis=1
    )

    return {

        "mean_token_distance":
            float(
                np.mean(token_distances)
            ),

        "max_token_distance":
            float(
                np.max(token_distances)
            ),

        "std_token_distance":
            float(
                np.std(token_distances)
            ),
    }


# ============================================================
# Technique 3:
# Attention Entropy Analysis
# ============================================================

def compute_entropy_distribution(
    matrix
):

    vec = np.abs(
        matrix.flatten()
    )

    total = np.sum(vec)

    if total == 0:
        return 0.0

    prob = vec / total

    return float(
        entropy(prob + 1e-12)
    )


def detect_attention_entropy_change(
    before_matrix,
    after_matrix
):

    before_matrix, after_matrix = (
        align_matrices(
            before_matrix,
            after_matrix
        )
    )

    before_entropy = (
        compute_entropy_distribution(
            before_matrix
        )
    )

    after_entropy = (
        compute_entropy_distribution(
            after_matrix
        )
    )

    entropy_change = abs(
        after_entropy
        - before_entropy
    )

    return {

        "before_entropy":
            float(before_entropy),

        "after_entropy":
            float(after_entropy),

        "entropy_change":
            float(entropy_change),
    }


# ============================================================
# Technique 4:
# CKA Similarity
# ============================================================

def gram_linear(x):

    return x @ x.T


def center_gram(gram):

    n = gram.shape[0]

    if n <= 1:
        return gram

    identity = np.eye(n)

    ones = (
        np.ones((n, n)) / n
    )

    return (
        (identity - ones)
        @ gram
        @ (identity - ones)
    )


def cka_similarity(x, y):

    x, y = align_matrices(x, y)

    gx = center_gram(
        gram_linear(x)
    )

    gy = center_gram(
        gram_linear(y)
    )

    numerator = np.sum(gx * gy)

    denominator = np.sqrt(
        np.sum(gx * gx)
        *
        np.sum(gy * gy)
    )

    if denominator == 0:
        return 0.0

    return (
        numerator / denominator
    )


def detect_cka_change(
    before_matrix,
    after_matrix
):

    cka_score = cka_similarity(
        before_matrix,
        after_matrix
    )

    restructuring_magnitude = (
        1.0 - cka_score
    )

    return {

        "cka_similarity":
            float(cka_score),

        "restructuring_magnitude":
            float(
                restructuring_magnitude
            ),
    }


# ============================================================
# Technique 5:
# Subspace-oriented Representational Analysis
# ============================================================

def detect_subspace_change(
    before_matrix,
    after_matrix,
    n_components=10
):

    before_matrix, after_matrix = (
        align_matrices(
            before_matrix,
            after_matrix
        )
    )

    min_samples = min(
        before_matrix.shape[0],
        after_matrix.shape[0]
    )

    min_features = min(
        before_matrix.shape[1],
        after_matrix.shape[1]
    )

    max_components = min(
        min_samples,
        min_features,
        n_components
    )

    if max_components < 2:

        before_vec = flatten_matrix(
            before_matrix
        )

        after_vec = flatten_matrix(
            after_matrix
        )

        similarity = safe_cosine(
            before_vec,
            after_vec
        )

        return {

            "subspace_similarity":
                similarity,

            "restructuring_magnitude":
                float(1.0 - similarity),

            "n_components":
                0,
        }

    scaler = StandardScaler()

    combined = np.vstack([
        before_matrix,
        after_matrix
    ])

    combined_scaled = (
        scaler.fit_transform(
            combined
        )
    )

    before_scaled = combined_scaled[
        :before_matrix.shape[0]
    ]

    after_scaled = combined_scaled[
        before_matrix.shape[0]:
    ]

    pca = PCA(
        n_components=max_components
    )

    before_subspace = (
        pca.fit_transform(
            before_scaled
        )
    )

    after_subspace = (
        pca.transform(
            after_scaled
        )
    )

    similarity = safe_cosine(
        flatten_matrix(before_subspace),
        flatten_matrix(after_subspace)
    )

    restructuring_magnitude = (
        1.0 - similarity
    )

    return {

        "subspace_similarity":
            float(similarity),

        "restructuring_magnitude":
            float(
                restructuring_magnitude
            ),

        "n_components":
            int(max_components),
    }


# ============================================================
# Build Summary Row
# ============================================================

def build_summary_row(
    pair_id,
    num_layers,
    layer_results
):

    cosine_values = []
    distance_values = []
    entropy_values = []
    cka_values = []
    subspace_values = []

    for layer_key, result in layer_results.items():

        cosine_values.append(
            result["cosine_similarity"][
                "restructuring_magnitude"
            ]
        )

        distance_values.append(
            result["activation_distance"][
                "mean_token_distance"
            ]
        )

        entropy_values.append(
            result["attention_entropy"][
                "entropy_change"
            ]
        )

        cka_values.append(
            result["cka_similarity"][
                "restructuring_magnitude"
            ]
        )

        subspace_values.append(
            result["subspace_analysis"][
                "restructuring_magnitude"
            ]
        )

    return {

        "pair_id":
            pair_id,

        "num_layers":
            num_layers,

        "Cosine_Similarity":
            cosine_values,

        "Layerwise_Distance":
            distance_values,

        "Attention_Entropy_Delta":
            entropy_values,

        "CKA_Similarity":
            cka_values,

        "Subspace_Similarity":
            subspace_values,

        "Cosine_Mean":
            float(np.mean(cosine_values)),

        "Distance_Mean":
            float(np.mean(distance_values)),

        "Entropy_Mean":
            float(np.mean(entropy_values)),

        "CKA_Mean":
            float(np.mean(cka_values)),

        "Subspace_Mean":
            float(np.mean(subspace_values)),
    }


# ============================================================
# Main Detection Pipeline
# ============================================================

def process_code_pair(
    pair_path
):

    with open(
        pair_path,
        "r",
        encoding="utf-8"
    ) as f:

        pair_data = json.load(f)

    pair_id = pair_data.get(
        "pair_id",
        pair_path.stem
    )

    print(f"Processing: {pair_id}")

    before_layers = pair_data[
        "before_activations"
    ]

    after_layers = pair_data[
        "after_activations"
    ]

    results = {}

    num_layers = min(
        len(before_layers),
        len(after_layers)
    )

    for layer_index in range(num_layers):

        before_matrix = np.asarray(
            before_layers[layer_index],
            dtype=np.float32
        )

        after_matrix = np.asarray(
            after_layers[layer_index],
            dtype=np.float32
        )

        layer_results = {}

        layer_results[
            "cosine_similarity"
        ] = detect_cosine_change(
            before_matrix,
            after_matrix
        )

        layer_results[
            "activation_distance"
        ] = detect_activation_distance(
            before_matrix,
            after_matrix
        )

        layer_results[
            "attention_entropy"
        ] = detect_attention_entropy_change(
            before_matrix,
            after_matrix
        )

        layer_results[
            "cka_similarity"
        ] = detect_cka_change(
            before_matrix,
            after_matrix
        )

        layer_results[
            "subspace_analysis"
        ] = detect_subspace_change(
            before_matrix,
            after_matrix
        )

        results[
            f"layer_{layer_index}"
        ] = layer_results

    summary_row = build_summary_row(
        pair_id,
        num_layers,
        results
    )

    return {

        "pair_id":
            pair_id,

        "num_layers":
            num_layers,

        "layers":
            results,

    }, summary_row


# ============================================================
# Run Full Dataset
# ============================================================

def main():

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Input directory not found: "
            f"{INPUT_DIR}"
        )

    pair_files = sorted(
        INPUT_DIR.glob(
            "*_token_activations.json"
        )
    )

    print(
        f"Found pair files: "
        f"{len(pair_files)}"
    )

    all_results = []
    summary_rows = []

    processed = 0
    failed = 0

    for pair_path in pair_files:

        try:

            result, summary_row = (
                process_code_pair(
                    pair_path
                )
            )

            all_results.append(
                result
            )

            summary_rows.append(
                summary_row
            )

            processed += 1

        except Exception as e:

            print(
                f"ERROR: "
                f"{pair_path.name}"
            )

            print(e)

            failed += 1

    save_json(
        all_results,
        OUTPUT_PATH
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False
    )

    print("\nDone.")

    print(
        f"Processed: {processed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Saved detailed results to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary results to: "
        f"{SUMMARY_OUTPUT_PATH}"
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()