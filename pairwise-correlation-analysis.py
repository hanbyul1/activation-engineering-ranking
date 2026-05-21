import os
import json
import numpy as np
import pandas as pd

from pathlib import Path

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

TOKEN_DIR = (
    BASE_DIR
    / "activation_analysis_outputs"
    / "token_level_activations"
)

OUTPUT_DIR = (
    BASE_DIR
    / "activation_analysis_outputs"
    / "technique_complementarity"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

# ============================================================
# TECHNIQUE FUNCTIONS
# ============================================================

def cosine_score(before, after):
    """
    Mean cosine similarity across aligned tokens.
    """

    min_tokens = min(
        before.shape[0],
        after.shape[0]
    )

    if min_tokens < 2:
        return 0.0

    before = before[:min_tokens]
    after = after[:min_tokens]

    sims = []

    for i in range(min_tokens):

        sim = cosine_similarity(
            before[i].reshape(1, -1),
            after[i].reshape(1, -1)
        )[0][0]

        sims.append(sim)

    return float(np.mean(sims))

# ------------------------------------------------------------

def activation_distance(before, after):
    """
    Mean Euclidean distance across aligned tokens.
    """

    min_tokens = min(
        before.shape[0],
        after.shape[0]
    )

    if min_tokens < 2:
        return 0.0

    before = before[:min_tokens]
    after = after[:min_tokens]

    distances = np.linalg.norm(
        after - before,
        axis=1
    )

    return float(np.mean(distances))

# ------------------------------------------------------------

def attention_entropy(x):
    """
    Entropy over normalized activation magnitudes.
    """

    values = np.abs(x.flatten())

    if values.sum() == 0:
        return 0.0

    p = values / values.sum()

    p = np.clip(
        p,
        1e-12,
        1.0
    )

    return float(
        -np.sum(p * np.log(p))
    )

# ------------------------------------------------------------

def entropy_difference(before, after):

    eb = attention_entropy(before)
    ea = attention_entropy(after)

    return float(abs(ea - eb))

# ------------------------------------------------------------

def upper_triangle_values(matrix):

    idx = np.triu_indices_from(
        matrix,
        k=1
    )

    return matrix[idx]

# ------------------------------------------------------------

def subspace_similarity(before, after):
    """
    Subspace-oriented representational analysis.
    """

    min_tokens = min(
        before.shape[0],
        after.shape[0]
    )

    if min_tokens < 3:
        return 0.0

    before = before[:min_tokens]
    after = after[:min_tokens]

    before_rdm = cosine_distances(before)
    after_rdm = cosine_distances(after)

    before_vals = upper_triangle_values(
        before_rdm
    )

    after_vals = upper_triangle_values(
        after_rdm
    )

    rho, _ = spearmanr(
        before_vals,
        after_vals
    )

    if np.isnan(rho):
        return 0.0

    return float(rho)

# ------------------------------------------------------------

def center_gram(K):

    n = K.shape[0]

    H = np.eye(n) - np.ones((n, n)) / n

    return H @ K @ H

# ------------------------------------------------------------

def linear_cka(before, after):
    """
    Linear CKA similarity.
    """

    min_tokens = min(
        before.shape[0],
        after.shape[0]
    )

    if min_tokens < 3:
        return 0.0

    before = before[:min_tokens]
    after = after[:min_tokens]

    K = before @ before.T
    L = after @ after.T

    Kc = center_gram(K)
    Lc = center_gram(L)

    hsic = np.sum(Kc * Lc)

    norm_k = np.sqrt(
        np.sum(Kc * Kc)
    )

    norm_l = np.sqrt(
        np.sum(Lc * Lc)
    )

    denom = norm_k * norm_l

    if denom < 1e-12:
        return 0.0

    return float(hsic / denom)

# ============================================================
# LOAD TOKEN ACTIVATIONS
# ============================================================

files = sorted(
    TOKEN_DIR.glob("*_token_activations.json")
)

print(
    "\nToken activation files:",
    len(files)
)

# ============================================================
# GENERATE REPRESENTATIONS
# ============================================================

pairwise_results = []

for file_path in files:

    with open(file_path, "r") as f:

        record = json.load(f)

    pair_id = record["pair_id"]

    before_layers = record["before_activations"]
    after_layers = record["after_activations"]

    num_layers = min(
        len(before_layers),
        len(after_layers)
    )

    print(
        "\nProcessing:",
        pair_id
    )

    for layer_idx in range(num_layers):

        before = np.array(
            before_layers[layer_idx],
            dtype=np.float32
        )

        after = np.array(
            after_layers[layer_idx],
            dtype=np.float32
        )

        row = {

            "Pair_ID":
                pair_id,

            "Layer":
                layer_idx,

            "CosineSimilarity":
                cosine_score(
                    before,
                    after
                ),

            "LayerwiseActivationDistance":
                activation_distance(
                    before,
                    after
                ),

            "AttentionEntropy":
                entropy_difference(
                    before,
                    after
                ),

            "SubspaceSimilarity":
                subspace_similarity(
                    before,
                    after
                ),

            "CKASimilarity":
                linear_cka(
                    before,
                    after
                ),
        }

        pairwise_results.append(row)

# ============================================================
# SAVE REPRESENTATIONS
# ============================================================

pairwise_df = pd.DataFrame(
    pairwise_results
)

representation_csv = (
    OUTPUT_DIR
    / "pairwise_layerwise_representations.csv"
)

pairwise_df.to_csv(
    representation_csv,
    index=False
)

print("\nSaved:")
print(representation_csv)

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

techniques = [

    "CosineSimilarity",

    "LayerwiseActivationDistance",

    "AttentionEntropy",

    "SubspaceSimilarity",

    "CKASimilarity",
]

pearson_matrix = pd.DataFrame(
    index=techniques,
    columns=techniques
)

spearman_matrix = pd.DataFrame(
    index=techniques,
    columns=techniques
)

for t1 in techniques:

    for t2 in techniques:

        x = pairwise_df[t1].values
        y = pairwise_df[t2].values

        pr, _ = pearsonr(x, y)
        sr, _ = spearmanr(x, y)

        pearson_matrix.loc[t1, t2] = pr
        spearman_matrix.loc[t1, t2] = sr

# ============================================================
# SAVE MATRICES
# ============================================================

pearson_csv = (
    OUTPUT_DIR
    / "pairwise_pearson_correlations.csv"
)

spearman_csv = (
    OUTPUT_DIR
    / "pairwise_spearman_correlations.csv"
)

pearson_matrix.to_csv(
    pearson_csv
)

spearman_matrix.to_csv(
    spearman_csv
)

print("\nSaved:")
print(pearson_csv)
print(spearman_csv)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n=== Pearson Correlations ===")
print(pearson_matrix)

print("\n=== Spearman Correlations ===")
print(spearman_matrix)