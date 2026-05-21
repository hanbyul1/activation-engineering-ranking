import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import entropy

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "bigcode/starcoderbase-1b"

BASE_DIR = Path(__file__).parent

DATASET_PATH = (
    BASE_DIR
    / "CC_regenerated_validated"
    / "cc_dataset_inline.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "activation_analysis_outputs"
)

OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_TOKENS = 512

# ============================================================
# Load Model
# ============================================================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
).to(DEVICE)

model.eval()

print("Model loaded.")

# ============================================================
# Load Dataset
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8",
) as f:

    dataset = json.load(f)

print("Pairs loaded:", len(dataset))

# ============================================================
# Helper Functions
# ============================================================

def mean_pool(
    hidden_state,
    attention_mask,
):

    mask = attention_mask.unsqueeze(-1)

    pooled = (
        hidden_state * mask
    ).sum(dim=1)

    pooled = pooled / (
        mask.sum(dim=1) + 1e-12
    )

    return pooled

# ------------------------------------------------------------

def sanitize_vector(x):

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    x = x.astype(np.float64)

    norm = np.linalg.norm(x)

    if norm > 0:
        x = x / norm

    return x

# ------------------------------------------------------------

def normalize_vector(vec):

    vec = np.nan_to_num(
        vec,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    vec = vec.astype(np.float64)

    return vec / (
        np.linalg.norm(vec) + 1e-12
    )

# ------------------------------------------------------------

def extract_model_outputs(code):

    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
        )

    return outputs, inputs

# ------------------------------------------------------------

def get_layerwise_embeddings(
    outputs,
    inputs,
):

    embeddings = []

    for layer_hidden in outputs.hidden_states:

        pooled = mean_pool(
            layer_hidden,
            inputs["attention_mask"],
        )

        vec = (
            pooled
            .squeeze(0)
            .cpu()
            .numpy()
        )

        vec = sanitize_vector(vec)

        embeddings.append(vec)

    return embeddings

# ------------------------------------------------------------

def compute_attention_entropy(attentions):

    if attentions is None:
        return []

    entropies = []

    for attn in attentions:

        if attn is None:

            entropies.append(0.0)
            continue

        try:

            attn_np = (
                attn
                .squeeze(0)
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            layer_entropy = []

            for head_matrix in attn_np:

                probs = (
                    head_matrix.mean(axis=0)
                )

                probs = probs / (
                    probs.sum() + 1e-12
                )

                probs = np.nan_to_num(
                    probs,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                layer_entropy.append(
                    entropy(probs + 1e-12)
                )

            entropies.append(
                float(np.mean(layer_entropy))
            )

        except Exception:

            entropies.append(0.0)

    return entropies

# ------------------------------------------------------------

def linear_CKA(X, Y):

    X = sanitize_vector(
        X.flatten()
    ).reshape(-1, 1)

    Y = sanitize_vector(
        Y.flatten()
    ).reshape(-1, 1)

    X = X - X.mean(
        axis=0,
        keepdims=True,
    )

    Y = Y - Y.mean(
        axis=0,
        keepdims=True,
    )

    X = X / (
        np.linalg.norm(X) + 1e-12
    )

    Y = Y / (
        np.linalg.norm(Y) + 1e-12
    )

    hsic = np.sum(
        (X.T @ Y) ** 2
    )

    var1 = np.sum(
        (X.T @ X) ** 2
    )

    var2 = np.sum(
        (Y.T @ Y) ** 2
    )

    return float(
        hsic
        /
        (
            np.sqrt(var1 * var2)
            + 1e-12
        )
    )

# ============================================================
# Storage
# ============================================================

records = []

# ============================================================
# Main Extraction Loop
# ============================================================

for pair in tqdm(dataset):

    pair_id = pair["pair_id"]

    before_code = pair["before_code"]
    after_code = pair["after_code"]

    # --------------------------------------------------------
    # Forward Pass
    # --------------------------------------------------------

    before_outputs, before_inputs = (
        extract_model_outputs(
            before_code
        )
    )

    after_outputs, after_inputs = (
        extract_model_outputs(
            after_code
        )
    )

    # --------------------------------------------------------
    # Hidden-State Embeddings
    # --------------------------------------------------------

    before_embeddings = (
        get_layerwise_embeddings(
            before_outputs,
            before_inputs,
        )
    )

    after_embeddings = (
        get_layerwise_embeddings(
            after_outputs,
            after_inputs,
        )
    )

    # ========================================================
    # Technique 1:
    # Cosine Similarity
    # ========================================================

    cosine_scores = []

    for b, a in zip(
        before_embeddings,
        after_embeddings,
    ):

        score = cosine_similarity(
            b.reshape(1, -1),
            a.reshape(1, -1),
        )[0][0]

        cosine_scores.append(
            float(score)
        )

    # ========================================================
    # Technique 2:
    # Layer-wise Activation Distance
    # ========================================================

    layer_distances = []

    for b, a in zip(
        before_embeddings,
        after_embeddings,
    ):

        dist = np.linalg.norm(
            b - a
        )

        layer_distances.append(
            float(dist)
        )

    # ========================================================
    # Technique 3:
    # Attention Entropy Analysis
    # ========================================================

    before_entropy = (
        compute_attention_entropy(
            before_outputs.attentions
        )
    )

    after_entropy = (
        compute_attention_entropy(
            after_outputs.attentions
        )
    )

    entropy_delta = [

        float(b - a)

        for b, a in zip(
            before_entropy,
            after_entropy,
        )
    ]

    # ========================================================
    # Technique 4:
    # Representational Subspace Analysis
    # ========================================================

    subspace_similarity = []
    subspace_distance = []

    for b, a in zip(
        before_embeddings,
        after_embeddings,
    ):

        b_norm = normalize_vector(b)

        a_norm = normalize_vector(a)

        sim = float(
            np.dot(b_norm, a_norm)
        )

        dist = float(
            1.0 - sim
        )

        subspace_similarity.append(sim)

        subspace_distance.append(dist)

    # ========================================================
    # Technique 5:
    # CKA Similarity
    # ========================================================

    cka_scores = []

    for b, a in zip(
        before_embeddings,
        after_embeddings,
    ):

        score = linear_CKA(
            b.reshape(-1, 1),
            a.reshape(-1, 1),
        )

        cka_scores.append(
            float(score)
        )

    # ========================================================
    # Save Per-Pair Record
    # ========================================================

    records.append({

        "pair_id":
            pair_id,

        "Cosine_Similarity":
            cosine_scores,

        "Layerwise_Distance":
            layer_distances,

        "Attention_Entropy_Delta":
            entropy_delta,

        "Subspace_Similarity":
            subspace_similarity,

        "Subspace_Distance":
            subspace_distance,

        "CKA_Similarity":
            cka_scores,
    })

# ============================================================
# Save Activation Metrics
# ============================================================

json_path = (
    OUTPUT_DIR
    / "activation_metrics.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        records,
        f,
        indent=2,
    )

# ============================================================
# Save Summary
# ============================================================

summary = {

    "model_name":
        MODEL_NAME,

    "total_pairs":
        len(records),

    "techniques_covered": [

        "Cosine similarity",

        "Layer-wise activation distance",

        "Attention entropy analysis",

        "Representational subspace analysis",

        "CKA similarity",
    ],

    "num_hidden_state_layers":

        len(
            records[0][
                "Cosine_Similarity"
            ]
        )

        if records else None,

    "num_attention_layers":

        len(
            records[0][
                "Attention_Entropy_Delta"
            ]
        )

        if records else None,
}

summary_path = (
    OUTPUT_DIR
    / "summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )

# ============================================================
# Finished
# ============================================================

print("\nActivation extraction complete.")

print(
    "Results saved to:",
    OUTPUT_DIR,
)

print(
    "Activation metrics:",
    json_path,
)

print(
    "Summary:",
    summary_path,
)