import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances

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

ACTIVATION_DIR = (
    BASE_DIR
    / "activation_analysis_outputs"
)

ACTIVATION_METRICS_FILE = (
    ACTIVATION_DIR
    / "activation_metrics.json"
)

OUTPUT_JSON = (
    ACTIVATION_DIR
    / "subspace_pairwise_scores.json"
)

UPDATED_METRICS_FILE = (
    ACTIVATION_DIR
    / "activation_metrics_with_subspace.json"
)

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
).to(DEVICE)

model.eval()

print("Model loaded.")

# ============================================================
# Load Dataset
# ============================================================

with open(DATASET_PATH, "r", encoding="utf-8") as f:

    dataset = json.load(f)

print("Pairs loaded:", len(dataset))

# ============================================================
# Load Activation Metrics
# ============================================================

with open(
    ACTIVATION_METRICS_FILE,
    "r",
    encoding="utf-8",
) as f:

    activation_metrics = json.load(f)

print(
    "Activation metric records loaded:",
    len(activation_metrics),
)

# ============================================================
# Helper Functions
# ============================================================

def sanitize_matrix(x):

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    x = x.astype(np.float64)

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    return x / (norms + 1e-12)

# ------------------------------------------------------------

def upper_triangle_values(matrix):

    idx = np.triu_indices_from(
        matrix,
        k=1,
    )

    return matrix[idx]

# ------------------------------------------------------------

def extract_hidden_states(code):

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
        )

    hidden_states = []

    for layer_hidden in outputs.hidden_states:

        token_matrix = (
            layer_hidden
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        token_matrix = sanitize_matrix(
            token_matrix
        )

        hidden_states.append(
            token_matrix
        )

    return hidden_states

# ------------------------------------------------------------

def compute_token_subspace_similarity(
    before_tokens,
    after_tokens,
):

    min_tokens = min(
        before_tokens.shape[0],
        after_tokens.shape[0],
    )

    if min_tokens < 3:
        return 0.0

    before_tokens = before_tokens[:min_tokens]
    after_tokens = after_tokens[:min_tokens]

    before_subspace = cosine_distances(
        before_tokens
    )

    after_subspace = cosine_distances(
        after_tokens
    )

    before_values = upper_triangle_values(
        before_subspace
    )

    after_values = upper_triangle_values(
        after_subspace
    )

    if len(before_values) < 2:
        return 0.0

    rho, _ = spearmanr(
        before_values,
        after_values,
    )

    if np.isnan(rho):
        return 0.0

    return float(rho)

# ------------------------------------------------------------

def compute_pair_subspace_similarity(
    before_code,
    after_code,
):

    before_layers = extract_hidden_states(
        before_code
    )

    after_layers = extract_hidden_states(
        after_code
    )

    num_layers = min(
        len(before_layers),
        len(after_layers),
    )

    subspace_scores = []

    for layer_idx in range(num_layers):

        score = compute_token_subspace_similarity(
            before_layers[layer_idx],
            after_layers[layer_idx],
        )

        subspace_scores.append(
            float(score)
        )

    return subspace_scores

# ============================================================
# Compute Pair-specific Subspace-oriented
# Representational Analysis
# ============================================================

subspace_records = []

for pair in tqdm(dataset):

    pair_id = pair["pair_id"]

    before_code = pair["before_code"]
    after_code = pair["after_code"]

    subspace_scores = (
        compute_pair_subspace_similarity(
            before_code,
            after_code,
        )
    )

    subspace_records.append({

        "pair_id":
            pair_id,

        "Subspace_Similarity":
            subspace_scores,
    })

# ============================================================
# Save Pair-specific Scores
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        subspace_records,
        f,
        indent=2,
    )

# ============================================================
# Merge into Activation Metrics
# ============================================================

subspace_lookup = {

    str(record["pair_id"]):
        record["Subspace_Similarity"]

    for record in subspace_records
}

for record in activation_metrics:

    pair_id = str(
        record.get("pair_id")
    )

    if pair_id in subspace_lookup:

        record["Subspace_Similarity"] = (
            subspace_lookup[pair_id]
        )

    else:

        record["Subspace_Similarity"] = []

with open(
    UPDATED_METRICS_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        activation_metrics,
        f,
        indent=2,
    )

# ============================================================
# Summary
# ============================================================

all_scores = [

    score

    for record in subspace_records

    for score in record[
        "Subspace_Similarity"
    ]
]

print("\n================================================")

print(
    "Pair-specific Token-level "
    "Subspace-oriented Representational "
    "Analysis Complete"
)

print("================================================")

print(
    "Pairs analyzed:",
    len(subspace_records),
)

if all_scores:

    print(
        "Mean subspace similarity:",
        round(
            float(np.mean(all_scores)),
            4,
        ),
    )

    print(
        "Min subspace similarity:",
        round(
            float(np.min(all_scores)),
            4,
        ),
    )

    print(
        "Max subspace similarity:",
        round(
            float(np.max(all_scores)),
            4,
        ),
    )

print("\nSaved files:")

print("-", OUTPUT_JSON)
print("-", UPDATED_METRICS_FILE)