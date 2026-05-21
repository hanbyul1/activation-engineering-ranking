import json
import numpy as np

from pathlib import Path
from tqdm import tqdm

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "layerwise_embeddings.json"
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

print(
    "Pairs loaded:",
    len(dataset),
)

# ============================================================
# Helper Functions
# ============================================================

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
            .detach()
            .cpu()
            .numpy()
        )

        vec = sanitize_vector(vec)

        embeddings.append(vec)

    return embeddings

# ============================================================
# Storage
# ============================================================

embedding_records = []

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
    # Layer-wise Embeddings
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

    # --------------------------------------------------------
    # Save Record
    # --------------------------------------------------------

    embedding_records.append({

        "pair_id":
            pair_id,

        "app":
            pair.get("app"),

        "source_file":
            pair.get("source_file"),

        "function_name":
            pair.get("function_name"),

        "generation_strategy":
            pair.get("generation_strategy"),

        "num_layers":
            len(before_embeddings),

        "embedding_dimension":
            int(
                before_embeddings[0].shape[0]
            ),

        "before_embeddings": [

            emb.tolist()

            for emb in before_embeddings
        ],

        "after_embeddings": [

            emb.tolist()

            for emb in after_embeddings
        ],
    })

# ============================================================
# Save Embeddings
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        embedding_records,
        f,
        indent=2,
    )

# ============================================================
# Summary
# ============================================================

print("\n========================================")
print("Layer-wise Embedding Extraction Complete")
print("========================================")

print(
    "Total pairs processed:",
    len(embedding_records),
)

if len(embedding_records) > 0:

    print(
        "Layers per sample:",
        embedding_records[0]["num_layers"],
    )

    print(
        "Embedding dimension:",
        embedding_records[0]["embedding_dimension"],
    )

print(
    "Saved embeddings to:",
    OUTPUT_FILE,
)