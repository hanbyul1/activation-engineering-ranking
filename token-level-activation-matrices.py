import os
import gc
import json
import torch
import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "/Users/dae-kyookim/.cache/huggingface/hub/"
    "models--bigcode--starcoderbase-1b/"
    "snapshots/"
    "182f0165fdf8da9c9935901eec65c94337f01c11"
)

INPUT_JSON = (
    "CC_regenerated_validated/"
    "cc_dataset_inline.json"
)

OUTPUT_DIR = "activation_analysis_outputs/token_level_activations"

MAX_LENGTH = 128
CHECKPOINT_EVERY = 1

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.set_grad_enabled(False)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.to(DEVICE)
model.eval()

NUM_LAYERS = model.config.n_layer + 1

print("\nModel loaded.")
print("Layers including embedding layer:", NUM_LAYERS)
print("Max tokens:", MAX_LENGTH)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_token_level_activations(code_text):
    """
    Extract token-level activation matrices.

    Returns:
        List of [tokens x hidden_dim] matrices,
        one for each hidden-state layer.
    """

    inputs = tokenizer(
        code_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True
        )

    layer_activations = []

    for layer_tensor in outputs.hidden_states:

        layer_matrix = (
            layer_tensor[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        layer_activations.append(
            layer_matrix.tolist()
        )

    del outputs
    del inputs

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return layer_activations


def save_pair_activation(pair_id, before_activations, after_activations):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{pair_id}_token_activations.json"
    )

    result = {
        "pair_id": pair_id,
        "before_activations": before_activations,
        "after_activations": after_activations
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f)

    return output_path

# ============================================================
# LOAD CODE PAIRS
# ============================================================

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    code_pairs = json.load(f)

print("\nLoaded code pairs:", len(code_pairs))

# ============================================================
# EXTRACTION LOOP
# ============================================================

saved_files = []

for idx, pair in enumerate(code_pairs):

    pair_id = pair["pair_id"]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{pair_id}_token_activations.json"
    )

    if os.path.exists(output_path):
        print(f"\nSkipping existing file: {pair_id}")
        saved_files.append(output_path)
        continue

    before_code = pair["before_code"]
    after_code = pair["after_code"]

    print("\n===================================")
    print("Processing:", pair_id)
    print("Pair:", idx + 1, "/", len(code_pairs))
    print("===================================")

    before_activations = extract_token_level_activations(
        before_code
    )

    after_activations = extract_token_level_activations(
        after_code
    )

    saved_path = save_pair_activation(
        pair_id,
        before_activations,
        after_activations
    )

    saved_files.append(saved_path)

    print("Saved:", saved_path)

    del before_activations
    del after_activations

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_path = os.path.join(
    OUTPUT_DIR,
    "manifest.json"
)

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "num_pairs": len(saved_files),
            "max_tokens": MAX_LENGTH,
            "num_layers_including_embedding": NUM_LAYERS,
            "files": saved_files
        },
        f,
        indent=2
    )

# ============================================================
# SHAPE VERIFICATION
# ============================================================

if saved_files:
    with open(saved_files[0], "r", encoding="utf-8") as f:
        sample = json.load(f)

    sample_before = np.array(
        sample["before_activations"][0],
        dtype=np.float32
    )

    print("\nExample activation shape:")
    print(sample_before.shape)

print("\n===================================")
print("Extraction complete.")
print("Saved directory:")
print(OUTPUT_DIR)
print("Manifest:")
print(manifest_path)
print("===================================")