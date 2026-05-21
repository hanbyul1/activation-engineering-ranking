# Activation Engineering Ranking for Maintainability-Oriented Code Restructuring

This repository contains the implementation, datasets, activation outputs, and evaluation artifacts for the study:

> **Ranking Activation-Space Analysis Techniques for Maintainability-Oriented Code Restructuring**

The project investigates whether software maintainability characteristics become systematically reflected within transformer activation spaces and evaluates activation-space analysis techniques for identifying maintainability-relevant restructuring behavior in code-oriented large language models (LLMs).

---

# Repository Structure

```text
activation-engineering-ranking/
│
├── CC_regenerated_validated/
│   ├── cc_dataset_inline.json
│   ├── cc_indicators.csv
│   ├── cc_indicators.jsonl
│   ├── cc_metadata.csv
│   ├── cc_validation_report.csv
│   └── pairs.zip
│
├── activation_analysis_outputs/
│   ├── bootstrap_ranking_stability/
│   ├── cc_maintainability_alignment/
│   ├── cc_maintainability_patterns/
│   ├── representational_effectiveness/
│   ├── technique_complementarity/
│   ├── activation_metrics.json
│   ├── activation_metrics_detected_changes.json
│   ├── activation_metrics_detected_changes_summary.csv
│   ├── activation_metrics_with_subspace.json
│   └── subspace_pairwise_scores.json
│
├── activation-extraction-applying-techniques.py
├── bootstrap-ranking-stability.py
├── cc-dataset-generation.py
├── cc-oriented-indicator-generation.py
├── collecting-embeddings.py
├── detect_cc_maintainability_patterns.py
├── detect_representational_changes.py
├── evaluate_cc_maintainability_alignment.py
├── pairwise-correlation-analysis.py
├── subspace-analysis.py
├── technique-ranking.py
└── token-level-activation-matrices.py
```

---

# Project Overview

The repository implements a full activation-space analysis pipeline for evaluating maintainability-sensitive restructuring behavior in transformer-based code LLMs.

The workflow includes:

1. Construction of CC-oriented restructuring datasets
2. Hidden-state activation extraction
3. Activation-space analysis
4. Representational restructuring detection
5. Maintainability-alignment evaluation
6. Technique ranking
7. Bootstrap-based stability validation
8. Token-level restructuring pattern discovery

The evaluation uses contrastive before/after restructuring pairs generated from real-world iOS applications.

---

# Main Components

## Dataset Generation

### `cc-dataset-generation.py`

Generates maintainability-oriented before/after restructuring pairs targeting cyclomatic complexity (CC) reduction.

Outputs include:

- Base implementations
- Restructured implementations
- Metadata
- Validation artifacts

Generated files are stored in:

```text
CC_regenerated_validated/
```

---

## Maintainability Indicator Extraction

### `cc-oriented-indicator-generation.py`

Extracts CC-oriented restructuring indicators including:

- Delta CC
- Relative CC Reduction
- Delta Branch Count
- Delta Nesting Depth
- Structural Decomposition Ratio
- Decision Density Change
- Complexity Labels
- Complexity Rankings

Outputs:

```text
cc_indicators.csv
cc_indicators.jsonl
```

---

## Hidden-State Activation Extraction

### `collecting-embeddings.py`

Extracts transformer hidden-state embeddings from the evaluated code-oriented LLM.

### `token-level-activation-matrices.py`

Constructs token-level hidden-state activation matrices across transformer layers.

### `activation-extraction-applying-techniques.py`

Applies the selected activation-space analysis techniques to extracted hidden representations.

---

# Activation-Space Analysis Techniques

The repository evaluates the following techniques:

- Cosine Similarity
- Layer-wise Activation Distance
- Attention Entropy Analysis
- Subspace-oriented Representational Analysis
- CKA Similarity Analysis

---

# Representational Analysis

## `detect_representational_changes.py`

Detects restructuring-sensitive representational differences between before/after activation states.

Outputs:

```text
activation_metrics_detected_changes.json
activation_metrics_detected_changes_summary.csv
```

---

## `subspace-analysis.py`

Performs PCA-based subspace-oriented representational analysis and computes latent subspace restructuring behavior.

Outputs:

```text
subspace_pairwise_scores.json
activation_metrics_with_subspace.json
```

---

# Technique Evaluation and Ranking

## `evaluate_cc_maintainability_alignment.py`

Evaluates maintainability alignment using:

- Pearson correlation
- Spearman correlation
- Pairwise discrimination accuracy
- Layer-wise stability
- Cross-sample variance

Outputs:

```text
cc_maintainability_alignment/
```

---

## `technique-ranking.py`

Ranks activation-space analysis techniques according to aggregate maintainability-alignment effectiveness.

Outputs:

```text
representational_effectiveness/
```

Generated summaries include:

```text
layerwise_effectiveness.csv
representational_effectiveness_summary.csv
representational_effectiveness_summary.json
```

---

# Pattern Discovery

## `detect_cc_maintainability_patterns.py`

Discovers restructuring-sensitive activation-space patterns from token-level activations.

Outputs:

```text
cc_maintainability_patterns/
```

Including:

```text
cc_maintainability_patterns.json
cc_maintainability_patterns_summary.csv
cc_maintainability_layerwise_summary.csv
cc_maintainability_technique_summary.csv
```

---

# Technique Complementarity Analysis

## `pairwise-correlation-analysis.py`

Evaluates complementarity and redundancy across activation-space analysis techniques.

Outputs:

```text
technique_complementarity/
```

Including:

```text
pairwise_pearson_correlations.csv
pairwise_spearman_correlations.csv
pairwise_layerwise_representations.csv
```

---

# Bootstrap-Based Stability Validation

## `bootstrap-ranking-stability.py`

Performs bootstrap resampling to evaluate ranking stability under dataset perturbation.

Outputs:

```text
bootstrap_ranking_stability/
```

Including:

```text
bootstrap_rank_distribution.csv
bootstrap_rank_summary.csv
bootstrap_rank_summary.json
```

---

# Activation Analysis Outputs

The `activation_analysis_outputs/` directory stores all generated activation-space analysis artifacts including:

- Representational effectiveness summaries
- Pattern detection outputs
- Maintainability-alignment evaluations
- Bootstrap stability evaluations
- Pairwise complementarity analysis
- Token-level restructuring metrics

---

# Dataset

The dataset consists of:

- 100 maintainability-oriented Swift before/after restructuring pairs
- Generated from four real-world iOS applications:
  - MenuCal
  - NutriCompass
  - DocuSense
  - PartGuard

The restructuring transformations primarily target:

- Cyclomatic complexity reduction
- Branch simplification
- Helper-function extraction
- Nesting reduction
- Structural decomposition

---

# Requirements

Typical dependencies include:

```text
Python 3.10+
PyTorch
Transformers
NumPy
Pandas
Scikit-learn
SciPy
Matplotlib
```

---

# Example Workflow

## 1. Generate dataset

```bash
python cc-dataset-generation.py
```

## 2. Generate restructuring indicators

```bash
python cc-oriented-indicator-generation.py
```

## 3. Extract hidden activations

```bash
python collecting-embeddings.py
```

## 4. Apply activation-space analysis techniques

```bash
python activation-extraction-applying-techniques.py
```

## 5. Detect restructuring changes

```bash
python detect_representational_changes.py
```

## 6. Evaluate maintainability alignment

```bash
python evaluate_cc_maintainability_alignment.py
```

## 7. Rank techniques

```bash
python technique-ranking.py
```

## 8. Validate ranking stability

```bash
python bootstrap-ranking-stability.py
```

## 9. Discover restructuring patterns

```bash
python detect_cc_maintainability_patterns.py
```

---

# Research Objective

The repository investigates whether:

- software maintainability characteristics become systematically encoded within transformer activation spaces, and
- activation-space analysis techniques can identify restructuring-sensitive representational behavior associated with software maintainability.

---

# License

This repository is intended for academic and research purposes.
