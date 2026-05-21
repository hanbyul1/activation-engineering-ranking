import re
import json
import csv
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

PAIR_DIR = Path("CC_regenerated_validated/pairs")
OUTPUT_DIR = Path("CC_regenerated_validated")

# ============================================================
# Utility Functions
# ============================================================

def strip_comments_and_strings(code: str) -> str:

    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.S)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)

    return code


def count_pattern(code: str, pattern: str) -> int:
    return len(re.findall(pattern, code))

# ============================================================
# CC Estimation
# ============================================================

def estimate_cc(code: str) -> int:

    code = strip_comments_and_strings(code)

    score = 1

    decision_patterns = [
        r'\bif\b',
        r'\bfor\b',
        r'\bwhile\b',
        r'\bguard\b',
        r'\bcatch\b',
    ]

    for pattern in decision_patterns:
        score += count_pattern(code, pattern)

    score += count_pattern(code, r'\bcase\b')
    score += count_pattern(code, r'&&|\|\|')
    score += count_pattern(code, r'\?')

    return score

# ============================================================
# Structural Indicators
# ============================================================

def count_branches(code: str) -> int:

    code = strip_comments_and_strings(code)

    patterns = [
        r'\bif\b',
        r'\belse\s+if\b',
        r'\bfor\b',
        r'\bwhile\b',
        r'\bswitch\b',
        r'\bcase\b',
        r'\bguard\b',
    ]

    total = 0

    for pattern in patterns:
        total += count_pattern(code, pattern)

    return total


def estimate_max_nesting_depth(code: str) -> int:

    code = strip_comments_and_strings(code)

    lines = code.splitlines()

    max_depth = 0
    current_depth = 0

    for line in lines:

        stripped = line.strip()

        opening = stripped.count("{")
        closing = stripped.count("}")

        current_depth += opening

        max_depth = max(max_depth, current_depth)

        current_depth -= closing

    return max_depth


def count_loc(code: str) -> int:

    return len([
        line for line in code.splitlines()
        if line.strip()
    ])


def count_decision_density(code: str) -> float:

    loc = count_loc(code)

    if loc == 0:
        return 0.0

    return round(estimate_cc(code) / loc, 4)

# ============================================================
# Decomposition Indicators
# ============================================================

def count_helper_functions(code: str) -> int:

    matches = re.findall(
        r'\bfunc\s+[A-Za-z_][A-Za-z0-9_]*',
        code
    )

    return max(0, len(matches) - 1)


def count_function_calls(code: str) -> int:

    code = strip_comments_and_strings(code)

    matches = re.findall(
        r'[A-Za-z_][A-Za-z0-9_]*\(',
        code
    )

    excluded = {
        "if",
        "for",
        "while",
        "switch",
        "return",
        "guard",
    }

    count = 0

    for match in matches:

        name = match[:-1]

        if name not in excluded:
            count += 1

    return count

# ============================================================
# Pair Discovery
# ============================================================

before_files = sorted(PAIR_DIR.glob("*_before.swift"))

pair_records = []

for before_path in before_files:

    pair_id = before_path.stem.replace("_before", "")

    after_path = PAIR_DIR / f"{pair_id}_after.swift"

    if not after_path.exists():
        continue

    before_code = before_path.read_text(encoding="utf-8")
    after_code = after_path.read_text(encoding="utf-8")

    # --------------------------------------------------------
    # Before Metrics
    # --------------------------------------------------------

    cc_before = estimate_cc(before_code)

    branch_before = count_branches(before_code)

    nesting_before = estimate_max_nesting_depth(before_code)

    loc_before = count_loc(before_code)

    density_before = count_decision_density(before_code)

    helper_before = count_helper_functions(before_code)

    calls_before = count_function_calls(before_code)

    # --------------------------------------------------------
    # After Metrics
    # --------------------------------------------------------

    cc_after = estimate_cc(after_code)

    branch_after = count_branches(after_code)

    nesting_after = estimate_max_nesting_depth(after_code)

    loc_after = count_loc(after_code)

    density_after = count_decision_density(after_code)

    helper_after = count_helper_functions(after_code)

    calls_after = count_function_calls(after_code)

    # --------------------------------------------------------
    # Derived Indicators
    # --------------------------------------------------------

    delta_cc = cc_before - cc_after

    relative_reduction = round(
        delta_cc / cc_before,
        4,
    ) if cc_before > 0 else 0.0

    delta_branch = branch_before - branch_after

    delta_nesting = nesting_before - nesting_after

    decomposition_ratio = round(
        (helper_after + calls_after + 1) /
        (helper_before + calls_before + 1),
        4
    )

    # --------------------------------------------------------
    # Structural Transformation Labels
    # --------------------------------------------------------

    if (
        cc_before > cc_after
        and decomposition_ratio > 1.0
    ):

        complexity_label = "MONOLITHIC_TO_DECOMPOSED"

    elif cc_before > cc_after:

        complexity_label = "HIGH_TO_LOW_CC"

    else:

        complexity_label = "UNCHANGED"

    # --------------------------------------------------------
    # Store Record
    # --------------------------------------------------------

    pair_records.append({

        "pair_id": pair_id,

        # ----------------------------------------------------
        # CC Metrics
        # ----------------------------------------------------

        "CC_before": cc_before,
        "CC_after": cc_after,
        "Delta_CC": delta_cc,
        "Relative_CC_Reduction": relative_reduction,

        # ----------------------------------------------------
        # Branch Metrics
        # ----------------------------------------------------

        "Branch_Count_Before": branch_before,
        "Branch_Count_After": branch_after,
        "Delta_Branch_Count": delta_branch,

        # ----------------------------------------------------
        # Nesting Metrics
        # ----------------------------------------------------

        "Nesting_Depth_Before": nesting_before,
        "Nesting_Depth_After": nesting_after,
        "Delta_Nesting_Depth": delta_nesting,

        # ----------------------------------------------------
        # LOC Metrics
        # ----------------------------------------------------

        "LOC_Before": loc_before,
        "LOC_After": loc_after,

        # ----------------------------------------------------
        # Density Metrics
        # ----------------------------------------------------

        "Decision_Density_Before": density_before,
        "Decision_Density_After": density_after,

        # ----------------------------------------------------
        # Decomposition Metrics
        # ----------------------------------------------------

        "Helper_Functions_Before": helper_before,
        "Helper_Functions_After": helper_after,

        "Function_Calls_Before": calls_before,
        "Function_Calls_After": calls_after,

        "Structural_Decomposition_Ratio": decomposition_ratio,

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        "Complexity_Label": complexity_label,
    })

# ============================================================
# Complexity Ranking
# ============================================================

sorted_by_cc = sorted(
    pair_records,
    key=lambda x: x["CC_before"],
    reverse=True
)

for rank, record in enumerate(sorted_by_cc, start=1):
    record["Complexity_Rank"] = rank

# ============================================================
# Save CSV
# ============================================================

csv_path = OUTPUT_DIR / "cc_indicators.csv"

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(pair_records[0].keys()),
    )

    writer.writeheader()
    writer.writerows(pair_records)

# ============================================================
# Save JSONL
# ============================================================

jsonl_path = OUTPUT_DIR / "cc_indicators.jsonl"

with jsonl_path.open(
    "w",
    encoding="utf-8",
) as file:

    for record in pair_records:
        file.write(json.dumps(record) + "\n")

# ============================================================
# Summary
# ============================================================

avg_before = round(
    sum(r["CC_before"] for r in pair_records) / len(pair_records),
    2,
)

avg_after = round(
    sum(r["CC_after"] for r in pair_records) / len(pair_records),
    2,
)

avg_delta = round(
    sum(r["Delta_CC"] for r in pair_records) / len(pair_records),
    2,
)

avg_decomposition = round(
    sum(r["Structural_Decomposition_Ratio"] for r in pair_records)
    / len(pair_records),
    2,
)

print("\nCC Indicator Derivation Complete")
print("--------------------------------")
print("Pairs Processed:", len(pair_records))
print("Average CC Before:", avg_before)
print("Average CC After:", avg_after)
print("Average CC Reduction:", avg_delta)
print("Average Structural Decomposition Ratio:", avg_decomposition)
print("CSV Saved:", csv_path)
print("JSONL Saved:", jsonl_path)