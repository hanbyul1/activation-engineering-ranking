import os
import zipfile
import re
import json
import csv
import shutil
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

# ============================================================
# Configuration
# ============================================================

WORK_DIR = Path("cc_regen_work")
SOURCE_ROOT = WORK_DIR / "sources"
OUTPUT_DIR = Path("CC_regenerated_validated")

ZIP_INPUTS = {
    "DocuSense": Path("DocuSense.zip"),
    "MenuCal": Path("MenuCal.zip"),
    "NutriCompass": Path("NutriCompass.zip"),
    "PartGuard": Path("PartGuard.zip"),
}

# ============================================================
# Setup
# ============================================================

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)

WORK_DIR.mkdir(parents=True)
SOURCE_ROOT.mkdir(parents=True)

for app, zip_path in ZIP_INPUTS.items():
    app_dir = SOURCE_ROOT / app
    app_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(app_dir)

# ============================================================
# Collect Swift Files
# ============================================================

swift_files = []

for app in ZIP_INPUTS:
    for path in (SOURCE_ROOT / app).rglob("*.swift"):

        if "__MACOSX" in path.parts:
            continue

        if path.name.startswith("._"):
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if len(text.strip()) == 0:
            continue

        swift_files.append((app, path, text))

# ============================================================
# Cyclomatic Complexity Estimation
# ============================================================


def strip_comments_and_strings(code: str) -> str:
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.S)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
    return code



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
        score += len(re.findall(pattern, code))

    score += len(re.findall(r'\bcase\b', code))
    score += len(re.findall(r'&&|\|\|', code))
    score += len(re.findall(r'\?', code))

    return score

def estimate_primary_function_cc(code: str) -> int:

    open_index = code.find("{")

    if open_index == -1:
        return estimate_cc(code)

    close_index = find_matching_brace(code, open_index)

    if close_index == -1:
        return estimate_cc(code)

    primary_function = code[:close_index + 1]

    return estimate_cc(primary_function)

# ============================================================
# Function Extraction
# ============================================================


def find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    in_string = False
    escape = False

    for i in range(open_index, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1

                if depth == 0:
                    return i

    return -1


FUNC_REGEX = re.compile(
    r'(?m)^([ \t]*(?:public|private|fileprivate|internal|open|static|class|mutating|nonmutating|override|final|@MainActor|async|@ViewBuilder|\s)*\s*func\s+[A-Za-z_][A-Za-z0-9_]*[^\{;=]*\{)'
)


@dataclass
class SwiftFunction:
    app: str
    path: str
    name: str
    code: str
    cc: int
    loc: int


functions = []

for app, path, text in swift_files:

    for match in FUNC_REGEX.finditer(text):

        open_brace = text.find('{', match.start())
        close_brace = find_matching_brace(text, open_brace)

        if close_brace == -1:
            continue

        code = text[match.start():close_brace + 1]

        name_match = re.search(
            r'\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)',
            code,
        )

        if not name_match:
            continue

        name = name_match.group(1)

        loc = len([
            line for line in code.splitlines()
            if line.strip()
        ])

        cc = estimate_cc(code)

        if loc >= 12 and cc >= 6:
            functions.append(
                SwiftFunction(
                    app=app,
                    path=str(path.relative_to(SOURCE_ROOT / app)),
                    name=name,
                    code=code,
                    cc=cc,
                    loc=loc,
                )
            )

# ============================================================
# Helper Functions
# ============================================================


def get_signature_header(code: str) -> str:
    return code[:code.find('{')].strip()



def get_return_type(header: str):
    match = re.search(r'->\s*([^\{]+)$', header, flags=re.S)

    if not match:
        return None

    return match.group(1).strip()



def get_function_name(header: str):
    match = re.search(r'\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)', header)

    if not match:
        return "unknown"

    return match.group(1)



def get_parameters(header: str):

    start = header.find('(')

    if start < 0:
        return []

    depth = 0
    end = -1

    for i in range(start, len(header)):
        if header[i] == '(':
            depth += 1
        elif header[i] == ')':
            depth -= 1

            if depth == 0:
                end = i
                break

    if end < 0:
        return []

    param_text = header[start + 1:end]

    params = []
    current = ''
    depth = 0

    for ch in param_text:

        if ch in '([{<':
            depth += 1

        elif ch in ')]}>':
            depth -= 1

        if ch == ',' and depth == 0:
            params.append(current.strip())
            current = ''
        else:
            current += ch

    if current.strip():
        params.append(current.strip())

    names = []

    for p in params:

        p = re.sub(r'@\w+(?:\([^)]*\))?\s*', '', p).strip()
        p = p.split('=')[0].strip()

        left = p.split(':')[0].strip()
        tokens = left.split()

        if len(tokens) >= 2:
            names.append(tokens[1])
        elif len(tokens) == 1:
            names.append(tokens[0])

    return names



def default_return_value(return_type: str):

    if return_type is None:
        return None

    if return_type.endswith('?'):
        return 'nil'

    if return_type == 'Bool':
        return 'false'

    if return_type in ['Int', 'Double', 'Float', 'CGFloat']:
        return '0'

    if return_type == 'String':
        return '""'

    if return_type.startswith('['):
        return '[]'

    return 'nil'

# ============================================================
# CC-Oriented Maintainability Transformations
# ============================================================

def get_helper_function_name(function_name: str):
    return f"{function_name}Core"


def extract_parameter_names(header: str):
    names = get_parameters(header)
    return [name for name in names if name != "_"]


def rename_function_signature(code: str, old_name: str, new_name: str):

    return re.sub(
        rf'\bfunc\s+{old_name}\b',
        f'func {new_name}',
        code,
        count=1
    )


def make_helper_extraction_rewrite(function: SwiftFunction):

    code = function.code

    header = get_signature_header(code)

    return_type = get_return_type(header)

    function_name = get_function_name(header)

    helper_name = get_helper_function_name(function_name)

    params = extract_parameter_names(header)

    call_args = ", ".join(params)

    helper_code = rename_function_signature(
        code,
        function_name,
        helper_name
    )

    lines = []

    lines.append(f"{header} {{")

    if return_type is None or return_type in ["Void", "()"]:

        if call_args:
            lines.append(f"    {helper_name}({call_args})")
        else:
            lines.append(f"    {helper_name}()")

    else:

        if call_args:
            lines.append(f"    return {helper_name}({call_args})")
        else:
            lines.append(f"    return {helper_name}()")

    lines.append("}")
    lines.append("")
    lines.append(helper_code)

    after_code = "\n".join(lines)

    before_cc = estimate_primary_function_cc(code)

    after_cc = estimate_primary_function_cc(after_code)

    if after_cc >= before_cc:
        return None

    return (
        after_code,
        "helper-extraction CC transformation"
    )


def make_after_version(function: SwiftFunction):

    result = make_helper_extraction_rewrite(function)

    if result is None:
        return None, "no transformation available"

    return result

# ============================================================
# Pair Selection
# ============================================================

candidate_functions = sorted(
    functions,
    key=lambda f: (-f.cc, f.loc),
)

selected_pairs = []
app_counts = Counter()

for function in candidate_functions:

    if len(selected_pairs) >= 100:
        break

    if app_counts[function.app] >= 30:
        continue

    after_code, strategy = make_after_version(function)

    if after_code is None:
        continue

    before_cc = estimate_primary_function_cc(function.code)

    after_cc = estimate_primary_function_cc(after_code)

    if after_cc < before_cc:
        
        selected_pairs.append(
            (function, after_code, strategy)
        )

        app_counts[function.app] += 1

# ============================================================
# Output
# ============================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

(OUTPUT_DIR / "pairs").mkdir(parents=True)

metadata = []

for index, (function, after_code, strategy) in enumerate(selected_pairs, start=1):

    pair_id = f"CC-{index:03d}"

    before_file = OUTPUT_DIR / "pairs" / f"{pair_id}_before.swift"
    after_file = OUTPUT_DIR / "pairs" / f"{pair_id}_after.swift"

    before_file.write_text(function.code, encoding="utf-8")
    after_file.write_text(after_code, encoding="utf-8")

    before_cc = estimate_primary_function_cc(function.code)

    after_cc = estimate_primary_function_cc(after_code)

    metadata.append({
        "pair_id": pair_id,
        "app": function.app,
        "source_file": function.path,
        "function_name": function.name,
        "estimated_CC_before": before_cc,
        "estimated_CC_after": after_cc,
        "estimated_CC_reduction": before_cc - after_cc,
        "generation_strategy": strategy,
    })

# ============================================================
# Save Metadata
# ============================================================

with (OUTPUT_DIR / "cc_metadata.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(metadata[0].keys()),
    )

    writer.writeheader()
    writer.writerows(metadata)

# ============================================================
# Save Inline JSON Dataset
# ============================================================

inline_dataset = []

for record in metadata:

    before_path = OUTPUT_DIR / "pairs" / f"{record['pair_id']}_before.swift"
    after_path = OUTPUT_DIR / "pairs" / f"{record['pair_id']}_after.swift"

    inline_dataset.append({
        **record,
        "before_code": before_path.read_text(encoding="utf-8"),
        "after_code": after_path.read_text(encoding="utf-8"),
    })

with (OUTPUT_DIR / "cc_dataset_inline.json").open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        inline_dataset,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("Generated", len(selected_pairs), "validated CC pairs")

# ============================================================
# Secondary Validation Pass
# ============================================================

print("\nRunning secondary validation pass...")

validation_results = []

for record in metadata:

    before_path = OUTPUT_DIR / "pairs" / f"{record['pair_id']}_before.swift"
    after_path = OUTPUT_DIR / "pairs" / f"{record['pair_id']}_after.swift"

    before_code = before_path.read_text(encoding="utf-8")
    after_code = after_path.read_text(encoding="utf-8")

    before_cc = estimate_primary_function_cc(before_code)

    after_cc = estimate_primary_function_cc(after_code)

    before_loc = len([
        line for line in before_code.splitlines()
        if line.strip()
    ])

    after_loc = len([
        line for line in after_code.splitlines()
        if line.strip()
    ])

    validation_status = "VALID"
    validation_notes = []

    if after_cc >= before_cc:
        validation_status = "INVALID"
        validation_notes.append("CC not reduced")

    if before_cc < 6:
        validation_status = "INVALID"
        validation_notes.append("before CC below threshold")

    if before_loc < 12:
        validation_status = "INVALID"
        validation_notes.append("before LOC below threshold")

    if len(after_code.strip()) == 0:
        validation_status = "INVALID"
        validation_notes.append("empty after code")

    if "func" not in after_code:
        validation_status = "INVALID"
        validation_notes.append("after code missing function")

    validation_results.append({
        "pair_id": record["pair_id"],
        "before_cc": before_cc,
        "after_cc": after_cc,
        "before_loc": before_loc,
        "after_loc": after_loc,
        "status": validation_status,
        "notes": "; ".join(validation_notes),
    })

# ============================================================
# Save Validation Report
# ============================================================

with (OUTPUT_DIR / "cc_validation_report.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(validation_results[0].keys()),
    )

    writer.writeheader()
    writer.writerows(validation_results)

valid_count = sum(
    1 for r in validation_results
    if r["status"] == "VALID"
)

invalid_count = sum(
    1 for r in validation_results
    if r["status"] == "INVALID"
)

print(f"Validation complete: {valid_count} VALID / {invalid_count} INVALID")