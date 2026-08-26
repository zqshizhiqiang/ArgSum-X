"""
Structure-Aware Evaluation for Argumentative Text Summarization (ATS)

This module implements the evaluation pipeline used for assessing
abstractive summaries in the Argumentative Text Summarization (ATS) task.
It evaluates how well generated summaries preserve the underlying
argument structure defined by ground-truth structured annotations.

==================================================
INPUT FILES
==================================================

The evaluation uses two JSON files:

--------------------------------------------------
1. Prediction + Reference Summaries
--------------------------------------------------


This file contains, for each example:
    - "predict": model-generated summary
    - "label": ground-truth human summary

Format (list of examples):
[
    {
        "index": 0,
        "predict": "<generated summary>",
        "label": "<reference summary>"
    },
    ...
]

Example:
predict:
"Modern architecture in the city center should not be restricted..."

label:
"Modern buildings should not be restricted by traditionalist views..."

--------------------------------------------------
2. Structured Argument Annotations
--------------------------------------------------


This file contains the ground-truth structured argument representation
for each example.

Each entry includes:
    - "structured_summary": structured argument graph in textual form

Format:
[
    {
        "index": 0,
        "structured_summary": "Major claim:\n...\nClaim 1:\n..."
    },
    ...
]

The structured summary encodes:
    - Major claim
    - Claims
    - Premises
    - Counterarguments
    - Rebuttals

==================================================
EVALUATION PIPELINE
==================================================

1. Sentence Segmentation

Both "predict" and "label" summaries are split into sentences.

--------------------------------------------------

2. Semantic Mapping (NLI-based)

Each sentence is mapped to argument units (ADUs) from the structured summary
using a Natural Language Inference (NLI) model.

This produces a mapping:
    sentence → set of argument units

--------------------------------------------------

3. Structure Metrics (Normalized)

For each metric, scores are computed separately for:

    - predicted summary ("predict")
    - reference summary ("label")

Final scores are normalized as:

    normalized_score = predict_score / label_score

A score of:
    - 1.0  → matches reference performance
    - <1.0 → worse than reference
    - >1.0 → exceeds reference

--------------------------------------------------

4. Reported Structure Metrics


- MajorClaimCoverage
  Measures whether the major claim in the structured summary is expressed
  in the summary.

- LocalConsistency
  Measures validity of transitions between adjacent sentences based on
  reachable argument relations.

- GlobalStructureF1
  F1 score comparing global precedence relations (argument ordering)
  between summary and structured argument graph.

- CounterargumentCoverage
  Measures how many counterarguments in the structured summary are realized
  in the summary.

--------------------------------------------------

5. Argument Flow Agreement

Separately evaluates alignment between:

    - predicted summary ("predict")
    - reference summary ("label")

Based on sentence-level precedence relations.

Outputs:
    global_abs_flow_p   (precision)
    global_abs_flow_r   (recall)
    global_abs_flow_f1  (F1 score)

==================================================
OUTPUT FORMAT
==================================================

Structure-aware metrics (normalized):

    {
        "MajorClaimCoverage": float,
        "LocalConsistency": float,
        "GlobalStructureF1": float,
        "CounterargumentCoverage": float
    }

Argument flow agreement:

    === Argument Flow Agreement (predict vs label) ===
    global_abs_flow_p: float
    global_abs_flow_r: float
    global_abs_flow_f1: float

==================================================
NOTES
==================================================

- All structure metrics are normalized relative to the reference summary.

- The structured summary serves as a canonical argument graph.

- The NLI model is used as a semantic alignment function between
  sentences and argument units.


==================================================
"""



import json
import re
import torch
from collections import defaultdict, deque
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


########################################
# Configuration
########################################

MODEL_NAME = "roberta-large-mnli"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

MAX_HOPS = 3          # reachability depth

########################################
# Load NLI Model
########################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

@torch.no_grad()
def entail(premise, hypothesis):
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)
    logits = model(**inputs).logits
    return LABELS[int(torch.argmax(logits, dim=-1))]

########################################
# Utilities
########################################

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sentence_split(text):
    text = text.replace("\n", " ")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


########################################
# Structured Summary Parsing
########################################

def parse_structured_summary(text):
    """
    Returns:
      units: dict {uid: {text, role, parent}}
      edges: adjacency list for reachability
      major_id
      claim_ids
    """
    units = {}
    edges = defaultdict(list)

    uid = 0
    major_id = None
    current_claim = None
    claim_ids = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("Major claim"):
            continue

        if major_id is None:
            units[uid] = {"text": line, "role": "major", "parent": None}
            major_id = uid
            uid += 1
            continue

        if line.startswith("Claim"):
            units[uid] = {
                "text": line.split(":", 1)[1].strip(),
                "role": "claim",
                "parent": major_id
            }
            edges[major_id].append(uid)
            current_claim = uid
            claim_ids.add(uid)
            uid += 1
            continue

        if line.startswith("Premise:"):
            units[uid] = {
                "text": line.replace("Premise:", "").strip(),
                "role": "premise",
                "parent": current_claim
            }
            edges[current_claim].append(uid)
            uid += 1
            continue

        if line.startswith("Counterargument:") and "None" not in line:
            units[uid] = {
                "text": line.replace("Counterargument:", "").strip(),
                "role": "counterargument",
                "parent": current_claim
            }
            edges[current_claim].append(uid)
            uid += 1
            continue

        if line.startswith("Rebuttal:") and "None" not in line:
            units[uid] = {
                "text": line.replace("Rebuttal:", "").strip(),
                "role": "rebuttal",
                "parent": current_claim
            }
            edges[uid].append(current_claim)
            uid += 1

    return units, edges, major_id, claim_ids


########################################
# Reachability Graph
########################################

def compute_reachable(edges, max_hops):
    reachable = defaultdict(set)

    for start in edges:
        queue = deque([(start, 0)])
        visited = set([start])
        while queue:
            node, dist = queue.popleft()
            if dist >= max_hops:
                continue
            for nxt in edges.get(node, []):
                if nxt not in visited:
                    visited.add(nxt)
                    reachable[start].add(nxt)
                    queue.append((nxt, dist + 1))
    return reachable

########################################
# Mapping Construction
########################################

def build_mapping(summary, units):
    sentences = sentence_split(summary)
    mappings = []

    for s in sentences:
        mapped = set()
        for uid, u in units.items():
            if entail(s, u["text"]) == "ENTAILMENT":
                mapped.add(uid)
        mappings.append(mapped)

    return mappings

########################################
# Structure Evaluation
########################################

def evaluate_counterarguments(mappings, units, edges):
    """
    Evaluates counterargument handling.
    """
    ca_ids = {uid for uid, u in units.items() if u["role"] == "counterargument"}

    result = {
        "CounterargumentRealized": 0,
        "TotalCounterarguments": len(ca_ids)
    }

    if not ca_ids:
        return result  # no counterarguments in gold

    # Find realized units
    realized_units = set().union(*mappings)

    for ca in ca_ids:
        if ca not in realized_units:
            continue

        result["CounterargumentRealized"] += 1


    return result


def reachable_to_relations(reachable):
    """
    reachable: dict[u] -> set of v

    Returns:
        A set of (u, v) pairs allowed by structured summary.
    """
    relations = set()
    for u, vs in reachable.items():
        for v in vs:
            relations.add((u, v))
    return relations


def evaluate_structure(mappings, units, edges, reachable, major_id, claim_ids):
    result = {}

    # ---- Preconditions ----
    result["MajorClaimRealized"] = any(major_id in s for s in mappings)
    # result["AnyClaimRealized"] = any(len(s & claim_ids) > 0 for s in mappings)

    # ---- Transition Validity ----
    reachable_ok = 0
    total = max(0, len(mappings) - 1)

    for i in range(len(mappings) - 1):
        U = mappings[i]
        V = mappings[i + 1]

        reach = False

        for u in U:
            for v in V:
                if v == u or v in reachable.get(u, []):
                    reach = True

        if reach:
            reachable_ok += 1

    result["ReachableTransitionValidity"] = reachable_ok / total if total else 1.0

    #  Global Structural Metric  ----
    pred_precedence = extract_precedence_relations(mappings)
    gold_precedence = reachable_to_relations(reachable)

    struct_scores = compare_precedence_sets(
        pred_precedence,
        gold_precedence
    )

    result["GlobalStructurePrecision"] = struct_scores["p"]
    result["GlobalStructureRecall"] = struct_scores["r"]
    result["GlobalStructureF1"] = struct_scores["f1"]


    # ---- Counterargument ----
    ca_result = evaluate_counterarguments(mappings, units, edges)

    result.update({
        "CounterargumentRealizationRate": (
            ca_result["CounterargumentRealized"] /
            ca_result["TotalCounterarguments"]
            if ca_result["TotalCounterarguments"] else 1.0
        )
    })


    return result


def extract_precedence_relations(mappings):
    """
    mappings: List[Set[adu_id]]
      One set per sentence, in order.

    Returns:
      A set of (u, v) meaning u precedes v
    """
    precedences = set()

    for i, cur_set in enumerate(mappings):
        for j in range(i + 1, len(mappings)):
            next_set = mappings[j]
            for u in cur_set:
                for v in next_set:
                    if u != v:
                        precedences.add((u, v))

    return precedences


def compare_precedence_sets(pred_pre, ref_pre):
    """
    Computes Precision / Recall / F1 for precedence relations.
    """
    if not pred_pre and not ref_pre:
        return {"p": 1.0, "r": 1.0, "f1": 1.0}

    if not pred_pre or not ref_pre:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}

    correct = pred_pre & ref_pre

    precision = len(correct) / len(pred_pre)
    recall = len(correct) / len(ref_pre)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"p": precision, "r": recall, "f1": f1}

########################################
# Dataset-Level Evaluation
########################################

def evaluate_dataset(dataset_path, summary_path, field):
    dataset = load_json(dataset_path)
    summaries = load_json(summary_path)

    agg = defaultdict(float)
    n = len(dataset)

    for ex, sm in tqdm(zip(dataset, summaries), total=n):
        units, edges, major_id, claim_ids = parse_structured_summary(
            ex["structured_summary"]
        )
        reachable = compute_reachable(edges, MAX_HOPS)
        mappings = build_mapping(sm[field], units)
        r = evaluate_structure(
            mappings, units, edges, reachable, major_id, claim_ids
        )

        for k, v in r.items():
            agg[k] += v

    return {k: v / n for k, v in agg.items()}

def evaluate_argument_flow_two_files(
    summary_json_path,
    structured_json_path
):
    """
    Compare argument flow between:
    - predict (generated summary)
    - label (ground-truth summary)

    Uses structured summaries (same order) to define gold ADUs.
    """
    summaries = load_json(summary_json_path)
    structured = load_json(structured_json_path)

    assert len(summaries) == len(structured), \
        "Summary file and structured file must have the same length"

    agg = defaultdict(float)
    n = 0

    # assume both are lists OR both are dicts with same order
    if isinstance(summaries, dict):
        summaries = list(summaries.values())
    if isinstance(structured, dict):
        structured = list(structured.values())

    for ex_sum, ex_struct in tqdm(zip(summaries, structured), total=len(summaries)):
        if "predict" not in ex_sum or "label" not in ex_sum:
            continue
        if "structured_summary" not in ex_struct:
            continue

        generated_summary = ex_sum["predict"]
        reference_summary = ex_sum["label"]
        structured_summary = ex_struct["structured_summary"]

        # 1. Parse gold ADUs from structured summary
        gold_units, _, _, _ = parse_structured_summary(structured_summary)

        # 2. Map both summaries to gold ADUs
        gen_mappings = build_mapping(generated_summary, gold_units)
        ref_mappings = build_mapping(reference_summary, gold_units)

        # 3. Extract argument-flow precedence relations
        gen_pre = extract_precedence_relations(gen_mappings)
        ref_pre = extract_precedence_relations(ref_mappings)

        # 4. Compare flows
        scores = compare_precedence_sets(gen_pre, ref_pre)

        agg["global_abs_flow_p"] += scores["p"]
        agg["global_abs_flow_r"] += scores["r"]
        agg["global_abs_flow_f1"] += scores["f1"]
        n += 1

    return {k: v / max(n, 1) for k, v in agg.items()}

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def normalize_results(pred, gt):
    final = {}

    final["MajorClaimCoverage"] = safe_div(
        pred["MajorClaimRealized"], gt["MajorClaimRealized"]
    )

    final["LocalConsistency"] = safe_div(
        pred["ReachableTransitionValidity"], gt["ReachableTransitionValidity"]
    )

    final["GlobalStructureF1"] = safe_div(
        pred["GlobalStructureF1"], gt["GlobalStructureF1"]
    )

    final["CounterargumentCoverage"] = safe_div(
        pred["CounterargumentRealizationRate"],
        gt["CounterargumentRealizationRate"]
    )

    return final

########################################
# Main
########################################

if __name__ == "__main__":
    print("Device:", DEVICE)

    print("\n=== IMPROVED STRUCTURE EVALUATION: MODEL ===")
    predict_results = evaluate_dataset(
        "ats_dataset_test.json",
        "model_outputs/ats/llama3.2-1b/ats_summary.json",
        "predict"
        )
    ground_truth_results = evaluate_dataset(
        "ats_dataset_test.json",
        'model_outputs/ats/llama3.2-1b/ats_summary.json',
        "label"
    )

    final_results = normalize_results(predict_results, ground_truth_results)

    print(final_results)

    summary_json = "model_outputs/ats/llama3.2-1b/ats_summary.json"
    structured_json = "ats_dataset_test.json"   # structured_summary

    flow_results = evaluate_argument_flow_two_files(
        summary_json,
        structured_json
    )

    print("\n=== Argument Flow Agreement (predict vs label) ===")
    for k, v in flow_results.items():
        print(f"{k}: {v:.4f}")



