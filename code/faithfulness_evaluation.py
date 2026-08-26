"""
Faithfulness Evaluation for Argumentative Summaries

This script evaluates the faithfulness of generated summaries by detecting
hallucinated spans using an NLI-based approach.

Inputs:
    - llama_ats_summary_span.json (example)
      Contains predicted summaries ("predict") and reference summaries ("label")
      represented as lists of spans.

    - ats_dataset_test.json
      Contains the ground truth structured summaries (argument units such as
      claims, premises, rebuttals), which serve as the evidence source.

Method:
    1. Extract structured argument units from each ground truth structured summary.
    2. For each span (predict or label):
        - Use NLI to compare the span against all structured units.
        - A span is considered:
            • faithful if it is entailed by at least one unit
            • hallucinated if it contradicts or is unsupported by all units
    3. Compute hallucination statistics:
        - total_spans
        - hallucinated_spans
        - hallucination_rate = hallucinated_spans / total_spans

Output:
    A final faithfulness score defined as:

        faithfulness = label_hallucination_rate / predict_hallucination_rate

    This measures how the model’s hallucination level compares to the
    ground truth reference.

Interpretation:
    - Score ≈ 1.0 → model matches reference faithfulness
    - Score < 1.0 → model produces more hallucinations (worse)
    - Score > 1.0 → model produces fewer hallucinations (better)

"""


import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODEL_NAME = "roberta-large-mnli"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

STRUCTURED_JSON = "ats_dataset_test.json"
SUMMARY_JSON = "llama_ats_summary_span.json"

# ------------------------------------------------------------
# Load NLI model
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

@torch.no_grad()
def nli(premise: str, hypothesis: str) -> str:
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)
    logits = model(**inputs).logits
    return LABELS[int(torch.argmax(logits, dim=-1))]

# ------------------------------------------------------------
# Structured summary utilities
# ------------------------------------------------------------
def extract_structured_units(structured_summary_text: str):
    units = []
    for line in structured_summary_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        content = re.sub(
            r"^(major claim|claim|premise|counterargument|rebuttal)\s*[:-]?\s*",
            "",
            line,
            flags=re.IGNORECASE
        )
        if content.strip():
            units.append(content.strip())
    return units


def normalize_spans(x, field_name="unknown"):
    """
    Normalize predict/label into a list of span strings.

    Handles:
    - list[str]
    - JSON-encoded list stored as a string
    - plain string (treated as a single span)
    """
    # Case 1: already a list
    if isinstance(x, list):
        return x

    # Case 2: string that may be JSON list
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Case 3: plain string → single span
        return [s]

    raise ValueError(f"{field_name} must be list or string, got {type(x)}")

# ------------------------------------------------------------
# Span-based mapping baseline
# ------------------------------------------------------------
def map_spans_to_units(spans, structured_units):
    """
    Each span is faithful iff it is entailed by AT LEAST ONE
    structured argument unit.
    """
    hallucinated = 0
    total = len(spans)

    for span in spans:
        supported = False
        for g in structured_units:
            if nli(g, span) == "ENTAILMENT":
                supported = True
                break
        if not supported:
            hallucinated += 1

    return total, hallucinated

# ------------------------------------------------------------
# Dataset-level evaluation
# ------------------------------------------------------------
def evaluate_dataset():
    summaries = json.load(open(SUMMARY_JSON, encoding="utf-8"))
    structured = json.load(open(STRUCTURED_JSON, encoding="utf-8"))

    assert len(summaries) == len(structured), "Dataset size mismatch"

    stats = {
        "predict": {"total_units": 0, "hallucinated": 0},
        "label": {"total_units": 0, "hallucinated": 0}
    }

    for i in tqdm(range(len(summaries))):
        struct_units = extract_structured_units(
            structured[i]["structured_summary"]
        )

        for field in ["predict", "label"]:
            spans = normalize_spans(summaries[i][field], field_name=field)

            total, hallucinated = map_spans_to_units(spans, struct_units)

            stats[field]["total_units"] += total
            stats[field]["hallucinated"] += hallucinated

    results = {}
    for field in ["predict", "label"]:
        total = stats[field]["total_units"]
        hall = stats[field]["hallucinated"]
        results[field] = {
            "total_spans": total,
            "hallucinated_spans": hall,
            "hallucination_rate": hall / total if total > 0 else 0.0
        }

    return results

def compute_faithfulness_score(results):
    pred_rate = results["predict"]["hallucination_rate"]
    label_rate = results["label"]["hallucination_rate"]

    if pred_rate == 0:
        return 0.0

    return label_rate / pred_rate

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Device:", DEVICE)

    print("\n=== Span-based faithfulness baseline ===")
    results = evaluate_dataset()
    faithfulness_score = compute_faithfulness_score(results)
    print(f'faithfulness score: {faithfulness_score}')


