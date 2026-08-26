# Semantic Evaluation
#
# Metrics:
#   - ROUGE-1 / ROUGE-2 / ROUGE-L
#   - BERTScore (Precision, Recall, F1)
#   - SBERT cosine similarity
#
# Reference-based evaluation:
#   Generated summary ("predict")
#   Human reference summary ("label")
# ============================================================

import json
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------------------------------------------
# ROUGE
# ------------------------------------------------------------

def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = {"r1": [], "r2": [], "rL": []}

    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        scores["r1"].append(s["rouge1"].fmeasure)
        scores["r2"].append(s["rouge2"].fmeasure)
        scores["rL"].append(s["rougeL"].fmeasure)

    return {
        "rouge1": np.mean(scores["r1"]),
        "rouge2": np.mean(scores["r2"]),
        "rougeL": np.mean(scores["rL"]),
    }

# ------------------------------------------------------------
# BERTScore
# ------------------------------------------------------------

def compute_bertscore(preds, refs):
    P, R, F1 = bertscore(
        preds,
        refs,
        lang="en",
        model_type="roberta-large",
        rescale_with_baseline=True
    )

    return {
        "bertscore_P": P.mean().item(),
        "bertscore_R": R.mean().item(),
        "bertscore_F1": F1.mean().item(),
    }

# ------------------------------------------------------------
# SBERT cosine similarity
# ------------------------------------------------------------

def compute_sbert(preds, refs):
    model = SentenceTransformer("all-mpnet-base-v2")

    emb_preds = model.encode(preds, convert_to_tensor=True)
    emb_refs  = model.encode(refs,  convert_to_tensor=True)

    cosine_scores = util.cos_sim(emb_preds, emb_refs).diagonal()
    return float(cosine_scores.mean())

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def evaluate_stage(json_path):
    data = load_json(json_path)

    if isinstance(data, dict):
        data = list(data.values())

    preds = [ex["predict"] for ex in data]
    refs  = [ex["label"]   for ex in data]

    print("Computing ROUGE...")
    rouge = compute_rouge(preds, refs)

    print("Computing BERTScore...")
    bert  = compute_bertscore(preds, refs)

    print("Computing SBERT similarity...")
    sbert = compute_sbert(preds, refs)

    print("\n=== Stage 4: Semantic & Content Metrics ===")
    for k, v in rouge.items():
        print(f"{k}: {v:.4f}")

    for k, v in bert.items():
        print(f"{k}: {v:.4f}")

    print(f"sbert_cosine: {sbert:.4f}")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    # JSON file must contain: predict, label
    evaluate_stage("ats_summary.json")
