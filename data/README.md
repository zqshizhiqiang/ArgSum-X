# ArgSum-X: A Structure-Aware Benchmark for Argument Summarization and Key Point Analysis

ArgSum-X is a dataset for **argument summarization** that jointly supports:

- **Argumentative Text Summarization (ATS)**
- **Key Point Analysis (KPA)**

It provides **multi-stage annotations** that capture the transformation from raw argumentative text to structured and abstractive summaries.

For full details (dataset construction, annotation process, statistics, and evaluation),
please refer to the accompanying paper.

---

## Overview

ArgSum-X is built on argumentative essays and includes:

- Argumentative discourse unit segmentation (ADUs)
- Argument component classification (ACC)
- Structured summaries
- Abstractive summaries
- Key point generation
- Argument–key point alignment


---

## Dataset Splits

The dataset contains **1,000 essays**:

- **Train (800 essays)**
  - `train_0-399`: LLM-annotated (no human correction)
  - `train_400-799`: LLM + human-corrected annotations

- **Dev (100 essays)**
  - Fully corrected annotations

- **Test (100 essays)**
  - Double-corrected with adjudication (highest quality)

All splits share the same format.

---

## Data Format

Each instance contains the following fields:

### 1. Basic Information
- `index`: sample ID  
- `topic`: essay prompt  
- `essay`: raw essay text  

---

### 2. ADU segmentation
- `essay_tagged`: essay annotated with ADU tags (`<AC0> ... </AC0>`)


### 3. Argument Component Classification

- `argument_components`: mapping from ADU IDs to labels

Labels:
- `major claim`
- `claim`
- `premise`
- `counterargument`
- `rebuttal`

These annotations provide the **semantic roles of argument units**.

---

### 4. Structured Summary

- `structured_summary`: hierarchical argument representation

Includes:
- Major claim
- Claims
- Premises
- Counterarguments
- Rebuttals

This acts as an **intermediate structured representation** between text and abstractive summaries.

---

### 5. Abstractive Summary

- `abstractive_summary`: natural-language summary of the essay

Generated based on the structured summary to preserve **argumentative meaning and organization**.

---

### 6. Key Point Analysis (KPA)

- `key_points`:
  - `pro_keypoints`
  - `con_keypoints`

Key points are generated from the topic and represent **reusable argumentative statements**.

---

### 7. Argument–Key Point Matching

- `argument_kps`: alignment between ADUs and key points

Each entry includes:
- argument text (ADU)
- stance (`pro` / `con`)
- matched key points

This links **discourse-level arguments to abstract key points**.

---

## Annotation Framework

ArgSum-X follows a **multi-stage annotation pipeline**:

1. ADU segmentation (discourse units)
2. Argument component classification
3. Structured summary generation
4. Abstractive summary generation
5. Key point generation
6. Key point matching

Annotations are produced using a **hybrid LLM–human workflow** with quality tiers.

---

## Annotation Quality Tiers

The dataset includes multiple levels of annotation quality:

| Tier | Description |
|------|------------|
| LLM-only | Fully automatic annotations (train_0-399) |
| Single-corrected | Human-corrected segmentation & labels |
| Fully corrected | All stages reviewed (dev set) |
| Adjudicated | Multiple annotators + adjudication (test set) |

This enables research on:
- robustness to annotation noise
- effect of annotation quality

---

## Intended Use

This dataset can be used for:

- Argument mining (ADU segmentation, ACC)
- Argument structure prediction
- Argumentative summarization (ATS)
- Key Point Analysis (KPA)
- Structure-aware generation
- Faithfulness and reasoning evaluation
- Multi-task training across structured representations

---


## Additional Details

Please refer to the paper for:

- Dataset statistics
- Annotation guidelines
- Quality analysis
- Inter-annotator agreement
- Evaluation framework (structure, faithfulness, semantic similarity)
- Experimental results

---

## Citation

(To be added)

## License
This dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.