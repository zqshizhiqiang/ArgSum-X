# ArgSum-X: Argumentative Summarization Dataset

ArgSum-X is a dataset for **argument summarization** that jointly supports:

- **Argumentative Text Summarization (ATS)**
- **Key Point Analysis (KPA)**

It provides **multi-stage annotations** capturing the transformation from raw argumentative text to structured and abstractive representations.

For full details on dataset construction, annotation process, and evaluation framework, please refer to the accompanying paper.

---

## Repository Structure
```
.
├── code/
│   ├── structure_evaluation.py
│   ├── faithfulness_evaluation.py
│   └── semantic_evaluation.py
│
├── data/
│   ├── ats_kpa_dataset_train_0-399.json
│   ├── ats_kpa_dataset_train_400-799.json
│   ├── ats_kpa_dataset_dev.json
│   ├── ats_kpa_dataset_test.json
│   └── README.md
```
---

## Dataset Overview

ArgSum-X contains **1,000 argumentative essays** with multi-layer annotations.

### Splits

- **Train (800 essays)**
  - `train_0-399`: LLM-annotated data without human correction
  - `train_400-799`: human-corrected annotations

- **Dev (100 essays)**
  - fully corrected annotations across all stages

- **Test (100 essays)**
  - double-corrected annotations with adjudication (highest quality)

All dataset files follow the same format. For detailed field descriptions, see `data/README.md`.

---

## Annotation Pipeline

The dataset is constructed using a **multi-stage annotation framework**:

1. Argumentative discourse unit (ADU) segmentation  
2. Argument component classification (ACC)  
3. Structured summary generation  
4. Abstractive summary generation  
5. Key point generation  
6. Key point matching  

Annotations are produced using a **hybrid LLM–human workflow** with multiple quality tiers.

---

## Evaluation Code

The `code/` folder provides scripts for evaluating generated summaries across three complementary dimensions:

### Structure Evaluation
- `structure_evaluation.py`
- Measures preservation of **argumentative structure**
- Includes:
  - argument unit coverage  
  - local argumentative flow consistency  
  - global structure consistency  

### Faithfulness Evaluation
- `faithfulness_evaluation.py`
- Evaluates whether summary content is **supported by the underlying argument structure**
- Detects hallucinated content using span-level analysis  

### Semantic Similarity
- `semantic_evaluation.py`
- Computes standard summarization metrics such as:
  - ROUGE  
  - BERTScore  

---

## Intended Use

ArgSum-X can support research in:

- Argument mining (ADU segmentation and component classification)  
- Argumentative text summarization (ATS)  
- Key Point Analysis (KPA)  
- Structure-aware text generation  
- Faithfulness evaluation grounded in argument structure  
- Learning under heterogeneous annotation quality  

---


## Additional Details

Please refer to the paper for:

- Dataset statistics  
- Annotation methodology  
- Quality analysis  
- Evaluation framework  
- Experimental results  

---

## Citation

(To be added)

---

## License

(To be added)

