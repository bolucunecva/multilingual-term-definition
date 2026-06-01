# MULDER: A Multilingual Dataset for Term-Definition Extraction in Scientific Literature

 [Paper](#) · [Code](https://github.com/bolucunecva/multilingual-term-definition) · [Dataset](https://zenodo.org/records/20475284)

> Necva Bölücü, Yunus Can Bilge, Zehra Yücel, Dilber Çetintaş  
> CSIRO · Hacettepe University · Necmettin Erbakan University · Malatya Turgut Ozal University

---

## Overview

Scientific literature is growing rapidly, making it increasingly difficult for researchers to identify conceptually relevant work. Existing scientific retrieval systems often rely on lexical overlap or embedding similarity. While useful, these approaches may retrieve papers that are broadly semantically similar rather than papers that explicitly share the same concepts, methods, or research tasks.

We introduce **MULDER** (**MUL**tilingual **D**efinition **E**xtraction for **R**etrieval), a multilingual dataset for term-definition extraction in scientific literature. MULDER focuses on English and Turkish machine learning abstracts and supports concept-aware scientific retrieval through explicit term-definition pairs and cross-lingual context links.

The dataset and benchmark are designed to study how term-definition structures can support:

- concept-aware scientific retrieval;
- multilingual terminology alignment;
- retrieval-augmented definition extraction;
- robustness analysis under irrelevant retrieval contexts;
- evaluation of open-source LLMs for scientific information extraction.

```text
Scientific abstract
        │
        ▼
Term-definition extraction
        │
        ├── Term span
        ├── Definition span
        └── Cross-lingual context link
        │
        ▼
Concept-aware scientific retrieval
```

---

## Motivation

Modern scientific search systems can identify semantically related papers, but semantic similarity alone is not always sufficient. In machine learning literature, terminology evolves quickly and technical expressions are often overloaded across subfields.

For example, the same abbreviation can refer to different concepts depending on context. This problem becomes harder in multilingual scientific settings, where equivalent concepts may appear under different surface forms across languages.

MULDER addresses this problem by representing scientific documents through explicit **term-definition pairs**, enabling more fine-grained and concept-specific retrieval.

---

## Dataset

MULDER contains English and Turkish machine learning abstracts with human-annotated term-definition pairs. It is accompanied by an ArXiv augmentation dataset designed for retrieval-oriented evaluation.

### MULDER

| Property | Value |
|---|---:|
| Domain | Machine Learning |
| Languages | English, Turkish |
| Abstracts | 344 |
| English abstracts | 172 |
| Turkish abstracts | 172 |
| Gold term-definition pairs | 348 |
| English annotations | 177 |
| Turkish annotations | 171 |
| Abstracts with at least one pair | ~48% |
| Inter-annotator agreement before adjudication | 0.985 |
| Inter-annotator agreement after adjudication | 0.99 |

### ArXiv Augmentation Dataset

| Property | Value |
|---|---:|
| Source | ArXiv ML abstracts |
| Final abstracts | 777 |
| Unique ML terms | 80 |
| Abstracts with at least one pair | 73.2% |
| Abstracts with more than one pair | 36.8% |
| Abstracts with no pair | 26.8% |

The ArXiv augmentation dataset was created using ML terms extracted from the English portion of MULDER. Candidate abstracts were retrieved from approximately 2.6 million ArXiv abstracts and then annotated for term-definition extraction.

---

## Example

![MULDER example](assets/mulder_example.png)

Each annotated instance contains:

```json
{
  "abstract_id": "example-001",
  "language": "en",
  "term": "Decision Tree",
  "definition": "a supervised learning algorithm used for classification and regression",
  "term_span": [0, 13],
  "definition_span": [20, 91],
  "aligned_context_id": "example-001-tr"
}
```

---

## Evaluation Settings

We define six complementary experiments covering monolingual extraction, multilingual contextual extraction, retrieval-augmented extraction, and robustness under irrelevant retrieval contexts.

| Experiment | Setting | Description |
|---|---|---|
| Exp1 | Monolingual joint extraction | Extract term-definition pairs from one abstract |
| Exp2 | Monolingual target-term extraction | Extract the definition for a given target term |
| Exp3 | Multilingual joint extraction | Extract pairs using the source abstract plus aligned second-language context |
| Exp4 | Multilingual target-term extraction | Extract a definition for a target term using multilingual context |
| Exp5 | Relevant retrieval-augmented extraction | Extract definitions from retrieved ArXiv abstracts containing the target term |
| Exp6 | Irrelevant retrieval robustness | Test whether models hallucinate definitions when retrieved contexts do not contain the target term |

---

## Models

We evaluate 13 open-source LLMs in zero-shot settings.

| Model Family | Models |
|---|---|
| AM-Thinking | AM-Thinking-v1 |
| Gemma | Gemma-3-1B, Gemma-3-4B, Gemma-3-12B, Gemma-3-27B, Gemma-4-E2B, Gemma-4-E4B, Gemma-4-31B |
| GPT-OSS | GPT-OSS-20B, GPT-OSS-120B |
| Olmo | Olmo-3-32B-Think |
| Qwen3 | Qwen3-14B, Qwen3-32B |

Experiments are run using **vLLM** on NVIDIA H100 GPUs with temperature 0, maximum input length 30k, and maximum new tokens 4096.

---

## Main Results

### Joint Term-Definition Extraction

Multilingual context improves joint term-definition extraction across nearly all models.

| Model | Exp1 Term F1 | Exp1 Def. F1 | Exp3 Term F1 | Exp3 Def. F1 |
|---|---:|---:|---:|---:|
| AM-Thinking-v1 | 0.4152 | 0.4159 | 0.7428 | 0.6556 |
| Gemma-3-1B | 0.3753 | 0.2093 | 0.3092 | 0.1016 |
| Gemma-3-4B | 0.4261 | 0.2653 | 0.4675 | 0.3758 |
| Gemma-3-12B | 0.4309 | 0.2662 | 0.4727 | 0.4432 |
| Gemma-3-27B | 0.4020 | 0.2120 | 0.4543 | 0.3453 |
| Gemma-4-E2B | 0.3945 | 0.1664 | 0.3915 | 0.2074 |
| Gemma-4-E4B | 0.4007 | 0.1328 | 0.3962 | 0.1514 |
| Gemma-4-31B | 0.2918 | 0.2872 | 0.7468 | 0.6625 |
| GPT-OSS-20B | 0.2927 | 0.2759 | 0.5564 | 0.6151 |
| GPT-OSS-120B | 0.3507 | 0.3858 | 0.6897 | 0.6482 |
| Olmo-3-32B-Think | 0.3596 | 0.3624 | 0.5472 | 0.5543 |
| Qwen3-14B | 0.2885 | 0.2658 | 0.6015 | 0.6024 |
| Qwen3-32B | 0.3971 | 0.3631 | 0.6722 | 0.6389 |

---

### Definition Extraction for a Target Term

Providing the target term substantially improves performance compared with joint extraction.

| Model | Exp2 EM | Exp2 Token F1 | Exp4 EM | Exp4 Token F1 |
|---|---:|---:|---:|---:|
| AM-Thinking-v1 | 0.8391 | 0.8832 | 0.8391 | 0.8761 |
| Gemma-3-1B | 0.0259 | 0.2336 | 0.0230 | 0.1601 |
| Gemma-3-4B | 0.2241 | 0.6780 | 0.0661 | 0.3737 |
| Gemma-3-12B | 0.1810 | 0.7700 | 0.1063 | 0.6901 |
| Gemma-3-27B | 0.6925 | 0.8246 | 0.5345 | 0.7348 |
| Gemma-4-E2B | 0.0833 | 0.6260 | 0.0632 | 0.6234 |
| Gemma-4-E4B | 0.4080 | 0.7495 | 0.2213 | 0.7150 |
| Gemma-4-31B | 0.7500 | 0.7994 | 0.6753 | 0.7265 |
| GPT-OSS-20B | 0.6523 | 0.7540 | 0.6236 | 0.7491 |
| GPT-OSS-120B | 0.7557 | 0.8026 | 0.7816 | 0.8376 |
| Olmo-3-32B-Think | 0.7586 | 0.7838 | 0.6810 | 0.7061 |
| Qwen3-14B | 0.5747 | 0.7405 | 0.5460 | 0.7385 |
| Qwen3-32B | 0.7414 | 0.8348 | 0.7155 | 0.8212 |

---

### Retrieval-Augmented Definition Extraction

Exp5 evaluates extraction from relevant retrieved contexts. Exp6 evaluates hallucination under irrelevant retrieved contexts.

| Model | Exp5 Token F1 | Exp6 HP ↓ | Exp6 HFPR ↓ |
|---|---:|---:|---:|
| AM-Thinking-v1 | 0.5001 | 0.0250 | 0.0050 |
| Gemma-3-1B | 0.0612 | 0.2875 | 0.0575 |
| Gemma-3-4B | 0.3280 | 0.2500 | 0.0500 |
| Gemma-3-12B | 0.4696 | 0.1500 | 0.0300 |
| Gemma-3-27B | 0.4676 | 0.1875 | 0.0375 |
| Gemma-4-E2B | 0.3974 | 0.1250 | 0.0250 |
| Gemma-4-E4B | 0.5711 | 0.0750 | 0.0150 |
| Gemma-4-31B | 0.5826 | 0.0125 | 0.0025 |
| GPT-OSS-20B | 0.5185 | 0.0750 | 0.0150 |
| GPT-OSS-120B | 0.5341 | 0.0125 | 0.0025 |
| Olmo-3-32B-Think | 0.2974 | 0.0125 | 0.0025 |
| Qwen3-14B | 0.4187 | 0.0125 | 0.0025 |
| Qwen3-32B | 0.4222 | 0.0500 | 0.0100 |

---

## Key Findings

- Jointly extracting term-definition pairs is challenging for current open-source LLMs.
- Providing the target term makes definition extraction substantially easier.
- Multilingual context consistently improves joint term-definition extraction.
- Retrieval-augmented extraction remains difficult because definitional content is sparse.
- Some LLMs over-generate plausible definitions in noisy contexts.
- Larger models generally benefit more from multilingual context and show stronger robustness under irrelevant retrieval settings.

---

## Repository Structure

```text
.
├── dataset/
│   ├── mulder_dataset.json
│   └── arxiv_dataset.json
│
│
├── experiments/
│   ├── run_experiments.py
│
├── requirements.txt
└── README.md
```

---

## Usage

### Installation

```bash
conda create -n mulder python=3.12
conda activate mulder
pip install -r requirements.txt
```

### Run Experiments

```bash
python run_experiments.py \
    --model                  $LOCAL_MODEL \
    --data                   ../dataset/all_experiments.json \
    --arxiv_data             ../dataset/arxiv_experiments.json \
    --outdir                 results/$LOCAL_MODEL \
    --exps                   1\ # 1 or 2 or 3 or 4 or 5 or 6
    --guideline              "../guideline.md" \
    --gpu_memory_utilization 0.95 \
    --max_model_len          10000
```

---

## Data Format

### Joint Term-Definition Extraction

```json
{
  "id": "mulder-en-001",
  "language": "en",
  "abstract": "...",
  "term_definition_pairs": [
    {
      "term": "Decision Tree",
      "definition": "a supervised learning algorithm used for classification and regression"
    }
  ]
}
```

### Multilingual Context

```json
{
  "id": "mulder-en-001",
  "language": "en",
  "abstract": "...",
  "aligned_language": "tr",
  "aligned_abstract": "...",
  "term_definition_pairs": []
}
```

### Retrieval-Augmented Context

```json
{
  "target_term": "attention mechanism",
  "retrieved_abstracts": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ],
  "gold_definitions": [
    "..."
  ]
}
```

---

## Metrics

| Metric | Used In | Description |
|---|---|---|
| Term F1 | Exp1, Exp3 | Token-level F1 for extracted terms |
| Definition F1 | Exp1, Exp3 | Token-level F1 for extracted definitions |
| Exact Match | Exp2, Exp4 | Exact match between predicted and gold definitions |
| Token F1 | Exp2, Exp4, Exp5 | Token-level overlap between predicted and gold definitions |
| HP | Exp6 | Hallucination probability under irrelevant context |
| HFPR | Exp6 | Hallucination false positive rate |

---
