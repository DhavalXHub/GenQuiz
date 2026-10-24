# GenQuiz Dataset Deep-Dive & Engineering Guide

> **Document Version:** 1.0.0
> **Target Artifact:** `genquiz_dataset.csv` and SFT Pipeline (`data/train_sft.jsonl`, `data/val_sft.jsonl`)
> **Project:** GenQuiz — Adaptive AI Quiz & Assessment Platform

---

## 1. Executive Summary & Purpose

The GenQuiz platform relies on a dual-purpose data foundation centered on `genquiz_dataset.csv`. In this architecture, data is utilized across three distinct operational layers:
1. **Supervised Fine-Tuning (SFT):** Transformed via `prepare_dataset.py` into 40 training and 10 validation ChatML conversation pairs for PEFT/LoRA adapter training of `Qwen2.5-1.5B-Instruct`.
2. **Few-Shot In-Context Exemplar Bank:** Queried at inference time by `dataset.py:get_few_shot_examples()` to provide few-shot demonstrations to Groq or local LLMs, illustrating the expected JSON schema, option structures, and Bloom explanations.
3. **Standalone Demo Mode Fallback Pool:** Queried by `dataset.py:filter_dataset()` when neither live API credentials nor local GPU weights are available, providing a graceful fallback quiz for offline testing.

---

## 2. Dataset Schema & Column-by-Column Specification

The core dataset `genquiz_dataset.csv` consists of exactly **50 records** and **16 structured columns** (with 14 core metadata and question attributes, plus ID and provenance).

| Column Name | Data Type | Nullable | Description & Domain Rules | Example |
| :--- | :--- | :--- | :--- | :--- |
| `question_id` | String | No | Unique identifier format `Q001`–`Q050`. Serves as primary key. | `Q001` |
| `subject` | Categorical | No | Broad discipline area (e.g., Computer Science, AI). | `Computer Science` |
| `topic` | Categorical | No | Core domain topic (e.g., Transformers, Machine Learning, Cybersecurity). | `Machine Learning` |
| `subtopic` | Categorical | No | Granular subject focus (e.g., Attention Mechanism, Loss Functions). | `Self-Attention` |
| `context` | Text | No | Grounding paragraph or factual reference text from which question is derived. | `Self-attention allows models to weigh the importance of different words...` |
| `question` | Text | No | The natural language question prompt presented to the student. | `What is the primary role of the query vector in self-attention?` |
| `question_type` | Categorical | No | Supported item type: `MCQ`, `Short Answer`, `True/False`, `Fill in the Blank`. | `MCQ` |
| `option_a` | Text | Conditional | Option text A (Required for MCQ and True/False; empty for Short Answer). | `Calculate attention weights` |
| `option_b` | Text | Conditional | Option text B (Required for MCQ and True/False; empty for Short Answer). | `Store memory representations` |
| `option_c` | Text | Conditional | Option text C (Required for 4-option MCQ; empty for True/False and Short Answer). | `Normalize activations` |
| `option_d` | Text | Conditional | Option text D (Required for 4-option MCQ; empty for True/False and Short Answer). | `Provide positional biases` |
| `correct_answer` | Text | No | For MCQ: Exact option letter (`A`, `B`, `C`, `D`). For Short Answer: Expected phrase. | `A` |
| `explanation` | Text | No | Pedagogical explanation justifying why the answer is correct and options wrong. | `The query vector is multiplied with key vectors to compute compatibility...` |
| `difficulty` | Categorical | No | Item difficulty rating: `Easy`, `Medium`, `Hard`. | `Medium` |
| `bloom_level` | Categorical | No | Cognitive skill tier from Bloom's Revised Taxonomy (`Remember`, `Understand`, etc.). | `Understand` |
| `provenance` | Text | No | Origin and verification metadata. | `AI-assisted educational content; requires verification` |

---

## 3. Dataset Distribution & Empirical Statistics

Based on the verified audit report (`reports/sft_dataset_validation_report.json`):

### 3.1 Distribution by Question Type (Total: 50)
- **Multiple Choice Questions (MCQ):** 25 (50.0%)
- **Short Answer:** 10 (20.0%)
- **True / False:** 8 (16.0%)
- **Fill in the Blank:** 7 (14.0%)

### 3.2 Distribution by Difficulty
- **Medium:** 23 (46.0%)
- **Easy:** 15 (30.0%)
- **Hard:** 12 (24.0%)

### 3.3 Distribution by Bloom's Taxonomy Level
- **Understand:** 15 (30.0%)
- **Remember:** 14 (28.0%)
- **Apply:** 9 (18.0%)
- **Analyze:** 7 (14.0%)
- **Evaluate:** 5 (10.0%)

### 3.4 Text Length Characteristics
- **Average Context Length:** 162.8 characters (~25–35 words per context snippet).
- **Average Question Length:** 98.4 characters (~15–20 words).
- **Average Explanation Length:** 145.2 characters (~22–30 words).

---

## 4. The SFT Preparation Pipeline (`prepare_dataset.py`)

The transformation of raw CSV tabular data into instruction-tuning datasets is managed deterministically by `prepare_dataset.py`.

### 4.1 Stratified Train/Val Split
To prevent evaluation bias and distribution skew, `prepare_dataset.py` implements multi-key stratification:
- **Split Ratio:** 80% Train (40 examples), 20% Validation (10 examples).
- **Stratification Keys:** Combined `(question_type, difficulty)` tuples to guarantee that minority question types (e.g., Fill in the Blank, True/False) and difficulty levels are proportionally represented in both splits.
- **Leakage Prevention:** Zero overlapping `question_id` primary keys exist between `train_sft.jsonl` and `val_sft.jsonl`. Verified by automated test `test_grounding_and_alignment.py`.

### 4.2 Formatting into ChatML / Qwen Conversation Format
Each row is compiled into a 3-turn OpenAI/ChatML conversation structure:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert assessment generator. Generate structured educational questions based on the provided text, topic, and difficulty. Return valid JSON only."
    },
    {
      "role": "user",
      "content": "Topic: Machine Learning\nDifficulty: Medium\nType: MCQ\nContext: Self-attention allows models to weigh the importance of different words...\nGenerate 1 question."
    },
    {
      "role": "assistant",
      "content": "{\n  \"question_id\": \"Q001\",\n  \"question\": \"What is the primary role of the query vector in self-attention?\",\n  \"question_type\": \"MCQ\",\n  \"options\": {\"A\": \"Calculate attention weights\", \"B\": \"Store memory representations\", \"C\": \"Normalize activations\", \"D\": \"Provide positional biases\"},\n  \"correct_answer\": \"A\",\n  \"explanation\": \"The query vector is multiplied with key vectors to compute compatibility...\",\n  \"difficulty\": \"Medium\",\n  \"bloom_level\": \"Understand\"\n}"
    }
  ]
}
```

---

## 5. Provenance, Quality Limitations & Ethical Assessment

### 5.1 Synthetic Origin & Provenance Statement
All 50 records in `genquiz_dataset.csv` carry the explicit provenance marker:
> `AI-assisted educational content; requires verification`

**Scientific Honesty Requirement:**
In academic defenses and project reports, this dataset must NOT be claimed as human-curated or gold-standard pedagogical data. It represents synthetic bootstrap data generated with AI assistance to seed the LoRA format alignment.

### 5.2 Strengths
1. **Schema Consistency:** Every record adheres strictly to the target JSON schema expected by the downstream Pydantic validator (`QuizSchema`).
2. **Context-Grounded Answers:** Answers and explanations are directly derivable from the accompanying `context` field, teaching the model context-conditioning.
3. **Zero ID Duplication:** Verified unique primary keys across all 50 entries.

### 5.3 Known Deficiencies & Critical Examiner Risks
1. **Sample Size Constraint (N=50):** 40 training examples are sufficient for **format-tuning** (teaching the model to produce valid JSON with Bloom taxonomy keys), but insufficient for **domain-knowledge infusion**. The base model's pre-trained knowledge base (`Qwen2.5-1.5B`) provides all domain facts.
2. **Distributional Bias:** 50% of the dataset is MCQs, while Short Answer (20%), True/False (16%), and Fill in the Blank (14%) have relatively small sample sizes in the validation split (1 to 2 items each).
3. **Short Context Windows:** Average context snippet length is ~163 characters, which is much shorter than real-world multi-page PDF document extraction chunks (typically 1,000–3,000 characters).

---

## 6. Engineering Roadmap for Dataset Expansion

To upgrade GenQuiz to production enterprise quality, the following improvements are recommended:
1. **Scale to 1,000+ Verified Items:** Expand dataset to 800 training / 200 validation items across 10 engineering domains.
2. **Human-in-the-Loop Subject Matter Verification:** Introduce pedagogical peer review by domain professors to eliminate distractor ambiguities.
3. **Adversarial Distractors:** Train negative sampling into MCQ options to avoid trivially identifiable wrong answers.
4. **Multi-paragraph Long Contexts:** Increase context snippet lengths from ~160 characters to 1,500–2,500 characters to mirror realistic multi-page PDF document extraction chunks.
