# GenQuiz Seed Dataset Methodology & Provenance Report

## 1. Executive Summary

The **GenQuiz Dataset** (`genquiz_dataset.csv`) is an academically curated seed benchmark comprising 50 standardized assessment records across core domains of **Generative Artificial Intelligence**. Rather than serving as a massive pretraining corpus, this dataset functions within the GenQuiz architecture as:
1. **An In-Context Few-Shot Exemplar Store:** Supplying high-quality schema and formatting patterns directly into the LLM system prompt.
2. **A Benchmark for Pedagogical Alignment:** Modeling Bloom's Revised Taxonomy levels (Remember, Understand, Apply, Analyze, Evaluate) and difficulty tiers (Easy, Medium, Hard).
3. **An Out-of-the-Box Demo Engine:** Providing verifiable, deterministic assessment items when external API keys are unavailable.

---

## 2. Dataset Schema & Field Definitions

Each record in `genquiz_dataset.csv` adheres to a strict 16-column relational schema validated via Pydantic:

| Column Name | Type | Description | Required? |
| :--- | :--- | :--- | :--- |
| `question_id` | String | Unique record identifier (e.g., `GAI001` to `GAI050`) | Yes |
| `subject` | String | High-level educational domain (`Generative AI`) | Yes |
| `topic` | String | Primary curriculum area (e.g., `Transformer architecture`, `RAG`) | Yes |
| `subtopic` | String | Specific concept covered (e.g., `Self-attention`, `Dense retrieval`) | Optional |
| `context` | String | Contextual excerpt or background scenario grounding the question | Optional |
| `question` | String | The actual assessment prompt text | Yes |
| `question_type`| String | `MCQ`, `True/False`, `Short Answer`, or `Fill in the Blank` | Yes |
| `option_a` | String | Text for Option A (populated for MCQs) | Conditional |
| `option_b` | String | Text for Option B (populated for MCQs) | Conditional |
| `option_c` | String | Text for Option C (populated for MCQs) | Conditional |
| `option_d` | String | Text for Option D (populated for MCQs) | Conditional |
| `correct_answer`| String | Ground truth answer string or exact matching option | Yes |
| `explanation` | String | Pedagogical rationale justifying why the answer is correct | Yes |
| `difficulty` | String | Cognitive challenge level: `Easy`, `Medium`, or `Hard` | Yes |
| `bloom_level` | String | Bloom's Revised Taxonomy level (`Remember` to `Evaluate`) | Yes |
| `source` | String | Provenance notice and review status | Yes |

---

## 3. Dataset Creation & Drafting Process

### A. Topic Selection
Topics were selected based on standard graduate and advanced undergraduate Generative AI curricula:
- **Foundations:** Generative vs. discriminative models, autoregressive modeling, output modalities.
- **Architectures:** Transformer encoder/decoder, Self-attention, Multi-head attention, Positional encodings.
- **Inference & Sampling:** Temperature, top-k, top-p (nucleus) sampling.
- **Fine-Tuning:** Full fine-tuning vs. Parameter-Efficient Fine-Tuning (PEFT, LoRA).
- **Retrieval-Augmented Generation (RAG):** Dense vector embeddings, vector databases, chunking strategies, hybrid retrieval.
- **Safety, Governance & Evaluation:** Hallucinations, prompt injections, toxic content, ROUGE, BLEU, and LLM-as-a-judge.

### B. Initial Drafting
Initial drafts were synthesized with educational AI assistance to ensure consistent conceptual coverage across all 17 subtopics. Each draft item was framed around clear problem statements and unambiguous distractors (incorrect options).

### C. Source Literature & Ground Truth References
General scientific consensus was established based on foundational peer-reviewed literature:
1. *Attention Is All You Need* (Vaswani et al., NeurIPS 2017) — Self-attention & Transformer equations.
2. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., NeurIPS 2020) — RAG frameworks.
3. *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., ICLR 2022) — PEFT decomposition.
4. *Language Models are Few-Shot Learners* (Brown et al., NeurIPS 2020) — Few-shot prompting and in-context learning.

---

## 4. Manual Verification vs. Automated Validation

To maintain strict academic integrity, we explicitly distinguish between what has been automated and what was reviewed:

### What Automated Validation Proves (`dataset.py`)
Our automated test suite runs programmatic checks confirming:
- **Schema Conformity:** Zero missing mandatory columns.
- **Uniqueness:** Zero duplicate `question_id` values and zero duplicate question texts.
- **MCQ Option-Answer Alignment:** Every MCQ record's `correct_answer` mathematically equals one of `option_a`, `option_b`, `option_c`, or `option_d`.
- **Constraint Boundaries:** `question_type` strictly belongs to the allowed enum set; `difficulty` belongs to `{Easy, Medium, Hard}`; `bloom_level` belongs to the 6 valid Bloom categories.

### What Automated Validation Does NOT Prove
- Automated checks cannot verify whether an explanation is conceptually lucid or pedagogically optimal.
- Automated checks cannot detect subtle ambiguities in phrasing or subjective grading criteria.

### Manual Spot-Check History
All 50 records were reviewed for scientific accuracy, typo suppression, and alignment of distractors before final release. Every record is flagged with `AI-assisted educational content; requires verification` in the `source` field to transparently reflect its provenance.

---

## 5. Integration into the GenQuiz Generation Pipeline

The dataset is not a static warehouse; it directly drives few-shot generation:

```
[User Prompt: e.g. "8 questions on Transformers, Medium difficulty"]
                         │
                         ▼
             [DatasetManager.get_few_shot_examples()]
                         │
      ┌──────────────────┴──────────────────┐
      ▼                                     ▼
(Topic Match: "Transformer")      (Difficulty Match: "Medium")
      └──────────────────┬──────────────────┘
                         ▼
        [Select 2 to 3 Matched Exemplars]
                         │
                         ▼
   [Injected into LLM System Prompt as Few-Shot Context]
                         │
                         ▼
   [LLM Instruction: "Emulate schema and depth; formulate
    NOVEL questions grounded strictly in uploaded PDF"]
```

### Protection Against Verbatim Copying
The system prompt explicitly commands the model:
> *"Do NOT copy example questions verbatim. Formulate brand new questions based solely on the uploaded PDF excerpt while following the formatting and depth demonstrated in the examples."*

---

## 6. Dataset Distributions & Summary Statistics

As confirmed by `dataset.py` execution:
- **Total Records:** 50
- **Question Types:**
  - Multiple Choice (MCQ): 25 (50.0%)
  - Short Answer: 10 (20.0%)
  - True/False: 8 (16.0%)
  - Fill in the Blank: 7 (14.0%)
- **Difficulty Breakdown:**
  - Medium: 23 (46.0%)
  - Easy: 15 (30.0%)
  - Hard: 12 (24.0%)
- **Bloom's Taxonomy Levels:**
  - Understand: 15 (30.0%)
  - Remember: 14 (28.0%)
  - Apply: 9 (18.0%)
  - Analyze: 7 (14.0%)
  - Evaluate: 5 (10.0%)

---

## 7. Supervised Fine-Tuning (SFT) Conversion & Stratified Splits

To enable genuine open-source model training, `prepare_dataset.py` transforms `genquiz_dataset.csv` into a standardized conversational Supervised Fine-Tuning corpus without modifying or duplicating the original dataset:

### A. Conversational ChatML Transformation
Each SFT example teaches the generative task:
$$\text{Source Context} + \text{Specifications (Topic, Type, Difficulty, Bloom)} \longrightarrow \text{Ground Truth Question JSON}$$

- **System Message:** Instructs the LLM on its role as an assessment generation engine.
- **User Message:** Contains the source context passage (average 432 characters) along with target parameters (topic, question type, difficulty, Bloom level).
- **Assistant Message:** The structured target JSON conforming to the Pydantic schema (`question_id`, `question`, `question_type`, `options`, `correct_answer`, `explanation`, `topic`, `difficulty`, `bloom_level`).

### B. Stratified Split Strategy (No Data Leakage)
To prevent evaluation leakage and maintain identical class balances across both subsets:
- **Total Valid SFT Records:** 50
- **Train Split (80%):** 40 examples
- **Validation Split (20%):** 10 examples
- **Stratification Criterion:** Stratified deterministically across all four `question_type` categories:
  - MCQ: 20 train / 5 val
  - Short Answer: 8 train / 2 val
  - True/False: 6 train / 2 val
  - Fill in the Blank: 6 train / 1 val
- **Leakage Verification:** ID set intersection is verified as empty (0 overlapping IDs).
- **No Artificial Duplication:** The 50 records yield exactly 50 distinct SFT examples without synthetic oversampling or fabrication.

---

## 8. Future Dataset Expansion Roadmap

For future academic iterations:
1. **Crowdsourced Educator Annotations:** Implementing an interface for professors to submit and peer-review questions with version control.
2. **Item Response Theory (IRT):** Calibrating empirical item difficulty based on aggregated student response logs rather than qualitative labels.
3. **Multimodal Item Support:** Incorporating architecture diagrams, attention heatmaps, and code snippets into the question context schema.
