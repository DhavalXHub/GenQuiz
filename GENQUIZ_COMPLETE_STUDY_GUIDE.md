# GenQuiz — Complete Mastery & Viva Defense Study Guide
## From Zero to Expert: Theory, Code, Architecture, LoRA Fine-Tuning & Defense

> **Author:** Antigravity (Senior AI/ML Professor & Research Engineer)
> **Target Project:** GenQuiz — Adaptive AI Quiz & Assessment Platform
> **Workspace:** `D:\GenQuiz`
> **Primary Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
> **Fine-Tuning Method:** 4-bit QLoRA (PEFT 0.21.2)
> **Target Audience:** Computer Science & AI/ML Engineering Students, Reviewers, Viva Examiners
> **Status:** Empirically Verified against Repository Code & Training Logs

---

## Table of Contents
1. [Chapter 1: Understanding GenQuiz from Zero](#chapter-1-understanding-genquiz-from-zero)
2. [Chapter 2: The Complete System Architecture & Pipelines](#chapter-2-the-complete-system-architecture--pipelines)
3. [Chapter 3: Repository & Source Code Deep Dive](#chapter-3-repository--source-code-deep-dive)
4. [Chapter 4: The Dataset in Depth (CSV & SFT Splits)](#chapter-4-the-dataset-in-depth-csv--sft-splits)
5. [Chapter 5: Language Models & LLM Foundations](#chapter-5-language-models--llm-foundations)
6. [Chapter 6: Qwen2.5-1.5B, PyTorch & LoRA/QLoRA Deep Dive](#chapter-6-qwen25-15b-pytorch--loraqlora-deep-dive)
7. [Chapter 7: Step-by-Step Walkthrough of GenQuiz_FineTuning.ipynb](#chapter-7-step-by-step-walkthrough-of-genquiz_finetuningipynb)
8. [Chapter 8: PDF Processing, Document Slicing & Grounding](#chapter-8-pdf-processing-document-slicing--grounding)
9. [Chapter 9: Prompt Engineering, JSON Schemas & Guardrails](#chapter-9-prompt-engineering-json-schemas--guardrails)
10. [Chapter 10: Answer Evaluation, Scoring & Adaptive Diagnostics](#chapter-10-answer-evaluation-scoring--adaptive-diagnostics)
11. [Chapter 11: Acronyms & Core Technical Concepts](#chapter-11-acronyms--core-technical-concepts)
12. [Chapter 12: Environment, Dependencies & Hardware Management](#chapter-12-environment-dependencies--hardware-management)
13. [Chapter 13: Honest Technical Limitations & Future Roadmap](#chapter-13-honest-technical-limitations--future-roadmap)
14. [Chapter 14: Comprehensive Viva & Defense Question Bank](#chapter-14-comprehensive-viva--defense-question-bank)
15. [Chapter 15: Presentation & Live Demonstration Scripts](#chapter-15-presentation--live-demonstration-scripts)
16. [Chapter 16: 15-Stage Hands-on Mastery Plan & Exercises](#chapter-16-15-stage-hands-on-mastery-plan--exercises)

---

# Chapter 1: Understanding GenQuiz from Zero

### 1.1 What is GenQuiz?
GenQuiz is an intelligent educational assessment platform. In plain English, it takes an educational document (such as a lecture PDF, textbook chapter, or research paper), reads its contents, generates customized multiple-choice and short-answer questions tailored to specific cognitive levels, evaluates the student's answers, and provides a diagnostic report showing their strong and weak areas.

### 1.2 Real-World Analogies
- **Analogy 1 (The Dedicated Teaching Assistant):** Imagine a human Teaching Assistant (TA) who reads a 20-page textbook chapter, drafts 5 targeted questions on key definitions and math derivations, waits for students to write down answers, grades them with partial credit, and points out exactly which paragraph students should re-read. GenQuiz performs that exact workflow in seconds using generative AI.
- **Analogy 2 (The Diagnostic Medical Scan):** Instead of giving you a single grade ("You got a B"), GenQuiz functions like a medical scan: it tells you that your recall of terminology is 100%, but your analytical understanding of attention mechanisms is at 40%, recommending specific remedial review.

### 1.3 The Elevator Pitches
- **30-Second Pitch (For quick introduction):**
  > "GenQuiz is an adaptive assessment platform powered by a fine-tuned open-source language model (Qwen2.5-1.5B with LoRA). It ingests syllabus PDFs, generates curriculum-aligned quizzes mapped to Bloom's Taxonomy, semantically grades short answers, and provides personalized diagnostic feedback with zero dependency on proprietary cloud APIs."
- **2-Minute Pitch (For lab evaluations):**
  > "Traditional computer-based testing relies on static question banks or rigid string matching. GenQuiz introduces an end-to-end generative AI assessment pipeline. It processes unstructured educational PDFs with PyMuPDF, extracts prioritized topic contexts, and feeds them into a locally fine-tuned Qwen2.5-1.5B-Instruct model adapted with 4-bit QLoRA. The generated quizzes adhere to strict Pydantic JSON schemas. Student evaluations combine deterministic scoring for MCQs with LLM-based 3-tier semantic rubrics for short answers, concluding with adaptive mastery analytics that flag weak knowledge areas. It operates fully locally on consumer GPUs with automatic batching for quizzes over 8 questions."
- **5-Minute Pitch (For formal project presentations):**
  > Covers the problem statement (manual exam authoring overhead, lack of adaptive feedback), the technical innovation (parameter-efficient fine-tuning with 18.46M trainable weights, in-memory keyword-prioritized document slicing, cross-domain hallucination defense), the empirical training results (loss reduction from 2.705 to 0.643 across 50 steps), and the practical impact on autonomous educational evaluation.

### 1.4 Precise Technical Definitions
- **Artificial Intelligence (AI):** The broad computational field dedicated to creating systems capable of performing tasks that typically require human cognition (pattern recognition, reasoning, problem solving).
- **Generative AI (GenAI):** A specialized subset of AI focused on generating new content (text, code, synthetic questions) modeled after the probability distribution of training data.
- **Large Language Model (LLM):** A deep transformer neural network trained on vast text corpora using self-supervised causal language modeling objectives to predict subsequent tokens.
- **Supervised Fine-Tuning (SFT):** The process of updating a pre-trained model's parameters on curated pairs of prompts and target completions to enforce specific behavioral personas or structured formats.
- **Adaptive Learning:** An educational paradigm that continuously monitors learner performance metrics and dynamically tailors instruction or assessments to remediate detected knowledge gaps.

### 1.5 What Did YOU Build? Core Engineering Contributions
1. **Curated & Validated SFT Dataset Pipeline:** Structured `genquiz_dataset.csv` into stratified 80/20 train/validation splits (`data/train_sft.jsonl`, `data/val_sft.jsonl`) with zero cross-set leakage.
2. **QLoRA Supervised Fine-Tuning Adapter:** Fine-tuned `Qwen2.5-1.5B-Instruct` using 4-bit NF4 quantization and PEFT LoRA across all 7 linear projections, optimizing 18.46M parameters in 127 seconds.
3. **In-Memory Grounding & Document Slicing Engine (`pdf_processor.py`):** Built context extraction that prioritizes keyword density and domain synonyms without expensive vector database overhead.
4. **Pydantic Guardrails & Hallucination Defense (`quiz_generator.py`):** Implemented schema validation, placeholder rejection, and cross-domain question-answer alignment verification.
5. **Safe Automatic Inference Batching (`local_model_service.py`):** Orchestrated seamless batching for quizzes exceeding the 8-question local safety ceiling, preserving exact question distributions and eliminating duplicate items via Jaccard similarity.
6. **Hybrid Assessment & Diagnostic Remediation (`quiz_evaluator.py`, `export_utils.py`):** Combined deterministic MCQ matching with a 3-tier semantic evaluation rubric and downloadable PDF performance reporting.

---

# Chapter 2: The Complete System Architecture & Pipelines

### 2.1 The End-to-End Pipeline
The GenQuiz system transitions through five distinct phases from document ingestion to report export.

```
+-----------------------------------------------------------------------------------+
|                                 1. INPUT STAGE                                    |
|   User uploads PDF + Selects Topic, Difficulty (Easy/Med/Hard), Question Count    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        2. DOCUMENT PROCESSING & SLICING                           |
|   pdf_processor.py extracts text, audits keyword coverage, and extracts context   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        3. INFERENCE & BATCH ORCHESTRATION                         |
|   quiz_generator.py formats ChatML prompt                                         |
|   If count <= 8: Single generation via local_model_service.py / llm_service.py     |
|   If count > 8: generate_quiz_local_batched() splits into chunks of <= 8          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      4. VALIDATION & ALIGNMENT DEFENSE                            |
|   Pydantic schema parses JSON, checks options & placeholders                      |
|   validate_question_answer_alignment() catches cross-domain mismatches            |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      5. EVALUATION, ADAPTIVE REMEDIATION & EXPORT                 |
|   MCQs evaluated via Exact Match; Short Answers evaluated via 3-tier Semantic LLM |
|   Scores < 75% trigger weak-topic remediation; export_utils.py compiles PDF report|
+-----------------------------------------------------------------------------------+
```

### 2.2 Worked Real-World Examples

#### Scenario A: The Correctly Grounded Flow (Transformers Document)
1. **User Action:** Uploads `Transformers_Explained.pdf`, requests 5 questions on "Transformers" (3 MCQs, 2 Short Answer, Medium difficulty).
2. **Document Processing:** `extract_text_from_pdf()` extracts 18,500 characters. `extract_prioritized_context()` identifies 42 occurrences of terms like "attention", "query", "key", "transformer" and extracts the top 3,000-character context window. Keyword coverage is verified at 100%.
3. **Prompt Compilation:** Assembles a ChatML prompt instructing the model to generate 3 MCQs and 2 Short Answers based solely on the extracted context.
4. **Local Generation:** `local_model_service.py` executes Qwen2.5-1.5B + LoRA adapter with `temperature=0.4`. Emits clean JSON adhering to `QuizSchema`.
5. **Validation:** Pydantic confirms 5 items, all 4 MCQ options present, correct answers populated, no boilerplate placeholders. Alignment check confirms questions and answers both reference Transformers.
6. **User Interaction & Grading:** User answers. MCQ answers are string-compared deterministically. Short answer regarding self-attention is evaluated via semantic rubric; student receives 1.0 full score for stating query-key dot product role. Overall score 80%; no remediation triggered.

#### Scenario B: The Cross-Domain Collision Detection (Cybercrime PDF vs Transformers Prompt)
1. **User Action:** User previously tested Transformers, then uploads `Cybercrime_Laws.pdf` but leaves the prompt topic as "Transformers".
2. **Topic Coverage Audit:** `pdf_processor.py:audit_topic_coverage()` scans the Cybercrime text for "transformer", "attention", "feedforward", and domain synonyms. It finds **0 keyword matches**.
3. **Defensive Interception:** Rather than passing unrelated text to the LLM (which previously produced hallucinated answers combining cybercrime questions with positional encoding answers), the system immediately raises `TopicMismatchError`.
4. **User Feedback:** The UI alerts the user: *"TopicMismatchError: Topic 'Transformers' has 0 matches in the uploaded PDF. Please select a topic actually discussed in the document."* No corrupted quiz is presented.

---

# Chapter 3: Repository & Source Code Deep Dive

Let's examine every core Python file in `D:\GenQuiz`, its responsibility, and its exact implementation details.

### 3.1 `app.py` — The Streamlit Presentation & State Engine
- **Role:** Web UI, session state management, configuration controls, and interactive quiz rendering.
- **Key Session Variables:**
  - `st.session_state["quiz_data"]`: Stores currently active parsed quiz questions.
  - `st.session_state["quiz_submitted"]`: Boolean flag controlling review screen rendering.
  - `st.session_state["eval_results"]`: Stores detailed scores, rubric breakdowns, and remediation notes.
  - `st.session_state["active_engine"]`: Indicates `local_ft` (fine-tuned model) or `cloud_llm` (Groq API).
- **Safety Architecture:** Enforces clean separation between generation and evaluation states. Provides clear model status indicators (Local Model Ready vs Missing API Key).

### 3.2 `pdf_processor.py` — Text Extraction & Context Windowing
- **Role:** Ingests raw PDF bytes, extracts textual content, filters noise, and scores passages for relevance.
- **Core Functions:**
  - `extract_text_from_pdf(pdf_file)`: Uses `fitz.open(stream=..., filetype="pdf")` to iterate pages, extracting UTF-8 text. Rejects scans with < 50 characters.
  - `audit_topic_coverage(text, topic)`: Scans for direct tokens and expanded domain synonyms. Returns match count and coverage boolean.
  - `extract_prioritized_context(text, topic, max_chars=3500)`: Implements sliding window keyword scoring. Scores each paragraph by keyword hit frequency and returns the highest-scoring contiguous windows to fit the LLM context.

### 3.3 `dataset.py` — Dataset Loading, Filtering & Few-Shot Bank
- **Role:** Manages `genquiz_dataset.csv`.
- **Core Functions:**
  - `load_dataset()`: Reads CSV, verifies 14 required columns, and validates row contents.
  - `get_few_shot_examples(topic, question_type, n=2)`: Retrieves representative question examples formatted as JSON to include as few-shot exemplars inside prompts.
  - `filter_dataset(...)`: Powers the offline demo mode when neither local GPU nor cloud API is available.

### 3.4 `prepare_dataset.py` — SFT Dataset Pipeline
- **Role:** Transforms tabular CSV into format-aligned JSONL training and validation splits.
- **Core Functions:**
  - `stratified_split(df, test_size=0.20)`: Uses `(question_type, difficulty)` multi-key stratification to partition 50 rows into 40 train and 10 validation records.
  - `format_chatml_record(row)`: Converts a CSV row into a 3-turn ChatML dictionary (`system`, `user`, `assistant`).
  - `validate_splits(train_path, val_path)`: Verifies zero overlapping `question_id` values, ensuring zero data leakage.

### 3.5 `quiz_generator.py` — Prompt Assembly, Orchestration & Guardrails
- **Role:** Brain of question synthesis and validation.
- **Core Functions & Schemas:**
  - `QuizSchema` / `QuestionItemSchema`: Pydantic models enforcing schema adherence.
  - `build_generation_prompt(...)`: Constructs instruction prompts with topic constraints, Bloom taxonomy levels, and target JSON formats.
  - `validate_question_answer_alignment(question, answer, topic, context)`: Intercepts cross-domain hallucinations (e.g., matching security questions with neural network answers).

### 3.6 `local_model_service.py` — Qwen2.5-1.5B + LoRA Local Inference & Safe Batching
- **Role:** GPU-accelerated local generation and batching orchestration.
- **Core Parameters:** Base model `Qwen/Qwen2.5-1.5B-Instruct`, adapter path `fine_tuned_adapter`, `temperature=0.4`, `max_new_tokens=512`, `LOCAL_MAX_QUESTIONS = 8`.
- **Key Functions:**
  - `load_local_model()`: Loads 4-bit quantized base model via `BitsAndBytesConfig`, mounts PEFT LoRA adapter, attaches tokenizer.
  - `generate_quiz_local(prompt, max_questions=8)`: Executes single-pass generation. Enforces the 8-question safety ceiling.
  - `generate_quiz_local_batched(pdf_text, topic, difficulty, question_types, total_count)`: Automatically splits requests > 8 into batches $\le 8$, preserves exact question type distributions, and deduplicates questions using token Jaccard similarity.

### 3.7 `llm_service.py` — Cloud Fallback Engine (Groq API)
- **Role:** Provides high-speed cloud fallback using `llama-3.3-70b-versatile` or `mixtral-8x7b-32768`.
- **Key Functions:**
  - `generate_quiz(prompt, model)`: Calls Groq client with JSON-mode output enabled.
  - `is_groq_available()`: Safely detects if `GROQ_API_KEY` is present in the environment without logging or leaking its secret value.

### 3.8 `quiz_evaluator.py` — Grading Engine & Cognitive Diagnostics
- **Role:** Evaluates submitted answers and generates adaptive remediation insights.
- **Key Functions:**
  - `evaluate_mcq(student_answer, correct_answer)`: Deterministic exact string match after case folding and whitespace stripping.
  - `evaluate_short_answer(student_ans, expected_ans, explanation)`: Executes a 3-tier semantic evaluation rubric ($1.0, 0.5, 0.0$) via LLM; falls back to token-overlap scoring if the evaluator LLM fails.
  - `generate_adaptive_feedback(eval_summary)`: Computes topic and Bloom-level scores. Flags any topic scoring < 75% as requiring remediation and generates specific review recommendations.

### 3.9 `export_utils.py` — PDF Performance Report Generator
- **Role:** Generates downloadable, publication-quality PDF assessment reports using ReportLab.
- **Features:** Includes overall percentage score, topic-by-topic mastery breakdown, Bloom taxonomy cognitive radar, and a detailed question-by-question review table showing student answers, correct answers, and pedagogical feedback.

---

# Chapter 4: The Dataset in Depth (CSV & SFT Splits)

### 4.1 Dataset Fundamentals: What is a Dataset in SFT?
In Supervised Fine-Tuning (SFT), a dataset is not just a collection of numbers. It is a set of paired **demonstrations** showing the model:
1. What a well-formed prompt looks like.
2. What an ideal, perfectly structured output looks like.

### 4.2 Comprehensive 50-Row Audit (`genquiz_dataset.csv`)
- **Total Records:** 50
- **Total Columns:** 16 (14 metadata/question fields + ID + Provenance)
- **Primary Key:** `question_id` (`Q001` to `Q050`, zero duplicates)
- **Question Types:**
  - MCQ: 25 rows (50%)
  - Short Answer: 10 rows (20%)
  - True/False: 8 rows (16%)
  - Fill in the Blank: 7 rows (14%)
- **Difficulty Breakdown:**
  - Medium: 23 rows (46%)
  - Easy: 15 rows (30%)
  - Hard: 12 rows (24%)
- **Bloom's Taxonomy Levels:**
  - Understand: 15 rows (30%)
  - Remember: 14 rows (28%)
  - Apply: 9 rows (18%)
  - Analyze: 7 rows (14%)
  - Evaluate: 5 rows (10%)

### 4.3 The Three Distinct Roles of the Dataset in GenQuiz
It is vital to explain to an examiner that `genquiz_dataset.csv` serves three separate purposes:
1. **SFT Training & Validation Data:** 40 rows are converted into `data/train_sft.jsonl` and 10 rows into `data/val_sft.jsonl` to train the LoRA adapter.
2. **Few-Shot In-Context Learning Bank:** During live inference, `dataset.py:get_few_shot_examples()` pulls 2–3 rows matching the requested topic to demonstrate the expected JSON structure inside the prompt.
3. **Offline Demo Fallback Pool:** If no GPU and no API key are present, `dataset.py:filter_dataset()` filters these 50 rows to present a realistic offline quiz.

### 4.4 Stratified Train/Val Split & Leakage Prevention
To ensure rigorous evaluation:
- Partitioned 80/20 into 40 training items and 10 validation items.
- Stratified across both `question_type` and `difficulty` to preserve proportional balance.
- Automated tests (`test_grounding_and_alignment.py`) verify that the set intersection of `train_ids` and `val_ids` is empty ($	ext{Train} \cap 	ext{Val} = \emptyset$).

### 4.5 Provenance Statement & Quality Limitations
All records carry the explicit provenance marker:
`AI-assisted educational content; requires verification`

**What to tell an examiner:**
> "The 50 records represent a synthetic bootstrap dataset. They are sufficient to train the model on format adherence and schema compliance (producing valid JSON with Bloom levels), but are not large enough to infuse broad factual domain knowledge. Factual grounding is supplied at runtime through the uploaded PDF."

---

# Chapter 5: Language Models & LLM Foundations

### 5.1 What is a Language Model?
At its mathematical core, a causal language model is a conditional probability estimator over token sequences. Given a sequence of $t-1$ tokens, the model computes a probability distribution over the vocabulary $V$ for the next token $x_t$:
$$P(x_t | x_1, x_2, \dots, x_{t-1})$$
Generating a complete response is simply repeating this prediction step-by-step (autoregressive decoding).

### 5.2 Tokens & Tokenization
LLMs do not read characters or English words directly; they process **tokens**—discrete subword units mapped to integer IDs.
- For example, the word `"Transformers"` might be tokenized into two tokens: `["Transform", "ers"]` $	o [42561, 654]$.
- `Qwen2.5` uses a byte-level Byte-Pair Encoding (BPE) tokenizer with a vocabulary size of **151,646 tokens**, allowing it to represent technical, mathematical, and multilingual terms with high efficiency.

### 5.3 The Transformer Architecture & Self-Attention
Transformers replace recurrent connections with Self-Attention.
For an input sequence embedded into matrix $X$, the model projects $X$ into Queries ($Q$), Keys ($K$), and Values ($V$):
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
The scaled dot-product attention is calculated as:
$$	ext{Attention}(Q, K, V) = 	ext{softmax}\left(rac{Q K^T}{\sqrt{d_k}}
ight) V$$
- $Q K^T$ computes pairwise compatibility between every word and every other word in the sequence.
- $\sqrt{d_k}$ scales the dot products to prevent softmax gradients from vanishing.
- The softmax operation normalizes similarities into attention weights (probabilities summing to 1).
- Multiplying by $V$ produces a context-aware representation for each token.

### 5.4 Sampling Hyperparameters: Temperature, Top-p, Top-k
During generation, logits $z \in \mathbb{R}^{|V|}$ are converted to probabilities using temperature scaling:
$$P(x_i) =rac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- **Temperature ($T$):** In GenQuiz, we set $T=0.4$. Lower temperature makes the distribution sharper (more deterministic), which is essential for valid JSON syntax while retaining enough diversity for natural question phrasing.
- **Top-p (Nucleus Sampling):** Samples only from the smallest set of tokens whose cumulative probability exceeds $p$ (e.g., $p=0.9$).
- **Top-k Sampling:** Limits sampling to the top $k$ highest-probability tokens.

### 5.5 Model Quantization: FP32, FP16, INT8, and 4-Bit NF4
- **FP32 (32-bit Floating Point):** Standard full precision (4 bytes per parameter). A 1.5B model requires $pprox 6 	ext{ GB}$ of VRAM just to store weights.
- **FP16 / BF16 (16-bit):** Half precision (2 bytes per parameter). Requires $pprox 3 	ext{ GB}$ of VRAM.
- **4-Bit NormalFloat (NF4):** Quantizes weights into 4 bits (0.5 bytes per parameter). A 1.5B model requires only $pprox 0.8	ext{--}1.2 	ext{ GB}$ of VRAM, making inference feasible on modest consumer GPUs and even shared environments.

---

# Chapter 6: Qwen2.5-1.5B, PyTorch & LoRA/QLoRA Deep Dive

### 6.1 Why Qwen2.5-1.5B-Instruct?
1. **Model Specs:** 1.54 billion parameters, 28 Transformer layers, hidden dimension 1,536, 12 attention heads, 2 key-value heads (Grouped Query Attention), 32,768 native context length.
2. **Instruction Following:** Pre-trained on 18 trillion tokens with strong performance in reasoning, coding, and structured JSON generation.
3. **Hardware Efficiency:** Small enough to fit into 4-bit VRAM alongside PyTorch runtime overhead on a single 4 GB or 6 GB consumer GPU.

### 6.2 The Mathematics of LoRA (Low-Rank Adaptation)
In standard fine-tuning, all weights $W_0 \in \mathbb{R}^{d 	imes k}$ are updated by calculating full gradient matrices $\Delta W \in \mathbb{R}^{d 	imes k}$. For large models, storing optimizer states for $\Delta W$ in AdamW requires $3	imes$ to $4	imes$ the memory of the model itself.

LoRA hypothesizes that weight updates have a low "intrinsic dimension". It decomposes $\Delta W$ into two low-rank matrices:
$$\Delta W = B \cdot A$$
where:
- $B \in \mathbb{R}^{d 	imes r}$ (initialized to all zeros)
- $A \in \mathbb{R}^{r 	imes k}$ (initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$)
- $r \ll \min(d, k)$ is the rank.

During training:
- $W_0$ remains completely frozen (no gradients computed for $W_0$).
- Gradients are computed only for $A$ and $B$.
- The forward pass computes:
$$h = W_0 x +rac{lpha}{r}(B \cdot A) x$$
where $lpha$ is a constant scaling hyperparameter.

### 6.3 GenQuiz LoRA Configuration (`adapter_config.json`)
- **Rank ($r$):** 16
- **Alpha ($lpha$):** 32
- **Scaling Factor:** $rac{lpha}{r} =rac{32}{16} = 2.0$
- **Dropout:** 0.05
- **Target Modules:** All 7 linear projection layers in every Transformer block:
  - Attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - MLP projections: `gate_proj`, `up_proj`, `down_proj`
- **Parameter Accounting:**
  - Base Model Parameters: 1,562,179,072
  - Trainable LoRA Parameters: **18,464,768**
  - Trainable Percentage: **1.1820%**

### 6.4 What Makes QLoRA Different from LoRA?
QLoRA (Dettmers et al., 2023) introduces three innovations:
1. **4-Bit NormalFloat (NF4):** Quantizes frozen weights to an information-theoretically optimal distribution for normally distributed weights.
2. **Double Quantization:** Quantizes the quantization constants themselves, saving ~0.37 bits per parameter.
3. **Paged Optimizers:** Uses CUDA unified memory to page memory spikes to CPU RAM, preventing out-of-memory errors.

In GenQuiz, base weights are stored in 4-bit NF4, while forward and backward passes compute adapter gradients in 16-bit brain floating point (`bfloat16`).

---

# Chapter 7: Step-by-Step Walkthrough of GenQuiz_FineTuning.ipynb

Let's dissect the 19 notebook cells in `GenQuiz_FineTuning.ipynb`, understanding exactly what each step executes.

### 7.1 Setup, Environment & Dataset Ingestion (Cells 1 to 7)
- **Cell 1–2 (Environment Verification):** Checks for GPU presence (`nvidia-smi`), verifies CUDA availability, installs dependencies (`transformers`, `peft`, `bitsandbytes`, `trl`, `accelerate`).
- **Cell 3–4 (Dataset Inspection):** Loads `genquiz_dataset.csv`, displays row count (50), audits distribution across question types and Bloom levels.
- **Cell 5–7 (Data Preparation):** Executes stratified splitting (80% train / 20% validation) and outputs `data/train_sft.jsonl` (40 rows) and `data/val_sft.jsonl` (10 rows) using the ChatML format.

### 7.2 Model Loading & Quantization Configuration (Cells 8 to 11)
- **Cell 8 (BitsAndBytesConfig):**
  ```python
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True
  )
  ```
- **Cell 9 (Base Model Instantiation):** Loads `Qwen/Qwen2.5-1.5B-Instruct` in 4-bit precision with device mapping `{"": 0}`.
- **Cell 10 (LoRA Configuration):** Defines `LoraConfig(r=16, lora_alpha=32, target_modules=[...], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")`.
- **Cell 11 (Adapter Injection):** Calls `get_peft_model()`. Displays parameter footprint: 18.46M trainable parameters out of 1.56B total.

### 7.3 Training Execution via TRL SFTTrainer (Cells 12 to 14)
- **Cell 12 (SFTConfig & Training):**
  - Epochs: 5.0
  - Batch size per device: 1
  - Gradient accumulation steps: 4 (Effective batch size = 4)
  - Optimizer: `paged_adamw_8bit`
  - Learning rate: $2 	imes 10^{-4}$ with 10 warmup steps and cosine annealing decay
  - Precision: `bf16=True`, `gradient_checkpointing=True`
  - Total Training Steps: 50
  - Total Runtime: 127.52 seconds on GPU

### 7.4 Empirical Training Metrics & Loss Analysis
From `genquiz_checkpoints/checkpoint-50/trainer_state.json`:

| Step | Epoch | Training Loss | Validation Loss | Mean Train Token Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 5 | 0.5 | 2.705 | — | 54.6% |
| 10 | 1.0 | 1.520 | 1.624 | 68.2% |
| 20 | 2.0 | 0.963 | 0.959 | 78.4% |
| 30 | 3.0 | 0.742 | 0.919 | 81.2% |
| 40 | 4.0 | 0.678 | 0.916 | 83.9% |
| 50 | 5.0 | 0.643 | 0.919 | 84.6% |

**Scientific Interpretation:**
1. **Convergence:** Training loss drops sharply from 2.705 to 0.643, and token accuracy rises from 54.6% to 84.6%, proving the model rapidly mastered the target JSON schema.
2. **Generalization Plateau:** Validation loss drops from 1.624 to 0.919 by epoch 3, then flattens across epochs 4 and 5 ($0.916 	o 0.919$). This confirms the model extracted maximal format structure from the 40 training items without severe overfit.

### 7.5 Export & Adapter Serialization (Cells 15 to 19)
- **Cell 15–16:** Saves the LoRA adapter to `fine_tuned_adapter/` (`adapter_model.safetensors`, `adapter_config.json`).
- **Cell 17–19:** Executes sample inference verifying JSON format adherence and packs artifacts for deployment.

---

# Chapter 8: PDF-Based Generation & Grounding

### 8.1 PDF Extraction via PyMuPDF (`fitz`)
In `pdf_processor.py`, `extract_text_from_pdf()` processes binary PDF streams:
```python
doc = fitz.open(stream=pdf_bytes, filetype="pdf")
text = ""
for page in doc:
    text += page.get_text() + "\n"
```
PyMuPDF was selected because it is 10x faster than `pypdf`, handles multi-column layouts effectively, and runs without external C-binary system dependencies.

### 8.2 In-Memory Keyword Prioritization & Document Slicing
Large textbooks exceed typical prompt limits. Instead of naive truncation, `pdf_processor.py:extract_prioritized_context()` performs:
1. **Paragraph Chunking:** Splits text into natural paragraph windows.
2. **Relevance Scoring:** Computes hit counts for the requested topic and its semantic synonyms.
3. **Density-Based Slicing:** Selects and concatenates the highest-scoring paragraphs up to 3,500 characters, ensuring the LLM receives the densest factual context for question generation.

### 8.3 RAG vs Context-Window Stuffing: Clear Architectural Distinction
- **Vector RAG:** Ingests documents, splits into chunks, computes dense embeddings, stores in ChromaDB/FAISS, queries via cosine similarity, and passes top-$k$ chunks to the LLM. Essential for 500-page libraries.
- **GenQuiz Approach (Direct Prioritized Windowing):** For single lecture PDFs (5–30 pages), GenQuiz audits keyword coverage and extracts prioritized context directly in RAM in under 50 milliseconds. This eliminates vector database latency, embedding model dependencies, and index persistence failures.

---

# Chapter 9: Prompt Engineering, JSON Schemas & Guardrails

### 9.1 ChatML Structured Prompt Template
GenQuiz structures prompts using Qwen's ChatML format:
```
<|im_start|>system
You are an expert assessment generator. Generate structured educational questions based on the provided text, topic, and difficulty. Return valid JSON only.
<|im_end|>
<|im_start|>user
Topic: Transformers
Difficulty: Medium
Types: 3 MCQ, 2 Short Answer
Context:
[Extracted PDF text snippet...]

Output MUST be a JSON object matching this schema:
{
  "quiz_title": "...",
  "questions": [
    {
      "question_id": "Q1",
      "question": "...",
      "question_type": "MCQ",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct_answer": "A",
      "explanation": "...",
      "difficulty": "Medium",
      "bloom_level": "Understand"
    }
  ]
}
<|im_end|>
<|im_start|>assistant
```

### 9.2 Pydantic Validation Engine (`quiz_generator.py`)
Outputs are parsed and validated via Pydantic models:
1. **Field Existence:** Confirms `question`, `options`, `correct_answer`, `explanation`, `difficulty`, and `bloom_level` are present.
2. **MCQ Option Integrity:** Verifies that options contains distinct entries for keys `A`, `B`, `C`, and `D`.
3. **Placeholder Screening:** Rejects outputs containing boilerplate like `"Insert question here"`, `"Option A text"`, or `"Lorem ipsum"`.

### 9.3 Cross-Domain Hallucination Screening
`validate_question_answer_alignment()` verifies that generated questions and answers are semantically aligned with the requested topic and source context. If a prompt requested "Cybercrime" but the model emits an answer referencing "positional encoding", the item is intercepted and rejected.

---

# Chapter 10: Answer Evaluation, Scoring & Adaptive Diagnostics

### 10.1 Deterministic Objective Scoring (`evaluate_mcq`)
Multiple-choice and True/False questions must have zero grading ambiguity:
- Normalized String Matching:
  ```python
  norm_student = student_answer.strip().upper()
  norm_correct = correct_answer.strip().upper()
  is_correct = (norm_student == norm_correct)
  score = 1.0 if is_correct else 0.0
  ```
- No LLM randomness, zero cost, instant deterministic execution.

### 10.2 Semantic Short-Answer Evaluation (`evaluate_short_answer`)
Traditional token matching fails when a student explains a concept correctly using alternative vocabulary. GenQuiz implements a **3-tier semantic evaluation rubric** via LLM:

| Tier | Score | Criteria |
| :---: | :---: | :--- |
| **Correct** | **1.0** | Accurately explains core concept; demonstrates clear understanding; no critical factual errors. |
| **Partially Correct** | **0.5** | Mentions key terms or partial ideas, but lacks depth, precision, or contains minor inaccuracies. |
| **Incorrect** | **0.0** | Fails to address the question, demonstrates severe misconceptions, or leaves answer blank. |

**Transparent Fallback:** If the evaluator LLM is unavailable or fails, `quiz_evaluator.py` executes token-overlap Jaccard scoring and explicitly tags the result in the UI as `[Fallback Token Evaluation]`.

### 10.3 Adaptive Cognitive Diagnostics & Remediation
- **Mastery Threshold:** Any topic with a score $< 75\%$ is categorized as `needs_review`.
- **Diagnostic Breakdown:** Scores are aggregated by both **Topic** and **Bloom's Taxonomy Level** (e.g., *Remember: 100%*, *Analyze: 33%*).
- **Remedial Guidance:** The platform outputs actionable study recommendations pointing the student to specific conceptual areas for review before re-taking the quiz.

---

# Chapter 11: Acronyms & Core Technical Concepts

*(Note: For the exhaustive 30+ term glossary with dual layman/technical definitions, refer directly to `GENQUIZ_GLOSSARY.md`.)*

### Core Acronym Quick-Reference:
- **LLM:** Large Language Model (deep autoregressive Transformer).
- **SFT:** Supervised Fine-Tuning (instruction conditioning on paired examples).
- **PEFT:** Parameter-Efficient Fine-Tuning (adapting models by training < 2% of weights).
- **LoRA:** Low-Rank Adaptation ($\Delta W = B \cdot A$).
- **QLoRA:** Quantized Low-Rank Adaptation (4-bit NF4 quantized base model + LoRA).
- **GQA:** Grouped Query Attention (reduces KV cache memory footprint).
- **NF4:** NormalFloat4 (quantile-optimized 4-bit representation for Gaussian weights).
- **ChatML:** Chat Markup Language (`<|im_start|>` / `<|im_end|>`).
- **EM:** Exact Match (deterministic grading metric).
- **RAG:** Retrieval-Augmented Generation.

---

# Chapter 12: Environment, Dependencies & Hardware Management

### 12.1 Hardware Architecture: VRAM vs System RAM
- **System RAM:** Host memory used by CPU and Python runtime (`D:\AI-Training\.venv`).
- **GPU VRAM:** High-Bandwidth Memory on the graphics card where model weights, KV cache, and activations must reside during forward/backward passes.
- **GenQuiz Footprint:**
  - Base `Qwen2.5-1.5B` in 4-bit NF4: $pprox 1.1	ext{--}1.2 	ext{ GB}$ VRAM.
  - LoRA Adapter weights: $pprox 74 	ext{ MB}$.
  - KV Cache & Activation overhead: $pprox 500	ext{--}800 	ext{ MB}$ during inference.
  - **Total Minimum GPU VRAM Required:** **2.5 GB**, fitting easily on an NVIDIA RTX 3050/3060/4060 or Google Colab T4.

### 12.2 Key Dependency Matrix (`requirements.txt`)
- `torch`: Deep learning runtime and CUDA kernel acceleration.
- `transformers`: Hugging Face model architecture and tokenizer definitions.
- `peft==0.21.2`: Parameter-efficient fine-tuning LoRA adapter management.
- `bitsandbytes`: 4-bit NF4 quantization and CUDA matrix multiplication kernels.
- `trl`: Transformer Reinforcement Learning library providing `SFTTrainer`.
- `pydantic`: Runtime schema validation and data parsing.
- `pymupdf` (`fitz`): High-speed PDF text parsing.
- `streamlit`: Web dashboard and interactive state rendering.
- `reportlab`: PDF assessment report generation.

---

# Chapter 13: Honest Technical Limitations & Future Roadmap

To earn top marks from viva examiners, you must demonstrate scientific maturity by honestly discussing your system's current boundaries:

### 13.1 Honest Current Limitations
1. **Small SFT Dataset Size ($N=50$):** 40 training items are sufficient for format and schema alignment, but cannot teach new academic subjects. All factual knowledge originates from pre-training and the uploaded PDF context.
2. **Text-Only PDF Extraction:** `fitz.get_text()` extracts digital text streams. Scanned PDFs containing pure images require an external OCR preprocessor (e.g., Tesseract).
3. **Linear Context Slicing:** Long documents (> 50 pages) are sliced based on keyword density rather than semantic dense vector embeddings.
4. **Single-Node Local Inference:** Local generation throughput is bounded by single-GPU compute speeds (~20–35 tokens/second on an RTX 3060).

### 13.2 Future Engineering Roadmap
1. **Dataset Scaling:** Expand SFT corpus to 1,000+ peer-reviewed items across multiple STEM disciplines.
2. **Hybrid OCR Integration:** Incorporate EasyOCR or Tesseract for seamless processing of scanned historical documents.
3. **Dense Vector RAG Pipeline:** Add ChromaDB or FAISS for enterprise multi-document curriculum indexing.
4. **Multi-Agent Evaluation:** Implement a 2-agent debate protocol for grading complex essay responses.

---

# Chapter 14: Comprehensive Viva & Defense Question Bank

*(Note: For complete detailed answers across all 4 tiers, refer directly to `GENQUIZ_VIVA_QUESTIONS.md`.)*

### Key Viva Defense Points to Remember:
1. **"Did you retrain the whole model?"**
   $	o$ **No.** We performed parameter-efficient fine-tuning (PEFT) using LoRA, freezing all 1.56 billion base parameters and training only 18.46 million adapter parameters (1.1820% of the model).
2. **"Does uploading a PDF retrain the model?"**
   $	o$ **No.** Uploading a PDF provides *in-context conditioning* at runtime. The model weights are unchanged.
3. **"Why not use full fine-tuning?"**
   $	o$ Full fine-tuning would require $pprox 18	ext{--}24 	ext{ GB}$ of VRAM for optimizer states and would risk catastrophic forgetting of general English reasoning capabilities.
4. **"How do you know the model didn't just memorize the 40 training questions?"**
   $	o$ Validation loss on 10 held-out examples dropped from 1.624 to 0.919 and plateaued. Furthermore, at runtime, the model generates completely new questions grounded in newly uploaded PDFs.

---

# Chapter 15: Presentation & Live Demonstration Scripts

### 15.1 Recommended Presentation Slide Structure (10 Slides)
- **Slide 1:** Project Title, Student Details, Guide Name, "GenQuiz: Adaptive AI Assessment Platform".
- **Slide 2:** Problem Statement (Assessment creation overhead, lack of personalized feedback).
- **Slide 3:** Proposed Solution & High-Level Architecture (PDF $	o$ LoRA LLM $	o$ Pydantic $	o$ Evaluation).
- **Slide 4:** Dataset Engineering (`genquiz_dataset.csv`, schema, stratified 80/20 SFT split).
- **Slide 5:** Supervised Fine-Tuning & QLoRA (Math $\Delta W = B \cdot A$, 4-bit NF4, 18.46M params).
- **Slide 6:** Empirical Training Results (Loss curve from 2.705 to 0.643, accuracy 84.6%).
- **Slide 7:** Document Ingestion & Hallucination Guardrails (Keyword audit, alignment checks).
- **Slide 8:** Adaptive Evaluation Engine (MCQ exact match, 3-tier semantic rubric, 75% threshold).
- **Slide 9:** Live Demonstration & System Outputs (Streamlit UI, results dashboard, PDF report).
- **Slide 10:** Conclusion, Limitations & Future Scope.

### 15.2 Step-by-Step Live Demonstration Script (5-Minute Protocol)
1. **Step 1 (Launch Application):** Run `streamlit run app.py`. Highlight that the local fine-tuned model loads onto GPU VRAM with memory optimization.
2. **Step 2 (Upload Document):** Upload a sample PDF (e.g., `Transformers_Explained.pdf`). Show that text is extracted and keyword coverage verified.
3. **Step 3 (Generate Quiz):** Request 5 questions (3 MCQs, 2 Short Answer, Medium difficulty). Show real-time generation and valid JSON rendering.
4. **Step 4 (Demonstrate Batching):** Request 10 questions. Show in console logs that the orchestrator transparently splits the request into two safe batches of 5 and merges them without duplicates.
5. **Step 5 (Interactive Assessment):** Submit answers (deliberately answer one MCQ incorrectly and write a conceptual short answer). Show exact match grading for MCQ and semantic rubric grading for short answer.
6. **Step 6 (Adaptive Diagnostics & Export):** Point out the topic mastery breakdown, the <75% remedial recommendation, and download the generated PDF assessment report.

---

# Chapter 16: 15-Stage Hands-on Mastery Plan & Exercises

Follow this sequential hands-on plan to verify your understanding before your viva:

- **Stage 1 (Codebase Familiarization):** Open and read `GENQUIZ_ARCHITECTURE.md`. Trace every box to the source code.
- **Stage 2 (Dataset Inspection):** Open `genquiz_dataset.csv` in Excel or VS Code. Verify all 16 columns and 50 rows.
- **Stage 3 (Data Pipeline Verification):** Run `python prepare_dataset.py`. Inspect `data/train_sft.jsonl` and `data/val_sft.jsonl`.
- **Stage 4 (Tokenizer Experimentation):** In a Python REPL, load Qwen tokenizer and encode a sample prompt. Observe token IDs and ChatML boundary tags.
- **Stage 5 (LoRA Architecture Inspection):** Open `fine_tuned_adapter/adapter_config.json`. Verify rank $r=16$, $lpha=32$, and target modules.
- **Stage 6 (Training Metrics Review):** Open `genquiz_checkpoints/checkpoint-50/trainer_state.json`. Trace training and evaluation loss across the 50 steps.
- **Stage 7 (PDF Extraction Testing):** Run `python -c "import fitz; print(fitz.__doc__)"`. Write a 3-line script extracting text from a sample PDF.
- **Stage 8 (Prompt Engineering Verification):** Inspect `quiz_generator.py:build_generation_prompt()`. Trace how context and JSON formatting rules are injected.
- **Stage 9 (Pydantic Schema Validation):** Test passing an invalid dictionary to `QuizSchema` in Python; observe how validation errors are caught.
- **Stage 10 (Local Generation Execution):** Run local inference on 3 questions; observe VRAM consumption in Task Manager or `nvidia-smi`.
- **Stage 11 (Batching Mechanics Test):** Run `pytest tests/test_local_batching.py`. Verify that 15 questions split into 8 and 7.
- **Stage 12 (Hallucination Defense Test):** Run `pytest tests/test_grounding_and_alignment.py`. Verify cross-domain collision rejection.
- **Stage 13 (Deterministic vs Semantic Grading):** Trace `quiz_evaluator.py:evaluate_mcq()` vs `evaluate_short_answer()`.
- **Stage 14 (Full Test Suite Execution):** Run `pytest`. Confirm all 128 tests pass with 100% success.
- **Stage 15 (Mock Viva Simulation):** Have a peer ask you 5 random questions from `GENQUIZ_VIVA_QUESTIONS.md`. Answer without reading notes.
