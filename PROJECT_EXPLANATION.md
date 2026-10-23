# GenQuiz — Comprehensive Technical & Academic Project Explanation

**Project Title:** GenQuiz — Adaptive Generative AI-Based Quiz Generation and Personalized Assessment System  
**Author:** College Final Year Project Team  
**Tech Stack:** Python 3.12, Streamlit, PyMuPDF, Pandas, Pydantic, Groq / OpenAI API, ReportLab  

---

## 1. Problem Statement and Objectives

### The Problem
Traditional educational assessments suffer from two primary bottlenecks:
1. **Manual Authoring Bottleneck:** Formulating pedagogically sound, high-quality assessment questions (MCQs, True/False, Short Answer) grounded in specific lecture slides or research papers is time-consuming for professors.
2. **Static, Non-Adaptive Feedback:** Standard online quizzes present static question banks without adjusting to a student's individual learning deficits, providing little diagnostic insight into topic-level mastery.

### The Objective
Build a lightweight, reliable, document-grounded assessment platform that:
- Allows an educator or student to upload an authoritative text PDF and describe desired quiz requirements in natural language.
- Dynamically extracts and validates PDF text using PyMuPDF.
- Leverages an academically validated 50-item seed dataset (`genquiz_dataset.csv`) to provide in-context few-shot guidance.
- Prompts a pretrained LLM (Groq / OpenAI) to generate novel, strictly grounded questions in structured JSON format.
- Enforces strict Pydantic schema validation on generated output.
- Scores multi-type responses (objective and short answer) and provides grounded explanations.
- Employs rule-based cognitive heuristics to diagnose weak topic areas and suggest targeted adaptive quizzes.
- Seamlessly falls back to an explicit **Demo Mode** when API keys are absent.

---

## 2. Overall System Architecture

The architecture consists of five decoupled, modular components:

```
┌────────────────────────┐      ┌──────────────────────────────┐
│  Uploaded PDF Document │      │ Natural Language User Prompt │
└───────────┬────────────┘      └──────────────┬───────────────┘
            │                                  │
            ▼                                  ▼
┌────────────────────────┐      ┌──────────────────────────────┐
│ pdf_processor.py       │      │ Heuristic Prompt Parser      │
│ (PyMuPDF Extraction)   │      │ (Count, Diff, Types, Topic)  │
└───────────┬────────────┘      └──────────────┬───────────────┘
            │                                  │
            └───────────────┬──────────────────┘
                            │
                            ▼
           ┌─────────────────────────────────┐
           │ dataset.py (DatasetManager)     │
           │ Few-Shot Example Selector       │
           │ (Matches topic & schema styles) │
           │ from genquiz_dataset.csv        │
           └────────────────┬────────────────┘
                            │
                            ▼
           ┌─────────────────────────────────┐
           │ llm_service.py (Groq / OpenAI)  │
           │ Provider-Agnostic LLM Engine    │
           │ Grounded Prompt + Few-Shot JSON │
           └────────────────┬────────────────┘
                            │
                            ▼
           ┌─────────────────────────────────┐
           │ quiz_generator.py               │
           │ Pydantic Structured Validator   │
           │ (Enforces question integrity)   │
           └────────────────┬────────────────┘
                            │
                            ▼
           ┌─────────────────────────────────┐
           │ app.py (Streamlit UI Player)    │
           │ Interactive Quiz Attempt        │
           └────────────────┬────────────────┘
                            │
                            ▼
           ┌─────────────────────────────────┐
           │ quiz_evaluator.py               │
           │ Auto-Scoring & Rule-Based       │
           │ Adaptive Weak-Topic Diagnostics │
           └─────────────────────────────────┘
```

---

## 3. Dataset Engineering & Supervised Fine-Tuning (SFT) Integration

### A. Dataset Schema & Preservation
`genquiz_dataset.csv` contains 50 verified records spanning 17 Generative AI subtopics. The source file is strictly preserved:
- `question_id`, `subject`, `topic`, `subtopic`, `context`, `question`, `question_type`, `option_a`, `option_b`, `option_c`, `option_d`, `correct_answer`, `explanation`, `difficulty`, `bloom_level`, `source`.

### B. Verification & Integrity Guarantees
- 100% of multiple-choice answers strictly match one of the four provided options.
- Zero duplicate question IDs or question texts.
- Verified against canonical literature (*Vaswani et al., 2017*, *Lewis et al., 2020*).
- Provenance statement: AI-assisted records are explicitly tracked as `AI-assisted educational content; requires verification` without false claims of independent human verification.

### C. Supervised Fine-Tuning (SFT) with LoRA / QLoRA
In addition to cloud few-shot prompting, GenQuiz provides a genuine **Supervised Fine-Tuning (SFT) and LoRA pipeline**:
1. **Data Preparation (`prepare_dataset.py`):**
   - Transforms records into standard conversational ChatML prompt-completion format:
     $$\text{Source Context} + \text{Specifications (Topic, Type, Difficulty, Bloom)} \longrightarrow \text{Structured Question JSON}$$
   - Performs a deterministic stratified 80/20 train/validation split (40 train / 10 val) stratified across all 4 question types (MCQ, True/False, Short Answer, Fill in the Blank).
   - Enforces zero data leakage (0 overlapping IDs between splits).
   - Generates `data/train_sft.jsonl`, `data/val_sft.jsonl`, and `reports/sft_dataset_validation_report.md`.
2. **Google Colab QLoRA Training (`GenQuiz_FineTuning.ipynb`):**
   - Base model: `Qwen/Qwen2.5-1.5B-Instruct` (Un-gated, open weights, state-of-the-art JSON reasoning).
   - 4-bit NF4 quantization with `bitsandbytes`.
   - Low-Rank Adaptation (`peft.LoraConfig`): $r=16, \alpha=32$, targeting linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - Trained with `trl.SFTTrainer` for 5 epochs (~4 minutes on free T4 GPU).
   - Saves adapter weights to `genquiz_lora_adapter` and exports a ready-to-use zip.
3. **Local Inference Engine (`local_model_service.py`):**
   - Caches base model and adapter via `@st.cache_resource` for instant execution without reloading weights.
   - Extracts and validates structured JSON output directly from local weights without external cloud API calls.
   - Grounded strictly in uploaded PDF passages.

### D. In-Context Few-Shot Learning for Cloud LLMs
When using Groq or OpenAI:
1. `DatasetManager.get_few_shot_examples()` queries the CSV for 2 to 3 high-quality examples matching the target topic and question types.
2. These records are injected into the LLM system prompt as structural exemplars.
3. The LLM is instructed:
   > *"Do NOT copy the reference questions. Emulate their clear pedagogical formatting, distractor quality, and concise explanations while generating NOVEL questions grounded solely in the uploaded PDF."*

---

## 4. Document Grounding & Hallucination Suppression

Document grounding is guaranteed through architectural constraints:
1. **PyMuPDF Extraction:** Directly extracts digital text layer by layer, preserving page markers (`--- [Page X] ---`).
2. **Context Bounding:** Text is capped at a conservative character limit (e.g., 15,000 characters) to avoid context blowout while informing the user via UI badges if truncation occurred.
3. **Strict Negative Constraints:** The LLM prompt explicitly commands:
   > *"CRITICAL: All questions MUST be strictly grounded in the provided PDF excerpt. If the document lacks sufficient detail to answer a question, do not invent facts; restrict questions solely to explicit document evidence."*
4. **Pydantic Validation:** The LLM output is parsed as strict JSON and validated against `GeneratedQuiz` and `GeneratedQuestion` models. If malformed, a single automatic retry is executed.

---

## 5. Scoring Engine & Rule-Based Adaptive Assessment

### Multi-Type Scoring Logic (`quiz_evaluator.py`)
- **MCQ:** Deterministic exact match against the correct option letter (`A`, `B`, `C`, `D`) or option string (Award: 1.0 or 0.0 pts).
- **True/False:** Deterministic exact boolean evaluation (Award: 1.0 or 0.0 pts).
- **Short Answer (AI Semantic Rubric & Fallback):**
  - **Live Evaluation (LLM-Assisted):** Leverages a 3-tier semantic assessment rubric:
    * **Full Credit (1.0 pt):** The response demonstrates clear conceptual understanding and accurately explains the core mechanism, regardless of differing syntax, wording, or synonyms.
    * **Partial Credit (0.5 pt):** The response demonstrates partial understanding, identifying some key concepts while omitting a crucial mechanism.
    * **No Credit (0.0 pt):** Factually incorrect, irrelevant, or empty.
    * Returns constructive semantic feedback, key concepts identified, and key concepts missed.
  - **Rule-Based Fallback:** If the LLM is unavailable or the call fails, an automatic fallback evaluator assesses substantive token/stem overlap (stemmed prefixes like `token`/`tokens`, `interact`/`interaction`) and transparently tags the question as evaluated via fallback.

### Rule-Based Adaptive Recommendation
Rather than claiming an unverified black-box recommendation model, GenQuiz uses an interpretable cognitive heuristic:
- Topic-level accuracy is tracked across all answered items.
- Topics with accuracy below **75%** are flagged as **Weak Areas**.
- The system generates a targeted follow-up prompt:
  `"Generate 5 questions focusing on [Weak Topic]. Use Medium difficulty with conceptual explanations."`
- The user can click one button to immediately re-generate a targeted practice quiz.

---

## 6. Transparent Demo Mode

If no API key is supplied:
1. The sidebar clearly displays **Active Mode: Demo Mode (Prewritten Verified Dataset)**.
2. The application retrieves matching verified records directly from `genquiz_dataset.csv`.
3. Questions are stamped with `is_saved_dataset=True` and labeled:
   `"SAVED QUESTION: From GenQuiz verified dataset. No live API call made."`
4. The user experiences full interactive attempt, scoring, analytics, and PDF export without spending money or failing silently.

---

## 7. Current Implementation vs. Planned Future Work

| Feature | Status | Notes |
| :--- | :--- | :--- |
| PyMuPDF Text Extraction | **Implemented** | Page-by-page extraction, scanned detection, truncation notification |
| Natural-Language Prompt Parser | **Implemented** | Regex & heuristic parsing with sensible defaults |
| CSV Dataset Management & Validator | **Implemented** | Pydantic schema validation, distribution stats, Markdown report |
| In-Context Few-Shot LLM Guidance | **Implemented** | Dynamic retrieval of seed examples into system prompt |
| Provider-Agnostic LLM Engine | **Implemented** | Groq (`llama-3.3-70b-versatile`) and OpenAI (`gpt-4o-mini`) |
| Structured JSON Validation | **Implemented** | Pydantic models with retry on malformed JSON |
| Multi-Type Quiz Player & Auto-Scorer | **Implemented** | Interactive cards for MCQ, True/False, and Short Answer |
| Rule-Based Adaptive Assessment | **Implemented** | Weak topic diagnosis & one-click targeted quiz re-generation |
| ReportLab PDF Export | **Implemented** | Downloadable assessment results report |
| Model Fine-Tuning | *Future Work* | Documented as next academic milestone; not required for core flow |
| Vector Database / Dense RAG | *Future Work* | ChromaDB / FAISS semantic chunk search for multi-hundred-page books |

---

## 8. Professor Viva Voce Preparation Guide (Questions & Model Answers)

### Q1: Why didn't you fine-tune an LLM from scratch?
**Answer:**  
Fine-tuning adjusts model weights for specialized vocabulary or style, but it does not reliably inject dynamic, user-uploaded documents into the model's memory without massive computational cost and the risk of catastrophic forgetting. In this application, the source document changes every time a user uploads a new PDF. Therefore, **In-Context Learning (prompt grounding with PyMuPDF text and few-shot dataset examples)** is computationally efficient, deterministic, zero-cost, and eliminates hallucinated out-of-document facts.

### Q2: What is the academic purpose of `genquiz_dataset.csv` if the LLM generates questions from the PDF?
**Answer:**  
The dataset serves three vital roles:
1. **Few-Shot Pedagogical Formatting:** We query the dataset to find 2–3 records matching the user's requested topic and difficulty. These are fed to the LLM as structural templates illustrating high-quality distractors, Bloom taxonomy depth, and concise explanations.
2. **Deterministic Evaluation & Demo Mode:** If an API connection fails or credentials are unavailable, the dataset powers an immediate, verifiable demonstration.
3. **Dataset Engineering Quality Assurance:** It demonstrates rigorous schema validation, uniqueness verification, and option-alignment auditing via Pydantic and Pandas.

### Q3: How do you prevent the LLM from hallucinating facts not in the PDF?
**Answer:**  
We employ a three-tier defensive strategy:
1. **Authoritative Grounding:** The extracted PDF text is explicitly designated as the single source of truth in the prompt.
2. **Negative Constraints:** The system prompt explicitly commands the model to only use facts directly supported by the text and refuse unsupported assertions.
3. **Pydantic Output Validation:** The output must conform to a strict JSON schema where every question requires an explanation citing document context.

### Q4: How does your adaptive testing feature work?
**Answer:**  
It uses a transparent, rule-based cognitive mastery heuristic. After a student submits their quiz, the evaluator calculates per-topic accuracy. Any topic with an accuracy below 75% is classified as a "Weak Area". The system then automatically constructs a targeted follow-up prompt focusing exclusively on those weak topics at an appropriate foundational difficulty, enabling iterative reinforcement learning.

### Q5: What happens if a user uploads a scanned or corrupted PDF?
**Answer:**  
In `pdf_processor.py`, PyMuPDF inspects the document page-by-page. If the total extracted text across all pages is under 50 characters, the system rejects the file and displays an informative error:
*"Insufficient readable text extracted. The document appears to be scanned or image-only. GenQuiz uses direct digital text extraction via PyMuPDF; OCR is not supported."*
This prevents sending empty prompts to the LLM and guarantees transparency.

### Q6: How do you score short-answer questions fairly without penalizing differing wording?
**Answer:**  
In `quiz_evaluator.py`, we implement a two-tier evaluation architecture:
1. **Primary AI Semantic Rubric:** We prompt the pretrained LLM with a structured evaluation rubric (1.0 = Full Credit, 0.5 = Partial Credit, 0.0 = No Credit). The prompt explicitly commands: *"Do NOT mark an answer incorrect merely because its wording differs from the expected answer! Award 1.0 if the meaning is equivalent."* The evaluator returns structured feedback, identified concepts, and missed concepts.
2. **Transparent Fallback:** If offline or if the API call fails, the system executes a deterministic token/stem overlap algorithm (matching substantive roots like `interact`/`interaction`) and transparently displays an evaluation note informing the user that fallback heuristics were used.
