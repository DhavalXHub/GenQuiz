# GenQuiz — Adaptive Generative AI Quiz & Assessment System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit 1.49](https://img.shields.io/badge/Streamlit-1.49-red.svg)](https://streamlit.io/)
[![HuggingFace PEFT](https://img.shields.io/badge/PEFT-0.21-orange.svg)](https://github.com/huggingface/peft)
[![PyMuPDF 2.0](https://img.shields.io/badge/PyMuPDF-2.0-green.svg)](https://pymupdf.readthedocs.io/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-purple.svg)](https://docs.pydantic.dev/)
[![Tests: 40 Passed](https://img.shields.io/badge/Tests-40%20Passed-brightgreen.svg)]()

> **GenQuiz** is an end-to-end, document-grounded, generative assessment platform supporting both cloud LLMs (Groq, OpenAI) and a **genuine supervised fine-tuned (SFT) open-source LLM backend using LoRA/QLoRA**. Upload any lecture note or research paper PDF, enter natural-language instructions, and generate pedagogically sound quizzes strictly grounded in document text.

---

## 🚀 Key Features & Architectural Capabilities

1. **Supervised Fine-Tuned Open-Source LLM Backend (QLoRA / PEFT):**
   - Fine-tuned on `genquiz_dataset.csv` using 4-bit QLoRA and Hugging Face `trl.SFTTrainer`.
   - Target Base Model: `Qwen/Qwen2.5-1.5B-Instruct` (Un-gated, open weights, state-of-the-art JSON reasoning).
   - Local inference via `local_model_service.py` with `@st.cache_resource` memory caching for instant reruns without reloading weights.
   - Zero external API calls required when local model backend is active.

2. **Rigorous SFT Dataset Engineering (`prepare_dataset.py`):**
   - Ingests 50 seed items and validates across all 4 question types (MCQ, True/False, Short Answer, Fill in the Blank).
   - Conversational ChatML SFT format:
     $$\text{Source Context} + \text{Specifications (Topic, Type, Difficulty)} \longrightarrow \text{Structured Question JSON}$$
   - Stratified 80/20 train/validation split (40 train / 10 val) with zero data leakage.
   - Comprehensive validation report (`reports/sft_dataset_validation_report.md`).

3. **Self-Contained Google Colab Notebook (`GenQuiz_FineTuning.ipynb`):**
   - Runnable end-to-end on Google Colab's free T4 GPU (~4 min training time).
   - Exports the trained LoRA adapter directory and one-click download zip.

4. **Robust Semantic Short-Answer Evaluation:**
   - Evaluates short answers against expected solutions using an LLM-assisted 3-tier rubric (Full Credit 1.0, Partial Credit 0.5, No Credit 0.0).
   - Does NOT mark answers incorrect merely because wording differs from the reference!
   - Transparent, deterministic token/stem fallback when offline.

5. **Document Grounding & Coverage Reporting:**
   - Inspects PDF text page-by-page, prioritizes sections matching the requested topic, and displays an explicit warning if the PDF does not contain sufficient coverage of the requested topic.

6. **Deterministic MCQ & True/False Scoring:**
   - Exact, objective evaluation of Multiple Choice and True/False questions.

7. **Theme-Native WCAG Contrast:**
   - Accessible rendering across both dark mode and light mode themes.

8. **Adaptive Remediation & PDF Export:**
   - Rule-based topic mastery diagnostics, weak-topic re-quizzing, and downloadable ReportLab PDF assessment reports.

---

## 🛠️ System Architecture

```
[Uploaded Text PDF] ──> [PyMuPDF Parser] ──> [Topic Prioritizer]
                                                    │
[User Instructions] ──> [Prompt Parser]             │
                                │                   │
                                └───┬───────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             Provider-Agnostic LLM Engine               │
       ├────────────────────────────────────────────────────────┤
       │ 1. Fine-Tuned Local Model (Qwen2.5-1.5B + LoRA Adapter)│
       │ 2. Groq (Llama 3.3 70B / Llama 3.1 8B Instant)         │
       │ 3. OpenAI (GPT-4o-mini / GPT-4o)                       │
       │ 4. Demo Mode (Verified Prewritten CSV Dataset)         │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             Structured Pydantic Validator              │
       │     (Enforces Counts, Types, Options, Explanations)     │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │       Interactive Quiz Player & Semantic Evaluator     │
       │   (Semantic SA Rubric, Mastery Analytics, PDF Report)  │
       └────────────────────────────────────────────────────────┘
```

---

## 💻 Setup & Installation (Windows 11 / PowerShell)

### 1. Open PowerShell & Navigate to Project Directory
```powershell
cd D:\GenQuiz
```

### 2. Install Required Dependencies
```powershell
python -m pip install -r requirements.txt
```

---

## 🧠 Fine-Tuning Your Open-Source Model (Google Colab)

### Step 1: Generate SFT Training and Validation Data
Run `prepare_dataset.py` locally to build the stratified JSONL datasets:
```powershell
python prepare_dataset.py
```
This generates:
- `data/train_sft.jsonl` (40 examples)
- `data/val_sft.jsonl` (10 examples)
- `reports/sft_dataset_validation_report.md`

### Step 2: Open and Run `GenQuiz_FineTuning.ipynb` in Colab
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload `D:\GenQuiz\GenQuiz_FineTuning.ipynb`.
3. Set Hardware Accelerator to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
4. Run cells sequentially:
   - Cell 1: Installs `transformers`, `peft`, `accelerate`, `bitsandbytes`, `trl`, `datasets`.
   - Cell 2: Upload `data/train_sft.jsonl` and `data/val_sft.jsonl`.
   - Cell 3: Loads `Qwen/Qwen2.5-1.5B-Instruct` in 4-bit NF4.
   - Cell 4: Configures LoRA (`r=16, alpha=32`).
   - Cell 5–6: Trains for 5 epochs using `SFTTrainer` (~4 minutes).
   - Cell 7–8: Tests grounded generation and validates JSON.
   - Cell 9: Zips and downloads `genquiz_lora_adapter.zip`.

### Step 3: Deploy Adapter in GenQuiz
Extract `genquiz_lora_adapter.zip` into `D:\GenQuiz\fine_tuned_adapter\`:
```
D:\GenQuiz\fine_tuned_adapter\
  ├── adapter_config.json
  ├── adapter_model.safetensors
  └── README.md
```

Launch GenQuiz and choose **'Fine-Tuned Local Model (QLoRA / PEFT)'** in the sidebar. The system will automatically detect the adapter, load it onto your GPU/CPU, and generate questions locally without any API keys!

---

## 🔑 Cloud LLM API Configuration (Groq / OpenAI)

If using Cloud LLMs:
1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Configure `.env` with your free Groq API key from [console.groq.com/keys](https://console.groq.com/keys):
   ```ini
   GROQ_API_KEY=gsk_your_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   LLM_PROVIDER=groq
   ```
3. Or enter your key directly into the sidebar in the Streamlit UI.

---

## 🧪 Running Automated Unit Tests

Run the full pytest suite (40 unit tests):

```powershell
python -m pytest tests -v
```

Expected output:
```
tests/test_dataset.py::test_dataset_loads_successfully PASSED            [  2%]
tests/test_dataset.py::test_all_required_columns_present PASSED          [  5%]
tests/test_dataset.py::test_no_duplicate_ids PASSED                      [  7%]
tests/test_dataset.py::test_mcq_answers_match_options PASSED             [ 10%]
tests/test_dataset.py::test_invalid_csv_detection PASSED                 [ 12%]
tests/test_dataset.py::test_few_shot_retrieval PASSED                    [ 15%]
tests/test_dataset.py::test_topic_stem_matching_transformers PASSED      [ 17%]
tests/test_dataset.py::test_unrelated_topic_does_not_falsely_return_records PASSED [ 20%]
tests/test_dataset.py::test_dataset_filtering PASSED                     [ 22%]
tests/test_local_model.py::test_clean_local_json_text PASSED             [ 25%]
tests/test_local_model.py::test_adapter_status_missing PASSED            [ 27%]
tests/test_local_model.py::test_adapter_status_detected PASSED           [ 30%]
tests/test_local_model.py::test_llm_service_local_provider_initialization PASSED [ 32%]
tests/test_local_model.py::test_local_model_missing_adapter_raises_clean_error PASSED [ 35%]
tests/test_local_model.py::test_quiz_generator_local_mode_missing_adapter PASSED [ 37%]
tests/test_pdf_processor.py::test_text_pdf_extraction PASSED             [ 40%]
tests/test_pdf_processor.py::test_empty_or_scanned_pdf_handling PASSED   [ 42%]
tests/test_pdf_processor.py::test_large_pdf_truncation_safety PASSED     [ 45%]
tests/test_pdf_processor.py::test_prioritize_topic_passages PASSED       [ 47%]
tests/test_pdf_processor.py::test_topic_not_in_pdf_returns_warning PASSED [ 50%]
tests/test_quiz_evaluator.py::test_prompt_parsing PASSED                 [ 52%]
tests/test_quiz_evaluator.py::test_medium_difficulty_parsing_not_hard PASSED [ 55%]
tests/test_quiz_evaluator.py::test_missing_prompt_applies_defaults PASSED [ 57%]
tests/test_quiz_evaluator.py::test_pydantic_question_validation PASSED   [ 60%]
tests/test_quiz_evaluator.py::test_exact_deterministic_mcq_scoring PASSED [ 62%]
tests/test_quiz_evaluator.py::test_semantically_equivalent_short_answer PASSED [ 65%]
tests/test_quiz_evaluator.py::test_partially_correct_short_answer PASSED [ 67%]
tests/test_quiz_evaluator.py::test_incorrect_short_answer PASSED         [ 70%]
tests/test_quiz_evaluator.py::test_empty_short_answer PASSED             [ 72%]
tests/test_quiz_evaluator.py::test_llm_evaluation_failure_transparent_fallback PASSED [ 75%]
tests/test_quiz_evaluator.py::test_question_count_and_type_breakdown_validation PASSED [ 77%]
tests/test_quiz_evaluator.py::test_rule_based_adaptive_recommendation PASSED [ 80%]
tests/test_quiz_evaluator.py::test_demo_mode_retrieves_topic_relevant_questions PASSED [ 82%]
tests/test_quiz_evaluator.py::test_live_mode_without_key_raises_informative_error PASSED [ 85%]
tests/test_sft_dataset.py::test_mcq_conversion PASSED                    [ 87%]
tests/test_sft_dataset.py::test_true_false_conversion PASSED             [ 90%]
tests/test_sft_dataset.py::test_short_answer_conversion PASSED            [ 92%]
tests/test_sft_dataset.py::test_malformed_mcq_rejected PASSED            [ 95%]
tests/test_sft_dataset.py::test_stratified_split_no_leakage PASSED       [ 97%]
tests/test_sft_dataset.py::test_full_dataset_pipeline_execution PASSED   [100%]
============================= 40 passed in 1.95s ==============================
```

---

## 🏃‍♂️ Launching the Application

Run the Streamlit application:

```powershell
python -m streamlit run app.py
```

The browser will open to `http://localhost:8501`.
