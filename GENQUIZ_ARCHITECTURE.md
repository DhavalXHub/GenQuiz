# GenQuiz — System Architecture & Workflow Specification

This document provides a verified, comprehensive technical specification of the architecture, data flow, component interactions, and execution pathways of **GenQuiz — Adaptive Generative AI-Based Quiz Generation and Personalized Assessment System**.

All descriptions, diagrams, and file references reflect the verified implementation in repository `D:\GenQuiz`.

---

## 1. High-Level Architecture Overview

GenQuiz connects an educational document ingestion pipeline, natural-language prompt interpretation, multi-backend question generation (with automatic batching for local inference), strict structural and semantic validation, interactive assessment delivery, semantic evaluation, and PDF report export.

```mermaid
flowchart TD
    subgraph UI_Layer ["1. User Interface (app.py)"]
        User(["👤 Student / Instructor"])
        PDF_In["📄 PDF Upload (.pdf)"]
        Prompt_In["💬 Natural-Language Prompt Input"]
        Mode_Select["⚙️ Backend Selector: Local PEFT / Groq / OpenAI / Demo"]
        Quiz_UI["📝 Interactive Quiz Player"]
        Eval_UI["📊 Cognitive Mastery & Solution Dashboard"]
        Export_UI["📥 Download PDF Assessment Report"]
    end

    subgraph Ingestion_Layer ["2. Document Processing (pdf_processor.py)"]
        PyMuPDF["PyMuPDF (fitz) Direct Digital Extraction"]
        Topic_Expander["Semantic Topic Expander (_TOPIC_DOMAIN_SYNONYMS)"]
        Ranker["Page-by-Page Keyword & Term Density Ranker"]
        Passage_Builder["Context Window Assembler (max_chars=15000)"]
    end

    subgraph Parser_Layer ["3. Prompt Parsing & Intent (quiz_generator.py)"]
        NL_Parser["parse_user_prompt() (Regex & Intent Rules)"]
        Topic_Guard["Topic Coverage Checker & TopicMismatchError Guard"]
    end

    subgraph Data_Layer ["4. Dataset & In-Context Guidance (dataset.py, genquiz_dataset.csv)"]
        CSV_Data[("genquiz_dataset.csv (50 Verified Records)")]
        FewShot_Retriever["Few-Shot Exemplar Matcher (Topic & Type)"]
    end

    subgraph Generation_Layer ["5. Multi-Backend Generation Orchestration"]
        Router{"Backend Provider"}

        subgraph Local_Backend ["Local Open-Source Pipeline (local_model_service.py)"]
            Batch_Splitter["split_request_into_batches() (max_batch=8)"]
            Dedup_Engine["normalize_question_for_dedup() & are_questions_duplicate()"]
            Model_Cache["load_cached_peft_model() (st.cache_resource)"]
            Qwen_Base["Qwen/Qwen2.5-1.5B-Instruct (Base Causal LM)"]
            LoRA_Adapter["LoRA Adapter Weights (fine_tuned_adapter/)"]
            Inference_Loop["generate_single_question_local() (max_new=512, temp=0.4)"]
        end

        subgraph Cloud_Backend ["Cloud LLM Pipeline (llm_service.py)"]
            Groq_API["Groq API (llama-3.3-70b-versatile)"]
            OpenAI_API["OpenAI API (gpt-4o-mini)"]
        end

        subgraph Demo_Backend ["Offline Demo Mode (quiz_generator.py)"]
            Demo_Sampler["_generate_demo_quiz() (Direct CSV Retrieval)"]
        end
    end

    subgraph Validation_Layer ["6. Schema & Grounding Verification (quiz_generator.py)"]
        Pydantic_Val["Pydantic GeneratedQuestion Validator"]
        QA_Alignment["validate_question_answer_alignment()"]
        Distractor_Check["MCQ 4-Option & Non-Generic Distractor Check"]
        Domain_Collision["Cross-Domain Collision & Context Grounding Check"]
        Harmonizer["_harmonize_questions() (Type & Count Balancing)"]
    end

    subgraph Assessment_Layer ["7. Evaluation & Analytics (quiz_evaluator.py)"]
        MCQ_Eval["Deterministic Exact Option Matcher"]
        TF_Eval["Deterministic Boolean Matcher"]
        Semantic_SA["SemanticShortAnswerEvaluator (LLM Rubric + Heuristic Fallback)"]
        Topic_Analytics["Cognitive Mastery Aggregator (75% Threshold)"]
        Adaptive_Rec["Rule-Based Follow-up Quiz Prompt Generator"]
    end

    subgraph Export_Layer ["8. Document Synthesis (export_utils.py)"]
        ReportLab["ReportLab SimpleDocTemplate PDF Generator"]
    end

    %% Data flow connections
    User --> PDF_In & Prompt_In & Mode_Select
    PDF_In --> PyMuPDF --> Ranker --> Passage_Builder
    Prompt_In --> NL_Parser --> Topic_Guard
    Topic_Expander --> Ranker
    Topic_Guard -->|Topic Supported| Passage_Builder
    Topic_Guard -->|Topic Unsupported| UI_Layer

    Passage_Builder --> Router
    FewShot_Retriever --> Router
    CSV_Data --> FewShot_Retriever

    Mode_Select --> Router
    Router -->|Provider = Local| Batch_Splitter --> Model_Cache
    Model_Cache --> Qwen_Base & LoRA_Adapter --> Inference_Loop
    Dedup_Engine <--> Inference_Loop

    Router -->|Provider = Groq / OpenAI| Groq_API & OpenAI_API
    Router -->|Provider = Demo| Demo_Sampler
    CSV_Data --> Demo_Sampler

    Inference_Loop --> Pydantic_Val
    Groq_API & OpenAI_API --> Pydantic_Val
    Demo_Sampler --> Harmonizer

    Pydantic_Val --> QA_Alignment --> Distractor_Check --> Domain_Collision --> Harmonizer
    Harmonizer --> Quiz_UI

    Quiz_UI --> User
    User -->|Submit Answers| MCQ_Eval & TF_Eval & Semantic_SA
    MCQ_Eval & TF_Eval & Semantic_SA --> Topic_Analytics --> Adaptive_Rec --> Eval_UI
    Eval_UI --> Export_UI --> ReportLab
```

---

## 2. Core Execution Pathways

GenQuiz provides three strictly separated generation pathways. The system never transparently or silently falls back to external cloud APIs when local mode is selected.

```mermaid
stateDiagram-v2
    [*] --> ModeSelection

    state ModeSelection {
        [*] --> CheckProvider
        CheckProvider --> LocalPEFT : user selected 'Fine-Tuned Local Model'
        CheckProvider --> CloudLLM : user selected 'Groq' or 'OpenAI'
        CheckProvider --> DemoMode : user selected 'Demo Mode'
    }

    state LocalPEFT {
        [*] --> VerifyAdapter
        VerifyAdapter --> RaiseMissingAdapter : adapter missing
        VerifyAdapter --> CheckTopicCoverage : adapter valid
        CheckTopicCoverage --> RaiseTopicMismatch : topic not in PDF
        CheckTopicCoverage --> PlanBatches : topic covered

        state PlanBatches {
            [*] --> SplitIntoBatches : chunks of max 8
            SplitIntoBatches --> BatchLoop

            state BatchLoop {
                [*] --> LoadCachedModel : st.cache_resource
                LoadCachedModel --> GenerateSingle : temperature=0.4
                GenerateSingle --> LocalValidate : _validate_local_question
                LocalValidate --> DedupCheck : are_questions_duplicate
                DedupCheck --> AcceptQuestion : unique & valid
                DedupCheck --> RetrySingle : duplicate / invalid (max 2 retries)
                RetrySingle --> GenerateSingle
            }
        }
    }

    state CloudLLM {
        [*] --> VerifyCredentials
        VerifyCredentials --> RaiseMissingKey : no API key
        VerifyCredentials --> CloudCheckTopic : key valid
        CloudCheckTopic --> CloudTopicMismatch : topic not in PDF
        CloudCheckTopic --> BuildPrompts : topic covered
        BuildPrompts --> SendCloudAPI : Groq/OpenAI JSON mode
        SendCloudAPI --> ParseCloudJSON : Pydantic validation
    }

    state DemoMode {
        [*] --> QueryCSV : filter genquiz_dataset.csv
        QueryCSV --> MatchTopicTypes : topic & type filtering
        MatchTopicTypes --> PackageDemoQuiz : label SAVED-GAIxxx
    }
```

---

## 3. End-to-End Data Flow Pipeline

### Step 1: Document Upload & PyMuPDF Ingestion
- **File**: [`pdf_processor.py`](file:///d:/GenQuiz/pdf_processor.py)
- **Input**: Digital PDF stream or file path.
- **Process**:
  - `extract_text_from_pdf()` opens the document via `pymupdf.open()`.
  - Reads page-by-page text.
  - Scanned PDF detection: if total extracted characters $< 50$, flags document as non-selectable image/scanned PDF without OCR support.
  - Limits context to `max_chars = 15000` to prevent context window exhaustion.
- **Output**: Dictionary with `text`, `page_texts`, `page_count`, `preview`, `is_valid`.

### Step 2: Natural-Language Instruction Parsing
- **File**: [`quiz_generator.py`](file:///d:/GenQuiz/quiz_generator.py)
- **Function**: `parse_user_prompt(prompt_text)`
- **Parsing Strategy**:
  - **Difficulty**: Regex word-boundary check (`\b(medium|intermediate|moderate)\b` $\to$ `Medium`; `\b(easy|basic)\b` $\to$ `Easy`; `\b(hard|advanced)\b` $\to$ `Hard`). Explicitly prevents collision with the word *"difficulty"*.
  - **Count**: Regex matching digits before *"questions"* or after *"generate/create"*. Clamped to $[1, 30]$.
  - **Question Types**: Detects counts for MCQs, Short Answer, and True/False. If type counts are explicitly given (e.g. 9 MCQs and 4 Short Answer), total count is automatically updated to the sum ($9 + 4 = 13$).
  - **Topic Extraction**: Extracts topic phrase following *"about|on|regarding|covering"*, stripping filler phrases like *"from this pdf"*.

### Step 3: Semantic Topic Grounding & Topic Mismatch Guard
- **Files**: [`pdf_processor.py`](file:///d:/GenQuiz/pdf_processor.py), [`quiz_generator.py`](file:///d:/GenQuiz/quiz_generator.py)
- **Policy**:
  - `get_topic_related_terms(topic)` extracts substantive stems and domain synonyms via `_TOPIC_DOMAIN_SYNONYMS` (e.g., *Transformers* expands to *self-attention, multi-head, encoder, decoder*).
  - `prioritize_topic_passages()` ranks pages by keyword density.
  - **Mismatch Guard**: If total matches across all pages is 0 and mode is Live (Local or Cloud), `QuizGenerator` raises `TopicMismatchError`. The system **never** silently generates questions about an unrelated subject.

### Step 4: Multi-Batch Local Model Generation
- **File**: [`local_model_service.py`](file:///d:/GenQuiz/local_model_service.py)
- **Orchestration**:
  - `split_request_into_batches(target_count, type_counts, target_types, max_batch_size=8)` splits requests $> 8$ into safe chunks (e.g., 15 $\to$ 8 + 7; 20 $\to$ 8 + 8 + 4).
  - Preserves exact question-type ratios across batches.
  - Calls `generate_quiz_local()` sequentially with `existing_questions` passed for cross-batch deduplication.
  - Deduplication via `are_questions_duplicate()` (Jaccard token similarity $\ge 0.8$ or exact normalized text).
  - Single-question generation via `generate_single_question_local()` with prompt schema free of literal stubs, temperature $= 0.4$, and maximum 512 new tokens.

### Step 5: Question-Answer Alignment & Pydantic Validation
- **File**: [`quiz_generator.py`](file:///d:/GenQuiz/quiz_generator.py)
- **Validators**:
  - [`GeneratedQuestion`](file:///d:/GenQuiz/quiz_generator.py#L190): Rejects placeholder values (`"..."`, `"<question>"`) and questions $< 10$ characters.
  - `validate_question_answer_alignment()`:
    - **MCQs**: Exactly 4 options (A, B, C, D), distinct, non-generic (rejects *"First", "Second"*), and `correct_answer` $\in \{A, B, C, D\}$.
    - **True/False**: `correct_answer` $\in \{"True", "False"\}$.
    - **Short Answer**: Rejects cross-domain collisions (e.g., question discussing cybersecurity with an answer discussing transformer positional encoding) and ungrounded answers whose substantive tokens do not appear in the question or source passage.
  - `_harmonize_questions()`: Balances question types to match the requested distribution and assigns sequential IDs `Q1` to `QN`.

### Step 6: Interactive Quiz Attempt & Answer Collection
- **File**: [`app.py`](file:///d:/GenQuiz/app.py)
- **UI Architecture**:
  - Native Streamlit containers (`st.container(border=True)`) guarantee readable contrast across both Light Mode and Dark Mode.
  - User answers stored in `st.session_state.user_answers` keyed by `q.question_id`.

### Step 7: Dual-Mode Quiz Evaluation & Cognitive Mastery Analytics
- **File**: [`quiz_evaluator.py`](file:///d:/GenQuiz/quiz_evaluator.py)
- **Scoring Pipeline**:
  - **MCQ**: Exact option letter and normalized option text match ($1.0$ or $0.0$).
  - **True/False**: Exact boolean match ($1.0$ or $0.0$).
  - **Short Answer**: `SemanticShortAnswerEvaluator.evaluate_batch()`:
    - 3-tier scoring: Full Credit ($1.0$), Partial Credit ($0.5$), No Credit ($0.0$).
    - AI Semantic Rubric evaluates conceptual accuracy, key concepts identified, and key concepts missed.
    - Automatic transparent heuristic fallback based on substantive content-token overlap if offline or model call fails.
  - **Cognitive Mastery Analytics**:
    - Calculates topic-level accuracy.
    - Flags topics with accuracy $< 75\%$ as **Weak Areas**.
    - Generates rule-based adaptive recommendations proposing target prompts for foundational reinforcement.

### Step 8: PDF Assessment Report Export
- **File**: [`export_utils.py`](file:///d:/GenQuiz/export_utils.py)
- **Engine**: ReportLab `SimpleDocTemplate` creates printable, styled assessment reports containing student scores, cognitive mastery breakdown, solutions, and explanations.

---

## 4. Component Responsibility Matrix

| Component File | Primary Class / Functions | Responsibilities | Key Upstream | Key Downstream |
|---|---|---|---|---|
| [`app.py`](file:///d:/GenQuiz/app.py) | Streamlit UI Script | Main presentation layer, session state management, user input collection, progress alerts, dark/light contrast rendering. | User browser | All core modules |
| [`pdf_processor.py`](file:///d:/GenQuiz/pdf_processor.py) | `extract_text_from_pdf()`, `prioritize_topic_passages()`, `get_topic_related_terms()` | PyMuPDF extraction, context windowing, domain synonym expansion, page ranking by topic density. | PDF upload | `quiz_generator.py` |
| [`dataset.py`](file:///d:/GenQuiz/dataset.py) | `DatasetManager` | Loads and validates `genquiz_dataset.csv`, schema verification, few-shot exemplar retrieval, topic-matched record filtering. | `genquiz_dataset.csv` | `quiz_generator.py`, `app.py` |
| [`prepare_dataset.py`](file:///d:/GenQuiz/prepare_dataset.py) | `convert_csv_to_sft_format()`, `split_dataset_stratified()` | Validates 50 CSV rows, formats into conversational SFT JSONL with system/user/assistant turns, stratified 80/20 train/val split. | `genquiz_dataset.csv` | `data/train_sft.jsonl`, `data/val_sft.jsonl` |
| [`quiz_generator.py`](file:///d:/GenQuiz/quiz_generator.py) | `QuizGenerator`, `parse_user_prompt()`, `GeneratedQuiz`, `GeneratedQuestion`, `validate_question_answer_alignment()` | Natural language prompt interpretation, topic mismatch enforcement, multi-batch local routing, Pydantic schema validation, Q&A alignment check. | `pdf_processor.py`, `dataset.py`, `local_model_service.py`, `llm_service.py` | `app.py`, `quiz_evaluator.py` |
| [`local_model_service.py`](file:///d:/GenQuiz/local_model_service.py) | `load_cached_peft_model()`, `generate_single_question_local()`, `generate_quiz_local()`, `generate_quiz_local_batched()`, `split_request_into_batches()` | GPU/CPU loading of Qwen2.5-1.5B + LoRA adapter, single-batch inference (max 8), multi-batch orchestration (e.g. 15 or 20 questions), cross-batch deduplication. | `fine_tuned_adapter/` | `quiz_generator.py` |
| [`llm_service.py`](file:///d:/GenQuiz/llm_service.py) | `LLMService` | Cloud API interface for Groq (`llama-3.3-70b-versatile`) and OpenAI (`gpt-4o-mini`), credential detection, masked key display. | `.env` / UI key input | `quiz_generator.py`, `quiz_evaluator.py` |
| [`quiz_evaluator.py`](file:///d:/GenQuiz/quiz_evaluator.py) | `QuizEvaluator`, `SemanticShortAnswerEvaluator` | Deterministic objective evaluation, AI semantic short-answer rubric with fallback, cognitive mastery analysis, adaptive recommendations. | User answers, `GeneratedQuiz` | `app.py`, `export_utils.py` |
| [`export_utils.py`](file:///d:/GenQuiz/export_utils.py) | `generate_assessment_pdf()` | ReportLab document synthesis, score card tables, question solutions, and cognitive mastery formatting for printable export. | `GeneratedQuiz`, evaluation result | `app.py` (download button) |

---

## 5. Failure Modes & Safeguards

```mermaid
flowchart TD
    subgraph Failures ["Observed / Potential Failure Modes"]
        F1["Request exceeds 8 questions in local mode"]
        F2["Requested topic absent from PDF"]
        F3["Conflicting context causes Q&A mismatch"]
        F4["Model reproduces prompt stubs ('...', 'First')"]
        F5["Scanned or corrupted PDF uploaded"]
        F6["No API key configured for live cloud generation"]
        F7["Short answer wording differs from reference"]
    end

    subgraph Safeguards ["Implemented Architectural Safeguards"]
        S1["generate_quiz_local_batched() partitions requests into batches of <= 8"]
        S2["TopicMismatchError halts silent topic substitution; prompts user"]
        S3["validate_question_answer_alignment() detects cross-domain collision"]
        S4["Pydantic validator & _is_placeholder() reject stubs; triggers retry"]
        S5["Digital character count check (< 50 chars) flags unreadable scan"]
        S6["has_active_credentials() check blocks generation with clear guide"]
        S7["SemanticShortAnswerEvaluator applies 3-tier rubric or token fallback"]
    end

    F1 --> S1
    F2 --> S2
    F3 --> S3
    F4 --> S4
    F5 --> S5
    F6 --> S6
    F7 --> S7
```
