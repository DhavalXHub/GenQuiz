"""
app.py - GenQuiz: Adaptive Generative AI-Based Quiz Generation and Personalized Assessment System.

Main Streamlit Application:
- PDF Text Extraction & Validation (PyMuPDF) with topic passage prioritization
- Natural Language Prompt Interpretation (count, question types, difficulty, topic)
- Seed Dataset In-Context Guidance & Grounded LLM Generation
- Strict verification and harmonization of generated quizzes (counts, types, schemas)
- Robust Semantic Short-Answer Evaluation with LLM rubric and rule-based fallback
- Guaranteed Dark Mode & Light Mode UI Readability via Native Streamlit Containers
- Rule-Based Adaptive Assessment (Weak Topic Targeting)
- Academic Dataset Explorer, Integrity Validator & Provenance Hub
- PDF Assessment Report Export (ReportLab)
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Reload local environment variables
load_dotenv(override=True)

from dataset import DatasetManager
from pdf_processor import extract_text_from_pdf, prioritize_topic_passages
from llm_service import LLMService, mask_secret
from quiz_generator import QuizGenerator, parse_user_prompt, GeneratedQuiz
from quiz_evaluator import QuizEvaluator
from export_utils import generate_assessment_pdf

# Streamlit Page Setup
st.set_page_config(
    page_title="GenQuiz — Adaptive AI Quiz & Assessment",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS focused on headers, badges, and layout without breaking input contrast in dark mode
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.4rem;
    }
    .badge-demo {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #FDE68A;
    }
    .badge-live {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #A7F3D0;
    }
    .badge-local {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #C7D2FE;
    }
    .mode-callout-live {
        background-color: rgba(16, 185, 129, 0.12);
        border-left: 5px solid #10B981;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0 20px 0;
    }
    .mode-callout-local {
        background-color: rgba(79, 70, 229, 0.12);
        border-left: 5px solid #4F46E5;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0 20px 0;
    }
    .mode-callout-demo {
        background-color: rgba(245, 158, 11, 0.12);
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "dataset_manager" not in st.session_state:
    st.session_state.dataset_manager = DatasetManager("genquiz_dataset.csv")

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}

if "extracted_pdf_data" not in st.session_state:
    st.session_state.extracted_pdf_data = None

if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = (
        "Generate 5 questions about Transformers. "
        "Include 3 MCQs and 2 short-answer questions. "
        "Use medium difficulty and provide correct answers and explanations."
    )


# ==============================================================================
# SIDEBAR: CREDENTIALS & SYSTEM STATUS
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
st.sidebar.title("GenQuiz System")
st.sidebar.markdown("**Adaptive AI Assessment Platform**")
st.sidebar.divider()

st.sidebar.subheader("⚙️ LLM Configuration")

provider_choice = st.sidebar.selectbox(
    "Target LLM Provider",
    options=[
        "Groq (Recommended - Fast & Free)",
        "OpenAI",
        "Fine-Tuned Local Model (QLoRA / PEFT)",
        "Demo Mode (Offline / Prewritten Dataset)"
    ],
    index=0
)

# API Key handling & local detection
api_key_input = ""
if "Groq" in provider_choice:
    provider_name = "groq"
    env_groq = os.getenv("GROQ_API_KEY", "").strip()
    api_key_input = st.sidebar.text_input(
        "Groq API Key",
        value=env_groq,
        type="password",
        placeholder="gsk_...",
        help="Free key available at https://console.groq.com. Stored in memory only."
    )
elif "OpenAI" in provider_choice:
    provider_name = "openai"
    env_openai = os.getenv("OPENAI_API_KEY", "").strip()
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key",
        value=env_openai,
        type="password",
        placeholder="sk-...",
        help="Enter OpenAI key. Stored in memory only."
    )
elif "Fine-Tuned Local" in provider_choice:
    provider_name = "local"
else:
    provider_name = "demo"

# Initialize LLM Service with explicit provider check
llm_service = LLMService(
    provider=provider_name,
    api_key=api_key_input if provider_name in ["groq", "openai"] else None
)
provider_info = llm_service.get_active_provider_info()

# Credential / Adapter Status Card
st.sidebar.markdown("#### 🔒 Backend & Model Status")
if provider_name == "demo":
    st.sidebar.info("💡 **Mode: Demo Mode**\nUses verified records from `genquiz_dataset.csv`. Zero API key required.")
elif provider_name == "local":
    adapter_stat = provider_info.get("adapter_status", {})
    if adapter_stat.get("is_available"):
        import torch
        device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.sidebar.success(
            f"🟢 **Fine-Tuned Adapter Detected**\n\n"
            f"- **Base Model:** `{adapter_stat.get('base_model', 'Qwen/Qwen2.5-1.5B-Instruct')}`\n"
            f"- **LoRA Path:** `{adapter_stat.get('adapter_path')}`\n"
            f"- **Inference Device:** `{device_str}`\n"
            f"- **Mode:** Zero API keys (Local PEFT)"
        )
    else:
        st.sidebar.warning(
            f"🟡 **LoRA Adapter Not Found Locally**\n\n"
            f"Expected at: `{adapter_stat.get('adapter_path', 'fine_tuned_adapter/')}`\n\n"
            "**Steps to activate:**\n"
            "1. Train adapter using `GenQuiz_FineTuning.ipynb` in Google Colab (free T4 GPU).\n"
            "2. Download `genquiz_lora_adapter.zip`.\n"
            "3. Extract to `fine_tuned_adapter/` in this project directory.\n\n"
            "*Or switch to Groq, OpenAI, or Demo Mode above.*"
        )
        with st.sidebar.expander("📖 Colab Fine-Tuning Guide"):
            st.markdown("""
            1. Open [Google Colab](https://colab.research.google.com).
            2. Upload `GenQuiz_FineTuning.ipynb` from this repo.
            3. Set Runtime to **T4 GPU**.
            4. Upload `data/train_sft.jsonl` and `data/val_sft.jsonl`.
            5. Run all cells (~4 min).
            6. Download the zip and extract to `D:\\GenQuiz\\fine_tuned_adapter\\`.
            """)
elif provider_info["is_configured"]:
    st.sidebar.success(
        f"🟢 **API Key Detected**\n\n"
        f"- **Provider:** `{provider_info['provider'].upper()}`\n"
        f"- **Model:** `{provider_info['model']}`\n"
        f"- **Masked Key:** `{provider_info['masked_key']}`\n"
        f"- **Source:** {provider_info['key_source']}"
    )
else:
    st.sidebar.warning(
        f"🔴 **No API Key Configured**\n\n"
        f"Live generation for `{provider_name.upper()}` is inactive.\n\n"
        "Enter your key above, or set in `.env`, or choose Demo Mode."
    )
    with st.sidebar.expander("🔑 How to get a free Groq Key"):
        st.markdown("""
        1. Visit **[console.groq.com/keys](https://console.groq.com/keys)**
        2. Sign up / Log in (free, no credit card required)
        3. Click **Create API Key**
        4. Paste into the **Groq API Key** box above, or save in `.env`:
           ```ini
           GROQ_API_KEY=gsk_your_key_here
           LLM_PROVIDER=groq
           ```
        """)

st.sidebar.divider()
st.sidebar.markdown("### 📊 Project Architecture")
st.sidebar.markdown("""
- **Dataset:** 50 Verified Seed Items
- **Extraction:** PyMuPDF Digital Parsing
- **Grounding:** Strict Document Constraints
- **Evaluation:** AI Semantic Rubric + Fallback
- **Adaptivity:** Cognitive Mastery Heuristics
""")


# ==============================================================================
# MAIN NAVIGATION TABS
# ==============================================================================
tab_quiz, tab_dataset, tab_about = st.tabs([
    "🎯 Quiz Generator & Assessment",
    "📚 Our Dataset (Engineering & Quality)",
    "📖 Project Methodology & Architecture"
])


# ==============================================================================
# TAB 1: QUIZ GENERATOR & ASSESSMENT
# ==============================================================================
with tab_quiz:
    st.markdown('<div class="main-title">GenQuiz — Quiz Generation & Personalized Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload lecture slides or research papers, describe your desired quiz in natural language, and receive grounded, adaptive assessments.</div>', unsafe_allow_html=True)

    # 1. Input Section: Upload & Prompt
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown("#### 1. Upload Source Document (PDF)")
        uploaded_file = st.file_uploader(
            "Select a text-based PDF file",
            type=["pdf"],
            help="PyMuPDF will extract readable text. Scanned image-only PDFs are flagged."
        )

        if uploaded_file is not None:
            with st.spinner("Extracting text via PyMuPDF..."):
                pdf_bytes = uploaded_file.read()
                pdf_result = extract_text_from_pdf(pdf_bytes)
                st.session_state.extracted_pdf_data = pdf_result

            if not pdf_result["is_valid"]:
                st.error(f"⚠️ {pdf_result['error_message']}")
            else:
                st.success(f"✅ Extracted {pdf_result['page_count']} pages ({pdf_result['final_char_count']} characters).")
                if pdf_result["truncated"]:
                    st.warning(f"ℹ️ Note: Document truncated to first {pdf_result['final_char_count']} characters to stay within model context limits.")
                
                with st.expander("📄 Document Text Preview"):
                    st.text(pdf_result["preview"])
        else:
            st.info("💡 Quick Demo Tip: You can test using the sample PDF at `D:\\GenQuiz\\sample_documents\\attention_and_transformers.pdf`.")

    with col2:
        st.markdown("#### 2. Natural-Language Instructions")
        prompt_val = st.text_area(
            "Enter your quiz specifications:",
            value=st.session_state.prompt_input,
            height=130,
            help="Specify question count, topic focus, question types (MCQ, True/False, Short Answer), and difficulty."
        )
        st.session_state.prompt_input = prompt_val

        # Prompt parser preview
        parsed_preview = parse_user_prompt(prompt_val)
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Questions", parsed_preview["count"])
        col_p2.metric("Difficulty", parsed_preview["difficulty"])
        col_p3.metric("Topic Focus", parsed_preview["topic"][:14] + ("..." if len(parsed_preview["topic"]) > 14 else ""))

        if parsed_preview.get("type_counts"):
            counts_summary = ", ".join([f"{v} {k}" for k, v in parsed_preview["type_counts"].items()])
            st.caption(f"🎯 Requested Type Breakdown: **{counts_summary}**")

        if parsed_preview["defaults_applied"]:
            st.caption(f"Applied defaults for unspecified fields: {', '.join(parsed_preview['defaults_applied'])}")

    # 2. Execution Mode Selection & Generation
    st.markdown("---")
    st.markdown("#### 3. Select Generation Engine")

    col_mode1, col_mode2 = st.columns([2, 1])
    with col_mode1:
        mode_options = [
            "🌐 Live Generation (Ground questions in PDF via Pretrained LLM)",
            "💾 Demo Mode (Use Verified Prewritten CSV Dataset Questions)"
        ]
        default_idx = 0 if provider_info["is_configured"] and provider_name != "demo" else 1
        selected_mode_label = st.radio(
            "Choose Generation Mode:",
            options=mode_options,
            index=default_idx,
            horizontal=True
        )
        is_live_requested = "Live Generation" in selected_mode_label

    with col_mode2:
        can_generate = (
            st.session_state.extracted_pdf_data is not None and 
            st.session_state.extracted_pdf_data["is_valid"]
        )
        generate_clicked = st.button(
            "🚀 Generate Quiz",
            type="primary",
            disabled=not can_generate,
            use_container_width=True
        )

    if not can_generate:
        st.info("👆 Please upload a valid text PDF above to enable quiz generation.")

    # Validation: When Live Mode is requested but backend is not configured
    if generate_clicked and can_generate and is_live_requested and not provider_info["is_configured"]:
        if provider_name == "local":
            adapter_p = provider_info.get("adapter_status", {}).get("adapter_path", "fine_tuned_adapter")
            st.error(
                f"❌ **Fine-Tuned Local Model Selected, But Adapter Weights Not Found!**\n\n"
                f"The LoRA adapter directory `{adapter_p}` is missing or does not contain `adapter_config.json`.\n\n"
                "**How to proceed:**\n"
                "1. Open `GenQuiz_FineTuning.ipynb` in Google Colab (free T4 GPU).\n"
                "2. Run all cells to train the LoRA adapter on `genquiz_dataset.csv`.\n"
                "3. Download `genquiz_lora_adapter.zip` and extract into `fine_tuned_adapter/` in this project.\n"
                "4. Or, select **💾 Demo Mode** or Groq in the sidebar to generate immediately."
            )
        else:
            st.error(
                f"❌ **Live LLM Generation Requested, But No API Key Found!**\n\n"
                f"You selected Live Generation with **{provider_name.upper()}**, but no active API key was detected in `.env` or sidebar.\n\n"
                "**How to proceed:**\n"
                "1. Enter your API key in the sidebar on the left and click Generate again.\n"
                "2. Or, select **💾 Demo Mode** above to immediately test using our verified 50-item dataset."
            )

    elif generate_clicked and can_generate:
        generator = QuizGenerator(st.session_state.dataset_manager, llm_service)
        run_force_demo = (not is_live_requested)

        with st.spinner("Analyzing document, matching topic-relevant passages, and formulating questions..."):
            try:
                generated_quiz = generator.generate_quiz(
                    pdf_text_or_dict=st.session_state.extracted_pdf_data,
                    user_prompt=st.session_state.prompt_input,
                    force_demo=run_force_demo
                )
                st.session_state.quiz = generated_quiz
                st.session_state.evaluation = None
                st.session_state.user_answers = {}
                st.success("🎉 Quiz formulated and validated successfully!")
            except Exception as ex:
                st.error(f"Generation error: {str(ex)}")

    # 3. Quiz Player Section (Native Streamlit Containers for Perfect Theme Contrast)
    if st.session_state.quiz is not None:
        quiz: GeneratedQuiz = st.session_state.quiz
        st.markdown("---")

        # Mode Distinction Callout Banner
        if quiz.is_demo_mode:
            st.markdown(f"""
            <div class="mode-callout-demo">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="font-size: 1.1rem; color: #92400E;">💾 DEMO MODE (PREWRITTEN DATASET QUESTIONS)</strong>
                    <span class="badge-demo">Prewritten CSV Records</span>
                </div>
                <p style="margin: 6px 0 0 0; color: #78350F; font-size: 0.95rem;">
                    {quiz.generation_note}
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif "Fine-Tuned" in (quiz.generation_note or quiz.source_grounding):
            st.markdown(f"""
            <div class="mode-callout-local">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="font-size: 1.1rem; color: #312E81;">⚡ FINE-TUNED OPEN-SOURCE MODEL (QLORA)</strong>
                    <span class="badge-local">Local Model Grounded</span>
                </div>
                <p style="margin: 6px 0 0 0; color: #3730A3; font-size: 0.95rem;">
                    {quiz.generation_note}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="mode-callout-live">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="font-size: 1.1rem; color: #065F46;">✨ LIVE AI GENERATION (GROUNDED IN PDF)</strong>
                    <span class="badge-live">Live LLM Grounded</span>
                </div>
                <p style="margin: 6px 0 0 0; color: #047857; font-size: 0.95rem;">
                    {quiz.generation_note}
                </p>
            </div>
            """, unsafe_allow_html=True)

        if quiz.topic_warning:
            st.warning(f"⚠️ **Topic Coverage Notice:** {quiz.topic_warning}")

        # Quiz Header Metadata
        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            st.subheader(f"📝 {quiz.title}")
            st.markdown(f"**Topic Focus:** `{quiz.topic}` &nbsp;|&nbsp; **Difficulty:** `{quiz.difficulty}` &nbsp;|&nbsp; **Questions:** `{len(quiz.questions)}`")
        with col_q2:
            st.info(f"Dataset Exemplars Used: `{', '.join(quiz.dataset_example_ids[:3])}`")

        # Assessment Display Form
        with st.form("quiz_attempt_form"):
            for i, q in enumerate(quiz.questions, 1):
                # Native container ensures 100% theme-adaptive contrast in both Dark and Light modes
                with st.container(border=True):
                    col_card_hdr1, col_card_hdr2 = st.columns([4, 1])
                    with col_card_hdr1:
                        st.markdown(f"##### Question {i} &nbsp;•&nbsp; `{q.question_type}` &nbsp;•&nbsp; *{q.difficulty}*")
                    with col_card_hdr2:
                        if q.is_saved_dataset:
                            st.caption("💾 Saved CSV Item")
                        else:
                            st.caption("🤖 AI Grounded in PDF")

                    st.markdown(f"**{q.question}**")

                    ans_key = f"ans_{q.question_id}"

                    if q.question_type == "MCQ" and q.options:
                        opts_keys = list(q.options.keys())
                        opts_labels = [f"({k}) {q.options[k]}" for k in opts_keys]
                        
                        selected_opt = st.radio(
                            label="Select your answer:",
                            options=opts_labels,
                            key=ans_key,
                            index=None
                        )
                        if selected_opt:
                            letter = selected_opt[1]  # Extract letter from "(A) ..."
                            st.session_state.user_answers[q.question_id] = letter

                    elif q.question_type == "True/False":
                        tf_choice = st.radio(
                            label="Select True or False:",
                            options=["True", "False"],
                            key=ans_key,
                            index=None
                        )
                        if tf_choice:
                            st.session_state.user_answers[q.question_id] = tf_choice

                    elif q.question_type == "Short Answer":
                        sa_text = st.text_input(
                            label="Enter your answer (concise phrase):",
                            key=ans_key,
                            placeholder="Type concise answer here..."
                        )
                        if sa_text:
                            st.session_state.user_answers[q.question_id] = sa_text

            submit_quiz = st.form_submit_button("✅ Submit Answers & Evaluate", type="primary", use_container_width=True)

        if submit_quiz:
            with st.spinner("Evaluating responses using semantic assessment rubric..."):
                eval_result = QuizEvaluator.evaluate_quiz(
                    quiz=quiz,
                    user_answers=st.session_state.user_answers,
                    llm_service=llm_service
                )
                st.session_state.evaluation = eval_result
                st.rerun()

    # 4. Evaluation & Results Section (Guaranteed High Contrast in Dark & Light Modes)
    if st.session_state.evaluation is not None:
        eval_data = st.session_state.evaluation
        st.markdown("---")
        
        pct = eval_data["percentage"]
        pts = eval_data["earned_points"]
        tot = eval_data["total_questions"]
        status_label = "Passed ✅" if eval_data["passed"] else "Needs Practice ⚠️"

        # Native Score Banner via st.container(border=True)
        with st.container(border=True):
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("Earned Score", f"{pts:.1f} / {tot:.1f} pts")
            col_b2.metric("Percentage", f"{pct:.1f}%", delta="Passing (≥60%)" if eval_data["passed"] else "Below passing", delta_color="normal" if eval_data["passed"] else "inverse")
            col_b3.metric("Assessment Status", status_label)

        # Performance breakdown tabs
        res_tab1, res_tab2 = st.tabs(["📋 Question-by-Question Solutions & Feedback", "🧠 Cognitive Mastery & Adaptive Guidance"])

        with res_tab1:
            for i, item in enumerate(eval_data["detailed_results"], 1):
                with st.container(border=True):
                    sc = item["score"]
                    if sc >= 0.8:
                        st.success(f"**Question {i}: Full Credit ({sc:.1f} / 1.0 pts)** — {item['question_type']} • *{item['topic']}*")
                    elif sc >= 0.3:
                        st.warning(f"**Question {i}: Partial Credit ({sc:.1f} / 1.0 pts)** — {item['question_type']} • *{item['topic']}*")
                    else:
                        st.error(f"**Question {i}: No Credit ({sc:.1f} / 1.0 pts)** — {item['question_type']} • *{item['topic']}*")

                    st.markdown(f"##### {item['question']}")

                    col_ans1, col_ans2 = st.columns(2)
                    with col_ans1:
                        st.markdown(f"**Your Answer:** `{item['user_answer']}`")
                    with col_ans2:
                        st.markdown(f"**Expected Answer:** `{item['correct_answer']}`")

                    # Display semantic feedback for Short Answer questions
                    if item["question_type"] == "Short Answer":
                        st.markdown(f"💬 **Semantic Feedback:** {item.get('feedback', '')}")
                        
                        id_concepts = item.get("key_concepts_identified", [])
                        miss_concepts = item.get("key_concepts_missed", [])
                        if id_concepts:
                            st.markdown(f"✅ *Key concepts identified:* {', '.join(id_concepts)}")
                        if miss_concepts:
                            st.markdown(f"⚠️ *Key concepts missed:* {', '.join(miss_concepts)}")

                    st.info(f"💡 **Document Explanation:** {item['explanation']}")
                    st.caption(f"Evaluator: {item.get('evaluator_label')} &nbsp;|&nbsp; Target Difficulty: {item['difficulty']}")

        with res_tab2:
            st.markdown("#### Topic Mastery Analysis")
            topic_df = pd.DataFrame(eval_data["topic_summary"])
            st.dataframe(topic_df, use_container_width=True, hide_index=True)

            col_ad1, col_ad2 = st.columns(2)
            with col_ad1:
                st.markdown("##### 🟢 Mastered Areas (Accuracy ≥ 75%)")
                if eval_data["strong_topics"]:
                    for t in eval_data["strong_topics"]:
                        st.markdown(f"- **{t}**")
                else:
                    st.caption("No topics reached 75% accuracy yet. Continued revision recommended.")

            with col_ad2:
                st.markdown("##### 🔴 Weak / Reinforcement Areas (Accuracy < 75%)")
                if eval_data["weak_topics"]:
                    for t in eval_data["weak_topics"]:
                        st.markdown(f"- **{t}**")
                else:
                    st.caption("Great job! No weak topic areas detected.")

            # Rule-Based Adaptive Recommendation
            adapt = eval_data["adaptive_recommendation"]
            st.markdown("---")
            st.markdown("#### 🎯 Rule-Based Adaptive Next Steps")
            st.info(f"**Cognitive Guidance:** {adapt['rationale']}")
            st.markdown(f"**Recommended Prompt:** `{adapt['suggested_prompt']}`")

            col_action1, col_action2 = st.columns([1, 1])
            with col_action1:
                if st.button("🔄 Generate Targeted Quiz on Weak Topics", type="primary", use_container_width=True):
                    st.session_state.prompt_input = adapt["suggested_prompt"]
                    st.session_state.quiz = None
                    st.session_state.evaluation = None
                    st.rerun()

            with col_action2:
                pdf_bytes = generate_assessment_pdf(st.session_state.quiz, eval_data)
                st.download_button(
                    label="📥 Download Assessment Report (PDF)",
                    data=pdf_bytes,
                    file_name="GenQuiz_Assessment_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )


# ==============================================================================
# TAB 2: OUR DATASET (ENGINEERING & QUALITY)
# ==============================================================================
with tab_dataset:
    st.markdown('<div class="main-title">GenQuiz Verified Seed Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A transparent, validated academic dataset of 50 Generative AI seed questions used to format and guide in-context LLM generation.</div>', unsafe_allow_html=True)

    dm: DatasetManager = st.session_state.dataset_manager
    report = dm.validation_report

    # Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Records", report.get("total_records", 0))
    m2.metric("Subjects", len(report.get("subject_distribution", {})))
    m3.metric("Distinct Topics", len(report.get("topic_distribution", {})))
    m4.metric("MCQs", report.get("question_type_distribution", {}).get("MCQ", 0))
    m5.metric("Integrity Check", "100% Passed" if report.get("is_valid") else "Warning")

    st.markdown("---")

    # Dataset Distribution Charts
    c_dist1, c_dist2 = st.columns(2)
    with c_dist1:
        st.markdown("##### Question Type Distribution")
        type_df = pd.DataFrame(list(report.get("question_type_distribution", {}).items()), columns=["Type", "Count"])
        st.bar_chart(type_df.set_index("Type"))

    with c_dist2:
        st.markdown("##### Difficulty Distribution")
        diff_df = pd.DataFrame(list(report.get("difficulty_distribution", {}).items()), columns=["Difficulty", "Count"])
        st.bar_chart(diff_df.set_index("Difficulty"))

    # Bloom Taxonomy Distribution
    st.markdown("##### Bloom's Taxonomy Cognitive Levels")
    bloom_df = pd.DataFrame(list(report.get("bloom_distribution", {}).items()), columns=["Cognitive Level", "Count"])
    st.bar_chart(bloom_df.set_index("Cognitive Level"))

    st.markdown("---")

    # Search & Filterable Dataset Explorer
    st.markdown("### 🔍 Searchable Dataset Explorer")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    
    with f_col1:
        topic_options = ["All"] + sorted(list(report.get("topic_distribution", {}).keys()))
        selected_topic = st.selectbox("Filter Topic", topic_options)
    
    with f_col2:
        type_options = ["All"] + sorted(list(report.get("question_type_distribution", {}).keys()))
        selected_type = st.selectbox("Filter Question Type", type_options)

    with f_col3:
        diff_options = ["All", "Easy", "Medium", "Hard"]
        selected_diff = st.selectbox("Filter Difficulty", diff_options)

    with f_col4:
        search_kw = st.text_input("Keyword Search", placeholder="e.g. transformer, attention...")

    filtered_df = dm.filter_records(
        topic=selected_topic,
        question_type=selected_type,
        difficulty=selected_diff,
        search_query=search_kw
    )

    st.caption(f"Showing {len(filtered_df)} of {len(dm.df)} verified records")
    st.dataframe(
        filtered_df[["question_id", "topic", "question", "question_type", "correct_answer", "difficulty", "bloom_level"]],
        use_container_width=True,
        hide_index=True
    )

    # Detailed Record Inspection
    with st.expander("🔬 Inspect Full Record Details (JSON / Field View)"):
        selected_qid = st.selectbox("Select Question ID to Inspect", options=filtered_df["question_id"].tolist() if not filtered_df.empty else [])
        if selected_qid:
            record_row = dm.df[dm.df["question_id"] == selected_qid].iloc[0].to_dict()
            st.json(record_row)

    # Dataset Validation & Provenance Report Download
    st.markdown("---")
    st.markdown("### 📋 Academic Integrity & Verification Report")
    
    rep_col1, rep_col2 = st.columns([2, 1])
    with rep_col1:
        st.markdown("""
        **Verification Summary:**
        - **Schema:** All 16 standardized columns validated with Pydantic.
        - **Uniqueness:** 0 duplicate question IDs, 0 duplicate question strings.
        - **MCQ Alignment:** 100% of multiple-choice answers verified to match supplied options (A, B, C, D).
        - **True/False Constraints:** Strict boolean values enforced.
        - **In-Context Guidance:** 2-3 matched records are injected into the LLM system prompt to guide schema, tone, and depth without copying content.
        """)
    with rep_col2:
        md_report = dm.generate_markdown_report()
        st.download_button(
            label="📥 Download Validation Report (.md)",
            data=md_report,
            file_name="dataset_validation_report.md",
            mime="text/markdown",
            use_container_width=True
        )
        st.download_button(
            label="📥 Download Dataset CSV",
            data=dm.df.to_csv(index=False),
            file_name="genquiz_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==============================================================================
# TAB 3: PROJECT METHODOLOGY & ARCHITECTURE
# ==============================================================================
with tab_about:
    st.markdown('<div class="main-title">Project Architecture & Methodology</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🏛️ System Architecture Workflow
    
    ```
    [Uploaded PDF Document] --------> (PyMuPDF Extraction & Topic Passage Prioritization)
                                                │
                                                ▼
    [User Natural-Language Prompt] ----> (Regex & Semantic Heuristic Parser)
                                                │
                                                ▼
    [genquiz_dataset.csv (50 Seed Qs)] -> (Dataset In-Context Retrieval: Matched Few-Shot Examples)
                                                │
                                                ▼
    [Pretrained LLM (Groq / OpenAI)] <── (Authoritative Document Grounding + Few-Shot Prompt)
                                                │
                                                ▼
                                    (Pydantic Structured JSON Validation & Harmonization)
                                                │
                                                ▼
                                    (Interactive Quiz Player - Theme Adaptive)
                                                │
                                                ▼
                        (Semantic Short-Answer Evaluator + Deterministic MCQ Scorer)
                                                │
                                                ▼
                                    (Rule-Based Adaptive Mastery & Follow-Up Quiz)
    ```

    ---

    ### 🔑 Key Architectural Decisions

    1. **PDF Grounding over Vector Databases:**
       - Rather than introducing heavyweight vector databases (ChromaDB, Pinecone) or embedding models which require separate compute or paid APIs, GenQuiz performs direct PyMuPDF text extraction with page preservation.
       - The extracted text is injected as the authoritative source of truth, constraining the LLM to test solely what is taught in the document.

    2. **Role of `genquiz_dataset.csv` (Dataset Engineering):**
       - The 50-question curated dataset serves as an academic benchmark and few-shot formatting guide.
       - Matching records are dynamically fetched via topic and question-type filters and injected into the LLM's system prompt.
       - The LLM is instructed: *"Do not copy dataset questions. Emulate the pedagogical framing and schema while formulating novel questions grounded in the PDF."*
       - When no API key is provided, the system enters **Demo Mode**, transparently serving verified dataset records.

    3. **Semantic Short-Answer Rubric Evaluation:**
       - Rather than rigid string comparison that penalizes correct answers with different wording, GenQuiz leverages LLM-assisted semantic evaluation with a structured 3-tier rubric (Full Credit 1.0, Partial Credit 0.5, No Credit 0.0).
       - When offline or if API calls fail, an automatic transparent fallback evaluates substantive token/root overlap.

    4. **Rule-Based Adaptive Assessment:**
       - Rather than a complex black-box recommendation model, GenQuiz implements an interpretable cognitive mastery heuristic.
       - If a student's accuracy on a topic falls below 75%, it is flagged as a "Weak Area".
       - The system generates an automated follow-up prompt specifically targeting weak topics with foundational difficulty and step-by-step explanations.
    """)
