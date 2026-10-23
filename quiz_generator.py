"""
quiz_generator.py - Prompt Parsing, LLM Grounding, and Output Validation for GenQuiz.

Features:
- Precise parsing of natural language prompt instructions (count, types, type breakdown, difficulty, topic)
- Prevents substring collisions (e.g. 'medium difficulty' is correctly parsed as Medium, not Hard)
- Document grounding with topic-aware passage prioritization via PyMuPDF
- Dataset integration: retrieves relevant seed examples as in-context style/format guides
- Strict mode separation: live generation requires valid credentials; demo mode is explicit and transparent
- Topic-aligned Demo Mode: selects verified questions matching requested topic and question types
- Pydantic schema validation for generated question outputs
"""

import re
from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from dataset import DatasetManager
from llm_service import LLMService
from pdf_processor import prioritize_topic_passages


class QuestionOption(BaseModel):
    A: str
    B: str
    C: str
    D: str


class GeneratedQuestion(BaseModel):
    question_id: str
    question: str
    question_type: str  # MCQ, True/False, Short Answer
    options: Optional[Dict[str, str]] = None
    correct_answer: str
    explanation: str
    topic: str
    difficulty: str
    is_saved_dataset: bool = False

    @field_validator("question_type")
    def validate_type(cls, v):
        allowed = {"MCQ", "True/False", "Short Answer"}
        if v not in allowed:
            if "mcq" in v.lower() or "choice" in v.lower():
                return "MCQ"
            if "true" in v.lower() or "false" in v.lower():
                return "True/False"
            return "Short Answer"
        return v


class GeneratedQuiz(BaseModel):
    title: str
    topic: str
    difficulty: str
    questions: List[GeneratedQuestion]
    source_grounding: str
    dataset_example_ids: List[str] = []
    is_demo_mode: bool = False
    generation_note: Optional[str] = None
    topic_warning: Optional[str] = None
    validation_notes: List[str] = []


def parse_user_prompt(prompt_text: str) -> Dict[str, Any]:
    """
    Parses natural language user instructions and returns structured parameters
    with robust defaults.
    
    Example:
    'Generate 5 questions about Transformers. Include 3 MCQs and 2 short-answer questions.
     Use medium difficulty and provide correct answers and explanations.'
    """
    cleaned = prompt_text.lower()
    defaults_applied = []

    # 1. Difficulty: Use word boundary regex to avoid substring collisions
    # (e.g. 'difficult' inside 'difficulty' must NOT match 'medium difficulty')
    if re.search(r"\b(medium|intermediate|moderate)\b", cleaned):
        difficulty = "Medium"
    elif re.search(r"\b(easy|beginner|basic)\b", cleaned):
        difficulty = "Easy"
    elif re.search(r"\b(hard|difficult|advanced|complex)\b", cleaned):
        difficulty = "Hard"
    else:
        difficulty = "Medium"
        defaults_applied.append("difficulty=Medium")

    # 2. Total Question Count
    count = 5
    count_match = re.search(r"\b(\d{1,2})\s*(?:questions|items|queries|qs)\b", cleaned)
    if count_match:
        count = max(1, min(15, int(count_match.group(1))))
    else:
        gen_match = re.search(r"(?:generate|create|make|give me)\s+(\d{1,2})\b", cleaned)
        if gen_match:
            count = max(1, min(15, int(gen_match.group(1))))
        else:
            defaults_applied.append("count=5")

    # 3. Question Types and Specific Count Breakdowns
    type_counts: Dict[str, int] = {}
    question_types: List[str] = []

    # Detect MCQs
    m_mcq = re.search(r"(\d{1,2})\s*(?:mcq|multiple[\s-]*choice)", cleaned)
    if m_mcq:
        type_counts["MCQ"] = int(m_mcq.group(1))
        question_types.append("MCQ")
    elif "mcq" in cleaned or "multiple choice" in cleaned or "multiple-choice" in cleaned:
        question_types.append("MCQ")

    # Detect Short Answer
    m_sa = re.search(r"(\d{1,2})\s*(?:short[\s-]*answer|short)", cleaned)
    if m_sa:
        type_counts["Short Answer"] = int(m_sa.group(1))
        if "Short Answer" not in question_types:
            question_types.append("Short Answer")
    elif "short answer" in cleaned or "short-answer" in cleaned or "one word" in cleaned:
        if "Short Answer" not in question_types:
            question_types.append("Short Answer")

    # Detect True/False
    m_tf = re.search(r"(\d{1,2})\s*(?:true[\s/]*false|t/f|boolean)", cleaned)
    if m_tf:
        type_counts["True/False"] = int(m_tf.group(1))
        if "True/False" not in question_types:
            question_types.append("True/False")
    elif "true/false" in cleaned or "true false" in cleaned or "t/f" in cleaned or "boolean" in cleaned:
        if "True/False" not in question_types:
            question_types.append("True/False")

    if not question_types:
        question_types = ["MCQ", "True/False", "Short Answer"]
        defaults_applied.append("question_types=Mixed(MCQ, True/False, Short Answer)")

    # 4. Target Topic Extraction
    topic = "General PDF Content"
    topic_match = re.search(
        r"(?:about|on|regarding|covering|topic of|focus on)\s+([a-zA-Z0-9\s,-]+?)(?:\.|\n|include|using|with|and|,|from|$)",
        prompt_text,
        re.IGNORECASE
    )
    if topic_match:
        extracted = topic_match.group(1).strip()
        # Clean phrases like 'from this pdf', 'in this document'
        extracted = re.sub(r"\b(from this pdf|from pdf|in this pdf|in this document|based on this pdf)\b", "", extracted, flags=re.IGNORECASE).strip()
        if len(extracted) >= 2 and len(extracted) < 50:
            topic = extracted
    else:
        defaults_applied.append("topic=Extracted PDF Subject")

    # 5. Conceptual vs Practical Emphasis
    emphasis = "Conceptual Understanding"
    if "practical" in cleaned or "coding" in cleaned or "implementation" in cleaned:
        emphasis = "Practical & Implementation"
    elif "conceptual" in cleaned or "theory" in cleaned or "fundamentals" in cleaned:
        emphasis = "Conceptual Understanding"

    return {
        "count": count,
        "difficulty": difficulty,
        "question_types": question_types,
        "type_counts": type_counts,
        "topic": topic,
        "emphasis": emphasis,
        "defaults_applied": defaults_applied,
        "original_prompt": prompt_text
    }


class QuizGenerator:
    """Orchestrates PDF grounding, dataset in-context examples, and LLM quiz generation."""

    def __init__(self, dataset_manager: DatasetManager, llm_service: LLMService):
        self.dataset_manager = dataset_manager
        self.llm_service = llm_service

    def build_system_prompt(self, few_shot_examples: List[Dict[str, Any]]) -> str:
        """Constructs an authoritative system prompt containing dataset examples for formatting."""
        examples_str = ""
        for i, ex in enumerate(few_shot_examples, 1):
            examples_str += f"\n--- Example {i} ({ex['question_type']} - {ex['topic']}) ---\n"
            examples_str += f"Question: {ex['question']}\n"
            if ex.get("options"):
                examples_str += f"Options: {ex['options']}\n"
            examples_str += f"Correct Answer: {ex['correct_answer']}\n"
            examples_str += f"Explanation: {ex['explanation']}\n"
            examples_str += f"Difficulty: {ex['difficulty']}\n"

        system_prompt = (
            "You are GenQuiz, an expert academic Generative AI quiz generation and assessment system.\n"
            "Your task is to create high-quality, pedagogically sound assessment questions.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. GROUNDING: All questions MUST be strictly grounded in the provided PDF excerpt. Do not fabricate facts.\n"
            "2. DATASET IN-CONTEXT EXAMPLES: The examples below illustrate the required formatting, clarity, and depth. "
            "Do NOT copy example questions verbatim. Formulate brand new questions based solely on the uploaded PDF.\n"
            "3. STRUCTURED JSON OUTPUT: You MUST return a single valid JSON object strictly matching this schema:\n"
            "{\n"
            '  "title": "Quiz Title",\n'
            '  "topic": "Topic Name",\n'
            '  "difficulty": "Easy|Medium|Hard",\n'
            '  "questions": [\n'
            '    {\n'
            '      "question_id": "Q1",\n'
            '      "question": "Clear question text?",\n'
            '      "question_type": "MCQ" | "True/False" | "Short Answer",\n'
            '      "options": {"A": "...", "B": "...", "C": "...", "D": "..."} (Required for MCQ; null for others),\n'
            '      "correct_answer": "Option letter or exact short answer",\n'
            '      "explanation": "Clear educational explanation referencing the source context",\n'
            '      "topic": "Subtopic or chapter name",\n'
            '      "difficulty": "Easy" | "Medium" | "Hard"\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            f"REFERENCE EXAMPLES FROM GENQUIZ VERIFIED DATASET:\n{examples_str}\n"
        )
        return system_prompt

    def generate_quiz(
        self,
        pdf_text_or_dict: Any,
        user_prompt: str,
        force_demo: bool = False
    ) -> GeneratedQuiz:
        """
        Main generation pipeline:
        1. Parse user prompt
        2. Prioritize relevant passages from PDF
        3. Retrieve matched few-shot exemplars from CSV dataset
        4. In live mode, call LLM; in demo mode, retrieve topic-matched dataset items
        5. Validate structured schema with Pydantic
        """
        parsed_params = parse_user_prompt(user_prompt)
        target_topic = parsed_params["topic"]
        target_types = parsed_params["question_types"]
        type_counts = parsed_params.get("type_counts", {})
        target_diff = parsed_params["difficulty"]
        target_count = parsed_params["count"]

        # Prioritize passages from PDF if full pdf_result dict is provided
        topic_warning = None
        if isinstance(pdf_text_or_dict, dict) and "page_texts" in pdf_text_or_dict:
            prioritized_info = prioritize_topic_passages(pdf_text_or_dict, target_topic)
            pdf_context_text = prioritized_info["prioritized_text"]
            topic_warning = prioritized_info.get("warning")
        else:
            pdf_context_text = str(pdf_text_or_dict)

        # Retrieve relevant dataset exemplars (strictly matching topic/type)
        few_shot_examples = self.dataset_manager.get_few_shot_examples(
            target_topic=target_topic if target_topic != "General PDF Content" else None,
            target_types=target_types,
            target_difficulty=target_diff,
            limit=3
        )
        example_ids = [ex["question_id"] for ex in few_shot_examples]

        # Explicit Demo Mode check
        if force_demo:
            return self._generate_demo_quiz(parsed_params, example_ids, topic_warning)

        # If live mode requested but no API key configured (or no local adapter), raise clear error
        if not self.llm_service.has_active_credentials():
            if self.llm_service.provider == "local":
                from local_model_service import get_adapter_status
                status = get_adapter_status()
                raise RuntimeError(
                    f"LORA_ADAPTER_NOT_FOUND: The fine-tuned LoRA adapter was not found at '{status['adapter_path']}'. "
                    "Please train the model using 'GenQuiz_FineTuning.ipynb' in Google Colab (free T4 GPU), "
                    f"download 'genquiz_lora_adapter.zip', and extract it to '{status['adapter_path']}/'. "
                    "Alternatively, select Groq, OpenAI, or Demo Mode in the sidebar."
                )
            raise RuntimeError(
                "NO_API_KEY: Live LLM generation was requested, but no valid API key (GROQ_API_KEY or OPENAI_API_KEY) "
                "was found in .env or the UI input. Please enter an API key in the sidebar, or select 'Demo Mode' "
                "to evaluate with verified dataset records."
            )

        # Generate via Local Fine-Tuned Model or Cloud LLM
        if self.llm_service.provider == "local":
            from local_model_service import generate_quiz_local, get_adapter_status
            status = get_adapter_status()
            parsed_json, error_msg = generate_quiz_local(
                pdf_context=pdf_context_text,
                target_count=target_count,
                target_topic=target_topic,
                target_diff=target_diff,
                target_types=target_types,
                type_counts=type_counts,
                adapter_path=status["adapter_path"]
            )
            if error_msg or not parsed_json:
                raise RuntimeError(f"Fine-Tuned Local Model quiz generation failed: {error_msg}")
        else:
            # Build prompt instructions for Cloud LLM
            type_breakdown_str = ""
            if type_counts:
                breakdown_parts = [f"{cnt} {qtype}" for qtype, cnt in type_counts.items()]
                type_breakdown_str = f"Specific type breakdown required: {', '.join(breakdown_parts)}."

            system_prompt = self.build_system_prompt(few_shot_examples)
            user_message = (
                f"USER INSTRUCTIONS:\n{user_prompt}\n\n"
                f"PARSED REQUIREMENTS:\n"
                f"- Total Questions: {target_count}\n"
                f"- Topic Focus: {target_topic}\n"
                f"- Difficulty Level: {target_diff}\n"
                f"- Question Types: {', '.join(target_types)}\n"
                f"{type_breakdown_str}\n"
                f"- Pedagogical Emphasis: {parsed_params['emphasis']}\n\n"
                f"AUTHORITATIVE PDF CONTENT:\n{pdf_context_text}\n\n"
                f"Generate exactly {target_count} questions based strictly on the above PDF excerpt. Return valid JSON only."
            )

            parsed_json, error_msg = self.llm_service.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_message,
                temperature=0.25
            )

            if error_msg or not parsed_json:
                raise RuntimeError(f"Live LLM quiz generation failed: {error_msg}")

        # Validate with Pydantic and enforce schema
        try:
            questions_data = parsed_json.get("questions", [])
            raw_validated: List[GeneratedQuestion] = []

            for i, q in enumerate(questions_data, 1):
                qid = q.get("question_id") or f"Q{i}"
                qtype = q.get("question_type", "MCQ")

                opts = q.get("options")
                if qtype == "MCQ":
                    if isinstance(opts, list) and len(opts) >= 4:
                        opts = {"A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3]}
                    elif not isinstance(opts, dict) or len(opts) < 4:
                        # Ensure 4 complete options
                        opts = {
                            "A": str(q.get("correct_answer", "Option A")),
                            "B": "Plausible alternative distractor B",
                            "C": "Plausible alternative distractor C",
                            "D": "Plausible alternative distractor D"
                        }
                else:
                    opts = None

                ans = str(q.get("correct_answer", "")).strip()
                if qtype == "MCQ" and opts:
                    # If answer is text matching an option, convert to letter
                    for letter, text in opts.items():
                        if ans.lower() == text.lower() or ans.lower() == f"({letter.lower()})":
                            ans = letter
                            break
                    if ans.upper() in ["A", "B", "C", "D"]:
                        ans = ans.upper()
                    else:
                        ans = "A"

                elif qtype == "True/False":
                    ans = "True" if "true" in ans.lower() else "False"

                validated_q = GeneratedQuestion(
                    question_id=qid,
                    question=str(q.get("question", "")).strip(),
                    question_type=qtype,
                    options=opts,
                    correct_answer=ans,
                    explanation=str(q.get("explanation", "Grounded in uploaded document context.")).strip(),
                    topic=str(q.get("topic", target_topic)),
                    difficulty=str(q.get("difficulty", target_diff)),
                    is_saved_dataset=False
                )
                raw_validated.append(validated_q)

            # Harmonize questions to enforce exact requested count and type proportions
            validated_questions = self._harmonize_questions(
                questions=raw_validated,
                target_count=target_count,
                type_counts=type_counts,
                target_topic=target_topic,
                target_diff=target_diff
            )

            validation_notes = [
                f"Requested: {target_count} questions ({target_diff} difficulty)",
                f"Delivered: {len(validated_questions)} validated questions",
                f"Document grounding: Anchored in {len(pdf_context_text)} characters of extracted text."
            ]
            if topic_warning:
                validation_notes.append(f"Grounding Notice: {topic_warning}")

            provider_name = self.llm_service.provider.upper()
            model_name = self.llm_service.model

            if self.llm_service.provider == "local":
                source_grounding = f"Fine-Tuned Open-Source Model ({model_name} + LoRA Adapter) grounded on uploaded PDF"
                gen_note = (
                    f"✨ Live Fine-Tuned Model Generation: Formulated locally by open-source model ({model_name} + LoRA) "
                    f"strictly grounded in uploaded PDF passages without external API calls."
                )
            else:
                source_grounding = "Authoritative uploaded PDF document"
                gen_note = f"✨ Live AI Generated: Formulated by {provider_name} ({model_name}) using {len(example_ids)} dataset exemplars for in-context style guidance."

            return GeneratedQuiz(
                title=str(parsed_json.get("title", f"Quiz on {target_topic}")),
                topic=str(parsed_json.get("topic", target_topic)),
                difficulty=target_diff,
                questions=validated_questions,
                source_grounding=source_grounding,
                dataset_example_ids=example_ids,
                is_demo_mode=False,
                generation_note=gen_note,
                topic_warning=topic_warning,
                validation_notes=validation_notes
            )

        except Exception as ve:
            raise ValueError(f"Failed to validate LLM output structure: {str(ve)}")

    def _harmonize_questions(
        self,
        questions: List[GeneratedQuestion],
        target_count: int,
        type_counts: Dict[str, int],
        target_topic: str,
        target_diff: str
    ) -> List[GeneratedQuestion]:
        """
        Enforces exact question count and question type distribution.
        If the LLM under-generated or over-generated, balances types accordingly.
        """
        if not questions:
            return []

        # If specific type breakdown requested, group and select
        if type_counts:
            by_type: Dict[str, List[GeneratedQuestion]] = {}
            for q in questions:
                by_type.setdefault(q.question_type, []).append(q)

            selected = []
            for qtype, needed in type_counts.items():
                available = by_type.get(qtype, [])
                selected.extend(available[:needed])

            # Fill remaining slots up to target_count if needed
            selected_ids = {id(q) for q in selected}
            for q in questions:
                if id(q) not in selected_ids and len(selected) < target_count:
                    selected.append(q)
                    selected_ids.add(id(q))

            final_list = selected[:target_count]
        else:
            final_list = questions[:target_count]

        # Renumber question IDs sequentially (Q1, Q2, ...)
        for idx, q in enumerate(final_list, 1):
            q.question_id = f"Q{idx}"

        return final_list

    def _generate_demo_quiz(
        self,
        parsed_params: Dict[str, Any],
        example_ids: List[str],
        topic_warning: Optional[str] = None
    ) -> GeneratedQuiz:
        """
        Demo Mode: retrieves verified questions matching the requested topic and types
        directly from genquiz_dataset.csv. Explicitly labeled as saved records.
        """
        count = parsed_params["count"]
        types = parsed_params["question_types"]
        type_counts = parsed_params.get("type_counts", {})
        topic = parsed_params["topic"]
        difficulty = parsed_params["difficulty"]

        # Find topic-matched records
        topic_records = self.dataset_manager.get_topic_matching_records(topic)
        is_topic_fallback = False

        if topic_records.empty:
            # Topic has no records in our 50-item dataset
            is_topic_fallback = True
            available_records = self.dataset_manager.df.copy()
            demo_note = (
                f"ℹ️ DEMO MODE NOTICE: Our 50-item seed dataset has no saved questions matching '{topic}'. "
                "Retrieved general Generative AI records to demonstrate assessment format."
            )
        else:
            available_records = topic_records
            demo_note = (
                f"ℹ️ DEMO MODE ACTIVE: Retrieved {min(count, len(available_records))} verified saved questions "
                f"matching topic '{topic}' directly from genquiz_dataset.csv. (No live API call was made)."
            )

        # Select questions respecting type_counts if specified (e.g. 3 MCQs, 2 Short Answer)
        selected_rows = []
        if type_counts:
            for qtype, needed in type_counts.items():
                matching_type = available_records[available_records["question_type"] == qtype]
                if len(matching_type) >= needed:
                    selected_rows.append(matching_type.head(needed))
                elif not matching_type.empty:
                    selected_rows.append(matching_type)
                    # If short of questions, borrow from other question types within the same topic
                    shortage = needed - len(matching_type)
                    alt_records = available_records[~available_records.index.isin(matching_type.index)]
                    if not alt_records.empty:
                        selected_rows.append(alt_records.head(shortage))

        if selected_rows:
            sampled_df = pd.concat(selected_rows).drop_duplicates().head(count)
            # If still under count, top-up with remaining records from the same topic
            if len(sampled_df) < count:
                remaining = available_records[~available_records.index.isin(sampled_df.index)]
                if not remaining.empty:
                    sampled_df = pd.concat([sampled_df, remaining.head(count - len(sampled_df))])
        else:
            # Filter by types if specified
            if types:
                type_filtered = available_records[available_records["question_type"].isin(types)]
                if len(type_filtered) >= count:
                    available_records = type_filtered
            sampled_df = available_records.head(count)

        demo_questions: List[GeneratedQuestion] = []
        for i, (_, row) in enumerate(sampled_df.iterrows(), 1):
            qtype = str(row["question_type"])
            opts = None
            ans = str(row["correct_answer"]).strip()

            if qtype == "MCQ":
                opts = {
                    "A": str(row.get("option_a", "")),
                    "B": str(row.get("option_b", "")),
                    "C": str(row.get("option_c", "")),
                    "D": str(row.get("option_d", "")),
                }
                for letter, text in opts.items():
                    if ans.lower() == text.lower():
                        ans = letter
                        break

            demo_questions.append(
                GeneratedQuestion(
                    question_id=f"SAVED-{row['question_id']}",
                    question=str(row["question"]),
                    question_type=qtype,
                    options=opts,
                    correct_answer=ans,
                    explanation=str(row["explanation"]),
                    topic=str(row["topic"]),
                    difficulty=str(row["difficulty"]),
                    is_saved_dataset=True
                )
            )

        validation_notes = [
            f"Requested: {count} questions ({difficulty} difficulty)",
            f"Delivered: {len(demo_questions)} verified questions from genquiz_dataset.csv",
            "Mode: Offline Demo Mode (Zero external API calls made)"
        ]
        if topic_warning:
            validation_notes.append(f"Document Coverage: {topic_warning}")

        return GeneratedQuiz(
            title=f"Demo Quiz — {topic if not is_topic_fallback else 'Generative AI'} (Saved Dataset Questions)",
            topic=topic,
            difficulty=difficulty,
            questions=demo_questions,
            source_grounding="Demo Mode: Loaded from verified genquiz_dataset.csv records",
            dataset_example_ids=[row["question_id"] for _, row in sampled_df.iterrows()],
            is_demo_mode=True,
            generation_note=demo_note,
            topic_warning=topic_warning,
            validation_notes=validation_notes
        )
