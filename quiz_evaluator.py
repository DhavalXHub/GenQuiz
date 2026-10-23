"""
quiz_evaluator.py - Scoring, Semantic Short-Answer Evaluation, and Adaptive Diagnostics for GenQuiz.

Features:
- Deterministic and exact scoring for MCQs and True/False questions
- Robust semantic evaluation for Short Answer questions via LLM with clear rubrics:
  * Full Credit (1.0): Conceptually equivalent, regardless of differing wording or synonyms
  * Partial Credit (0.5): Partial understanding or incomplete explanation
  * No Credit (0.0): Factually incorrect, irrelevant, or missing
- Transparent, reliable rule-based fallback when LLM evaluation is unavailable or fails
- Identifies key concepts addressed and missed with constructive pedagogical feedback
- Supports fractional scoring (e.g. 4.5 / 5.0) and percentage benchmarks
- Topic-level mastery tracking and rule-based adaptive recommendations
"""

import re
import string
from typing import Dict, Any, List, Optional
from quiz_generator import GeneratedQuiz, GeneratedQuestion
from llm_service import LLMService

# Standard English stopwords to exclude when analyzing concept token overlap
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def normalize_text(text: str) -> str:
    """Strip punctuation, convert hyphens/symbols to spaces, and collapse whitespace for fair matching."""
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", " ", text.lower().strip())
    return " ".join(text.split())


def evaluate_short_answer(user_ans: str, correct_ans: str) -> bool:
    """Backwards-compatible helper using the semantic fallback evaluator."""
    res = SemanticShortAnswerEvaluator._evaluate_fallback({
        "question_id": "compat",
        "question": "",
        "correct_answer": correct_ans,
        "explanation": "",
        "user_answer": user_ans
    })
    return res["score"] >= 0.5


def extract_content_tokens(text: str) -> List[str]:
    """Extract substantive keywords excluding common stopwords."""
    norm = normalize_text(text)
    tokens = [t for t in norm.split() if len(t) > 2 and t not in STOPWORDS]
    return tokens


class SemanticShortAnswerEvaluator:
    """
    Evaluates student short-answer responses using an LLM semantic rubric,
    with an automatic transparent fallback to keyword/token heuristics.
    """

    @staticmethod
    def evaluate_batch(
        items: List[Dict[str, Any]],
        llm_service: Optional[LLMService] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates a batch of short-answer questions.
        
        Args:
            items: list of dicts with:
                - question_id
                - question
                - correct_answer
                - explanation
                - user_answer
            llm_service: LLMService instance (optional)
            
        Returns:
            dict mapping question_id -> evaluation result dict
        """
        results = {}
        if not items:
            return results

        # Separate empty answers immediately
        evaluable_items = []
        for it in items:
            u_ans = str(it.get("user_answer", "")).strip()
            if not u_ans or u_ans.lower() in ["(not answered)", "none", "n/a", "no answer", "empty"]:
                results[it["question_id"]] = {
                    "score": 0.0,
                    "status": "Incorrect",
                    "feedback": "No response provided.",
                    "key_concepts_identified": [],
                    "key_concepts_missed": extract_content_tokens(it.get("correct_answer", ""))[:4],
                    "evaluator_type": "deterministic"
                }
            else:
                evaluable_items.append(it)

        if not evaluable_items:
            return results

        # Attempt LLM evaluation if service is configured
        llm_success = False
        if llm_service and llm_service.has_active_credentials():
            try:
                system_prompt = (
                    "You are an expert academic evaluator assessing student responses in an AI assessment system.\n"
                    "Evaluate each student response strictly on conceptual understanding and semantic equivalence.\n\n"
                    "EVALUATION RUBRIC:\n"
                    "- 1.0 (Correct / Full Credit): The student conveys the core principle, mechanism, or fact accurately, "
                    "even if using completely different wording, phrasing, or synonyms.\n"
                    "- 0.5 (Partial Credit): The student demonstrates partial understanding or identifies some key concepts, "
                    "but omits a crucial distinction or has a minor factual imperfection.\n"
                    "- 0.0 (Incorrect / No Credit): The response is factually incorrect, irrelevant, or misses the core concept.\n\n"
                    "CRITICAL DIRECTIVE: Do NOT mark an answer incorrect merely because its wording differs from the expected answer! "
                    "If the meaning is conceptually sound, award full credit (1.0).\n\n"
                    "Return a JSON object matching this schema:\n"
                    "{\n"
                    '  "evaluations": [\n'
                    '    {\n'
                    '      "question_id": "Q_ID",\n'
                    '      "score": 1.0,\n'
                    '      "status": "Correct" | "Partial" | "Incorrect",\n'
                    '      "feedback": "Constructive 1-2 sentence explanation of the score.",\n'
                    '      "key_concepts_identified": ["concept1", "concept2"],\n'
                    '      "key_concepts_missed": ["missing_concept"]\n'
                    '    }\n'
                    '  ]\n'
                    "}\n"
                )

                prompt_lines = ["Evaluate the following student responses against the expected solutions:"]
                for it in evaluable_items:
                    prompt_lines.append(
                        f"\n--- Item [{it['question_id']}] ---\n"
                        f"Question: {it['question']}\n"
                        f"Expected Answer: {it['correct_answer']}\n"
                        f"Context/Explanation: {it.get('explanation', '')}\n"
                        f"Student Response: {it['user_answer']}"
                    )

                user_prompt = "\n".join(prompt_lines)

                parsed_json, err = llm_service.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.1
                )

                if not err and parsed_json and "evaluations" in parsed_json:
                    for ev in parsed_json["evaluations"]:
                        qid = ev.get("question_id")
                        raw_score = float(ev.get("score", 0.0))
                        # Clamp score to [0.0, 1.0]
                        score = max(0.0, min(1.0, raw_score))
                        status = ev.get("status")
                        if not status:
                            status = "Correct" if score >= 0.8 else ("Partial" if score >= 0.4 else "Incorrect")

                        results[qid] = {
                            "score": score,
                            "status": status,
                            "feedback": ev.get("feedback", "Evaluated via AI semantic rubric."),
                            "key_concepts_identified": ev.get("key_concepts_identified", []),
                            "key_concepts_missed": ev.get("key_concepts_missed", []),
                            "evaluator_type": "llm"
                        }
                    llm_success = True
            except Exception:
                llm_success = False

        # Apply transparent fallback for any items not evaluated by LLM
        for it in evaluable_items:
            qid = it["question_id"]
            if qid not in results:
                results[qid] = SemanticShortAnswerEvaluator._evaluate_fallback(it)

        return results

    @staticmethod
    def _evaluate_fallback(item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based fallback evaluation using normalized string similarity and token overlap.
        Explicitly labeled so users and professors know fallback heuristics were used.
        """
        u_ans = str(item.get("user_answer", "")).strip()
        c_ans = str(item.get("correct_answer", "")).strip()
        exp = str(item.get("explanation", "")).strip()

        u_norm = normalize_text(u_ans)
        c_norm = normalize_text(c_ans)

        if not u_norm:
            return {
                "score": 0.0,
                "status": "Incorrect",
                "feedback": "Evaluated via fallback: No response provided.",
                "key_concepts_identified": [],
                "key_concepts_missed": extract_content_tokens(c_ans)[:4],
                "evaluator_type": "rule_based_fallback"
            }

        # Exact match
        if u_norm == c_norm:
            return {
                "score": 1.0,
                "status": "Correct",
                "feedback": "Evaluated via rule-based fallback: Exact text alignment with expected answer.",
                "key_concepts_identified": [c_ans],
                "key_concepts_missed": [],
                "evaluator_type": "rule_based_fallback"
            }

        # Substring containment
        if c_norm in u_norm or u_norm in c_norm:
            return {
                "score": 1.0,
                "status": "Correct",
                "feedback": "Evaluated via rule-based fallback: Core expected phrase is contained in student response.",
                "key_concepts_identified": [c_ans],
                "key_concepts_missed": [],
                "evaluator_type": "rule_based_fallback"
            }

        # Substantive token overlap analysis with stemming / prefix similarity
        c_raw = extract_content_tokens(c_ans) + extract_content_tokens(exp)[:6]
        u_raw = extract_content_tokens(u_ans)

        if not c_raw:
            c_raw = c_norm.split()
            u_raw = u_norm.split()

        # Match tokens directly or by substantive root prefix (e.g. interact vs interaction, token vs tokens)
        identified = []
        c_remaining = list(c_raw)
        for u_tok in u_raw:
            for c_tok in list(c_remaining):
                if u_tok == c_tok or (len(u_tok) >= 4 and len(c_tok) >= 4 and (u_tok.startswith(c_tok[:4]) or c_tok.startswith(u_tok[:4]))):
                    identified.append(c_tok)
                    c_remaining.remove(c_tok)
                    break

        unique_c = set(c_raw)
        unique_identified = set(identified)
        overlap_ratio = len(unique_identified) / len(unique_c) if unique_c else 0.0

        identified_list = list(unique_identified)
        missed_list = list(unique_c.difference(unique_identified))

        if overlap_ratio >= 0.40:
            return {
                "score": 1.0,
                "status": "Correct",
                "feedback": f"Evaluated via rule-based fallback: High conceptual term overlap ({int(overlap_ratio * 100)}%).",
                "key_concepts_identified": identified_list[:4],
                "key_concepts_missed": missed_list[:3],
                "evaluator_type": "rule_based_fallback"
            }
        elif overlap_ratio >= 0.20:
            return {
                "score": 0.5,
                "status": "Partial",
                "feedback": f"Evaluated via rule-based fallback: Partial conceptual overlap ({int(overlap_ratio * 100)}%).",
                "key_concepts_identified": identified_list[:3],
                "key_concepts_missed": missed_list[:4],
                "evaluator_type": "rule_based_fallback"
            }
        else:
            return {
                "score": 0.0,
                "status": "Incorrect",
                "feedback": "Evaluated via rule-based fallback: Key concepts from expected answer were not detected.",
                "key_concepts_identified": identified_list,
                "key_concepts_missed": missed_list[:4],
                "evaluator_type": "rule_based_fallback"
            }


class QuizEvaluator:
    """Evaluates student quiz submissions, aggregates analytics, and computes adaptive guidance."""

    @staticmethod
    def evaluate_quiz(
        quiz: GeneratedQuiz,
        user_answers: Dict[str, str],
        llm_service: Optional[LLMService] = None
    ) -> Dict[str, Any]:
        """
        Calculates scores with partial credit, compiles detailed question reports,
        and performs topic-level analytics.
        
        Args:
            quiz: GeneratedQuiz instance
            user_answers: dict mapping question_id -> user string answer
            llm_service: optional LLMService for semantic short-answer evaluation
            
        Returns:
            dict containing score summary, detailed question reports,
            topic analytics, and adaptive recommendations.
        """
        total_questions = len(quiz.questions)
        earned_points = 0.0
        detailed_results = []
        topic_stats: Dict[str, Dict[str, float]] = {}

        # Collect short answer items for batch semantic evaluation
        short_answer_items = []
        for q in quiz.questions:
            if q.question_type == "Short Answer":
                short_answer_items.append({
                    "question_id": q.question_id,
                    "question": q.question,
                    "correct_answer": q.correct_answer,
                    "explanation": q.explanation,
                    "user_answer": str(user_answers.get(q.question_id, "")).strip()
                })

        # Run semantic evaluation (LLM with rule-based fallback)
        sa_evaluations = SemanticShortAnswerEvaluator.evaluate_batch(
            items=short_answer_items,
            llm_service=llm_service
        )

        for q in quiz.questions:
            qid = q.question_id
            u_ans = str(user_answers.get(qid, "")).strip()
            c_ans = str(q.correct_answer).strip()
            qtype = q.question_type
            topic = q.topic or "General"
            
            if topic not in topic_stats:
                topic_stats[topic] = {"total": 0.0, "earned": 0.0}
            topic_stats[topic]["total"] += 1.0

            item_score = 0.0
            status = "Incorrect"
            eval_feedback = ""
            evaluator_label = "Deterministic Matching"
            identified_concepts = []
            missed_concepts = []

            # 1. Deterministic MCQ Evaluation
            if qtype == "MCQ":
                u_norm = u_ans.upper()
                c_norm = c_ans.upper()

                if u_norm == c_norm:
                    item_score = 1.0
                    status = "Correct"
                    eval_feedback = "Exact option match."
                elif q.options:
                    u_text = q.options.get(u_norm, "").strip().lower()
                    c_text = q.options.get(c_norm, c_ans).strip().lower()
                    if (u_text and c_text and u_text == c_text) or (u_ans.strip().lower() == c_text):
                        item_score = 1.0
                        status = "Correct"
                        eval_feedback = "Exact option text match."
                    else:
                        eval_feedback = f"Selected option '{u_ans}' does not match correct option '{c_ans}'."
                else:
                    eval_feedback = f"Selected answer does not match '{c_ans}'."

                evaluator_label = "Deterministic Exact Match"

            # 2. Deterministic True/False Evaluation
            elif qtype == "True/False":
                u_norm = u_ans.lower()
                c_norm = c_ans.lower()
                if u_norm in ["true", "false"] and u_norm == c_norm:
                    item_score = 1.0
                    status = "Correct"
                    eval_feedback = "Correct boolean statement."
                else:
                    eval_feedback = f"Statement is {c_ans}; you selected {u_ans or '(None)'}."
                evaluator_label = "Deterministic Boolean Match"

            # 3. Semantic Short-Answer Evaluation
            elif qtype == "Short Answer":
                sa_result = sa_evaluations.get(qid, {})
                item_score = sa_result.get("score", 0.0)
                status = sa_result.get("status", "Incorrect")
                eval_feedback = sa_result.get("feedback", "")
                identified_concepts = sa_result.get("key_concepts_identified", [])
                missed_concepts = sa_result.get("key_concepts_missed", [])
                eval_type = sa_result.get("evaluator_type", "rule_based_fallback")

                if eval_type == "llm":
                    evaluator_label = "AI Semantic Rubric"
                elif eval_type == "deterministic":
                    evaluator_label = "Deterministic (Empty Check)"
                else:
                    evaluator_label = "Rule-Based Fallback"

            earned_points += item_score
            topic_stats[topic]["earned"] += item_score

            opts_display = None
            if q.options:
                opts_display = {k: v for k, v in q.options.items()}

            detailed_results.append({
                "question_id": qid,
                "question": q.question,
                "question_type": qtype,
                "options": opts_display,
                "user_answer": u_ans if u_ans else "(Not answered)",
                "correct_answer": c_ans,
                "score": item_score,
                "max_score": 1.0,
                "status": status,
                "is_correct": item_score >= 0.8,
                "is_partial": (0.3 <= item_score < 0.8),
                "feedback": eval_feedback,
                "key_concepts_identified": identified_concepts,
                "key_concepts_missed": missed_concepts,
                "explanation": q.explanation,
                "topic": topic,
                "difficulty": q.difficulty,
                "evaluator_label": evaluator_label,
                "is_saved_dataset": q.is_saved_dataset
            })

        percentage = round((earned_points / total_questions * 100), 1) if total_questions > 0 else 0.0

        # Topic breakdown & mastery classification
        strong_topics = []
        weak_topics = []
        topic_summary = []

        for topic, counts in topic_stats.items():
            tot = counts["total"]
            pts = counts["earned"]
            acc = round((pts / tot * 100), 1) if tot > 0 else 0.0
            status_desc = "Strong" if acc >= 75.0 else ("Moderate" if acc >= 50.0 else "Needs Improvement")
            
            if acc >= 75.0:
                strong_topics.append(topic)
            else:
                weak_topics.append(topic)

            topic_summary.append({
                "topic": topic,
                "total": int(tot),
                "earned_points": round(pts, 1),
                "accuracy": acc,
                "status": status_desc
            })

        adaptive_recommendation = QuizEvaluator._generate_adaptive_guidance(
            weak_topics=weak_topics,
            percentage=percentage,
            total_questions=total_questions
        )

        return {
            "total_questions": total_questions,
            "earned_points": round(earned_points, 1),
            "correct_count": round(earned_points, 1),  # Backwards compatibility
            "percentage": percentage,
            "passed": percentage >= 60.0,
            "detailed_results": detailed_results,
            "topic_summary": topic_summary,
            "strong_topics": strong_topics,
            "weak_topics": weak_topics,
            "adaptive_recommendation": adaptive_recommendation,
            "is_demo_mode": quiz.is_demo_mode,
            "generation_note": quiz.generation_note
        }

    @staticmethod
    def _generate_adaptive_guidance(
        weak_topics: List[str],
        percentage: float,
        total_questions: int
    ) -> Dict[str, Any]:
        """Calculates rule-based adaptive recommendations for remediation."""
        if weak_topics:
            target_str = ", ".join(weak_topics[:3])
            if percentage < 60.0:
                recommended_diff = "Easy"
                rationale = (
                    f"Overall score was {percentage}%. Reinforced foundational practice is recommended "
                    f"for: {target_str}."
                )
            else:
                recommended_diff = "Medium"
                rationale = (
                    f"Good progress ({percentage}%). Targeted review on weak areas ({target_str}) "
                    "will solidify conceptual mastery."
                )
            recommended_prompt = (
                f"Generate 5 questions focusing on {target_str}. "
                f"Use {recommended_diff} difficulty with conceptual explanations."
            )
        else:
            recommended_diff = "Hard"
            rationale = (
                f"Excellent performance ({percentage}%)! All tested topics achieved high accuracy. "
                "Ready for higher-order Bloom taxonomy analysis and advanced questions."
            )
            recommended_prompt = (
                "Generate 5 advanced questions with Hard difficulty and analytical scenarios."
            )

        return {
            "recommended_difficulty": recommended_diff,
            "rationale": rationale,
            "suggested_prompt": recommended_prompt,
            "target_topics": weak_topics,
            "methodology": "Rule-based cognitive mastery heuristic (accuracy threshold = 75%)"
        }
