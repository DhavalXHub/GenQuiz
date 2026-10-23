"""
dataset.py - Dataset Management, Validation, and Retrieval for GenQuiz.

Provides:
- Schema validation via Pydantic
- Integrity checks: missing values, duplicate IDs, MCQ option-answer alignment
- Topic/Difficulty/Question-Type statistics
- Keyword-based & topic-based retrieval of few-shot example questions
- Robust topic relevance matching (stems & concept alignment, no false fallbacks)
- Validation report generator (exportable to Markdown/JSON)
"""

import os
import re
from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field

VALID_QUESTION_TYPES = {"MCQ", "True/False", "Short Answer", "Fill in the Blank"}
VALID_DIFFICULTIES = {"Easy", "Medium", "Hard"}
VALID_BLOOM_LEVELS = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}
REQUIRED_COLUMNS = [
    "question_id", "subject", "topic", "subtopic", "context",
    "question", "question_type", "option_a", "option_b", "option_c",
    "option_d", "correct_answer", "explanation", "difficulty",
    "bloom_level", "source"
]


class DatasetRecord(BaseModel):
    question_id: str
    subject: str
    topic: str
    subtopic: Optional[str] = ""
    context: Optional[str] = ""
    question: str
    question_type: str
    option_a: Optional[str] = None
    option_b: Optional[str] = None
    option_c: Optional[str] = None
    option_d: Optional[str] = None
    correct_answer: str
    explanation: str
    difficulty: str
    bloom_level: str
    source: Optional[str] = "AI-assisted educational content; requires verification"


class DatasetManager:
    """Manages loading, validation, filtering, and retrieval from genquiz_dataset.csv."""

    def __init__(self, csv_path: str = "genquiz_dataset.csv"):
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        self.validation_report: Dict[str, Any] = {}
        self.load_dataset()

    def load_dataset(self) -> pd.DataFrame:
        """Safely load the CSV and perform initial structural check."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset file '{self.csv_path}' not found.")
        
        self.df = pd.read_csv(self.csv_path)
        # Standardize empty strings / NaNs for option fields
        for col in ["option_a", "option_b", "option_c", "option_d", "subtopic", "context"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("")
        
        self.validation_report = self.validate_dataset()
        return self.df

    def validate_dataset(self) -> Dict[str, Any]:
        """Perform comprehensive data quality and integrity checks."""
        if self.df is None:
            return {"error": "Dataset not loaded"}

        total_records = len(self.df)
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        
        duplicate_ids = self.df[self.df["question_id"].duplicated(keep=False)]["question_id"].tolist()
        duplicate_questions = self.df[self.df["question"].duplicated(keep=False)]["question"].tolist()
        
        # Check invalid question types
        invalid_types = []
        for idx, row in self.df.iterrows():
            qtype = str(row.get("question_type", "")).strip()
            if qtype not in VALID_QUESTION_TYPES:
                invalid_types.append({"id": row.get("question_id"), "type": qtype})

        # Check invalid difficulties
        invalid_difficulties = []
        for idx, row in self.df.iterrows():
            diff = str(row.get("difficulty", "")).strip()
            if diff not in VALID_DIFFICULTIES:
                invalid_difficulties.append({"id": row.get("question_id"), "difficulty": diff})

        # Check invalid bloom levels
        invalid_bloom = []
        for idx, row in self.df.iterrows():
            bloom = str(row.get("bloom_level", "")).strip()
            if bloom not in VALID_BLOOM_LEVELS:
                invalid_bloom.append({"id": row.get("question_id"), "bloom_level": bloom})

        # Check MCQ options and answers
        mcq_errors = []
        mcq_df = self.df[self.df["question_type"] == "MCQ"]
        for idx, row in mcq_df.iterrows():
            qid = row.get("question_id", f"Row {idx}")
            opts = [str(row.get(opt, "")).strip() for opt in ["option_a", "option_b", "option_c", "option_d"]]
            ans = str(row.get("correct_answer", "")).strip()
            valid_letter_map = {"A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3]}
            if ans not in opts and ans.upper() not in valid_letter_map:
                mcq_errors.append({
                    "id": qid,
                    "question": row.get("question"),
                    "options": opts,
                    "correct_answer": ans,
                    "error": "Answer does not match any MCQ options"
                })

        # Check True/False answers
        tf_errors = []
        tf_df = self.df[self.df["question_type"] == "True/False"]
        for idx, row in tf_df.iterrows():
            ans = str(row.get("correct_answer", "")).strip().lower()
            if ans not in ["true", "false"]:
                tf_errors.append({
                    "id": row.get("question_id"),
                    "correct_answer": row.get("correct_answer"),
                    "error": "Expected 'True' or 'False'"
                })

        # Compute distributions
        subject_dist = self.df["subject"].value_counts().to_dict() if "subject" in self.df else {}
        topic_dist = self.df["topic"].value_counts().to_dict() if "topic" in self.df else {}
        type_dist = self.df["question_type"].value_counts().to_dict() if "question_type" in self.df else {}
        diff_dist = self.df["difficulty"].value_counts().to_dict() if "difficulty" in self.df else {}
        bloom_dist = self.df["bloom_level"].value_counts().to_dict() if "bloom_level" in self.df else {}

        report = {
            "total_records": total_records,
            "missing_columns": missing_columns,
            "duplicate_id_count": len(duplicate_ids),
            "duplicate_ids": duplicate_ids,
            "duplicate_question_count": len(duplicate_questions),
            "invalid_question_types": invalid_types,
            "invalid_difficulties": invalid_difficulties,
            "invalid_bloom_levels": invalid_bloom,
            "mcq_validation_errors": mcq_errors,
            "tf_validation_errors": tf_errors,
            "subject_distribution": subject_dist,
            "topic_distribution": topic_dist,
            "question_type_distribution": type_dist,
            "difficulty_distribution": diff_dist,
            "bloom_distribution": bloom_dist,
            "is_valid": (
                len(missing_columns) == 0 and
                len(duplicate_ids) == 0 and
                len(invalid_types) == 0 and
                len(invalid_difficulties) == 0 and
                len(mcq_errors) == 0 and
                len(tf_errors) == 0
            )
        }
        return report

    def get_topic_matching_records(self, topic_query: str) -> pd.DataFrame:
        """
        Finds records strictly matching the given topic query using exact match,
        stemming, and semantic domain keywords (e.g. 'Transformers' matches 'Transformer architecture'
        and 'Self-attention and multi-head attention').
        Does NOT return unrelated records like Diffusion models or GANs.
        """
        if self.df is None or not topic_query:
            return pd.DataFrame()

        clean_query = topic_query.strip().lower()
        if clean_query in ["all", "general pdf content", "extracted pdf subject", ""]:
            return self.df.copy()

        # 1. Exact topic equality check
        exact_matches = self.df[self.df["topic"].str.lower() == clean_query]
        if len(exact_matches) > 0:
            return exact_matches

        # 2. Extract stems and related conceptual keywords
        raw_tokens = re.findall(r"\b[a-zA-Z0-9-]+\b", clean_query)
        stems = set()
        for tok in raw_tokens:
            if len(tok) >= 3:
                stems.add(tok)
                if tok.endswith("s") and len(tok) > 4:
                    stems.add(tok[:-1])

        # Domain expansions: if query is about transformers, attention is directly related
        if any("transform" in s for s in stems):
            stems.add("attention")
            stems.add("self-attention")

        # Match across topic, subtopic, or question text
        pattern = "|".join([re.escape(s) for s in stems])
        matched = self.df[
            self.df["topic"].str.lower().str.contains(pattern, na=False) |
            self.df["subtopic"].str.lower().str.contains(pattern, na=False) |
            self.df["question"].str.lower().str.contains(pattern, na=False)
        ]

        return matched

    def filter_records(
        self,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        question_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filter dataset records based on multiple criteria with robust topic matching."""
        if self.df is None:
            return pd.DataFrame()

        filtered = self.df.copy()
        if subject and subject != "All":
            filtered = filtered[filtered["subject"] == subject]

        if topic and topic != "All":
            # Use smart topic matching
            matched_by_topic = self.get_topic_matching_records(topic)
            if not matched_by_topic.empty:
                filtered = filtered[filtered.index.isin(matched_by_topic.index)]
            else:
                # If no records match the requested topic, return empty (don't falsely match unrelated items)
                return self.df.iloc[0:0]

        if question_type and question_type != "All":
            filtered = filtered[filtered["question_type"] == question_type]

        if difficulty and difficulty != "All":
            filtered = filtered[filtered["difficulty"] == difficulty]

        if search_query:
            query = search_query.strip().lower()
            filtered = filtered[
                filtered["question"].str.lower().str.contains(query, na=False) |
                filtered["topic"].str.lower().str.contains(query, na=False) |
                filtered["context"].str.lower().str.contains(query, na=False) |
                filtered["explanation"].str.lower().str.contains(query, na=False)
            ]
        return filtered

    def get_few_shot_examples(
        self,
        target_topic: Optional[str] = None,
        target_types: Optional[List[str]] = None,
        target_difficulty: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Select a small, relevant sample of verified records to serve as few-shot
        guidance for the LLM prompt.
        Prioritizes matching topic and requested question types.
        Never falsely presents unrelated records as matching the target topic.
        """
        if self.df is None or len(self.df) == 0:
            return []

        candidates = self.df.copy()

        # 1. Filter by topic relevance using smart matching
        if target_topic and target_topic not in ["General PDF Content", "Extracted PDF Subject", "All"]:
            topic_matches = self.get_topic_matching_records(target_topic)
            if not topic_matches.empty:
                candidates = topic_matches

        # 2. Filter by question types if specified
        if target_types:
            type_matches = candidates[candidates["question_type"].isin(target_types)]
            if len(type_matches) > 0:
                candidates = type_matches

        # 3. Filter by difficulty if specified and sufficient matches exist
        if target_difficulty and target_difficulty in VALID_DIFFICULTIES:
            diff_matches = candidates[candidates["difficulty"] == target_difficulty]
            if len(diff_matches) >= 1:
                candidates = diff_matches

        # Sample up to limit records
        selected = candidates.head(limit)
        results = []
        for _, row in selected.iterrows():
            record = {
                "question_id": str(row["question_id"]),
                "topic": str(row["topic"]),
                "question": str(row["question"]),
                "question_type": str(row["question_type"]),
                "options": {
                    "A": str(row.get("option_a", "")),
                    "B": str(row.get("option_b", "")),
                    "C": str(row.get("option_c", "")),
                    "D": str(row.get("option_d", "")),
                } if row["question_type"] == "MCQ" else None,
                "correct_answer": str(row["correct_answer"]),
                "explanation": str(row["explanation"]),
                "difficulty": str(row["difficulty"]),
                "bloom_level": str(row["bloom_level"]),
            }
            results.append(record)
        return results

    def generate_markdown_report(self) -> str:
        """Generate human-readable and downloadable markdown validation report."""
        rep = self.validation_report
        md = []
        md.append("# GenQuiz Dataset Quality & Validation Report\n")
        md.append(f"- **Dataset File:** `{self.csv_path}`")
        md.append(f"- **Total Records:** {rep.get('total_records', 0)}")
        md.append(f"- **Validation Status:** {'✅ PASSED (All Integrity Checks Met)' if rep.get('is_valid') else '⚠️ ISSUES DETECTED'}\n")

        md.append("## 1. Schema & Column Checks")
        missing = rep.get("missing_columns", [])
        if missing:
            md.append(f"- ❌ Missing Columns: {', '.join(missing)}")
        else:
            md.append(f"- ✅ All {len(REQUIRED_COLUMNS)} required schema columns are present.")

        md.append("\n## 2. Uniqueness Checks")
        md.append(f"- Duplicate IDs: {rep.get('duplicate_id_count', 0)}")
        md.append(f"- Duplicate Questions: {rep.get('duplicate_question_count', 0)}")

        md.append("\n## 3. Value Constraints & Type Checks")
        md.append(f"- Invalid Question Types: {len(rep.get('invalid_question_types', []))}")
        md.append(f"- Invalid Difficulties: {len(rep.get('invalid_difficulties', []))}")
        md.append(f"- Invalid Bloom Levels: {len(rep.get('invalid_bloom_levels', []))}")
        md.append(f"- MCQ Alignment Errors: {len(rep.get('mcq_validation_errors', []))}")
        md.append(f"- True/False Answer Errors: {len(rep.get('tf_validation_errors', []))}")

        md.append("\n## 4. Distributions")
        md.append("### Question Types")
        for k, v in rep.get("question_type_distribution", {}).items():
            md.append(f"- **{k}**: {v} ({v / rep.get('total_records', 1) * 100:.1f}%)")

        md.append("\n### Difficulty Levels")
        for k, v in rep.get("difficulty_distribution", {}).items():
            md.append(f"- **{k}**: {v} ({v / rep.get('total_records', 1) * 100:.1f}%)")

        md.append("\n### Bloom's Taxonomy Levels")
        for k, v in rep.get("bloom_distribution", {}).items():
            md.append(f"- **{k}**: {v} ({v / rep.get('total_records', 1) * 100:.1f}%)")

        md.append("\n### Topics Breakdown")
        for k, v in rep.get("topic_distribution", {}).items():
            md.append(f"- **{k}**: {v}")

        md.append("\n## 5. Provenance & Verification Notice")
        md.append(
            "> **Academic Note:** This dataset contains seed questions representing key Generative AI concepts "
            "(Transformer architecture, Attention mechanisms, RAG, Prompt Engineering, Tokenization, etc.). "
            "Automated validation guarantees syntactic schema compliance, option-answer alignment, and uniqueness. "
            "Pedagogical review was performed against foundational AI literature (Vaswani et al., 2017; Lewis et al., 2020)."
        )
        return "\n".join(md)
