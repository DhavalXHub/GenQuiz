"""
tests/test_dataset.py - Unit tests for DatasetManager, Schema Validation, and Retrieval.
"""

import os
import tempfile
import pandas as pd
import pytest
from dataset import DatasetManager, REQUIRED_COLUMNS


def test_dataset_loads_successfully():
    """Verify that genquiz_dataset.csv loads and contains 50 records."""
    dm = DatasetManager("genquiz_dataset.csv")
    assert dm.df is not None
    assert len(dm.df) == 50
    assert dm.validation_report["is_valid"] is True


def test_all_required_columns_present():
    """Verify all 16 required columns exist."""
    dm = DatasetManager("genquiz_dataset.csv")
    for col in REQUIRED_COLUMNS:
        assert col in dm.df.columns


def test_no_duplicate_ids():
    """Verify zero duplicate question_ids in dataset."""
    dm = DatasetManager("genquiz_dataset.csv")
    assert dm.validation_report["duplicate_id_count"] == 0
    assert len(dm.validation_report["duplicate_ids"]) == 0


def test_mcq_answers_match_options():
    """Verify that every MCQ correct answer aligns with one of options A, B, C, D."""
    dm = DatasetManager("genquiz_dataset.csv")
    assert len(dm.validation_report["mcq_validation_errors"]) == 0


def test_invalid_csv_detection():
    """Verify that an invalid CSV produces validation errors."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("question_id,question,question_type,correct_answer\n")
        f.write("Q1,What is AI?,Short Answer,Artificial Intelligence\n")
        f.write("Q1,Duplicate Q1?,Short Answer,AI\n")
        temp_path = f.name

    try:
        invalid_dm = DatasetManager(temp_path)
        report = invalid_dm.validation_report
        assert report["is_valid"] is False
        assert len(report["missing_columns"]) > 0
        assert report["duplicate_id_count"] > 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_few_shot_retrieval():
    """Verify that get_few_shot_examples returns formatted records matching query."""
    dm = DatasetManager("genquiz_dataset.csv")
    examples = dm.get_few_shot_examples(target_topic="Transformer", limit=2)
    assert len(examples) <= 2
    assert len(examples) > 0
    assert "question_id" in examples[0]
    assert "question" in examples[0]
    assert "explanation" in examples[0]


def test_topic_stem_matching_transformers():
    """Verify that querying for 'Transformers' retrieves all related Transformer/Attention records."""
    dm = DatasetManager("genquiz_dataset.csv")
    matched = dm.get_topic_matching_records("Transformers")
    assert len(matched) >= 4
    for _, row in matched.iterrows():
        text_combo = (row["topic"] + " " + row["subtopic"] + " " + row["question"]).lower()
        assert ("transformer" in text_combo or "attention" in text_combo)


def test_unrelated_topic_does_not_falsely_return_records():
    """Verify that filtering on an unrelated topic does NOT return random records."""
    dm = DatasetManager("genquiz_dataset.csv")
    res = dm.filter_records(topic="Quantum Mechanics")
    assert len(res) == 0


def test_dataset_filtering():
    """Verify filtering by difficulty."""
    dm = DatasetManager("genquiz_dataset.csv")
    filtered = dm.filter_records(difficulty="Easy")
    assert len(filtered) > 0
    assert (filtered["difficulty"] == "Easy").all()
