"""
test_sft_dataset.py - Unit Tests for SFT Dataset Preparation and Validation.
"""

import os
import json
import tempfile
import pandas as pd
import pytest
from prepare_dataset import (
    validate_and_convert_row,
    stratified_train_val_split,
    prepare_sft_dataset,
    SFTQuestionOutput,
    SYSTEM_PROMPT
)


def test_mcq_conversion():
    row = pd.Series({
        "question_id": "TEST_MCQ_01",
        "context": "Self-attention computes dynamic weights between tokens in a sequence.",
        "question": "What is the core mechanism of self-attention?",
        "question_type": "MCQ",
        "option_a": "Computing token-to-token compatibility weights",
        "option_b": "Recurrent state transitions across time",
        "option_c": "Convolutional pooling filters",
        "option_d": "Static lookup table lookups",
        "correct_answer": "A",
        "explanation": "Self-attention dynamically scores pairs of input positions.",
        "difficulty": "Medium",
        "bloom_level": "Understand",
        "topic": "Self-Attention",
        "source": "AI-assisted educational content; requires verification"
    })

    sft_ex, meta, warns = validate_and_convert_row(row, 0)
    assert sft_ex is not None
    assert meta is not None
    assert len(sft_ex["messages"]) == 3
    assert sft_ex["messages"][0]["role"] == "system"
    assert sft_ex["messages"][1]["role"] == "user"
    assert sft_ex["messages"][2]["role"] == "assistant"

    # Verify assistant message is valid JSON matching schema
    parsed = json.loads(sft_ex["messages"][2]["content"])
    assert parsed["question_id"] == "TEST_MCQ_01"
    assert parsed["question_type"] == "MCQ"
    assert parsed["options"]["A"] == "Computing token-to-token compatibility weights"
    assert parsed["correct_answer"] == "A"


def test_true_false_conversion():
    row = pd.Series({
        "question_id": "TEST_TF_01",
        "context": "LoRA freezes the pretrained weights and trains small low-rank adapter matrices.",
        "question": "True or False: LoRA updates all pretrained model weights during fine-tuning.",
        "question_type": "True/False",
        "correct_answer": "False",
        "explanation": "LoRA freezes the original weights and inserts trainable rank decomposition matrices.",
        "difficulty": "Easy",
        "bloom_level": "Remember",
        "topic": "LoRA",
        "source": "AI-assisted educational content; requires verification"
    })

    sft_ex, meta, warns = validate_and_convert_row(row, 1)
    assert sft_ex is not None
    parsed = json.loads(sft_ex["messages"][2]["content"])
    assert parsed["question_type"] == "True/False"
    assert parsed["correct_answer"] == "False"
    assert parsed["options"] is None


def test_short_answer_conversion():
    row = pd.Series({
        "question_id": "TEST_SA_01",
        "context": "Temperature controls the randomness of token sampling during generation.",
        "question": "How does lowering temperature affect LLM generation?",
        "question_type": "Short Answer",
        "correct_answer": "It makes generation more deterministic and focused.",
        "explanation": "Lower temperatures sharpen the probability distribution over next tokens.",
        "difficulty": "Medium",
        "bloom_level": "Apply",
        "topic": "Sampling",
        "source": "AI-assisted educational content; requires verification"
    })

    sft_ex, meta, warns = validate_and_convert_row(row, 2)
    assert sft_ex is not None
    parsed = json.loads(sft_ex["messages"][2]["content"])
    assert parsed["question_type"] == "Short Answer"
    assert parsed["correct_answer"] == "It makes generation more deterministic and focused."
    assert parsed["options"] is None


def test_malformed_mcq_rejected():
    # Missing option_d (context is > 30 chars)
    row = pd.Series({
        "question_id": "TEST_BAD_01",
        "context": "Sufficient context for testing question validity and options.",
        "question": "Which option is correct?",
        "question_type": "MCQ",
        "option_a": "Choice A",
        "option_b": "Choice B",
        "option_c": "Choice C",
        "option_d": "",
        "correct_answer": "A",
        "explanation": "Explanation text.",
        "difficulty": "Medium",
        "bloom_level": "Understand",
        "topic": "Testing"
    })

    sft_ex, meta, warns = validate_and_convert_row(row, 3)
    assert sft_ex is None
    assert any("missing one or more options" in w.lower() for w in warns)


def test_stratified_split_no_leakage():
    # Create synthetic records of multiple types
    records = []
    for i in range(25):
        records.append(({"id": f"mcq_{i}"}, {"question_id": f"mcq_{i}", "question_type": "MCQ"}))
    for i in range(10):
        records.append(({"id": f"sa_{i}"}, {"question_id": f"sa_{i}", "question_type": "Short Answer"}))
    for i in range(8):
        records.append(({"id": f"tf_{i}"}, {"question_id": f"tf_{i}", "question_type": "True/False"}))
    for i in range(7):
        records.append(({"id": f"fib_{i}"}, {"question_id": f"fib_{i}", "question_type": "Fill in the Blank"}))

    train, val = stratified_train_val_split(records, train_ratio=0.8, seed=42)

    assert len(train) == 40
    assert len(val) == 10

    # Ensure zero data leakage
    train_ids = set([m["question_id"] for _, m in train])
    val_ids = set([m["question_id"] for _, m in val])
    assert len(train_ids.intersection(val_ids)) == 0

    # Verify all 4 types exist in both train and val
    val_types = set([m["question_type"] for _, m in val])
    assert "MCQ" in val_types
    assert "Short Answer" in val_types
    assert "True/False" in val_types
    assert "Fill in the Blank" in val_types


def test_full_dataset_pipeline_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        report_dir = os.path.join(tmpdir, "reports")

        report = prepare_sft_dataset(
            csv_path="genquiz_dataset.csv",
            output_dir=data_dir,
            report_dir=report_dir
        )

        assert report["total_source_rows"] == 50
        assert report["valid_sft_records"] == 50
        assert report["train_set_size"] == 40
        assert report["val_set_size"] == 10
        assert report["data_leakage_detected"] is False

        # Verify files were created
        assert os.path.exists(os.path.join(data_dir, "train_sft.jsonl"))
        assert os.path.exists(os.path.join(data_dir, "val_sft.jsonl"))
        assert os.path.exists(os.path.join(report_dir, "sft_dataset_validation_report.md"))
        assert os.path.exists(os.path.join(report_dir, "sft_dataset_validation_report.json"))
