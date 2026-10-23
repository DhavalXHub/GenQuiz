"""
tests/test_pdf_processor.py - Unit tests for PDF extraction and passage prioritization via PyMuPDF.
"""

import pymupdf
import pytest
from pdf_processor import extract_text_from_pdf, prioritize_topic_passages


def test_text_pdf_extraction():
    """Verify that selectable text is correctly extracted from a synthetic PDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Transformers rely on multi-head attention mechanisms to capture contextual dependencies.")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = extract_text_from_pdf(pdf_bytes)
    assert result["is_valid"] is True
    assert result["page_count"] == 1
    assert "Transformers rely on multi-head attention" in result["text"]
    assert result["truncated"] is False
    assert result["error_message"] is None


def test_empty_or_scanned_pdf_handling():
    """Verify that an empty or near-empty PDF (<50 chars) returns a clear validation error."""
    doc = pymupdf.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()

    result = extract_text_from_pdf(pdf_bytes, min_chars=50)
    assert result["is_valid"] is False
    assert result["error_message"] is not None
    assert "scanned document" in result["error_message"].lower() or "insufficient" in result["error_message"].lower()


def test_large_pdf_truncation_safety():
    """Verify that text exceeding max_chars is safely truncated with notification flag."""
    doc = pymupdf.open()
    for _ in range(5):
        page = doc.new_page()
        page.insert_text((50, 72), "Self-attention calculations across deep transformer layers for language generation.")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = extract_text_from_pdf(pdf_bytes, max_chars=100)
    assert result["is_valid"] is True
    assert result["truncated"] is True
    assert len(result["text"]) == 100
    assert result["original_char_count"] > 100


def test_prioritize_topic_passages():
    """Verify topic passage ranking and relevance detection."""
    doc = pymupdf.open()
    # Page 1: General
    p1 = doc.new_page()
    p1.insert_text((50, 72), "General computer science introductory concepts and basic data structures.")
    # Page 2: Specific to Transformers
    p2 = doc.new_page()
    p2.insert_text((50, 72), "The Transformer uses multi-head self-attention and positional encodings in its encoder.")
    pdf_bytes = doc.tobytes()
    doc.close()

    extracted = extract_text_from_pdf(pdf_bytes)
    prioritized = prioritize_topic_passages(extracted, topic="Transformers")
    assert prioritized["has_topic_match"] is True
    assert prioritized["match_count"] > 0
    # Page 2 should be ranked first
    assert prioritized["matched_pages"][0] == 2
    assert prioritized["warning"] is None


def test_topic_not_in_pdf_returns_warning():
    """Verify that an unmentioned topic produces an informative warning."""
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((50, 72), "Only discussions regarding classical sorting algorithms and binary trees.")
    pdf_bytes = doc.tobytes()
    doc.close()

    extracted = extract_text_from_pdf(pdf_bytes)
    prioritized = prioritize_topic_passages(extracted, topic="Transformers")
    assert prioritized["has_topic_match"] is False
    assert prioritized["warning"] is not None
    assert "0 keyword matches" in prioritized["warning"]
