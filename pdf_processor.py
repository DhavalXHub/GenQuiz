"""
pdf_processor.py - PDF Text Extraction and Analysis for GenQuiz using PyMuPDF.

Features:
- Extracts page-by-page text with page labels
- Validates readability (detects scanned image-only or empty PDFs)
- Handles large PDFs safely by limiting context window with explicit user notice
- Topic-aware passage prioritization: ranks pages by relevance to requested topic
- Detects if uploaded document lacks sufficient content for the requested topic
- Generates a preview snippet for UI inspection
- Never claims OCR support (explicitly warns if scanned)
"""

from typing import Dict, Any, Optional, Union, List
import io
import re
import pymupdf


def extract_text_from_pdf(
    pdf_source: Union[bytes, io.BytesIO, str],
    max_chars: int = 15000,
    min_chars: int = 50
) -> Dict[str, Any]:
    """
    Extracts text from a PDF file stream, bytes, or path using PyMuPDF.
    
    Args:
        pdf_source: File path, bytes, or BytesIO from file uploader.
        max_chars: Maximum characters to send to LLM context (prevents context blowout/cost).
        min_chars: Minimum threshold to distinguish readable text from empty/scanned PDFs.
        
    Returns:
        dict containing:
            - is_valid (bool)
            - text (str)
            - page_count (int)
            - page_texts (list of dicts with page number, text, and char count)
            - preview (str)
            - truncated (bool)
            - original_char_count (int)
            - final_char_count (int)
            - error_message (str or None)
    """
    try:
        if isinstance(pdf_source, bytes):
            doc = pymupdf.open(stream=pdf_source, filetype="pdf")
        elif isinstance(pdf_source, io.BytesIO):
            doc = pymupdf.open(stream=pdf_source.getvalue(), filetype="pdf")
        elif isinstance(pdf_source, str):
            doc = pymupdf.open(pdf_source)
        else:
            return {
                "is_valid": False,
                "text": "",
                "page_count": 0,
                "page_texts": [],
                "preview": "",
                "truncated": False,
                "original_char_count": 0,
                "final_char_count": 0,
                "error_message": "Unsupported PDF source format."
            }

        page_count = len(doc)
        if page_count == 0:
            return {
                "is_valid": False,
                "text": "",
                "page_count": 0,
                "page_texts": [],
                "preview": "",
                "truncated": False,
                "original_char_count": 0,
                "final_char_count": 0,
                "error_message": "The uploaded PDF has 0 pages or is corrupted."
            }

        extracted_pages: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        total_extracted_chars = 0

        for page_num in range(page_count):
            page = doc[page_num]
            page_text = page.get_text("text").strip()
            total_extracted_chars += len(page_text)
            extracted_pages.append({
                "page": page_num + 1,
                "text": page_text,
                "char_count": len(page_text)
            })
            if page_text:
                full_text_parts.append(f"--- [Page {page_num + 1}] ---\n{page_text}")

        doc.close()

        combined_text = "\n\n".join(full_text_parts).strip()

        # Check for scanned or unreadable PDF
        if total_extracted_chars < min_chars:
            return {
                "is_valid": False,
                "text": "",
                "page_count": page_count,
                "page_texts": extracted_pages,
                "preview": "",
                "truncated": False,
                "original_char_count": total_extracted_chars,
                "final_char_count": 0,
                "error_message": (
                    f"Insufficient readable text extracted ({total_extracted_chars} characters). "
                    "The uploaded PDF appears to be a scanned document or contains only images. "
                    "GenQuiz performs direct digital text extraction via PyMuPDF; OCR is not supported. "
                    "Please upload a standard text-based PDF."
                )
            }

        # Check for truncation if content exceeds conservative context limits
        truncated = False
        final_text = combined_text
        if len(combined_text) > max_chars:
            truncated = True
            final_text = combined_text[:max_chars]

        # Preview snippet (first 600 characters)
        preview = combined_text[:600] + ("..." if len(combined_text) > 600 else "")

        return {
            "is_valid": True,
            "text": final_text,
            "page_count": page_count,
            "page_texts": extracted_pages,
            "preview": preview,
            "truncated": truncated,
            "original_char_count": len(combined_text),
            "final_char_count": len(final_text),
            "error_message": None
        }

    except Exception as e:
        return {
            "is_valid": False,
            "text": "",
            "page_count": 0,
            "page_texts": [],
            "preview": "",
            "truncated": False,
            "original_char_count": 0,
            "final_char_count": 0,
            "error_message": f"PyMuPDF failed to process PDF: {str(e)}"
        }


# Domain-related terminology mapping for semantic topic expansion
_TOPIC_DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    "transformer": ["attention", "self-attention", "multi-head", "encoder", "decoder", "positional encoding", "bert", "gpt", "sequence-to-sequence"],
    "transformers": ["attention", "self-attention", "multi-head", "encoder", "decoder", "positional encoding", "bert", "gpt", "sequence-to-sequence"],
    "attention": ["self-attention", "multi-head", "query", "key", "value", "transformer"],
    "cybercrime": ["cyber", "malware", "phishing", "ransomware", "cybersecurity", "cyberterrorism", "fraud", "hacker", "unauthorized access", "computer crime"],
    "cybersecurity": ["cyber", "malware", "phishing", "ransomware", "cybercrime", "firewall", "encryption", "vulnerability", "intrusion", "threat"],
    "deep learning": ["neural network", "backpropagation", "gradient", "activation function", "weights", "bias", "loss function", "epoch"],
    "neural network": ["deep learning", "perceptron", "backpropagation", "neuron", "layer", "weights", "activation"],
    "diffusion": ["denoising", "score-based", "ddpm", "forward process", "reverse process", "gaussian noise"],
    "reinforcement learning": ["agent", "environment", "reward", "policy", "q-learning", "markov", "action-value"],
    "computer vision": ["convolution", "cnn", "pooling", "image classification", "object detection", "filter"],
    "nlp": ["natural language processing", "tokenizer", "token", "embedding", "corpus", "vocabulary", "language model"],
    "database": ["sql", "relational", "query", "schema", "table", "transaction", "index", "normalization"],
}

_TOPIC_STOPWORDS = frozenset([
    "about", "the", "and", "for", "with", "from", "in", "on", "of", "to", "a", "an",
    "this", "pdf", "document", "questions", "question", "quiz", "content", "covering",
    "regarding", "topic", "using", "use", "based", "give", "generate", "include"
])


def get_topic_related_terms(topic: str) -> List[str]:
    """
    Extract substantive keywords, stems, and domain synonyms for a requested topic.
    Avoids false mismatches when legitimate synonyms or related technical terms exist.
    """
    cleaned = topic.strip().lower()
    tokens = re.findall(r"\b[a-zA-Z0-9-]+\b", cleaned)
    terms = set()

    for tok in tokens:
        if len(tok) >= 3 and tok not in _TOPIC_STOPWORDS:
            terms.add(tok)
            # Add singular form if plural
            if tok.endswith("s") and len(tok) > 4:
                terms.add(tok[:-1])
            if tok.endswith("es") and len(tok) > 5:
                terms.add(tok[:-2])
            if tok.endswith("ing") and len(tok) > 5:
                terms.add(tok[:-3])

    # Check domain synonyms dictionary
    for key, syn_list in _TOPIC_DOMAIN_SYNONYMS.items():
        if key in cleaned or any(tok in key for tok in tokens if len(tok) >= 4):
            for syn in syn_list:
                terms.add(syn.lower())

    return sorted(list(terms))


def check_topic_in_text(text: str, topic: str) -> Dict[str, Any]:
    """
    Check if a topic (or its related terminology) is present in arbitrary text.
    Returns dict with 'has_match', 'match_count', 'matched_terms', and 'warning'.
    """
    cleaned_topic = topic.strip().lower()
    if cleaned_topic in ["", "general pdf content", "extracted pdf subject", "all"]:
        return {
            "has_match": True,
            "match_count": 0,
            "matched_terms": [],
            "warning": None
        }

    terms = get_topic_related_terms(topic)
    text_lower = text.lower()
    total_matches = 0
    matched_terms = []

    for term in terms:
        cnt = len(re.findall(r"\b" + re.escape(term) + r"\b", text_lower))
        if cnt > 0:
            total_matches += cnt
            matched_terms.append(term)

    if total_matches == 0:
        warning_msg = (
            f"The uploaded document does not appear to contain mentions of '{topic}' "
            "(0 keyword matches found across all pages). Questions will be grounded strictly in the available "
            "document text, but may not reflect your desired topic."
        )
        return {
            "has_match": False,
            "match_count": 0,
            "matched_terms": [],
            "warning": warning_msg
        }

    return {
        "has_match": True,
        "match_count": total_matches,
        "matched_terms": matched_terms,
        "warning": None
    }


def prioritize_topic_passages(
    pdf_result: Dict[str, Any],
    topic: str,
    max_chars: int = 15000
) -> Dict[str, Any]:
    """
    Inspects page-by-page text for keywords related to the requested topic.
    Ranks pages by topic relevance and assembles prioritized text context for the LLM.
    Also detects if the PDF lacks sufficient coverage of the requested topic.
    """
    if not pdf_result.get("is_valid") or not pdf_result.get("page_texts"):
        return {
            "prioritized_text": pdf_result.get("text", ""),
            "has_topic_match": False,
            "match_count": 0,
            "matched_pages": [],
            "warning": "PDF text is invalid or empty."
        }

    pages = pdf_result["page_texts"]
    cleaned_topic = topic.strip().lower()

    # Skip generic labels
    if cleaned_topic in ["", "general pdf content", "extracted pdf subject", "all"]:
        return {
            "prioritized_text": pdf_result["text"],
            "has_topic_match": True,
            "match_count": 0,
            "matched_pages": [p["page"] for p in pages],
            "warning": None
        }

    # Generate topic keywords & stems & domain terms
    keywords = get_topic_related_terms(topic)

    # Score each page by keyword density
    page_scores = []
    total_matches = 0
    for p in pages:
        p_text_lower = p["text"].lower()
        score = 0
        for kw in keywords:
            # Count word matches with boundary
            matches = len(re.findall(r"\b" + re.escape(kw) + r"\b", p_text_lower))
            score += matches
        total_matches += score
        page_scores.append({
            "page": p["page"],
            "text": p["text"],
            "score": score
        })

    # Sort pages by score descending (pages with most topic mentions first)
    ranked_pages = sorted(page_scores, key=lambda x: x["score"], reverse=True)
    matched_page_nums = [p["page"] for p in ranked_pages if p["score"] > 0]

    if total_matches == 0:
        warning_msg = (
            f"The uploaded document does not appear to contain mentions of '{topic}' "
            "(0 keyword matches found across all pages). Questions will be grounded strictly in the available "
            "document text, but may not reflect your desired topic."
        )
        return {
            "prioritized_text": pdf_result["text"],
            "has_topic_match": False,
            "match_count": 0,
            "matched_pages": [],
            "warning": warning_msg
        }

    # Assemble text from ranked pages first, then append other pages if room permits
    selected_parts = []
    accumulated_len = 0

    for p in ranked_pages:
        part_str = f"--- [Page {p['page']}] ---\n{p['text']}"
        if accumulated_len + len(part_str) <= max_chars:
            selected_parts.append(part_str)
            accumulated_len += len(part_str)
        else:
            # Add partial if room
            rem = max_chars - accumulated_len
            if rem > 200:
                selected_parts.append(part_str[:rem])
            break

    prioritized_text = "\n\n".join(selected_parts).strip()

    return {
        "prioritized_text": prioritized_text,
        "has_topic_match": True,
        "match_count": total_matches,
        "matched_pages": matched_page_nums,
        "warning": None
    }
