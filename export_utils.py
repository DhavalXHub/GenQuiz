"""
export_utils.py - Optional PDF Assessment Report Generator using ReportLab.

Enables downloading formatted PDF reports of generated quizzes and evaluation results.
"""

import io
from typing import Optional, Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from quiz_generator import GeneratedQuiz


def generate_assessment_pdf(
    quiz: GeneratedQuiz,
    evaluation: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Renders quiz questions and assessment scores into a formatted PDF document.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=8
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=14
    )
    section_style = ParagraphStyle(
        "DocSec",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=12,
        spaceAfter=6
    )
    q_title_style = ParagraphStyle(
        "QTitle",
        parent=styles["Normal"],
        fontSize=11,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#111827"),
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "QBody",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#374151"),
        spaceAfter=3
    )
    meta_style = ParagraphStyle(
        "QMeta",
        parent=styles["Italic"],
        fontSize=9,
        textColor=colors.HexColor("#6B7280"),
        spaceAfter=6
    )

    story = []

    # Title & Header
    story.append(Paragraph("GenQuiz — Assessment & Performance Report", title_style))
    story.append(Paragraph(f"Topic: <b>{quiz.topic}</b> | Target Difficulty: <b>{quiz.difficulty}</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=12))

    # Evaluation summary block if evaluated
    if evaluation:
        pct = evaluation.get("percentage", 0.0)
        corr = evaluation.get("earned_points", evaluation.get("correct_count", 0))
        tot = evaluation.get("total_questions", 0)
        status_text = "PASSED" if evaluation.get("passed") else "NEEDS REVIEW"
        status_color = "#16A34A" if evaluation.get("passed") else "#DC2626"

        summary_data = [
            [
                Paragraph("<b>Total Score:</b>", body_style),
                Paragraph(f"{corr:.1f} / {tot:.1f} ({pct}%)", body_style),
                Paragraph("<b>Assessment Status:</b>", body_style),
                Paragraph(f"<font color='{status_color}'><b>{status_text}</b></font>", body_style)
            ]
        ]
        t = Table(summary_data, colWidths=[90, 150, 120, 150])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

        # Topic Breakdown
        weak = evaluation.get("weak_topics", [])
        strong = evaluation.get("strong_topics", [])
        if weak or strong:
            story.append(Paragraph("<b>Cognitive Domain Performance:</b>", body_style))
            if strong:
                story.append(Paragraph(f"• <b>Mastered Topics:</b> {', '.join(strong)}", meta_style))
            if weak:
                story.append(Paragraph(f"• <b>Weak / Review Topics:</b> {', '.join(weak)}", meta_style))
            story.append(Spacer(1, 8))

    story.append(Paragraph("Questions & Grounded Solutions", section_style))

    # Render questions
    details_map = {}
    if evaluation:
        details_map = {item["question_id"]: item for item in evaluation.get("detailed_results", [])}

    for i, q in enumerate(quiz.questions, 1):
        q_text = f"<b>Question {i} ({q.question_type} — {q.topic})</b>"
        story.append(Paragraph(q_text, q_title_style))
        story.append(Paragraph(q.question, body_style))

        if q.options and q.question_type == "MCQ":
            for opt_key, opt_val in q.options.items():
                story.append(Paragraph(f"<b>[{opt_key}]</b> {opt_val}", body_style))

        # Check if user result available
        detail = details_map.get(q.question_id)
        if detail:
            u_ans = detail["user_answer"]
            c_ans = detail["correct_answer"]
            score = detail.get("score", 1.0 if detail.get("is_correct") else 0.0)
            
            if score >= 0.8:
                c_label = "<font color='#16A34A'><b>Correct (1.0 pt)</b></font>"
            elif score >= 0.3:
                c_label = f"<font color='#D97706'><b>Partial Credit ({score:.1f} pts)</b></font>"
            else:
                c_label = "<font color='#DC2626'><b>Incorrect (0.0 pts)</b></font>"

            story.append(Paragraph(f"<b>Your Answer:</b> {u_ans} — {c_label}", body_style))
            story.append(Paragraph(f"<b>Expected Answer:</b> {c_ans}", body_style))
            
            feedback = detail.get("feedback")
            if feedback:
                story.append(Paragraph(f"<b>Feedback:</b> {feedback}", meta_style))
        else:
            story.append(Paragraph(f"<b>Expected Answer:</b> {q.correct_answer}", body_style))

        story.append(Paragraph(f"<b>Document Explanation:</b> {q.explanation}", meta_style))
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
