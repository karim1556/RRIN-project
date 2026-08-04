"""
src/analysis/report_generator.py
==================================
Pillar 8 — AI Clinical Report Generator

Combines ALL pillar outputs into a structured clinical report:
  - Restoration summary (what the AI fixed)
  - Anomaly findings (what's wrong)
  - Biomarker panel (quantitative measurements)
  - Severity grading (ICDR level)
  - Confidence assessment
  - Clinical recommendation

Outputs JSON-serializable report data + optional PDF export.
"""

import io
import os
import numpy as np
from datetime import datetime
from typing import Optional


# ---- Report Data Structure ------------------------------------

def generate_clinical_report(
    restoration_metrics: Optional[dict] = None,
    change_analysis: Optional[dict] = None,
    anomaly_results: Optional[dict] = None,
    biomarker_results: Optional[dict] = None,
    gradcam_insight: Optional[str] = None,
    confidence_score: Optional[float] = None,
    patient_id: str = "N/A",
    patient_age: Optional[int] = None,
    eye_side: str = "N/A",
    image_quality_score: Optional[float] = None,
) -> dict:
    """
    Generate a comprehensive clinical report from all analysis pillar outputs.

    params:
        restoration_metrics — from change_analysis module
        change_analysis — full change analysis dict
        anomaly_results — from anomaly_detector module
        biomarker_results — from biomarkers module
        gradcam_insight — textual Grad-CAM interpretation
        confidence_score — overall AI confidence 0-1
        patient_id — identifier string
        patient_age — optional age for retinal age gap
        eye_side — "OS" (left), "OD" (right), or "N/A"
        image_quality_score — 0-1 quality score
    returns: dict with full structured report
    """
    report = {
        "report_id": f"RRIN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "generated_at": datetime.now().isoformat(),
        "version": "1.0.0",
        "disclaimer": (
            "This report is AI-generated and intended for screening assistance only. "
            "All findings must be verified by a qualified ophthalmologist before "
            "clinical decisions are made. This system is not a substitute for "
            "professional medical judgment."
        ),
    }

    # Patient Info
    report["patient"] = {
        "id": patient_id,
        "age": patient_age,
        "eye": eye_side,
        "exam_date": datetime.now().strftime("%Y-%m-%d"),
    }

    # Image Quality
    if image_quality_score is not None:
        quality_label = "EXCELLENT" if image_quality_score > 0.8 else \
                        "GOOD" if image_quality_score > 0.6 else \
                        "FAIR" if image_quality_score > 0.4 else "POOR"
        report["image_quality"] = {
            "score": round(image_quality_score, 2),
            "label": quality_label,
            "adequate_for_analysis": image_quality_score > 0.3,
        }
    else:
        report["image_quality"] = {
            "score": None,
            "label": "NOT ASSESSED",
            "adequate_for_analysis": True,
        }

    # Restoration Summary
    report["restoration"] = _build_restoration_section(restoration_metrics, change_analysis)

    # Anomaly Findings
    report["findings"] = _build_findings_section(anomaly_results)

    # Severity Assessment
    report["severity"] = _build_severity_section(anomaly_results)

    # Biomarker Panel
    report["biomarkers"] = _build_biomarker_section(biomarker_results, patient_age)

    # AI Explainability
    report["explainability"] = {
        "gradcam_insight": gradcam_insight or "Grad-CAM analysis not performed.",
    }

    # Confidence Assessment
    report["confidence"] = _build_confidence_section(
        anomaly_results, confidence_score
    )

    # Clinical Recommendation
    report["recommendation"] = _build_recommendation(
        anomaly_results, biomarker_results
    )

    # Narrative Summary (human-readable paragraph)
    report["narrative_summary"] = _generate_narrative(report)

    return report


# ---- Section Builders -----------------------------------------

def _build_restoration_section(
    metrics: Optional[dict],
    analysis: Optional[dict],
) -> dict:
    """Build the restoration summary section."""
    if analysis is None and metrics is None:
        return {"performed": False, "summary": "No restoration performed."}

    section = {"performed": True, "improvements": []}

    if analysis and "summary_text" in analysis:
        section["summary"] = analysis["summary_text"]
    elif metrics:
        section["summary"] = "Image restoration completed successfully."

    if analysis and "metrics" in analysis:
        m = analysis["metrics"]
        if m.get("noise_removal_percent", 0) > 5:
            section["improvements"].append({
                "type": "Noise Reduction",
                "percentage": m["noise_removal_percent"],
                "icon": "🟢",
            })
        if m.get("detail_recovery_percent", 0) > 5:
            section["improvements"].append({
                "type": "Detail Recovery",
                "percentage": m["detail_recovery_percent"],
                "icon": "🔵",
            })
        if m.get("illumination_correction_percent", 0) > 5:
            section["improvements"].append({
                "type": "Illumination Correction",
                "percentage": m["illumination_correction_percent"],
                "icon": "🟡",
            })
        if m.get("artifact_removal_percent", 0) > 3:
            section["improvements"].append({
                "type": "Artifact Removal",
                "percentage": m["artifact_removal_percent"],
                "icon": "🟠",
            })

        section["overall_change"] = m.get("change_coverage_percent", 0)
        section["structural_similarity"] = m.get("mean_structural_similarity", 0)

    return section


def _build_findings_section(anomaly_results: Optional[dict]) -> dict:
    """Build the anomaly findings section."""
    if anomaly_results is None:
        return {"analyzed": False, "findings": [], "total_count": 0}

    findings = anomaly_results.get("findings", [])
    counts = anomaly_results.get("finding_counts", {})
    optic_disc = anomaly_results.get("optic_disc", {})

    # Group findings by type with clinical descriptions
    finding_groups = {}
    type_info = {
        "microaneurysm": {
            "clinical_name": "Microaneurysms",
            "significance": "Earliest sign of diabetic retinopathy",
            "icon": "🔴",
        },
        "hard_exudate": {
            "clinical_name": "Hard Exudates",
            "significance": "Lipid deposits indicating vascular leakage",
            "icon": "🟡",
        },
        "hemorrhage": {
            "clinical_name": "Retinal Hemorrhages",
            "significance": "Bleeding from damaged blood vessels",
            "icon": "🟤",
        },
        "cotton_wool_spot": {
            "clinical_name": "Cotton Wool Spots",
            "significance": "Nerve fiber layer infarcts indicating ischemia",
            "icon": "⚪",
        },
        "neovascularization": {
            "clinical_name": "Neovascularization",
            "significance": "Abnormal new vessel growth — proliferative disease",
            "icon": "🟣",
        },
    }

    for finding in findings:
        ftype = finding.get("finding_type", "unknown")
        if ftype not in finding_groups:
            info = type_info.get(ftype, {"clinical_name": ftype, "significance": "", "icon": "⚪"})
            finding_groups[ftype] = {
                "type": ftype,
                "clinical_name": info["clinical_name"],
                "significance": info["significance"],
                "icon": info["icon"],
                "count": 0,
                "locations": [],
                "mean_confidence": 0,
            }
        group = finding_groups[ftype]
        group["count"] += 1
        group["locations"].append(finding.get("quadrant", "unknown"))
        group["mean_confidence"] += finding.get("confidence", 0)

    for group in finding_groups.values():
        if group["count"] > 0:
            group["mean_confidence"] = round(group["mean_confidence"] / group["count"], 2)
        group["locations"] = list(set(group["locations"]))

    # Optic disc assessment
    optic_disc_assessment = "Not detected"
    if optic_disc.get("detected"):
        cdr = optic_disc.get("cup_to_disc_ratio", 0)
        if cdr > 0.6:
            optic_disc_assessment = f"⚠️ Elevated CDR ({cdr}) — glaucoma evaluation recommended"
        elif cdr > 0.4:
            optic_disc_assessment = f"Borderline CDR ({cdr}) — monitor"
        else:
            optic_disc_assessment = f"✅ Normal CDR ({cdr})"

    return {
        "analyzed": True,
        "finding_groups": list(finding_groups.values()),
        "total_count": counts.get("total", len(findings)),
        "optic_disc": optic_disc_assessment,
        "optic_disc_details": optic_disc,
    }


def _build_severity_section(anomaly_results: Optional[dict]) -> dict:
    """Build the severity assessment section."""
    if anomaly_results is None:
        return {"assessed": False}

    severity = anomaly_results.get("severity", {})
    return {
        "assessed": True,
        "icdr_level": severity.get("level", 0),
        "label": severity.get("label", "Not Assessed"),
        "confidence": severity.get("confidence", 0),
        "description": severity.get("description", ""),
    }


def _build_biomarker_section(
    biomarker_results: Optional[dict],
    patient_age: Optional[int],
) -> dict:
    """Build the biomarker panel section."""
    if biomarker_results is None:
        return {"analyzed": False}

    markers = []

    # Tortuosity
    tort = biomarker_results.get("tortuosity", {})
    markers.append({
        "name": "Vessel Tortuosity",
        "value": tort.get("mean_tortuosity", "N/A"),
        "unit": "ratio",
        "normal_range": tort.get("normal_range", [1.05, 1.20]),
        "interpretation": tort.get("clinical_interpretation", ""),
        "in_range": _is_in_range(tort.get("mean_tortuosity"), tort.get("normal_range", [1.05, 1.20])),
    })

    # Fractal Dimension
    fd = biomarker_results.get("fractal_dimension", {})
    markers.append({
        "name": "Fractal Dimension",
        "value": fd.get("fractal_dimension", "N/A"),
        "unit": "D",
        "normal_range": fd.get("normal_range", [1.40, 1.70]),
        "interpretation": fd.get("clinical_interpretation", ""),
        "in_range": _is_in_range(fd.get("fractal_dimension"), fd.get("normal_range", [1.40, 1.70])),
    })

    # AVR
    avr = biomarker_results.get("arteriovenous_ratio", {})
    markers.append({
        "name": "Arteriovenous Ratio",
        "value": avr.get("avr", "N/A"),
        "unit": "",
        "normal_range": avr.get("normal_range", [0.67, 0.80]),
        "interpretation": avr.get("clinical_interpretation", ""),
        "in_range": _is_in_range(avr.get("avr"), avr.get("normal_range", [0.67, 0.80])),
    })

    # Retinal Age
    retinal_age = biomarker_results.get("retinal_age", {})
    age_data = {
        "estimated_age": retinal_age.get("estimated_age"),
        "confidence_interval": retinal_age.get("confidence_interval"),
        "age_gap": retinal_age.get("age_gap"),
        "gap_interpretation": retinal_age.get("gap_interpretation"),
    }

    return {
        "analyzed": True,
        "markers": markers,
        "retinal_age": age_data,
    }


def _is_in_range(value, normal_range) -> bool:
    """Check if a value falls within the normal range."""
    if value is None or not isinstance(value, (int, float)):
        return True
    if len(normal_range) != 2:
        return True
    return normal_range[0] <= value <= normal_range[1]


def _build_confidence_section(
    anomaly_results: Optional[dict],
    overall_confidence: Optional[float],
) -> dict:
    """Build the confidence assessment section."""
    if overall_confidence is not None:
        confidence = overall_confidence
    elif anomaly_results and anomaly_results.get("severity"):
        confidence = anomaly_results["severity"].get("confidence", 0.7)
    else:
        confidence = 0.7

    if confidence > 0.85:
        level = "HIGH"
        note = "AI analysis has high confidence in results."
    elif confidence > 0.65:
        level = "MODERATE"
        note = "Results should be verified by clinical examination."
    else:
        level = "LOW"
        note = "Low confidence — manual review strongly recommended."

    return {
        "overall_score": round(confidence, 2),
        "level": level,
        "note": note,
    }


def _build_recommendation(
    anomaly_results: Optional[dict],
    biomarker_results: Optional[dict],
) -> dict:
    """Build clinical recommendation based on all findings."""
    severity_level = 0
    if anomaly_results and anomaly_results.get("severity"):
        severity_level = anomaly_results["severity"].get("level", 0)

    glaucoma_flag = False
    if anomaly_results and anomaly_results.get("optic_disc", {}).get("glaucoma_flag"):
        glaucoma_flag = True

    # Build recommendation
    if severity_level >= 4:
        return {
            "urgency": "URGENT",
            "action": "Immediate referral to retina specialist",
            "follow_up": "Within 1-2 weeks",
            "notes": "Proliferative disease detected. Risk of vision loss without treatment.",
        }
    elif severity_level == 3:
        return {
            "urgency": "HIGH",
            "action": "Referral to retina specialist",
            "follow_up": "Within 2-4 weeks",
            "notes": "Severe non-proliferative changes. Close monitoring and possible treatment initiation.",
        }
    elif severity_level == 2:
        return {
            "urgency": "MODERATE",
            "action": "Ophthalmology follow-up",
            "follow_up": "3-6 months",
            "notes": "Moderate changes present. Regular monitoring recommended.",
        }
    elif severity_level == 1:
        return {
            "urgency": "LOW",
            "action": "Routine follow-up",
            "follow_up": "12 months",
            "notes": "Mild changes only. Annual screening sufficient." +
                     (" Glaucoma evaluation also recommended." if glaucoma_flag else ""),
        }
    else:
        notes = "No diabetic retinopathy findings."
        if glaucoma_flag:
            notes += " However, elevated cup-to-disc ratio warrants glaucoma evaluation."
        return {
            "urgency": "ROUTINE",
            "action": "Continue annual screening",
            "follow_up": "12 months",
            "notes": notes,
        }


# ---- Narrative Summary Generation -----------------------------

def _generate_narrative(report: dict) -> str:
    """Generate a human-readable narrative paragraph from the report data."""
    sentences = []

    # Opening
    eye = report["patient"].get("eye", "N/A")
    date = report["patient"].get("exam_date", "today")
    sentences.append(
        f"RetinaAI analysis performed on {date} for {eye} eye."
    )

    # Quality
    quality = report.get("image_quality", {})
    if quality.get("label"):
        sentences.append(f"Image quality assessed as {quality['label']}.")

    # Restoration
    restoration = report.get("restoration", {})
    if restoration.get("performed") and restoration.get("summary"):
        sentences.append(restoration["summary"])

    # Findings
    findings = report.get("findings", {})
    total = findings.get("total_count", 0)
    if total > 0:
        groups = findings.get("finding_groups", [])
        finding_strs = [
            f"{g['count']} {g['clinical_name'].lower()}"
            for g in groups if g["count"] > 0
        ]
        sentences.append(f"Anomaly detection identified {total} findings: {', '.join(finding_strs)}.")
    else:
        sentences.append("No retinal anomalies were detected.")

    # Severity
    severity = report.get("severity", {})
    if severity.get("assessed"):
        sentences.append(
            f"Severity graded as ICDR Level {severity['icdr_level']}: {severity['label']}."
        )

    # Biomarkers
    biomarkers = report.get("biomarkers", {})
    if biomarkers.get("analyzed"):
        retinal_age = biomarkers.get("retinal_age", {})
        if retinal_age.get("estimated_age"):
            sentences.append(
                f"Estimated retinal biological age: {retinal_age['estimated_age']} years."
            )
            if retinal_age.get("gap_interpretation"):
                sentences.append(retinal_age["gap_interpretation"] + ".")

    # Optic disc
    od = findings.get("optic_disc", "")
    if od and "⚠️" in str(od):
        sentences.append(str(od))

    # Recommendation
    rec = report.get("recommendation", {})
    if rec.get("action"):
        sentences.append(f"Recommendation: {rec['action']}. Follow-up: {rec.get('follow_up', 'as needed')}.")

    # Confidence
    conf = report.get("confidence", {})
    if conf.get("level"):
        sentences.append(f"Overall AI confidence: {conf['level']} ({conf.get('overall_score', 0):.0%}).")

    return " ".join(sentences)


# ---- PDF Export (Optional) ------------------------------------

def export_report_to_pdf(
    report: dict,
    output_path: str,
    original_image: Optional[np.ndarray] = None,
    annotated_image: Optional[np.ndarray] = None,
) -> bool:
    """
    Export the clinical report as a PDF document.
    Requires reportlab (optional dependency).

    params:
        report — structured report dict
        output_path — where to save the PDF
        original_image — optional image to embed
        annotated_image — optional annotated image to embed
    returns: True if successful, False otherwise
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
    except ImportError:
        print("reportlab not installed — PDF export unavailable. Install with: pip install reportlab")
        return False

    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                                topMargin=20*mm, bottomMargin=20*mm,
                                leftMargin=15*mm, rightMargin=15*mm)

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'ReportTitle', parent=styles['Title'],
            fontSize=18, textColor=HexColor('#00E5A0'),
            spaceAfter=10,
        )
        heading_style = ParagraphStyle(
            'SectionHead', parent=styles['Heading2'],
            fontSize=13, textColor=HexColor('#4FC3F7'),
            spaceBefore=15, spaceAfter=8,
        )
        body_style = ParagraphStyle(
            'ReportBody', parent=styles['Normal'],
            fontSize=10, leading=14,
        )
        disclaimer_style = ParagraphStyle(
            'Disclaimer', parent=styles['Normal'],
            fontSize=8, textColor=HexColor('#999999'),
            leading=10,
        )

        elements = []

        # Title
        elements.append(Paragraph("🧬 RetinaAI Clinical Analysis Report", title_style))
        elements.append(Spacer(1, 5*mm))

        # Patient Info
        patient = report.get("patient", {})
        info_text = (
            f"<b>Report ID:</b> {report.get('report_id', 'N/A')} &nbsp;&nbsp; "
            f"<b>Date:</b> {patient.get('exam_date', 'N/A')} &nbsp;&nbsp; "
            f"<b>Eye:</b> {patient.get('eye', 'N/A')} &nbsp;&nbsp; "
            f"<b>Age:</b> {patient.get('age', 'N/A')}"
        )
        elements.append(Paragraph(info_text, body_style))
        elements.append(Spacer(1, 5*mm))

        # Narrative Summary
        elements.append(Paragraph("Executive Summary", heading_style))
        elements.append(Paragraph(report.get("narrative_summary", ""), body_style))
        elements.append(Spacer(1, 3*mm))

        # Severity
        severity = report.get("severity", {})
        if severity.get("assessed"):
            elements.append(Paragraph("Severity Assessment", heading_style))
            sev_text = (
                f"<b>ICDR Level {severity.get('icdr_level', 'N/A')}:</b> "
                f"{severity.get('label', 'N/A')} "
                f"(Confidence: {severity.get('confidence', 0):.0%})"
            )
            elements.append(Paragraph(sev_text, body_style))
            if severity.get("description"):
                elements.append(Paragraph(severity["description"], body_style))
            elements.append(Spacer(1, 3*mm))

        # Findings
        findings = report.get("findings", {})
        if findings.get("analyzed"):
            elements.append(Paragraph("Anomaly Findings", heading_style))
            groups = findings.get("finding_groups", [])
            if groups:
                table_data = [["Finding", "Count", "Locations", "Confidence"]]
                for g in groups:
                    table_data.append([
                        g.get("clinical_name", ""),
                        str(g.get("count", 0)),
                        ", ".join(g.get("locations", [])),
                        f"{g.get('mean_confidence', 0):.0%}",
                    ])
                t = Table(table_data, colWidths=[45*mm, 20*mm, 50*mm, 25*mm])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a2744')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#333333')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#0a1628'), HexColor('#0f1d35')]),
                    ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#CCCCCC')),
                ]))
                elements.append(t)
            else:
                elements.append(Paragraph("No anomalies detected.", body_style))
            elements.append(Spacer(1, 3*mm))

        # Biomarkers
        biomarkers = report.get("biomarkers", {})
        if biomarkers.get("analyzed"):
            elements.append(Paragraph("Retinal Biomarker Panel", heading_style))
            markers = biomarkers.get("markers", [])
            if markers:
                table_data = [["Biomarker", "Value", "Normal Range", "Status"]]
                for m in markers:
                    val = m.get("value", "N/A")
                    if isinstance(val, float):
                        val = f"{val:.3f}"
                    nr = m.get("normal_range", [])
                    nr_str = f"{nr[0]}-{nr[1]}" if len(nr) == 2 else "N/A"
                    status = "✅ Normal" if m.get("in_range", True) else "⚠️ Abnormal"
                    table_data.append([m.get("name", ""), str(val), nr_str, status])

                t = Table(table_data, colWidths=[45*mm, 25*mm, 30*mm, 30*mm])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a2744')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#333333')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#0a1628'), HexColor('#0f1d35')]),
                    ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#CCCCCC')),
                ]))
                elements.append(t)

            # Retinal Age
            ra = biomarkers.get("retinal_age", {})
            if ra.get("estimated_age"):
                elements.append(Spacer(1, 2*mm))
                age_text = (
                    f"<b>Estimated Retinal Age:</b> {ra['estimated_age']} years "
                    f"(CI: {ra.get('confidence_interval', ['?', '?'])})"
                )
                elements.append(Paragraph(age_text, body_style))
                if ra.get("gap_interpretation"):
                    elements.append(Paragraph(ra["gap_interpretation"], body_style))
            elements.append(Spacer(1, 3*mm))

        # Recommendation
        rec = report.get("recommendation", {})
        if rec:
            elements.append(Paragraph("Clinical Recommendation", heading_style))
            rec_text = (
                f"<b>Urgency:</b> {rec.get('urgency', 'N/A')} &nbsp;&nbsp; "
                f"<b>Action:</b> {rec.get('action', 'N/A')} &nbsp;&nbsp; "
                f"<b>Follow-up:</b> {rec.get('follow_up', 'N/A')}"
            )
            elements.append(Paragraph(rec_text, body_style))
            if rec.get("notes"):
                elements.append(Paragraph(rec["notes"], body_style))
            elements.append(Spacer(1, 5*mm))

        # Disclaimer
        elements.append(Spacer(1, 10*mm))
        elements.append(Paragraph(report.get("disclaimer", ""), disclaimer_style))

        # Build PDF
        doc.build(elements)
        return True

    except Exception as e:
        print(f"PDF export failed: {e}")
        return False
