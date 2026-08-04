"""
api/routes/report.py
=====================
API endpoints for clinical report generation and export.
  POST /api/v1/report/generate      — Generate structured clinical report
  POST /api/v1/report/download-pdf   — Generate + download as PDF
"""

import io
import os
import base64
import tempfile
import numpy as np

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional

router = APIRouter(prefix="/report", tags=["Report"])


async def _load_upload_as_float(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into a float32 numpy array [0, 1]."""
    from PIL import Image
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


@router.post("/generate")
async def generate_report(
    file: UploadFile = File(..., description="Retinal fundus image"),
    patient_id: str = Form(default="N/A"),
    patient_age: Optional[int] = Form(default=None),
    eye_side: str = Form(default="N/A"),
):
    """
    Run the full analysis pipeline and generate a structured clinical report.
    Returns JSON with all findings, biomarkers, severity, and recommendation.
    """
    try:
        image = await _load_upload_as_float(file)

        # Run anomaly detection
        from src.analysis.anomaly_detector import detect_anomalies
        anomaly_result = detect_anomalies(image)

        # Run biomarkers
        from src.analysis.anomaly_detector import preprocess_for_detection, segment_vessels
        from src.analysis.vessel_topology import extract_skeleton
        from src.analysis.biomarkers import compute_all_biomarkers

        prep = preprocess_for_detection(image)
        vessels = segment_vessels(prep["green_clahe"], prep["fov_mask"])
        skeleton = extract_skeleton(vessels["vessel_mask"])
        optic_disc = anomaly_result.get("optic_disc", {})

        biomarker_result = compute_all_biomarkers(
            image,
            vessel_mask=vessels["vessel_mask"],
            skeleton=skeleton,
            optic_disc_x=optic_disc.get("x", 0),
            optic_disc_y=optic_disc.get("y", 0),
            optic_disc_radius=optic_disc.get("radius", 50),
            optic_disc_detected=optic_disc.get("detected", False),
            chronological_age=patient_age,
        )

        # Generate report
        from src.analysis.report_generator import generate_clinical_report
        report = generate_clinical_report(
            anomaly_results=anomaly_result,
            biomarker_results=biomarker_result,
            patient_id=patient_id,
            patient_age=patient_age,
            eye_side=eye_side,
        )

        return JSONResponse(content=report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.post("/download-pdf")
async def download_report_pdf(
    file: UploadFile = File(..., description="Retinal fundus image"),
    patient_id: str = Form(default="N/A"),
    patient_age: Optional[int] = Form(default=None),
    eye_side: str = Form(default="N/A"),
):
    """
    Generate a clinical report and return it as a downloadable PDF.
    Requires reportlab to be installed.
    """
    try:
        image = await _load_upload_as_float(file)

        # Run analysis
        from src.analysis.anomaly_detector import detect_anomalies
        anomaly_result = detect_anomalies(image)

        from src.analysis.report_generator import generate_clinical_report, export_report_to_pdf
        report = generate_clinical_report(
            anomaly_results=anomaly_result,
            patient_id=patient_id,
            patient_age=patient_age,
            eye_side=eye_side,
        )

        # Export to PDF
        pdf_path = tempfile.mktemp(suffix=".pdf")
        success = export_report_to_pdf(report, pdf_path)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="PDF export failed. Ensure reportlab is installed: pip install reportlab"
            )

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"RetinaAI_Report_{report.get('report_id', 'unknown')}.pdf",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
