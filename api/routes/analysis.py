"""
api/routes/analysis.py
=======================
API endpoints for the RetinaAI analysis pillars:
  POST /api/v1/analysis/enhancement-diff   — Change analysis (Pillar 2)
  POST /api/v1/analysis/detect-anomalies   — Anomaly detection (Pillar 3)
  POST /api/v1/analysis/vessel-topology    — 3D vessel graph (Pillar 4)
  POST /api/v1/analysis/biomarkers         — Retinal biomarkers (Pillar 5)
  POST /api/v1/analysis/explainability     — Grad-CAM heatmaps (Pillar 6)
  POST /api/v1/analysis/progression        — Multi-visit comparison (Pillar 7)
  POST /api/v1/analysis/full-pipeline      — Run all pillars in one call
"""

import io
import os
import time
import base64
import tempfile
import numpy as np

from api.schemas import (
    RestoreResponse, BatchRestoreRequest,
    ChangeAnalysisResponse, AnomalyDetectionResponse, VesselTopologyResponse,
    BiomarkerResponse, GradCAMResponse, ProgressionResponse,
    ClinicalReportResponse, FullPipelineResponse, ROIInspectRequest, ROIInspectResponse
)

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

router = APIRouter(prefix="/analysis", tags=["Analysis"])


# ---- Utility: image ↔ base64 ----------------------------------

def _sanitize_for_json(obj):
    """Recursively convert numpy types, Form objects, and non-serializable types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        if hasattr(obj, 'default'):
            return str(getattr(obj, 'default', 'N/A'))
        return str(obj)
    return obj


def _array_to_base64(arr: np.ndarray) -> str:
    """Convert a numpy image array to base64-encoded PNG data URL."""
    from PIL import Image
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        uint8_arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        uint8_arr = arr
    img = Image.fromarray(uint8_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


async def _load_upload_as_float(file: UploadFile, max_dim: int = 384) -> np.ndarray:
    """
    Read an uploaded file into a float32 numpy array [0, 1].
    Smartly resizes oversized images to max_dim (default 384px) to guarantee
    ultra-fast <1s PyTorch GAN inference and <100MB RAM usage on cloud containers.
    """
    from PIL import Image
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    return np.array(img, dtype=np.float32) / 255.0


# ---- Pillar 2: Enhancement Change Analysis ---------------------

@router.post("/enhancement-diff")
async def enhancement_diff(
    original: UploadFile = File(..., description="Original degraded image"),
    restored: UploadFile = File(..., description="Restored image"),
):
    """
    Compute AI change analysis between original and restored images.
    Shows what the restoration changed: noise removed, details recovered,
    illumination corrected, artifacts removed.
    """
    try:
        original_arr = await _load_upload_as_float(original)
        restored_arr = await _load_upload_as_float(restored)

        from src.analysis.change_analysis import compute_enhancement_analysis
        result = compute_enhancement_analysis(original_arr, restored_arr)

        return JSONResponse(content={
            "summary_text": result["summary_text"],
            "metrics": result["metrics"],
            "frequency_analysis": result["frequency_analysis"],
            "difference_map": _array_to_base64(result["difference_map_colored"]),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Change analysis failed: {str(e)}")


# ---- Pillar 3: Anomaly Detection ------------------------------

@router.post("/detect-anomalies")
async def detect_anomalies(
    file: UploadFile = File(..., description="Retinal fundus image"),
):
    """
    Run the full anomaly detection pipeline on a retinal image.
    Detects microaneurysms, hemorrhages, exudates, cotton wool spots.
    Returns annotated image with severity grading.
    """
    try:
        image = await _load_upload_as_float(file)

        from src.analysis.anomaly_detector import detect_anomalies as _detect
        result = _detect(image)

        return JSONResponse(content={
            "findings": result["findings"],
            "finding_counts": result["finding_counts"],
            "severity": result["severity"],
            "optic_disc": result["optic_disc"],
            "vessel_info": result["vessel_info"],
            "annotated_image": _array_to_base64(result["annotated_image"]),
            "vessel_mask": _array_to_base64(
                (result["vessel_mask"].astype(np.uint8) * 255)[..., None].repeat(3, axis=-1)
            ),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


# ---- Pillar 4: Vessel Topology ---------------------------------

@router.post("/vessel-topology")
async def vessel_topology(
    file: UploadFile = File(..., description="Retinal fundus image"),
):
    """
    Extract the retinal vessel tree topology as an interactive graph structure.
    Returns nodes (junctions/endpoints) and edges (vessel segments) with
    properties like width, tortuosity, and vessel type.
    """
    try:
        image = await _load_upload_as_float(file)

        from src.analysis.vessel_topology import extract_vessel_topology
        graph = extract_vessel_topology(image)

        return JSONResponse(content=graph)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vessel topology extraction failed: {str(e)}")


# ---- Pillar 5: Biomarkers -------------------------------------

@router.post("/biomarkers")
async def biomarkers(
    file: UploadFile = File(..., description="Retinal fundus image"),
    patient_age: Optional[int] = Form(default=None),
):
    """
    Compute quantitative retinal biomarkers: vessel tortuosity,
    fractal dimension, A/V ratio, vessel density, and estimated retinal age.
    """
    try:
        image = await _load_upload_as_float(file)

        # Get vessel info and optic disc first
        from src.analysis.anomaly_detector import preprocess_for_detection, segment_vessels, detect_optic_disc
        prep = preprocess_for_detection(image)
        vessels = segment_vessels(prep["green_clahe"], prep["fov_mask"])
        optic_disc = detect_optic_disc(prep["green_channel"], prep["fov_mask"])

        from src.analysis.vessel_topology import extract_skeleton
        skeleton = extract_skeleton(vessels["vessel_mask"])

        from src.analysis.biomarkers import compute_all_biomarkers
        result = compute_all_biomarkers(
            image,
            vessel_mask=vessels["vessel_mask"],
            skeleton=skeleton,
            width_map=vessels["width_map"],
            optic_disc_x=optic_disc.x,
            optic_disc_y=optic_disc.y,
            optic_disc_radius=optic_disc.radius,
            optic_disc_detected=optic_disc.detected,
            chronological_age=patient_age,
        )

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Biomarker analysis failed: {str(e)}")


# ---- Pillar 6: Grad-CAM Explainability ------------------------

@router.post("/explainability")
async def explainability(
    file: UploadFile = File(..., description="Retinal fundus image"),
    checkpoint_path: str = Form(default="checkpoints/best.pt"),
):
    """
    Generate Grad-CAM heatmaps showing where the neural network
    focused during restoration.
    """
    try:
        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

        image = await _load_upload_as_float(file)

        # Load generator
        from api.routes.inference import _get_generator
        generator = _get_generator(checkpoint_path)

        # Prepare input
        from src.utils.image_utils import build_four_channel_input_tensor
        from src.inference.restore import _pad_to_multiple
        from src.config import DEVICE

        padded, orig_shape = _pad_to_multiple(image, 32)
        input_tensor = build_four_channel_input_tensor(padded).unsqueeze(0).to(DEVICE)

        # Compute Grad-CAM
        from src.analysis.explainability import compute_gradcam, overlay_heatmap, generate_gradcam_insight

        heatmap = compute_gradcam(generator, input_tensor, "bottleneck")
        heatmap_cropped = heatmap[:orig_shape[0], :orig_shape[1]]

        # Generate overlay
        overlay = overlay_heatmap(image, heatmap_cropped, alpha=0.4)
        insight = generate_gradcam_insight(heatmap_cropped, orig_shape[0], orig_shape[1])

        return JSONResponse(content={
            "heatmap": _array_to_base64(overlay),
            "insight_text": insight,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {str(e)}")


# ---- Pillar 7: Disease Progression ----------------------------

@router.post("/progression")
async def progression(
    image_earlier: UploadFile = File(..., description="Earlier visit image"),
    image_later: UploadFile = File(..., description="Later visit image"),
    visit_date_earlier: Optional[str] = Form(default=None),
    visit_date_later: Optional[str] = Form(default=None),
):
    """
    Compare two retinal images from different visits to track
    disease progression. Reports new, resolved, and worsening findings.
    """
    try:
        earlier_arr = await _load_upload_as_float(image_earlier)
        later_arr = await _load_upload_as_float(image_later)

        from src.analysis.progression import analyze_progression
        result = analyze_progression(
            earlier_arr, later_arr,
            visit_date_earlier=visit_date_earlier,
            visit_date_later=visit_date_later,
        )

        # Remove non-serializable numpy arrays
        serializable = {
            "visit_dates": result["visit_dates"],
            "severity_comparison": result["severity_comparison"],
            "findings_comparison": result["findings_comparison"],
            "progression": result["progression"],
            "visit_difference": {
                "mean_change": result["visit_difference"]["mean_change"],
                "max_change": result["visit_difference"]["max_change"],
                "change_coverage_percent": result["visit_difference"]["change_coverage_percent"],
            },
        }

        return JSONResponse(content=serializable)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Progression analysis failed: {str(e)}")


# ---- Full Pipeline: All Pillars in One Call --------------------

@router.post("/full-pipeline")
async def full_pipeline(
    file: UploadFile = File(..., description="Retinal fundus image"),
    checkpoint_path: str = Form(default="checkpoints/best_inference.pt"),
    enhance_veins: bool = Form(default=True),
    patient_id: str = Form(default="N/A"),
    patient_age: Optional[int] = Form(default=None),
    eye_side: str = Form(default="N/A"),
):
    """
    Run the complete 9-pillar analysis pipeline in a single call.
    Uses checkpoints/best.pt to restore the retina image via GAN generator.
    """
    try:
        start_time = time.time()
        image = await _load_upload_as_float(file)

        # Sanitize parameters
        clean_patient_id = str(getattr(patient_id, 'default', patient_id)) if not isinstance(patient_id, str) else patient_id
        clean_eye_side = str(getattr(eye_side, 'default', eye_side)) if not isinstance(eye_side, str) else eye_side
        clean_patient_age = None
        if patient_age is not None and not hasattr(patient_age, 'default'):
            try:
                clean_patient_age = int(patient_age)
            except (ValueError, TypeError):
                clean_patient_age = None

        # ----- Pillar 1: Restoration with best.pt / best_inference.pt -----
        restored_image = image.copy()
        restoration_time = 0
        generator = None

        try:
            from api.routes.inference import _get_generator, _apply_clahe
            generator = _get_generator(checkpoint_path)
            from src.inference.restore import restore_image_array
            t0 = time.time()
            restored_image = restore_image_array(generator, image)
            restoration_time = (time.time() - t0) * 1000

            # Apply vein & vessel contrast enhancement (CLAHE)
            if enhance_veins:
                restored_image = _apply_clahe(restored_image)

        except Exception as model_err:
            print(f"⚠️ Model restoration notice: {model_err}")
            from api.routes.inference import _apply_clahe
            if enhance_veins:
                restored_image = _apply_clahe(image)

        # ----- Pillar 2: Change Analysis -----
        from src.analysis.change_analysis import compute_enhancement_analysis
        change_result = compute_enhancement_analysis(image, restored_image)

        # ----- Pillar 3: Anomaly Detection (on restored image) -----
        from src.analysis.anomaly_detector import detect_anomalies as _detect
        anomaly_result = _detect(restored_image)

        # ----- Pillar 4: Vessel Topology -----
        from src.analysis.vessel_topology import extract_vessel_topology
        vessel_graph = extract_vessel_topology(
            restored_image,
            vessel_mask=anomaly_result["vessel_mask"],
        )

        # ----- Pillar 5: Biomarkers -----
        from src.analysis.vessel_topology import extract_skeleton
        skeleton = extract_skeleton(anomaly_result["vessel_mask"])

        from src.analysis.biomarkers import compute_all_biomarkers
        optic_disc = anomaly_result.get("optic_disc", {})
        biomarker_result = compute_all_biomarkers(
            restored_image,
            vessel_mask=anomaly_result["vessel_mask"],
            skeleton=skeleton,
            optic_disc_x=optic_disc.get("x", 0),
            optic_disc_y=optic_disc.get("y", 0),
            optic_disc_radius=optic_disc.get("radius", 50),
            optic_disc_detected=optic_disc.get("detected", False),
            chronological_age=clean_patient_age,
        )

        # ----- Pillar 6: Grad-CAM -----
        gradcam_data = {"heatmap": None, "insight_text": "Model checkpoint not available for Grad-CAM analysis."}
        if generator is not None:
            try:
                from src.utils.image_utils import build_four_channel_input_tensor
                from src.inference.restore import _pad_to_multiple
                from src.config import DEVICE
                from src.analysis.explainability import compute_gradcam, overlay_heatmap, generate_gradcam_insight

                padded, orig_shape = _pad_to_multiple(image, 32)
                input_tensor = build_four_channel_input_tensor(padded).unsqueeze(0).to(DEVICE)
                heatmap = compute_gradcam(generator, input_tensor, "bottleneck")
                heatmap_cropped = heatmap[:orig_shape[0], :orig_shape[1]]
                overlay = overlay_heatmap(restored_image, heatmap_cropped, alpha=0.4)
                insight = generate_gradcam_insight(heatmap_cropped, orig_shape[0], orig_shape[1])
                gradcam_data = {
                    "heatmap": _array_to_base64(overlay),
                    "insight_text": insight,
                }
            except Exception as gc_err:
                print(f"Grad-CAM notice: {gc_err}")
                pass

        # ----- Pillar 8: Clinical Report -----
        from src.analysis.report_generator import generate_clinical_report
        report = generate_clinical_report(
            restoration_metrics=change_result.get("metrics"),
            change_analysis=change_result,
            anomaly_results=anomaly_result,
            biomarker_results=biomarker_result,
            gradcam_insight=gradcam_data.get("insight_text"),
            patient_id=clean_patient_id,
            patient_age=clean_patient_age,
            eye_side=clean_eye_side,
        )

        # ----- Pillar 9: Co-Pilot Suggestions -----
        from src.analysis.copilot import generate_suggested_questions
        severity_level = anomaly_result.get("severity", {}).get("level", 0)
        suggestions = generate_suggested_questions(
            anomaly_results=anomaly_result,
            biomarker_results=biomarker_result,
            severity_level=severity_level,
        )

        total_time = (time.time() - start_time) * 1000

        # Build response (convert numpy arrays to base64)
        response = {
            "original_image": _array_to_base64(image),
            "restored_image": _array_to_base64(restored_image),
            "restoration_time_ms": round(restoration_time, 1),
            "total_pipeline_time_ms": round(total_time, 1),
            "change_analysis": {
                "summary_text": change_result["summary_text"],
                "metrics": change_result["metrics"],
                "frequency_analysis": change_result["frequency_analysis"],
                "difference_map": _array_to_base64(change_result["difference_map_colored"]),
            },
            "anomaly_detection": {
                "findings": anomaly_result["findings"],
                "finding_counts": anomaly_result["finding_counts"],
                "severity": anomaly_result["severity"],
                "optic_disc": anomaly_result["optic_disc"],
                "vessel_info": anomaly_result["vessel_info"],
                "annotated_image": _array_to_base64(anomaly_result["annotated_image"]),
                "heatmap_image": _array_to_base64(anomaly_result["heatmap_image"]) if "heatmap_image" in anomaly_result else None,
            },
            "vessel_topology": vessel_graph,
            "biomarkers": biomarker_result,
            "gradcam": gradcam_data,
            "report": report,
            "copilot_suggestions": suggestions,
        }

        import gc
        gc.collect()

        return JSONResponse(content=_sanitize_for_json(response))

    except Exception as e:
        import gc
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Full pipeline failed: {str(e)}")


@router.post("/inspect-roi")
async def inspect_roi(req: ROIInspectRequest):
    """
    Perform Groq Vision AI analysis on a clinician-clicked Region of Interest (ROI).
    """
    try:
        from src.analysis.vision_inspector import inspect_retinal_roi
        result = inspect_retinal_roi(
            image_crop_base64=req.image_crop_base64,
            x_percent=req.x_percent,
            y_percent=req.y_percent,
            quadrant_name=req.quadrant,
            groq_api_key=req.groq_api_key,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ROI Inspection failed: {str(e)}")
