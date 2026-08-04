"""
api/schemas.py
==============
Pydantic data models for the FastAPI request and response bodies.

WHAT IS THIS? (for beginners)
  These are "blueprints" for the data that goes IN and OUT of the API.
  FastAPI uses them to automatically:
    - Validate that requests contain the required fields
    - Show you what the API expects in its documentation (/docs)
    - Convert Python objects to JSON automatically
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---- Enums ------------------------------------------------

class SplitName(str, Enum):
    train  = "train"
    val    = "val"
    test   = "test"
    holdout = "holdout"


class DatasetName(str, Enum):
    eyepacs  = "eyepacs"
    aptos    = "aptos"
    idrid    = "idrid"
    messidor2 = "messidor2"
    rfmid    = "rfmid"
    odir     = "odir"
    stare    = "stare"
    drive    = "drive"


class TrainingStatus(str, Enum):
    idle       = "idle"
    running    = "running"
    paused     = "paused"
    completed  = "completed"
    failed     = "failed"


# ---- Data Pipeline Schemas --------------------------------

class IngestRequest(BaseModel):
    """Request body for ingesting a dataset into the metadata database."""
    dataset_name: DatasetName = Field(..., description="Name of the dataset to ingest")
    dataset_path: str         = Field(..., description="Absolute path to the dataset folder on disk")

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_name": "eyepacs",
                "dataset_path": "/data/eyepacs"
            }
        }


class IngestResponse(BaseModel):
    """Response after ingesting a dataset."""
    dataset_name:     str
    images_ingested:  int
    message:          str


class QualityScoringRequest(BaseModel):
    """Request to compute quality scores for all unscored images."""
    recompute_all: bool = Field(
        default=False,
        description="If True, recompute scores for images that already have scores"
    )


class QualityScoringResponse(BaseModel):
    """Response after quality scoring."""
    images_scored:       int
    pseudo_clean_count:  int
    message:             str


class SplitRequest(BaseModel):
    """Request to assign train/val/test splits."""
    train_fraction:        float = Field(default=0.85, ge=0.5, le=0.95)
    val_fraction:          float = Field(default=0.10, ge=0.02, le=0.3)
    quality_quantile:      float = Field(default=0.75, ge=0.5, le=0.99)
    random_seed:           int   = Field(default=42)

    class Config:
        json_schema_extra = {
            "example": {
                "train_fraction": 0.85,
                "val_fraction": 0.10,
                "quality_quantile": 0.75,
                "random_seed": 42
            }
        }


class SplitResponse(BaseModel):
    """Response after split assignment."""
    split_counts:  dict[str, int]
    leakage_check: str
    message:       str


# ---- Training Schemas -------------------------------------

class TrainingRequest(BaseModel):
    """Request to start or resume training."""
    num_epochs:           int   = Field(default=200, ge=1, le=1000)
    batch_size:           int   = Field(default=4,   ge=1, le=32)
    learning_rate:        float = Field(default=2e-4, gt=0)
    run_domain_adaptation: bool = Field(default=False)
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to a checkpoint to resume from (None = start fresh)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "num_epochs": 200,
                "batch_size": 4,
                "learning_rate": 0.0002,
                "run_domain_adaptation": False,
                "resume_from_checkpoint": None
            }
        }


class TrainingStatusResponse(BaseModel):
    """Current training status."""
    status:          TrainingStatus
    current_epoch:   Optional[int]
    total_epochs:    Optional[int]
    best_ssim:       Optional[float]
    best_psnr:       Optional[float]
    latest_losses:   Optional[dict[str, float]]
    elapsed_seconds: Optional[float]
    message:         str


class TrainingStopResponse(BaseModel):
    """Response after stopping training."""
    message:       str
    final_epoch:   Optional[int]
    best_checkpoint: Optional[str]


# ---- Inference Schemas ------------------------------------

from pydantic import BaseModel, Field, ConfigDict

class RestoreResponse(BaseModel):
    """Response after restoring a single image."""
    model_config = ConfigDict(protected_namespaces=())
    output_path:      str
    input_filename:   str
    processing_time_ms: float
    model_checkpoint: str
    uncertainty_computed: bool = False


class BatchRestoreRequest(BaseModel):
    """Request for batch restoration of a folder."""
    input_folder:         str
    output_folder:        str
    checkpoint_path:      str   = Field(default="checkpoints/best.pt")
    compute_uncertainty:  bool  = Field(default=False)
    n_mc_samples:         int   = Field(default=10, ge=1, le=50)

    class Config:
        json_schema_extra = {
            "example": {
                "input_folder":   "data/to_restore",
                "output_folder":  "data/restored",
                "checkpoint_path": "checkpoints/best.pt",
                "compute_uncertainty": False,
                "n_mc_samples": 10
            }
        }


class BatchRestoreResponse(BaseModel):
    """Response after batch restoration completes."""
    images_processed:    int
    output_folder:       str
    output_paths:        list[str]
    processing_time_s:   float
    message:             str


# ---- Evaluation Schemas -----------------------------------

class EvaluationRequest(BaseModel):
    """Request to run final evaluation on the test set."""
    checkpoint_path:     str  = Field(default="checkpoints/best.pt")
    run_downstream_eval: bool = Field(default=False)


class EvaluationResponse(BaseModel):
    """Summary of test-set evaluation results."""
    mean_psnr:         Optional[float]
    std_psnr:          Optional[float]
    mean_ssim:         Optional[float]
    std_ssim:          Optional[float]
    mean_lpips:        Optional[float]
    mean_vessel_dice:  Optional[float]
    n_images_evaluated: int
    results_csv_path:  str
    message:           str


# ---- Generic API Response ---------------------------------

class HealthResponse(BaseModel):
    """Health check response."""
    status:           str
    device:           str
    checkpoint_exists: bool
    database_exists:  bool
    version:          str = "1.0.0"


# ---- Analysis Schemas (RetinaAI Pillars) -------------------

class AnomalyFinding(BaseModel):
    """A single detected retinal anomaly."""
    finding_type:  str
    x:             int
    y:             int
    radius:        int
    confidence:    float
    quadrant:      str
    description:   str
    color:         list[int] = [255, 0, 0]


class AnomalyDetectionResponse(BaseModel):
    """Response from the anomaly detection pipeline."""
    findings:        list[AnomalyFinding]
    finding_counts:  dict[str, int]
    severity:        dict[str, Any]
    optic_disc:      dict[str, Any]
    vessel_info:     dict[str, Any]
    annotated_image: Optional[str] = None  # base64 PNG
    vessel_mask:     Optional[str] = None  # base64 PNG


class ChangeAnalysisResponse(BaseModel):
    """Response from enhancement change analysis."""
    summary_text:       str
    metrics:            dict[str, Any]
    frequency_analysis: dict[str, Any]
    difference_map:     Optional[str] = None  # base64 PNG
    ssim_map:           Optional[str] = None  # base64 PNG


class VesselTopologyResponse(BaseModel):
    """Response from vessel topology extraction."""
    nodes:             list[dict[str, Any]]
    edges:             list[dict[str, Any]]
    topology_metrics:  dict[str, Any]
    image_dimensions:  dict[str, int]


class BiomarkerResult(BaseModel):
    """Single biomarker measurement."""
    name:                str
    value:               Any
    normal_range:        list[float]
    interpretation:      str
    in_range:            bool


class RetinalAgeEstimate(BaseModel):
    """Retinal biological age estimation."""
    estimated_age:        int
    confidence_interval:  list[int]
    age_gap:              Optional[float] = None
    gap_interpretation:   Optional[str] = None


class BiomarkerResponse(BaseModel):
    """Response from retinal biomarker analysis."""
    tortuosity:           dict[str, Any]
    fractal_dimension:    dict[str, Any]
    arteriovenous_ratio:  dict[str, Any]
    vessel_density:       dict[str, Any]
    retinal_age:          dict[str, Any]


class GradCAMResponse(BaseModel):
    """Response from Grad-CAM explainability analysis."""
    heatmap:         Optional[str] = None  # base64 PNG overlay
    insight_text:    str
    channel_heatmaps: Optional[dict[str, str]] = None  # per-channel base64


class ProgressionRequest(BaseModel):
    """Request for disease progression comparison."""
    visit_date_earlier:  Optional[str] = None
    visit_date_later:    Optional[str] = None


class ProgressionResponse(BaseModel):
    """Response from disease progression analysis."""
    visit_dates:          dict[str, str]
    severity_comparison:  dict[str, Any]
    findings_comparison:  dict[str, Any]
    progression:          dict[str, Any]
    visit_difference:     dict[str, Any]


class ClinicalReportResponse(BaseModel):
    """Full structured clinical report."""
    report_id:          str
    generated_at:       str
    patient:            dict[str, Any]
    image_quality:      dict[str, Any]
    restoration:        dict[str, Any]
    findings:           dict[str, Any]
    severity:           dict[str, Any]
    biomarkers:         dict[str, Any]
    confidence:         dict[str, Any]
    recommendation:     dict[str, Any]
    narrative_summary:  str
    disclaimer:         str


class CopilotMessage(BaseModel):
    """A message in the co-pilot chat."""
    question:      str = Field(..., description="Natural language question about the findings")
    groq_api_key:  Optional[str] = Field(default=None, description="Optional Groq API key for Llama-3.3-70b model")


class CopilotResponse(BaseModel):
    """Response from the AI co-pilot."""
    response:       str
    topic:          str
    timestamp:      str
    is_contextual:  bool
    suggestions:    list[str] = []


class ROIInspectRequest(BaseModel):
    """Request for interactive click-to-inspect ROI Groq Vision analysis."""
    image_crop_base64: str = Field(..., description="Base64 PNG crop of clicked ROI")
    x_percent:         float = Field(..., description="Relative X coordinate [0.0 - 1.0]")
    y_percent:         float = Field(..., description="Relative Y coordinate [0.0 - 1.0]")
    quadrant:          str = Field(default="Central Retina")
    groq_api_key:      Optional[str] = Field(default=None)


class ROIInspectResponse(BaseModel):
    """Response from Groq Vision ROI inspection."""
    x_percent:   float
    y_percent:   float
    quadrant:    str
    analysis:    str
    model_used:  str
    timestamp:   str
    status:      str


class FullPipelineRequest(BaseModel):
    """Request for the full analysis pipeline."""
    checkpoint_path:     str = Field(default="checkpoints/best.pt")
    compute_uncertainty: bool = Field(default=False)
    patient_id:          str = Field(default="N/A")
    patient_age:         Optional[int] = None
    eye_side:            str = Field(default="N/A")


class FullPipelineResponse(BaseModel):
    """Response from the full 9-pillar analysis pipeline."""
    original_image:       str        # base64 PNG
    restored_image:       str        # base64 PNG
    restoration_time_ms:  float
    change_analysis:      dict[str, Any]
    anomaly_detection:    dict[str, Any]
    vessel_topology:      dict[str, Any]
    biomarkers:           dict[str, Any]
    gradcam:              dict[str, Any]
    report:               dict[str, Any]
    copilot_suggestions:  list[str]
