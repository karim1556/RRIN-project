"""
api/routes/copilot.py
======================
API endpoints for the AI Co-Pilot chat interface.
  POST /api/v1/copilot/ask         — Ask a question about findings
  GET  /api/v1/copilot/suggestions — Get contextual suggested questions
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from api.schemas import CopilotMessage, CopilotResponse

router = APIRouter(prefix="/copilot", tags=["CoPilot"])

# In-memory session store for analysis context (per-session)
_analysis_context = {
    "anomaly_results": None,
    "biomarker_results": None,
    "report": None,
    "severity_level": 0,
}


def update_copilot_context(
    anomaly_results: Optional[dict] = None,
    biomarker_results: Optional[dict] = None,
    report: Optional[dict] = None,
):
    """Update the co-pilot's analysis context. Called after full pipeline runs."""
    if anomaly_results is not None:
        _analysis_context["anomaly_results"] = anomaly_results
        _analysis_context["severity_level"] = anomaly_results.get("severity", {}).get("level", 0)
    if biomarker_results is not None:
        _analysis_context["biomarker_results"] = biomarker_results
    if report is not None:
        _analysis_context["report"] = report


@router.post("/ask")
async def ask_copilot(message: CopilotMessage):
    """
    Ask the AI co-pilot a question about the current analysis findings.
    Returns a contextual response based on the ophthalmology knowledge base
    and the current image analysis results.
    """
    try:
        from src.analysis.copilot import generate_response, generate_suggested_questions

        response = generate_response(
            question=message.question,
            anomaly_results=_analysis_context["anomaly_results"],
            biomarker_results=_analysis_context["biomarker_results"],
            report=_analysis_context["report"],
            groq_api_key=message.groq_api_key,
        )

        # Add follow-up suggestions
        suggestions = generate_suggested_questions(
            anomaly_results=_analysis_context["anomaly_results"],
            biomarker_results=_analysis_context["biomarker_results"],
            severity_level=_analysis_context["severity_level"],
        )

        response["suggestions"] = suggestions

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Co-pilot response failed: {str(e)}")


@router.get("/suggestions")
async def get_suggestions():
    """
    Get contextual suggested questions based on current analysis findings.
    Returns a list of relevant questions the user can ask.
    """
    try:
        from src.analysis.copilot import generate_suggested_questions

        suggestions = generate_suggested_questions(
            anomaly_results=_analysis_context["anomaly_results"],
            biomarker_results=_analysis_context["biomarker_results"],
            severity_level=_analysis_context["severity_level"],
        )

        return JSONResponse(content={"suggestions": suggestions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate suggestions: {str(e)}")
