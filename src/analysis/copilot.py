"""
src/analysis/copilot.py
=========================
Pillar 9 — AI Co-Pilot Chat Engine

A rule-based contextual Q&A system that provides intelligent responses
about retinal findings. Works fully offline without external APIs.

Features:
  - Pre-built knowledge base of ophthalmology facts
  - Context-aware suggested questions based on current findings
  - Template-based responses that fill in dynamic values
  - Conversation history per session
"""

from typing import Optional
from datetime import datetime


# ---- Knowledge Base -------------------------------------------

KNOWLEDGE_BASE = {
    # Diabetic Retinopathy
    "microaneurysm": {
        "what": (
            "Microaneurysms are the earliest clinical sign of diabetic retinopathy. "
            "They appear as tiny (< 125μm) dark red dots in the retina, caused by "
            "weakening and ballooning of small retinal capillary walls due to chronic "
            "hyperglycemia. They may leak fluid (plasma) into surrounding tissue."
        ),
        "significance": (
            "Their presence alone classifies the eye as having at least mild "
            "non-proliferative diabetic retinopathy (NPDR). While individual "
            "microaneurysms are not vision-threatening, they indicate underlying "
            "microvascular damage and warrant regular monitoring."
        ),
        "action": (
            "For isolated microaneurysms: annual diabetic eye screening. "
            "If accompanied by other findings (exudates, hemorrhages): "
            "follow-up in 6 months. Optimize blood glucose control (HbA1c target < 7%)."
        ),
    },
    "hard_exudate": {
        "what": (
            "Hard exudates are bright yellow-white deposits of lipids and proteins "
            "that have leaked from damaged retinal blood vessels. They have sharp, "
            "well-defined edges and often form rings (circinate patterns) around "
            "areas of vascular leakage."
        ),
        "significance": (
            "Hard exudates near the macula (< 500μm from the fovea) indicate "
            "clinically significant macular edema (CSME), which is a common cause "
            "of vision loss in diabetic patients. Their presence suggests ongoing "
            "vascular leakage that may require treatment."
        ),
        "action": (
            "If perifoveal or macular: referral for OCT and possible anti-VEGF "
            "injection or laser photocoagulation. If peripheral: monitor at "
            "3-6 month intervals. Control lipid levels and blood pressure."
        ),
    },
    "hemorrhage": {
        "what": (
            "Retinal hemorrhages are areas of bleeding from damaged blood vessels. "
            "They can be dot-blot (deep retinal layers), flame-shaped (nerve fiber "
            "layer), or pre-retinal/vitreous. Their shape indicates which retinal "
            "layer is affected."
        ),
        "significance": (
            "Extensive hemorrhages in all 4 quadrants of the retina define severe "
            "NPDR (the '4-2-1 rule'). Hemorrhages indicate significant vascular "
            "damage and increased risk of progression to proliferative disease."
        ),
        "action": (
            "If mild: monitor every 6 months. If moderate-severe NPDR: referral "
            "to retina specialist. Consider panretinal photocoagulation (PRP) "
            "for high-risk characteristics."
        ),
    },
    "cotton_wool_spot": {
        "what": (
            "Cotton wool spots are fluffy, white, soft-edged patches caused by "
            "localized ischemia (lack of blood supply) in the nerve fiber layer. "
            "They represent micro-infarcts — tiny areas where retinal tissue has "
            "been deprived of oxygen."
        ),
        "significance": (
            "Their presence indicates retinal ischemia, which is a risk factor for "
            "progression to proliferative diabetic retinopathy. They're also seen "
            "in hypertensive retinopathy, HIV retinopathy, and collagen vascular diseases."
        ),
        "action": (
            "Investigate underlying cause (diabetes, hypertension, autoimmune). "
            "Consider fluorescein angiography to assess retinal perfusion. "
            "Monitor for development of neovascularization."
        ),
    },
    "neovascularization": {
        "what": (
            "Neovascularization is the growth of abnormal new blood vessels on the "
            "retinal surface or optic disc. These new vessels are fragile, thin-walled, "
            "and prone to bleeding. They grow in response to retinal ischemia as the "
            "retina releases VEGF (Vascular Endothelial Growth Factor)."
        ),
        "significance": (
            "This is the hallmark of proliferative diabetic retinopathy (PDR), "
            "the most severe stage. These vessels can bleed into the vitreous "
            "(vitreous hemorrhage) or cause tractional retinal detachment, both "
            "of which can cause sudden severe vision loss."
        ),
        "action": (
            "URGENT referral to retina specialist. Treatment options include: "
            "anti-VEGF injections (bevacizumab, ranibizumab, aflibercept), "
            "panretinal photocoagulation (PRP), or vitrectomy surgery for "
            "complications."
        ),
    },

    # Severity Levels
    "severity_0": {
        "what": "No apparent diabetic retinopathy. No microaneurysms or other diabetic lesions detected.",
        "action": "Continue annual diabetic eye screening. Maintain good glycemic control.",
    },
    "severity_1": {
        "what": (
            "Mild Non-Proliferative Diabetic Retinopathy (NPDR). Only microaneurysms "
            "are present. This is the earliest stage of diabetic eye disease."
        ),
        "action": "Annual screening. Optimize HbA1c, blood pressure, and lipid levels.",
    },
    "severity_2": {
        "what": (
            "Moderate NPDR. More than just microaneurysms — may include dot-blot "
            "hemorrhages, hard exudates, or cotton wool spots, but not meeting "
            "criteria for severe NPDR."
        ),
        "action": "Follow-up in 6 months. Consider referral if macular edema suspected.",
    },
    "severity_3": {
        "what": (
            "Severe NPDR. Extensive retinal changes meeting the '4-2-1 rule': "
            "hemorrhages in all 4 quadrants, OR venous beading in 2+ quadrants, "
            "OR IRMA in 1+ quadrant. High risk of progression to PDR (50% within 1 year)."
        ),
        "action": "Referral to retina specialist. Consider early PRP. Follow-up every 2-3 months.",
    },
    "severity_4": {
        "what": (
            "Proliferative Diabetic Retinopathy (PDR). Neovascularization detected. "
            "High risk of vitreous hemorrhage and tractional retinal detachment."
        ),
        "action": "URGENT referral. Anti-VEGF and/or PRP treatment required promptly.",
    },

    # Biomarkers
    "tortuosity": {
        "what": (
            "Vessel tortuosity measures how curved or winding the retinal blood "
            "vessels are. It's calculated as the ratio of actual vessel length to "
            "the straight-line distance between endpoints. Normal values are 1.05-1.20."
        ),
        "significance": (
            "Increased tortuosity is associated with systemic hypertension, diabetes, "
            "retinopathy of prematurity (ROP), and cardiovascular disease. "
            "It reflects chronic vascular remodeling due to hemodynamic stress."
        ),
    },
    "fractal_dimension": {
        "what": (
            "Fractal dimension (FD) quantifies the complexity of the retinal vascular "
            "tree. It's computed using box-counting: how the number of 'boxes' needed "
            "to cover the vessel network changes with box size. Normal range: 1.40-1.70."
        ),
        "significance": (
            "Low FD indicates sparse vasculature, possibly from ischemia or atrophic "
            "conditions. High FD indicates dense, complex branching. Changes in FD "
            "have been correlated with stroke risk and cognitive decline."
        ),
    },
    "avr": {
        "what": (
            "The Arteriovenous Ratio (AVR) is the ratio of the Central Retinal Artery "
            "Equivalent (CRAE) to the Central Retinal Vein Equivalent (CRVE). Normal "
            "range: 0.67-0.80. It's measured in Zone B (0.5-1.0 disc diameters from disc edge)."
        ),
        "significance": (
            "Low AVR (< 0.67) indicates arteriolar narrowing, which is a strong "
            "indicator of systemic hypertension and increased cardiovascular risk. "
            "High AVR may indicate venular dilation, associated with diabetes and inflammation."
        ),
    },
    "retinal_age": {
        "what": (
            "Retinal age is an estimate of the biological age of the retinal vasculature, "
            "computed from vessel features (tortuosity, fractal dimension, caliber). "
            "Research (Google Health/Nature Medicine) has shown retinal age can predict "
            "cardiovascular mortality independent of traditional risk factors."
        ),
        "significance": (
            "A positive 'retinal age gap' (retinal age > chronological age) suggests "
            "accelerated vascular aging and increased risk of cardiovascular events. "
            "Each year of retinal age gap increases mortality risk by approximately 2-3%."
        ),
    },

    # Glaucoma
    "glaucoma_cdr": {
        "what": (
            "The Cup-to-Disc Ratio (CDR) measures the size of the optic cup relative "
            "to the optic disc. Normal CDR is 0.2-0.5. Values > 0.6 are suspicious "
            "for glaucoma, where the optic nerve fibers are being lost."
        ),
        "significance": (
            "Elevated CDR is the hallmark structural sign of glaucoma, a leading cause "
            "of irreversible blindness. However, CDR alone is not diagnostic — "
            "asymmetry between eyes and progression over time are also important."
        ),
        "action": (
            "Refer for full glaucoma workup: IOP measurement, visual field testing "
            "(perimetry), OCT of the retinal nerve fiber layer (RNFL), and gonioscopy."
        ),
    },

    # General
    "referral": {
        "what": (
            "Based on the current findings, the following referral criteria apply:\n"
            "• Any proliferative disease → URGENT retina specialist referral\n"
            "• Severe NPDR → Retina specialist within 2-4 weeks\n"
            "• Moderate NPDR with macular edema → Ophthalmology within 1 month\n"
            "• Elevated CDR (>0.6) → Glaucoma specialist\n"
            "• Mild NPDR only → Annual screening"
        ),
    },
}


# ---- Context-Aware Suggestions --------------------------------

def generate_suggested_questions(
    anomaly_results: Optional[dict] = None,
    biomarker_results: Optional[dict] = None,
    severity_level: int = 0,
) -> list:
    """
    Generate context-aware suggested questions based on current findings.

    params:
        anomaly_results — from anomaly_detector
        biomarker_results — from biomarkers module
        severity_level — ICDR severity level
    returns: list of suggested question strings
    """
    suggestions = []

    # Always include general questions
    suggestions.append("What does this severity level mean?")

    # Finding-specific questions
    if anomaly_results:
        counts = anomaly_results.get("finding_counts", {})

        if counts.get("microaneurysm", 0) > 0:
            suggestions.append("What are microaneurysms and why do they matter?")
        if counts.get("hard_exudate", 0) > 0:
            suggestions.append("Are these exudates close to the macula?")
        if counts.get("hemorrhage", 0) > 0:
            suggestions.append("How serious are these hemorrhages?")
        if counts.get("cotton_wool_spot", 0) > 0:
            suggestions.append("What causes cotton wool spots?")

        # Optic disc
        optic_disc = anomaly_results.get("optic_disc", {})
        if optic_disc.get("glaucoma_flag"):
            suggestions.append("Should I be concerned about glaucoma?")

    # Biomarker questions
    if biomarker_results:
        tort = biomarker_results.get("tortuosity", {})
        if tort.get("mean_tortuosity", 0) > 1.2:
            suggestions.append("Why is the vessel tortuosity elevated?")

        retinal_age = biomarker_results.get("retinal_age", {})
        if retinal_age.get("age_gap") and retinal_age["age_gap"] > 3:
            suggestions.append("What does it mean that my retinal age is older than expected?")

    # Severity-specific
    if severity_level >= 2:
        suggestions.append("Should this patient be referred to a specialist?")
    if severity_level >= 3:
        suggestions.append("What treatment options are available?")

    # Always add
    suggestions.append("Explain the retinal age estimation")
    suggestions.append("What is the arteriovenous ratio?")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique[:8]  # Max 8 suggestions


# ---- Response Generation --------------------------------------

import os

DEFAULT_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def call_groq_copilot(question: str, context_str: str, groq_api_key: Optional[str] = None) -> Optional[str]:
    """
    Call Groq API using Llama-3.3-70b-versatile for ultra-fast clinical reasoning.
    """
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY") or DEFAULT_GROQ_API_KEY
    if not api_key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are RetinaAI Co-Pilot, an expert ophthalmic clinical AI assistant. "
                        "Provide concise, evidence-based clinical answers to the clinician's question based on the provided retinal analysis context."
                    )
                },
                {
                    "role": "user",
                    "content": f"Retinal Findings & Biomarkers Context:\n{context_str}\n\nQuestion: {question}"
                }
            ],
            temperature=0.3,
            max_tokens=600
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"⚠️ Groq API call error: {e}")
        return None


def generate_response(
    question: str,
    anomaly_results: Optional[dict] = None,
    biomarker_results: Optional[dict] = None,
    report: Optional[dict] = None,
    groq_api_key: Optional[str] = None,
) -> dict:
    """
    Generate an intelligent co-pilot response.
    First tries Groq API (Llama-3.3-70b), falling back to offline knowledge base.
    """
    question_lower = question.lower().strip()

    # Build context string
    context_parts = []
    if anomaly_results:
        sev = anomaly_results.get("severity", {})
        counts = anomaly_results.get("finding_counts", {})
        context_parts.append(f"ICDR Severity: Level {sev.get('level', 0)} ({sev.get('label', 'N/A')})")
        context_parts.append(f"Pathologies Detected: {counts}")
    if biomarker_results:
        tort = biomarker_results.get("tortuosity", {})
        fd = biomarker_results.get("fractal_dimension", {})
        context_parts.append(f"Vessel Tortuosity: {tort.get('mean_tortuosity', 'N/A')}")
        context_parts.append(f"Fractal Dimension: {fd.get('fractal_dimension', 'N/A')} D")
        context_parts.append(f"Retinal Age: {biomarker_results.get('retinal_age', {}).get('estimated_retinal_age', 'N/A')} yrs")

    context_str = "\n".join(context_parts) if context_parts else "No image context loaded."

    # Try Groq AI model first
    groq_reply = call_groq_copilot(question, context_str, groq_api_key)
    if groq_reply:
        return {
            "question": question,
            "topic": "Groq AI Clinical Assistant",
            "response": groq_reply,
            "source": "Groq (Llama-3.3-70b-versatile)",
            "timestamp": datetime.now().isoformat(),
        }

    # Keyword matching to knowledge base topics
    response_text = ""
    topic = ""
    references = []

    # Microaneurysm questions
    if any(kw in question_lower for kw in ["microaneurysm", "micro aneurysm", "small dot", "dark dot"]):
        kb = KNOWLEDGE_BASE["microaneurysm"]
        topic = "Microaneurysms"
        response_text = f"**{topic}**\n\n{kb['what']}\n\n**Clinical Significance:** {kb['significance']}\n\n**Recommended Action:** {kb['action']}"

        if anomaly_results:
            count = anomaly_results.get("finding_counts", {}).get("microaneurysm", 0)
            if count > 0:
                response_text += f"\n\n📊 **In this image:** {count} microaneurysm(s) detected."

    # Exudate questions
    elif any(kw in question_lower for kw in ["exudate", "bright spot", "yellow"]):
        kb = KNOWLEDGE_BASE["hard_exudate"]
        topic = "Hard Exudates"
        response_text = f"**{topic}**\n\n{kb['what']}\n\n**Clinical Significance:** {kb['significance']}\n\n**Recommended Action:** {kb['action']}"

        if anomaly_results:
            count = anomaly_results.get("finding_counts", {}).get("hard_exudate", 0)
            if count > 0:
                findings = [f for f in anomaly_results.get("findings", []) if f.get("finding_type") == "hard_exudate"]
                locations = set(f.get("quadrant", "") for f in findings)
                response_text += f"\n\n📊 **In this image:** {count} hard exudate(s) in {', '.join(locations)}."

    # Hemorrhage questions
    elif any(kw in question_lower for kw in ["hemorrhage", "bleeding", "bleed", "dark blotch"]):
        kb = KNOWLEDGE_BASE["hemorrhage"]
        topic = "Retinal Hemorrhages"
        response_text = f"**{topic}**\n\n{kb['what']}\n\n**Clinical Significance:** {kb['significance']}\n\n**Recommended Action:** {kb['action']}"

    # Cotton wool spots
    elif any(kw in question_lower for kw in ["cotton wool", "white patch", "fluffy"]):
        kb = KNOWLEDGE_BASE["cotton_wool_spot"]
        topic = "Cotton Wool Spots"
        response_text = f"**{topic}**\n\n{kb['what']}\n\n**Clinical Significance:** {kb['significance']}\n\n**Recommended Action:** {kb['action']}"

    # Severity questions
    elif any(kw in question_lower for kw in ["severity", "level", "icdr", "grade", "stage"]):
        level = 0
        if anomaly_results and anomaly_results.get("severity"):
            level = anomaly_results["severity"].get("level", 0)
        elif report and report.get("severity"):
            level = report["severity"].get("icdr_level", 0)

        kb = KNOWLEDGE_BASE.get(f"severity_{level}", {})
        topic = f"ICDR Level {level}"
        response_text = f"**{topic}**\n\n{kb.get('what', 'Information not available.')}"
        if kb.get("action"):
            response_text += f"\n\n**Recommended Action:** {kb['action']}"

    # Referral questions
    elif any(kw in question_lower for kw in ["refer", "specialist", "send to"]):
        kb = KNOWLEDGE_BASE["referral"]
        topic = "Referral Guidelines"
        response_text = f"**{topic}**\n\n{kb['what']}"

        if report and report.get("recommendation"):
            rec = report["recommendation"]
            response_text += f"\n\n📋 **For this patient:** {rec.get('action', 'N/A')} (Urgency: {rec.get('urgency', 'N/A')})"

    # Glaucoma questions
    elif any(kw in question_lower for kw in ["glaucoma", "cup to disc", "cdr", "optic nerve"]):
        kb = KNOWLEDGE_BASE["glaucoma_cdr"]
        topic = "Glaucoma Screening"
        response_text = f"**{topic}**\n\n{kb['what']}\n\n**Significance:** {kb['significance']}\n\n**Action:** {kb['action']}"

        if anomaly_results:
            od = anomaly_results.get("optic_disc", {})
            if od.get("detected"):
                cdr = od.get("cup_to_disc_ratio", 0)
                response_text += f"\n\n📊 **In this image:** CDR = {cdr}. {'⚠️ Elevated — referral recommended.' if cdr > 0.6 else '✅ Within normal limits.'}"

    # Tortuosity questions
    elif any(kw in question_lower for kw in ["tortuosity", "curved", "winding", "vessel curve"]):
        kb = KNOWLEDGE_BASE["tortuosity"]
        topic = "Vessel Tortuosity"
        response_text = f"**{topic}**\n\n{kb['what']}\n\n**Significance:** {kb['significance']}"

        if biomarker_results:
            tort = biomarker_results.get("tortuosity", {})
            response_text += f"\n\n📊 **In this image:** Mean tortuosity = {tort.get('mean_tortuosity', 'N/A')}. {tort.get('clinical_interpretation', '')}"

    # Fractal dimension
    elif any(kw in question_lower for kw in ["fractal", "complexity", "branching"]):
        kb = KNOWLEDGE_BASE["fractal_dimension"]
        topic = "Fractal Dimension"
        response_text = f"**{topic}**\n\n{kb['what']}\n\n**Significance:** {kb['significance']}"

    # AVR questions
    elif any(kw in question_lower for kw in ["avr", "arteriovenous", "artery vein ratio", "a/v ratio"]):
        kb = KNOWLEDGE_BASE["avr"]
        topic = "Arteriovenous Ratio"
        response_text = f"**{topic}**\n\n{kb['what']}\n\n**Significance:** {kb['significance']}"

        if biomarker_results:
            avr_data = biomarker_results.get("arteriovenous_ratio", {})
            response_text += f"\n\n📊 **In this image:** AVR = {avr_data.get('avr', 'N/A')}. {avr_data.get('clinical_interpretation', '')}"

    # Retinal age questions
    elif any(kw in question_lower for kw in ["retinal age", "biological age", "age estimate", "age gap"]):
        kb = KNOWLEDGE_BASE["retinal_age"]
        topic = "Retinal Age Estimation"
        response_text = f"**{topic}**\n\n{kb['what']}\n\n**Significance:** {kb['significance']}"

        if biomarker_results:
            ra = biomarker_results.get("retinal_age", {})
            if ra.get("estimated_age"):
                response_text += f"\n\n📊 **In this image:** Estimated retinal age = {ra['estimated_age']} years"
                if ra.get("age_gap") is not None:
                    response_text += f" (age gap: {ra['age_gap']:+.1f} years)"
                if ra.get("gap_interpretation"):
                    response_text += f". {ra['gap_interpretation']}"

    # Treatment questions
    elif any(kw in question_lower for kw in ["treatment", "treat", "cure", "option"]):
        topic = "Treatment Options"
        response_text = (
            "**Treatment Options for Diabetic Retinopathy**\n\n"
            "Treatment depends on the severity:\n\n"
            "• **Mild-Moderate NPDR:** Optimize systemic control (blood sugar, blood pressure, "
            "lipids). Regular monitoring.\n\n"
            "• **Severe NPDR / early PDR:** Panretinal Photocoagulation (PRP) — laser treatment "
            "to reduce oxygen demand and VEGF production.\n\n"
            "• **Macular edema:** Anti-VEGF injections (ranibizumab, aflibercept, bevacizumab) "
            "or focal/grid laser photocoagulation.\n\n"
            "• **Advanced PDR with complications:** Pars plana vitrectomy surgery for vitreous "
            "hemorrhage or tractional retinal detachment.\n\n"
            "Early detection and treatment can prevent up to 95% of vision loss from diabetic retinopathy."
        )

    # Generic / unmatched
    else:
        topic = "General"
        response_text = (
            "I can answer questions about the findings in this retinal image. "
            "Try asking about:\n\n"
            "• Specific findings (microaneurysms, exudates, hemorrhages)\n"
            "• Severity level and what it means\n"
            "• Whether a referral is needed\n"
            "• Retinal biomarkers (tortuosity, fractal dimension, AVR)\n"
            "• Retinal age estimation\n"
            "• Glaucoma screening (cup-to-disc ratio)\n"
            "• Available treatment options"
        )

    return {
        "response": response_text,
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "is_contextual": anomaly_results is not None or biomarker_results is not None,
    }
