"""
src/analysis/progression.py
=============================
Pillar 7 — Disease Progression Tracker

Multi-visit analysis for tracking retinal disease over time:
  1. Image registration — align images using optic disc as anchor
  2. Differential analysis — pixel-wise changes between visits
  3. Finding comparison — new vs resolved vs worsening anomalies
  4. Progression scoring — single metric from -1 (improving) to +1 (worsening)
  5. Timeline data — structured output for frontend timeline visualization
"""

import cv2
import numpy as np
from typing import Optional
from datetime import datetime


# ---- Image Registration (Optic Disc-Based) --------------------

def register_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    disc_a: Optional[dict] = None,
    disc_b: Optional[dict] = None,
) -> tuple:
    """
    Align two retinal images using the optic disc as the anchor point.
    Uses affine transformation based on disc center and feature matching.

    params:
        image_a — (H, W, 3) float32 reference image (earlier visit)
        image_b — (H, W, 3) float32 target image (later visit)
        disc_a — optic disc info for image_a {x, y, radius}
        disc_b — optic disc info for image_b {x, y, radius}
    returns: (aligned_b, transform_matrix) where aligned_b matches image_a's geometry
    """
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]

    # First resize image_b to match image_a
    if (h_a, w_a) != (h_b, w_b):
        image_b = cv2.resize(image_b, (w_a, h_a), interpolation=cv2.INTER_LINEAR)

    # If optic disc positions are available, use them for rigid alignment
    if disc_a and disc_b and disc_a.get("detected") and disc_b.get("detected"):
        dx = disc_a["x"] - disc_b["x"]
        dy = disc_a["y"] - disc_b["y"]

        # Scale factor from disc sizes
        scale = disc_a.get("radius", 80) / max(disc_b.get("radius", 80), 1)
        scale = max(0.8, min(1.2, scale))  # Clamp scale

        # Build affine transform: translate + scale around disc center
        center_b = (float(disc_b["x"]), float(disc_b["y"]))
        M = cv2.getRotationMatrix2D(center_b, 0, scale)
        M[0, 2] += dx
        M[1, 2] += dy

        aligned = cv2.warpAffine(image_b, M, (w_a, h_a),
                                  borderMode=cv2.BORDER_REFLECT_101)
        return aligned, M

    # Fallback: ORB feature matching
    gray_a = cv2.cvtColor((np.clip(image_a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor((np.clip(image_b, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=500)
    kp_a, desc_a = orb.detectAndCompute(gray_a, None)
    kp_b, desc_b = orb.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        # Can't register — return resized but unaligned
        return image_b, np.eye(2, 3, dtype=np.float32)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_a, desc_b)
    matches = sorted(matches, key=lambda m: m.distance)[:50]

    if len(matches) < 4:
        return image_b, np.eye(2, 3, dtype=np.float32)

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])

    M, inliers = cv2.estimateAffinePartial2D(pts_b, pts_a, method=cv2.RANSAC)
    if M is None:
        return image_b, np.eye(2, 3, dtype=np.float32)

    aligned = cv2.warpAffine(image_b, M, (w_a, h_a),
                              borderMode=cv2.BORDER_REFLECT_101)
    return aligned, M


# ---- Differential Analysis ------------------------------------

def compute_visit_difference(
    image_earlier: np.ndarray,
    image_later: np.ndarray,
) -> dict:
    """
    Compute pixel-wise differences between two registered visit images.

    params:
        image_earlier — (H, W, 3) float32, earlier visit
        image_later — (H, W, 3) float32, later visit (aligned)
    returns: dict with difference maps and metrics
    """
    h, w = image_earlier.shape[:2]

    # Per-pixel L2 difference
    diff = np.sqrt(np.sum((image_earlier.astype(np.float64) - image_later.astype(np.float64))**2, axis=2))

    # Significant change mask
    change_threshold = 0.08
    changed_mask = diff > change_threshold

    # Regions getting brighter vs darker (in green channel for vessel contrast)
    green_earlier = image_earlier[:, :, 1]
    green_later = image_later[:, :, 1]
    brightness_change = green_later.astype(np.float64) - green_earlier.astype(np.float64)

    brighter_mask = (brightness_change > 0.05) & changed_mask
    darker_mask = (brightness_change < -0.05) & changed_mask

    return {
        "difference_map": diff.astype(np.float32),
        "changed_mask": changed_mask,
        "brighter_regions": brighter_mask,
        "darker_regions": darker_mask,
        "mean_change": float(np.mean(diff)),
        "max_change": float(np.max(diff)),
        "change_coverage_percent": round(100.0 * np.sum(changed_mask) / (h * w), 1),
    }


# ---- Finding Comparison (New / Resolved / Worsening) ----------

def compare_findings(
    findings_earlier: list,
    findings_later: list,
    match_radius: int = 40,
) -> dict:
    """
    Compare anomaly findings between two visits.

    Matches findings by spatial proximity and type:
    - New: present in later, not in earlier
    - Resolved: present in earlier, not in later
    - Stable: present in both, similar size
    - Worsening: present in both, larger/higher confidence in later

    params:
        findings_earlier — list of finding dicts from earlier visit
        findings_later — list of finding dicts from later visit
        match_radius — max pixel distance to consider same finding
    returns: dict with categorized findings and summary
    """
    matched_earlier = set()
    matched_later = set()

    stable = []
    worsening = []

    # Match findings by proximity and type
    for j, f_later in enumerate(findings_later):
        best_match = None
        best_dist = match_radius + 1

        for i, f_earlier in enumerate(findings_earlier):
            if i in matched_earlier:
                continue
            if f_earlier["finding_type"] != f_later["finding_type"]:
                continue

            dist = np.sqrt(
                (f_earlier["x"] - f_later["x"])**2 +
                (f_earlier["y"] - f_later["y"])**2
            )

            if dist < best_dist:
                best_dist = dist
                best_match = i

        if best_match is not None:
            matched_earlier.add(best_match)
            matched_later.add(j)

            # Check if worsening (larger radius or higher confidence)
            f_earlier = findings_earlier[best_match]
            size_change = f_later.get("radius", 10) / max(f_earlier.get("radius", 10), 1)
            conf_change = f_later.get("confidence", 0.5) - f_earlier.get("confidence", 0.5)

            if size_change > 1.3 or conf_change > 0.15:
                worsening.append({
                    "finding": f_later,
                    "previous": f_earlier,
                    "size_change": round(size_change, 2),
                    "status": "worsening",
                })
            else:
                stable.append({
                    "finding": f_later,
                    "previous": f_earlier,
                    "status": "stable",
                })

    # New findings (in later but not matched)
    new_findings = [
        {"finding": findings_later[j], "status": "new"}
        for j in range(len(findings_later))
        if j not in matched_later
    ]

    # Resolved findings (in earlier but not matched)
    resolved_findings = [
        {"finding": findings_earlier[i], "status": "resolved"}
        for i in range(len(findings_earlier))
        if i not in matched_earlier
    ]

    return {
        "new": new_findings,
        "resolved": resolved_findings,
        "stable": stable,
        "worsening": worsening,
        "summary": {
            "new_count": len(new_findings),
            "resolved_count": len(resolved_findings),
            "stable_count": len(stable),
            "worsening_count": len(worsening),
        },
    }


# ---- Progression Score ----------------------------------------

def compute_progression_score(
    findings_comparison: dict,
    severity_earlier: dict,
    severity_later: dict,
) -> dict:
    """
    Compute a single progression score from -1 (improving) to +1 (worsening).

    Factors:
    - ICDR level change
    - New vs resolved finding ratio
    - Worsening finding count

    params:
        findings_comparison — output of compare_findings()
        severity_earlier — severity grade dict from earlier visit
        severity_later — severity grade dict from later visit
    returns: dict with score, trend label, and recommendation
    """
    summary = findings_comparison["summary"]

    # ICDR level change (strongest signal)
    level_change = severity_later.get("level", 0) - severity_earlier.get("level", 0)

    # Finding balance
    new_count = summary["new_count"]
    resolved_count = summary["resolved_count"]
    worsening_count = summary["worsening_count"]

    # Weighted score
    score = 0.0
    score += level_change * 0.4         # ICDR change is weighted heavily
    score += (new_count - resolved_count) * 0.05  # Net new findings
    score += worsening_count * 0.1      # Worsening findings

    # Clamp to [-1, 1]
    score = max(-1.0, min(1.0, score))

    # Trend label
    if score < -0.2:
        trend = "improving"
        recommendation = "Positive trend. Continue current treatment. Follow-up in 12 months."
    elif score < 0.1:
        trend = "stable"
        recommendation = "Disease appears stable. Routine follow-up in 6-12 months."
    elif score < 0.4:
        trend = "mildly_worsening"
        recommendation = "Mild progression noted. Consider closer monitoring (3-6 months)."
    else:
        trend = "significantly_worsening"
        recommendation = "Significant progression detected. Referral to retina specialist recommended."

    return {
        "score": round(score, 3),
        "trend": trend,
        "recommendation": recommendation,
        "icdr_change": level_change,
        "finding_balance": new_count - resolved_count,
    }


# ---- Master Progression Analysis Function ---------------------

def analyze_progression(
    image_earlier: np.ndarray,
    image_later: np.ndarray,
    findings_earlier: Optional[list] = None,
    severity_earlier: Optional[dict] = None,
    disc_earlier: Optional[dict] = None,
    disc_later: Optional[dict] = None,
    visit_date_earlier: Optional[str] = None,
    visit_date_later: Optional[str] = None,
) -> dict:
    """
    Run the full disease progression analysis between two visits.

    params:
        image_earlier, image_later — (H, W, 3) float32 in [0, 1]
        findings_earlier — pre-computed findings from earlier visit (or None to compute)
        severity_earlier — pre-computed severity from earlier visit (or None to compute)
        disc_earlier/later — optic disc info for registration
        visit_date_earlier/later — ISO date strings
    returns: dict with all progression analysis results
    """
    from src.analysis.anomaly_detector import detect_anomalies

    # Step 1: Register images
    aligned_later, transform = register_images(
        image_earlier, image_later, disc_earlier, disc_later
    )

    # Step 2: Compute visit difference
    visit_diff = compute_visit_difference(image_earlier, aligned_later)

    # Step 3: Get findings for both if not provided
    if findings_earlier is None or severity_earlier is None:
        result_earlier = detect_anomalies(image_earlier)
        findings_earlier = result_earlier["findings"] if findings_earlier is None else findings_earlier
        severity_earlier = result_earlier["severity"] if severity_earlier is None else severity_earlier

    result_later = detect_anomalies(aligned_later)
    findings_later = result_later["findings"]
    severity_later = result_later["severity"]

    # Step 4: Compare findings
    findings_comparison = compare_findings(findings_earlier, findings_later)

    # Step 5: Compute progression score
    progression = compute_progression_score(
        findings_comparison, severity_earlier, severity_later
    )

    return {
        "visit_dates": {
            "earlier": visit_date_earlier or "Unknown",
            "later": visit_date_later or "Unknown",
        },
        "registration": {
            "aligned_image": aligned_later,  # (H, W, 3) float32
            "transform_matrix": transform.tolist() if isinstance(transform, np.ndarray) else transform,
        },
        "visit_difference": {
            "mean_change": visit_diff["mean_change"],
            "max_change": visit_diff["max_change"],
            "change_coverage_percent": visit_diff["change_coverage_percent"],
            "difference_map": visit_diff["difference_map"],  # (H, W) float32
        },
        "severity_comparison": {
            "earlier": severity_earlier,
            "later": severity_later,
        },
        "findings_comparison": {
            "new": findings_comparison["new"],
            "resolved": findings_comparison["resolved"],
            "stable": findings_comparison["stable"],
            "worsening": findings_comparison["worsening"],
            "summary": findings_comparison["summary"],
        },
        "progression": progression,
    }
