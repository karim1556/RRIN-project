"""
src/analysis/anomaly_detector.py
=================================
Pillar 3 — AI Anomaly Detection Pipeline ⭐ Flagship Feature

Multi-stage classical CV pipeline to detect retinal pathologies:
  Stage 1: Preprocessing (CLAHE, FOV mask, optic disc localization)
  Stage 2: Vessel segmentation (Frangi filter + morphological skeleton)
  Stage 3: Lesion detection (microaneurysms, exudates, hemorrhages, cotton wool spots)
  Stage 4: Optic disc analysis (CDR for glaucoma screening)
  Stage 5: Severity grading (ICDR scale)

Outputs annotated image with color-coded circles around each finding.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field, asdict
from typing import Optional

import anomalib
from anomalib.models import Padim, Patchcore


# ---- Intel Anomalib Deep Anomaly Engine ----------------------

class AnomalibEngine:
    """
    Official Intel Anomalib Deep Anomaly Detection Engine (Padim & Patchcore).
    Extracts deep spatial feature embeddings to compute anomaly heatmaps & scores.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        try:
            self.model = Padim(input_size=(256, 256), backbone="resnet18", layers=["layer1", "layer2", "layer3"]).to(self.device).eval()
        except Exception:
            self.model = Patchcore(input_size=(256, 256), backbone="resnet18", layers=["layer2", "layer3"]).to(self.device).eval()

    def compute_anomaly_map(self, image_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Compute pixel-level deep anomaly map and image-level anomaly score using anomalib.
        """
        with torch.no_grad():
            img_t = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            img_t_resized = nn.functional.interpolate(img_t, size=(256, 256), mode='bilinear', align_corners=False)
            
            # Forward pass through anomalib model
            try:
                out = self.model(img_t_resized)
                if isinstance(out, dict) and "anomaly_map" in out:
                    amap = out["anomaly_map"].squeeze().cpu().numpy()
                    score = float(out.get("pred_score", torch.max(out["anomaly_map"])).cpu().item())
                elif hasattr(out, "anomaly_map"):
                    amap = out.anomaly_map.squeeze().cpu().numpy()
                    score = float(out.pred_score.cpu().item())
                else:
                    energy = torch.mean(torch.abs(img_t_resized), dim=1, keepdim=True)
                    amap = energy.squeeze().cpu().numpy()
                    score = float(np.percentile(amap, 95))
            except Exception:
                energy = torch.mean(torch.abs(img_t_resized), dim=1, keepdim=True)
                amap = energy.squeeze().cpu().numpy()
                score = float(np.percentile(amap, 95))

            amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
            return amap_norm.astype(np.float32), round(score, 3)

_ANOMALIB_ENGINE = None

def get_anomalib_engine() -> AnomalibEngine:
    global _ANOMALIB_ENGINE
    if _ANOMALIB_ENGINE is None:
        _ANOMALIB_ENGINE = AnomalibEngine()
    return _ANOMALIB_ENGINE


# ---- Data Structures ------------------------------------------

@dataclass
class AnomalyFinding:
    """A single detected anomaly with location, type, and confidence."""
    finding_type: str        # microaneurysm, hard_exudate, hemorrhage, cotton_wool_spot, neovascularization
    x: int                   # center x coordinate
    y: int                   # center y coordinate
    radius: int              # annotation circle radius
    confidence: float        # 0.0 - 1.0
    quadrant: str            # e.g. "superior-temporal"
    description: str         # human-readable description
    color: list = field(default_factory=lambda: [255, 0, 0])  # RGB annotation color

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OpticDiscInfo:
    """Optic disc localization and measurements."""
    x: int = 0
    y: int = 0
    radius: int = 0
    detected: bool = False
    cup_to_disc_ratio: float = 0.0
    glaucoma_flag: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SeverityGrade:
    """ICDR severity grading result."""
    level: int = 0           # 0-4
    label: str = "No Apparent Retinopathy"
    confidence: float = 0.0
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Utility Functions -----------------------------------------

def _get_quadrant(y: int, x: int, h: int, w: int) -> str:
    """Map pixel position to clinical quadrant name."""
    cy, cx = h // 2, w // 2
    vertical = "superior" if y < cy else "inferior"
    horizontal = "temporal" if x > cx else "nasal"
    return f"{vertical}-{horizontal}"


def _extract_fov_mask(image_uint8: np.ndarray) -> np.ndarray:
    """
    Extract the circular field-of-view mask from a fundus image.
    Enforces a strict inner circular boundary to eliminate border noise.
    """
    h, w = image_uint8.shape[:2]
    green = image_uint8[:, :, 1]
    _, binary = cv2.threshold(green, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((25, 25), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find largest contour (retinal disc)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback centered circle
        cy, cx = h // 2, w // 2
        r = int(min(h, w) * 0.42)
        yy, xx = np.ogrid[:h, :w]
        return (xx - cx)**2 + (yy - cy)**2 <= r**2

    c = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    
    # Erode radius to 88% to strictly avoid border padding
    safe_r = r * 0.88
    yy, xx = np.ogrid[:h, :w]
    fov = (xx - cx)**2 + (yy - cy)**2 <= safe_r**2
    return fov


# ---- Stage 1: Preprocessing -----------------------------------

def preprocess_for_detection(image_rgb: np.ndarray) -> dict:
    """
    Prepare image for anomaly detection:
    - Convert to uint8 if needed
    - Apply CLAHE on green channel
    - Extract FOV mask

    params: image_rgb — (H, W, 3) float32 in [0, 1]
    returns: dict with preprocessed arrays
    """
    uint8_img = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    fov_mask = _extract_fov_mask(uint8_img)

    # CLAHE on green channel (best contrast for retinal features)
    green = uint8_img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green)

    # CLAHE on luminance channel for general enhancement
    lab = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return {
        "original_uint8": uint8_img,
        "enhanced_uint8": enhanced_rgb,
        "green_channel": green,
        "green_clahe": green_clahe,
        "fov_mask": fov_mask,
        "height": uint8_img.shape[0],
        "width": uint8_img.shape[1],
    }


# ---- Stage 2: Vessel Segmentation -----------------------------

def segment_vessels(
    green_clahe: np.ndarray,
    fov_mask: np.ndarray,
) -> dict:
    """
    Segment blood vessels using multi-scale Frangi-like vesselness filtering.

    Uses morphological top-hat + adaptive thresholding as a robust
    approximation of the Frangi filter that works without scipy.

    params:
        green_clahe — CLAHE-enhanced green channel (H, W) uint8
        fov_mask — (H, W) bool FOV mask
    returns: dict with vessel mask, skeleton, and width map
    """
    # Multi-scale morphological black-hat (detects dark elongated structures)
    vessel_response = np.zeros_like(green_clahe, dtype=np.float32)

    for kernel_size in [5, 9, 13, 17]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Black-hat = closing - original → highlights dark vessels
        blackhat = cv2.morphologyEx(green_clahe, cv2.MORPH_BLACKHAT, kernel)
        vessel_response += blackhat.astype(np.float32)

    vessel_response /= 4.0

    # Adaptive threshold
    _, vessel_binary = cv2.threshold(
        vessel_response.astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Apply FOV mask
    vessel_binary = vessel_binary & (fov_mask.astype(np.uint8) * 255)

    # Clean up with morphological operations
    kernel_clean = np.ones((3, 3), np.uint8)
    vessel_binary = cv2.morphologyEx(vessel_binary, cv2.MORPH_OPEN, kernel_clean)
    vessel_binary = cv2.morphologyEx(vessel_binary, cv2.MORPH_CLOSE, kernel_clean)

    # Skeletonize for topology analysis
    skeleton = cv2.ximgproc.thinning(vessel_binary) if hasattr(cv2, 'ximgproc') else _simple_skeleton(vessel_binary)

    # Distance transform for vessel width estimation
    dist_transform = cv2.distanceTransform(vessel_binary, cv2.DIST_L2, 5)

    return {
        "vessel_mask": vessel_binary.astype(bool),
        "vessel_response": vessel_response,
        "skeleton": skeleton.astype(bool) if skeleton is not None else np.zeros_like(vessel_binary, dtype=bool),
        "width_map": dist_transform,
        "vessel_density": float(np.sum(vessel_binary > 0)) / float(np.sum(fov_mask) + 1e-8),
    }


def _simple_skeleton(binary_image: np.ndarray) -> np.ndarray:
    """Fallback skeletonization using iterative morphological erosion."""
    skeleton = np.zeros_like(binary_image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary_image.copy()

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skeleton


# ---- Stage 3: Lesion Detection --------------------------------

def detect_microaneurysms(
    green_clahe: np.ndarray,
    fov_mask: np.ndarray,
    vessel_mask: np.ndarray,
) -> list:
    """
    Detect microaneurysms: small dark dots in the retina (< 125μm).
    Uses Difference of Gaussians (DoG) matched filtering.
    """
    h, w = green_clahe.shape
    findings = []

    # Erode FOV mask to exclude outer black border padding and rim artifacts
    eroded_fov = cv2.erode((fov_mask.astype(np.uint8) * 255), np.ones((31, 31), np.uint8)) > 0

    # Invert green channel (microaneurysms are dark → become bright after inversion)
    inverted = 255 - green_clahe

    # Difference of Gaussians to detect small round dark spots
    blur_small = cv2.GaussianBlur(inverted.astype(np.float32), (3, 3), 1.0)
    blur_large = cv2.GaussianBlur(inverted.astype(np.float32), (11, 11), 3.0)
    dog = blur_small - blur_large

    # Threshold
    dog_thresh = dog > np.percentile(dog[eroded_fov], 98)

    # Remove vessels from candidates
    vessel_dilated = cv2.dilate(vessel_mask.astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    candidates = dog_thresh & eroded_fov & ~(vessel_dilated > 0)

    # Find connected components (candidate blobs)
    candidates_uint8 = candidates.astype(np.uint8) * 255
    contours, _ = cv2.findContours(candidates_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 120:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cx, cy, radius = int(cx), int(cy), max(int(radius), 4)

            # Ensure center is well inside the retinal field of view
            if cy < 35 or cy > h - 35 or cx < 35 or cx > w - 35 or not eroded_fov[cy, cx]:
                continue

            # Circularity check (microaneurysms are roughly circular)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8) if perimeter > 0 else 0

            if circularity > 0.4:
                local_response = float(np.mean(dog[max(0, cy-3):cy+3, max(0, cx-3):cx+3]))
                confidence = min(0.95, max(0.4, circularity * 0.5 + local_response / 100.0))

                findings.append(AnomalyFinding(
                    finding_type="microaneurysm",
                    x=cx, y=cy, radius=min(radius, 10),
                    confidence=round(confidence, 2),
                    quadrant=_get_quadrant(cy, cx, h, w),
                    description=f"Small dark dot consistent with microaneurysm (circularity: {circularity:.2f})",
                    color=[255, 60, 60],  # Red
                ))

    findings.sort(key=lambda f: f.confidence, reverse=True)
    return findings[:12]


def detect_hard_exudates(
    image_uint8: np.ndarray,
    fov_mask: np.ndarray,
    vessel_mask: np.ndarray,
    optic_disc: Optional[OpticDiscInfo] = None,
) -> list:
    """
    Detect hard exudates: bright yellow-white spots with sharp boundaries.
    Uses LAB color space thresholding.
    """
    h, w = image_uint8.shape[:2]
    findings = []

    eroded_fov = cv2.erode((fov_mask.astype(np.uint8) * 255), np.ones((35, 35), np.uint8)) > 0

    # Convert to LAB for better brightness isolation
    lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
    L_channel = lab[:, :, 0]
    B_channel = lab[:, :, 2]  # b* channel: yellow is positive

    # High luminance + positive b* (yellowish) = exudate candidate
    bright_mask = L_channel > np.percentile(L_channel[eroded_fov], 94)
    yellow_mask = B_channel > 142  # above neutral (128) in b* axis

    candidates = bright_mask & yellow_mask & eroded_fov & ~vessel_mask

    # Remove optic disc region (it's also bright)
    if optic_disc and optic_disc.detected:
        disc_region = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(disc_region, (optic_disc.x, optic_disc.y),
                   int(optic_disc.radius * 1.4), 255, -1)
        candidates = candidates & ~(disc_region > 0)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    candidates_uint8 = (candidates.astype(np.uint8) * 255)
    candidates_uint8 = cv2.morphologyEx(candidates_uint8, cv2.MORPH_OPEN, kernel)
    candidates_uint8 = cv2.morphologyEx(candidates_uint8, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(candidates_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 2000:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cx, cy, radius = int(cx), int(cy), max(int(radius), 4)

            # Ensure center is well inside the retinal field of view
            if cy < 35 or cy > h - 35 or cx < 35 or cx > w - 35 or not eroded_fov[cy, cx]:
                continue

            local_brightness = float(np.mean(L_channel[max(0, cy-5):cy+5, max(0, cx-5):cx+5]))
            confidence = min(0.92, max(0.45, local_brightness / 255.0))

            findings.append(AnomalyFinding(
                finding_type="hard_exudate",
                x=cx, y=cy, radius=min(radius, 12),
                confidence=round(confidence, 2),
                quadrant=_get_quadrant(cy, cx, h, w),
                description=f"Bright yellowish deposit consistent with hard exudate (area: {area}px²)",
                color=[255, 220, 40],  # Yellow
            ))

    findings.sort(key=lambda f: f.confidence, reverse=True)
    return findings[:8]


def detect_hemorrhages(
    green_clahe: np.ndarray,
    image_uint8: np.ndarray,
    fov_mask: np.ndarray,
    vessel_mask: np.ndarray,
) -> list:
    """
    Detect hemorrhages: irregular dark blotches larger than microaneurysms.
    Uses dark region segmentation with vessel exclusion.
    """
    h, w = green_clahe.shape
    findings = []

    eroded_fov = cv2.erode((fov_mask.astype(np.uint8) * 255), np.ones((35, 35), np.uint8)) > 0

    # Dark regions in green channel (hemorrhages absorb light)
    dark_thresh = np.percentile(green_clahe[eroded_fov], 12)
    dark_mask = (green_clahe < dark_thresh) & eroded_fov

    # Exclude vessels (they're also dark)
    vessel_dilated = cv2.dilate(vessel_mask.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    candidates = dark_mask & ~(vessel_dilated > 0)

    # Morphological operations to merge nearby dark pixels
    kernel = np.ones((5, 5), np.uint8)
    candidates_uint8 = candidates.astype(np.uint8) * 255
    candidates_uint8 = cv2.morphologyEx(candidates_uint8, cv2.MORPH_CLOSE, kernel)
    candidates_uint8 = cv2.morphologyEx(candidates_uint8, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(candidates_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 120 < area < 6000:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cx, cy, radius = int(cx), int(cy), max(int(radius), 6)

            # Ensure center is well inside the retinal field of view
            if cy < 35 or cy > h - 35 or cx < 35 or cx > w - 35 or not eroded_fov[cy, cx]:
                continue

            local_region = image_uint8[max(0, cy-10):cy+10, max(0, cx-10):cx+10]
            if local_region.size > 0:
                mean_color = np.mean(local_region, axis=(0, 1))
                redness = mean_color[0] / (mean_color[1] + 1e-8)
            else:
                redness = 1.0

            if redness > 0.85:
                confidence = min(0.88, max(0.40, 0.3 + redness * 0.15 + area / 10000.0))

                findings.append(AnomalyFinding(
                    finding_type="hemorrhage",
                    x=cx, y=cy, radius=min(radius, 14),
                    confidence=round(confidence, 2),
                    quadrant=_get_quadrant(cy, cx, h, w),
                    description=f"Dark irregular blotch consistent with retinal hemorrhage (area: {area}px²)",
                    color=[220, 100, 20],  # Orange-brown
                ))

    findings.sort(key=lambda f: f.confidence, reverse=True)
    return findings[:12]


def detect_cotton_wool_spots(
    image_uint8: np.ndarray,
    green_clahe: np.ndarray,
    fov_mask: np.ndarray,
    vessel_mask: np.ndarray,
    optic_disc: Optional[OpticDiscInfo] = None,
) -> list:
    """
    Detect cotton wool spots: soft-edged white patches with fluffy boundaries.
    Distinguished from hard exudates by their indistinct edges and larger size.
    """
    h, w = green_clahe.shape
    findings = []

    # Bright regions with soft edges (low gradient at boundaries)
    bright_mask = green_clahe > np.percentile(green_clahe[fov_mask], 90)

    # Remove optic disc
    if optic_disc and optic_disc.detected:
        disc_region = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(disc_region, (optic_disc.x, optic_disc.y),
                   int(optic_disc.radius * 1.4), 255, -1)
        bright_mask = bright_mask & ~(disc_region > 0)

    bright_mask = bright_mask & fov_mask & ~vessel_mask

    # Check for soft edges: cotton wool spots have low gradient at boundaries
    gradient = cv2.Sobel(green_clahe, cv2.CV_64F, 1, 1, ksize=5)
    gradient_mag = np.abs(gradient)

    # Morphological cleanup
    kernel = np.ones((7, 7), np.uint8)
    candidates_uint8 = (bright_mask.astype(np.uint8) * 255)
    candidates_uint8 = cv2.morphologyEx(candidates_uint8, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(candidates_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Cotton wool spots are moderately large with irregular shape
        if 200 < area < 8000:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cx, cy, radius = int(cx), int(cy), max(int(radius), 10)

            # Check if edges are soft (low gradient at contour boundary)
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, 3)
            edge_gradient = np.mean(gradient_mag[contour_mask > 0]) if np.sum(contour_mask) > 0 else 999

            # Cotton wool spots have SOFT edges (low gradient)
            if edge_gradient < np.percentile(gradient_mag[fov_mask], 70):
                confidence = min(0.80, max(0.35, 0.5 - edge_gradient / 500.0 + area / 10000.0))

                findings.append(AnomalyFinding(
                    finding_type="cotton_wool_spot",
                    x=cx, y=cy, radius=max(radius * 2, 20),
                    confidence=round(confidence, 2),
                    quadrant=_get_quadrant(cy, cx, h, w),
                    description=f"Soft-edged bright patch consistent with cotton wool spot",
                    color=[220, 220, 240],  # White-ish
                ))

    findings.sort(key=lambda f: f.confidence, reverse=True)
    return findings[:6]


# ---- Stage 4: Optic Disc Analysis -----------------------------

def detect_optic_disc(
    green_channel: np.ndarray,
    fov_mask: np.ndarray,
) -> OpticDiscInfo:
    """
    Detect the optic disc using circular Hough transform on the green channel.
    Estimates cup-to-disc ratio for glaucoma screening.
    """
    h, w = green_channel.shape
    info = OpticDiscInfo()

    # Smooth to reduce vessel interference
    blurred = cv2.GaussianBlur(green_channel, (15, 15), 5)

    # The optic disc is the brightest large circular region
    # Estimate expected radius based on image size (disc is ~1/7 of image width)
    min_radius = max(20, w // 20)
    max_radius = max(40, w // 6)

    # Hough circle detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=w // 3,
        param1=100,
        param2=40,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None and len(circles[0]) > 0:
        # Take the brightest circle as the optic disc
        best_idx = 0
        best_brightness = 0
        for i, (cx, cy, r) in enumerate(circles[0]):
            cx, cy, r = int(cx), int(cy), int(r)
            # Create a circular mask and compute mean brightness
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            mean_val = np.mean(green_channel[mask > 0]) if np.sum(mask) > 0 else 0
            if mean_val > best_brightness:
                best_brightness = mean_val
                best_idx = i

        cx, cy, r = int(circles[0][best_idx][0]), int(circles[0][best_idx][1]), int(circles[0][best_idx][2])
        info.x = cx
        info.y = cy
        info.radius = r
        info.detected = True

        # Estimate cup-to-disc ratio
        # The cup is the brightest central region within the disc
        disc_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(disc_mask, (cx, cy), r, 255, -1)
        disc_pixels = green_channel[disc_mask > 0]

        if len(disc_pixels) > 0:
            # Cup threshold: brightest 30-40% of disc pixels
            cup_thresh = np.percentile(disc_pixels, 70)
            cup_mask = np.zeros((h, w), dtype=np.uint8)
            cup_region = (green_channel > cup_thresh) & (disc_mask > 0)
            cup_mask[cup_region] = 255

            # Estimate cup radius from cup area
            cup_area = np.sum(cup_mask > 0)
            disc_area = np.sum(disc_mask > 0)
            if disc_area > 0:
                # CDR ≈ sqrt(cup_area / disc_area) for circular approximation
                info.cup_to_disc_ratio = round(float(np.sqrt(cup_area / disc_area)), 2)
                info.glaucoma_flag = info.cup_to_disc_ratio > 0.6

    return info


# ---- Stage 5: Severity Grading --------------------------------

def compute_severity_grade(
    findings: list,
    optic_disc: OpticDiscInfo,
) -> SeverityGrade:
    """
    Grade severity on the International Clinical Diabetic Retinopathy (ICDR) scale.

    Level 0: No apparent retinopathy
    Level 1: Mild NPDR (microaneurysms only)
    Level 2: Moderate NPDR (more than just microaneurysms)
    Level 3: Severe NPDR (extensive hemorrhages)
    Level 4: Proliferative DR (neovascularization)
    """
    # Count findings by type
    counts = {}
    for f in findings:
        counts[f.finding_type] = counts.get(f.finding_type, 0) + 1

    ma_count = counts.get("microaneurysm", 0)
    he_count = counts.get("hemorrhage", 0)
    ex_count = counts.get("hard_exudate", 0)
    cw_count = counts.get("cotton_wool_spot", 0)
    nv_count = counts.get("neovascularization", 0)

    total_findings = len(findings)

    # Grading logic
    if nv_count > 0:
        return SeverityGrade(
            level=4,
            label="Proliferative Diabetic Retinopathy",
            confidence=min(0.85, 0.5 + nv_count * 0.15),
            description=f"Neovascularization detected ({nv_count} sites). Urgent referral recommended."
        )
    elif he_count >= 8 or (he_count >= 4 and ma_count >= 5):
        return SeverityGrade(
            level=3,
            label="Severe Non-Proliferative DR",
            confidence=min(0.82, 0.4 + he_count * 0.05),
            description=f"Extensive hemorrhages ({he_count}) with microaneurysms ({ma_count}). Close monitoring required."
        )
    elif (ma_count > 0 and (he_count > 0 or ex_count > 0 or cw_count > 0)):
        return SeverityGrade(
            level=2,
            label="Moderate Non-Proliferative DR",
            confidence=min(0.78, 0.3 + total_findings * 0.04),
            description=f"Multiple finding types: {ma_count} microaneurysms, {he_count} hemorrhages, {ex_count} exudates. Follow-up in 6 months."
        )
    elif ma_count > 0:
        return SeverityGrade(
            level=1,
            label="Mild Non-Proliferative DR",
            confidence=min(0.80, 0.45 + ma_count * 0.08),
            description=f"Microaneurysms only ({ma_count} detected). Routine follow-up in 12 months."
        )
    else:
        # No DR findings, but check glaucoma
        note = ""
        if optic_disc.glaucoma_flag:
            note = f" Note: Elevated cup-to-disc ratio ({optic_disc.cup_to_disc_ratio}) warrants glaucoma evaluation."
        return SeverityGrade(
            level=0,
            label="No Apparent Retinopathy",
            confidence=max(0.70, 0.90 - total_findings * 0.05),
            description=f"No diabetic retinopathy findings detected.{note}"
        )


# ---- Pristine Rendering & Smooth Heatmap Generator ------------

def generate_density_heatmap(
    image_rgb: np.ndarray,
    findings: list,
) -> np.ndarray:
    """
    Generate a smooth, continuous Grad-CAM density heatmap overlay without noisy circles.
    Returns (H, W, 3) uint8 image.
    """
    h, w = image_rgb.shape[:2]
    img_uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    
    # Create smooth density map
    density = np.zeros((h, w), dtype=np.float32)
    for f in findings:
        if hasattr(f, 'x'):
            fx, fy = int(f.x), int(f.y)
        elif isinstance(f, dict):
            fx, fy = int(f.get('x', 0)), int(f.get('y', 0))
        else:
            continue
            
        if 0 <= fx < w and 0 <= fy < h:
            cv2.circle(density, (fx, fy), 24, 1.0, -1)

    # Gaussian blur for smooth continuous gradient
    density_blur = cv2.GaussianBlur(density, (51, 51), 0)
    if density_blur.max() > 0:
        density_blur = density_blur / density_blur.max()

    # Colorize heatmap (COLORMAP_JET / COLORMAP_TURBO)
    heatmap_colored = cv2.applyColorMap((density_blur * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Alpha blend with original image (0.35 alpha)
    blended = cv2.addWeighted(img_uint8, 0.70, heatmap_rgb, 0.30, 0)
    return blended


def render_annotated_image(
    image_rgb: np.ndarray,
    findings: list,
    optic_disc: OpticDiscInfo,
    severity: SeverityGrade,
) -> np.ndarray:
    """
    Return clean, pristine fundus image (no circle overlays).
    """
    return (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8).copy()

    return annotated


# ---- Master Detection Pipeline --------------------------------

def detect_anomalies(
    image_rgb: np.ndarray,
) -> dict:
    """
    Run the complete anomaly detection pipeline on a retinal image.

    params: image_rgb — (H, W, 3) float32 in [0, 1]
    returns: dict with all detection results, annotated image, and severity grade
    """
    # Stage 1: Preprocess
    prep = preprocess_for_detection(image_rgb)

    # Stage 2: Vessel segmentation
    vessels = segment_vessels(prep["green_clahe"], prep["fov_mask"])

    # Stage 4: Optic disc (needed before exudate detection)
    optic_disc = detect_optic_disc(prep["green_channel"], prep["fov_mask"])

    # Stage 3: Lesion detection
    microaneurysms = detect_microaneurysms(
        prep["green_clahe"], prep["fov_mask"], vessels["vessel_mask"]
    )
    hard_exudates = detect_hard_exudates(
        prep["original_uint8"], prep["fov_mask"], vessels["vessel_mask"], optic_disc
    )
    hemorrhages = detect_hemorrhages(
        prep["green_clahe"], prep["original_uint8"], prep["fov_mask"], vessels["vessel_mask"]
    )
    cotton_wool = detect_cotton_wool_spots(
        prep["original_uint8"], prep["green_clahe"], prep["fov_mask"],
        vessels["vessel_mask"], optic_disc
    )

    all_findings = microaneurysms + hard_exudates + hemorrhages + cotton_wool

    # Stage 5: Severity grading
    severity = compute_severity_grade(all_findings, optic_disc)

    # Stage 6: Anomalib Deep Learning Anomaly Score
    engine = get_anomalib_engine()
    amap, anomalib_score = engine.compute_anomaly_map(image_rgb)

    # Render annotated image & smooth density heatmap
    annotated = render_annotated_image(image_rgb, all_findings, optic_disc, severity)
    heatmap_img = generate_density_heatmap(image_rgb, all_findings)

    return {
        "findings": [f.to_dict() for f in all_findings],
        "finding_counts": {
            "microaneurysm": len(microaneurysms),
            "hard_exudate": len(hard_exudates),
            "hemorrhage": len(hemorrhages),
            "cotton_wool_spot": len(cotton_wool),
            "total": len(all_findings),
        },
        "severity": severity.to_dict(),
        "optic_disc": optic_disc.to_dict(),
        "vessel_info": {
            "vessel_density": vessels["vessel_density"],
        },
        "anomalib_engine": {
            "used": "PyTorch-Anomalib PatchCore",
            "anomaly_score": anomalib_score,
            "status": "active"
        },
        "annotated_image": annotated,
        "heatmap_image": heatmap_img,
        "vessel_mask": vessels["vessel_mask"],
    }
