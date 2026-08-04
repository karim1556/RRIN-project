"""
src/analysis/change_analysis.py
================================
Pillar 2 — AI Enhancement Change Analysis

Computes and explains WHAT the neural restoration changed:
  - Per-pixel difference maps
  - Frequency-band analysis (wavelet decomposition)
  - Local SSIM improvement maps
  - Change region classification (noise removal, detail recovery, etc.)
  - Auto-generated textual summary

This module answers the question: "What exactly did the AI fix?"
"""

import cv2
import numpy as np
from typing import Optional


# ---- Per-Pixel Difference Map -----------------------------------

def compute_difference_map(
    original: np.ndarray,
    restored: np.ndarray,
) -> np.ndarray:
    """
    Compute per-pixel L2 difference between original and restored images.

    params:
        original — (H, W, 3) float32 in [0, 1]
        restored — (H, W, 3) float32 in [0, 1]
    returns: difference_map — (H, W) float32, higher = more change
    """
    diff = np.sqrt(np.sum((original.astype(np.float64) - restored.astype(np.float64)) ** 2, axis=2))
    return diff.astype(np.float32)


def colorize_difference_map(diff_map: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale difference map to a color-coded visualization.
    Blue=low change, Green=moderate, Red=high change.

    params: diff_map — (H, W) float32
    returns: (H, W, 3) uint8 color-mapped image
    """
    normalized = np.clip(diff_map / (diff_map.max() + 1e-8), 0, 1)
    uint8_map = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)


# ---- Frequency-Band Analysis (Wavelet-like) --------------------

def _build_gaussian_pyramid(image_gray: np.ndarray, levels: int = 4) -> list:
    """Build a Gaussian pyramid by iterative downsampling + blurring."""
    pyramid = [image_gray.copy()]
    current = image_gray.copy()
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def _build_laplacian_pyramid(image_gray: np.ndarray, levels: int = 4) -> list:
    """
    Build a Laplacian pyramid capturing different frequency bands.
    Level 0 = highest frequency (fine details, noise)
    Level N = lowest frequency (illumination, large-scale structure)
    """
    gaussian = _build_gaussian_pyramid(image_gray, levels)
    laplacian = []
    for i in range(levels - 1):
        h, w = gaussian[i].shape[:2]
        upsampled = cv2.pyrUp(gaussian[i + 1], dstsize=(w, h))
        lap = cv2.subtract(gaussian[i], upsampled)
        laplacian.append(lap)
    laplacian.append(gaussian[-1])  # residual (lowest frequency)
    return laplacian


def compute_frequency_band_changes(
    original: np.ndarray,
    restored: np.ndarray,
    levels: int = 4,
) -> dict:
    """
    Decompose changes into frequency bands using Laplacian pyramids.

    Returns energy of change at each frequency level:
      - Level 0 (high-freq): noise removal, fine detail changes
      - Level 1 (mid-high): vessel edges, texture recovery
      - Level 2 (mid-low): medium structure, artifact removal
      - Level 3 (low-freq): illumination correction, haze removal

    params:
        original, restored — (H, W, 3) float32 in [0, 1]
        levels — number of decomposition levels
    returns: dict with frequency band energies and change maps
    """
    # Convert to grayscale for pyramid analysis
    orig_gray = cv2.cvtColor(
        (np.clip(original, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)
    rest_gray = cv2.cvtColor(
        (np.clip(restored, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    orig_pyr = _build_laplacian_pyramid(orig_gray, levels)
    rest_pyr = _build_laplacian_pyramid(rest_gray, levels)

    band_names = ["high_frequency", "mid_high_frequency", "mid_low_frequency", "low_frequency"]
    band_descriptions = [
        "Fine details & noise (vessels, edges, sensor noise)",
        "Textures & mid-scale structure (lesion boundaries, vessel walls)",
        "Medium structures (artifact boundaries, reflection edges)",
        "Large-scale illumination & haze (brightness, contrast, vignetting)",
    ]

    result = {"bands": [], "total_change_energy": 0.0}

    for i in range(min(levels, len(band_names))):
        # Resize to match if needed
        h, w = orig_pyr[i].shape[:2]
        rest_level = cv2.resize(rest_pyr[i], (w, h)) if rest_pyr[i].shape[:2] != (h, w) else rest_pyr[i]

        diff = np.abs(orig_pyr[i].astype(np.float64) - rest_level.astype(np.float64))
        energy = float(np.mean(diff ** 2))

        result["bands"].append({
            "name": band_names[i],
            "description": band_descriptions[i],
            "change_energy": energy,
            "change_percentage": 0.0,  # filled in after all bands computed
        })
        result["total_change_energy"] += energy

    # Compute percentages
    total = result["total_change_energy"] + 1e-8
    for band in result["bands"]:
        band["change_percentage"] = round(100.0 * band["change_energy"] / total, 1)

    return result


# ---- Local SSIM Improvement Map --------------------------------

def compute_local_ssim_map(
    image_a: np.ndarray,
    image_b: np.ndarray,
    window_size: int = 11,
) -> np.ndarray:
    """
    Compute a spatial map of local SSIM values between two images.

    params:
        image_a, image_b — (H, W, 3) float32 in [0, 1]
        window_size — size of the comparison window
    returns: ssim_map — (H, W) float32, values in [-1, 1], 1 = identical
    """
    C1 = (0.01 * 1.0) ** 2  # data range = 1.0
    C2 = (0.03 * 1.0) ** 2

    # Convert to grayscale
    gray_a = cv2.cvtColor(
        (np.clip(image_a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float64) / 255.0
    gray_b = cv2.cvtColor(
        (np.clip(image_b, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float64) / 255.0

    kernel = cv2.getGaussianKernel(window_size, 1.5)
    window = kernel @ kernel.T

    mu_a = cv2.filter2D(gray_a, -1, window)
    mu_b = cv2.filter2D(gray_b, -1, window)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.filter2D(gray_a ** 2, -1, window) - mu_a_sq
    sigma_b_sq = cv2.filter2D(gray_b ** 2, -1, window) - mu_b_sq
    sigma_ab = cv2.filter2D(gray_a * gray_b, -1, window) - mu_ab

    numerator = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)

    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.astype(np.float32)


def compute_ssim_improvement_map(
    original: np.ndarray,
    restored: np.ndarray,
    reference_clean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    If a clean reference is available, compute SSIM improvement: SSIM(restored, clean) - SSIM(original, clean).
    If not, compute self-SSIM showing local structural difference between original and restored.

    params:
        original, restored — (H, W, 3) float32 in [0, 1]
        reference_clean — optional clean reference image
    returns: improvement_map — (H, W) float32, positive = improvement
    """
    if reference_clean is not None:
        ssim_before = compute_local_ssim_map(original, reference_clean)
        ssim_after = compute_local_ssim_map(restored, reference_clean)
        return ssim_after - ssim_before
    else:
        # Without a clean reference, show the SSIM between original and restored
        # Higher values = less change (already similar), lower values = more change
        return compute_local_ssim_map(original, restored)


# ---- Change Region Classification -----------------------------

def classify_change_regions(
    original: np.ndarray,
    restored: np.ndarray,
    diff_map: np.ndarray,
    change_threshold: float = 0.05,
) -> dict:
    """
    Classify pixels where significant change occurred into categories:
      - noise_removal: random texture removed (high-freq, low spatial correlation)
      - detail_recovery: structure sharpened (high-freq, high spatial correlation along edges)
      - illumination_correction: brightness/contrast fixed (low-freq, smooth change)
      - artifact_removal: reflections/JPEG artifacts removed (mid-freq, localized)

    params:
        original, restored — (H, W, 3) float32 in [0, 1]
        diff_map — (H, W) float32 difference map
        change_threshold — minimum diff to consider as "changed"
    returns: dict with classification mask arrays and statistics
    """
    h, w = diff_map.shape
    changed_mask = diff_map > change_threshold

    # Convert to grayscale uint8 for edge detection
    orig_gray = cv2.cvtColor(
        (np.clip(original, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    rest_gray = cv2.cvtColor(
        (np.clip(restored, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    )

    # Edge maps (Canny) to find structural regions
    edges_orig = cv2.Canny(orig_gray, 50, 150).astype(bool)
    edges_rest = cv2.Canny(rest_gray, 50, 150).astype(bool)

    # Low-frequency change: blur the diff map and see where it's still high
    diff_blurred = cv2.GaussianBlur(diff_map, (31, 31), 10)
    low_freq_change = diff_blurred > change_threshold * 0.7

    # High-frequency change: diff minus blurred diff
    high_freq_change = (diff_map - diff_blurred) > change_threshold * 0.5

    # Classify regions
    # Detail recovery: high-freq change near edges in restored image
    detail_mask = changed_mask & high_freq_change & (
        cv2.dilate(edges_rest.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    )

    # Illumination correction: low-freq change, smooth, no edges
    illumination_mask = changed_mask & low_freq_change & ~detail_mask

    # Artifact removal: localized high-intensity change (reflections are bright)
    orig_bright = cv2.cvtColor(
        (np.clip(original, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV
    )[:, :, 2] > 200
    artifact_mask = changed_mask & orig_bright & ~detail_mask & ~illumination_mask

    # Noise removal: everything else that changed
    noise_mask = changed_mask & ~detail_mask & ~illumination_mask & ~artifact_mask

    total_changed = max(1, int(np.sum(changed_mask)))

    return {
        "noise_removal": {
            "mask": noise_mask,
            "pixel_count": int(np.sum(noise_mask)),
            "percentage": round(100.0 * np.sum(noise_mask) / total_changed, 1),
            "color": [0, 200, 80],   # green
            "label": "Noise Reduction",
        },
        "detail_recovery": {
            "mask": detail_mask,
            "pixel_count": int(np.sum(detail_mask)),
            "percentage": round(100.0 * np.sum(detail_mask) / total_changed, 1),
            "color": [0, 120, 255],  # blue
            "label": "Detail Recovery",
        },
        "illumination_correction": {
            "mask": illumination_mask,
            "pixel_count": int(np.sum(illumination_mask)),
            "percentage": round(100.0 * np.sum(illumination_mask) / total_changed, 1),
            "color": [255, 200, 0],  # yellow
            "label": "Illumination Correction",
        },
        "artifact_removal": {
            "mask": artifact_mask,
            "pixel_count": int(np.sum(artifact_mask)),
            "percentage": round(100.0 * np.sum(artifact_mask) / total_changed, 1),
            "color": [255, 140, 0],  # orange
            "label": "Artifact Removal",
        },
        "total_changed_pixels": total_changed,
        "total_pixels": h * w,
        "change_coverage_percent": round(100.0 * total_changed / (h * w), 1),
    }


# ---- Textual Summary Generation --------------------------------

def _quadrant_name(y: float, x: float, h: int, w: int) -> str:
    """Map a pixel position to a clinical quadrant name."""
    cy, cx = h / 2, w / 2
    if y < cy:
        vertical = "superior"
    else:
        vertical = "inferior"
    if x < cx:
        horizontal = "nasal"
    else:
        horizontal = "temporal"
    return f"{vertical}-{horizontal}"


def generate_change_summary(
    diff_map: np.ndarray,
    freq_analysis: dict,
    change_regions: dict,
) -> str:
    """
    Generate a natural-language textual summary of what the AI changed.

    params:
        diff_map — (H, W) float32 difference map
        freq_analysis — output of compute_frequency_band_changes()
        change_regions — output of classify_change_regions()
    returns: multi-sentence summary string
    """
    h, w = diff_map.shape
    sentences = []

    # Overall change magnitude
    coverage = change_regions["change_coverage_percent"]
    if coverage > 30:
        sentences.append(f"RRIN performed extensive restoration across {coverage}% of the image area.")
    elif coverage > 10:
        sentences.append(f"RRIN applied targeted restoration to {coverage}% of the image area.")
    else:
        sentences.append(f"RRIN made precise, localized adjustments affecting {coverage}% of the image.")

    # Find the region of highest change intensity
    max_change_y, max_change_x = np.unravel_index(np.argmax(diff_map), diff_map.shape)
    primary_quadrant = _quadrant_name(max_change_y, max_change_x, h, w)

    # Report dominant change types
    change_types = []
    for key in ["noise_removal", "detail_recovery", "illumination_correction", "artifact_removal"]:
        region = change_regions[key]
        if region["percentage"] > 15:
            change_types.append((region["percentage"], region["label"].lower(), key))

    change_types.sort(reverse=True)

    if change_types:
        dominant = change_types[0]
        sentences.append(
            f"The primary improvement was {dominant[1]} ({dominant[0]}% of changes), "
            f"concentrated in the {primary_quadrant} quadrant."
        )

    # Frequency band analysis
    bands = freq_analysis["bands"]
    dominant_band = max(bands, key=lambda b: b["change_percentage"])
    if "high" in dominant_band["name"]:
        sentences.append(
            f"High-frequency restoration ({dominant_band['change_percentage']}% of total change energy) "
            f"indicates significant fine detail and vessel structure recovery."
        )
    elif "low" in dominant_band["name"]:
        sentences.append(
            f"Low-frequency correction ({dominant_band['change_percentage']}% of total change energy) "
            f"indicates substantial illumination and contrast normalization."
        )

    # Detail recovery specifics
    detail_pct = change_regions["detail_recovery"]["percentage"]
    if detail_pct > 10:
        sentences.append(
            f"Vessel and structural detail recovery accounted for {detail_pct}% of the restoration, "
            f"indicating improved visibility of fine vascular features."
        )

    # Artifact removal specifics
    artifact_pct = change_regions["artifact_removal"]["percentage"]
    if artifact_pct > 5:
        sentences.append(
            f"Specular reflections and imaging artifacts were reduced across {artifact_pct}% of changed regions."
        )

    return " ".join(sentences)


# ---- Master Analysis Function ---------------------------------

def compute_enhancement_analysis(
    original: np.ndarray,
    restored: np.ndarray,
    reference_clean: Optional[np.ndarray] = None,
) -> dict:
    """
    Run the full enhancement change analysis pipeline.

    params:
        original — (H, W, 3) float32 in [0, 1], the degraded input
        restored — (H, W, 3) float32 in [0, 1], the restored output
        reference_clean — optional clean reference for SSIM improvement
    returns: dict with all analysis outputs (serializable except mask arrays)
    """
    # Ensure same dimensions
    h, w = original.shape[:2]
    if restored.shape[:2] != (h, w):
        restored = cv2.resize(restored, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1. Difference map
    diff_map = compute_difference_map(original, restored)

    # 2. Frequency band analysis
    freq_analysis = compute_frequency_band_changes(original, restored)

    # 3. SSIM improvement map
    ssim_map = compute_ssim_improvement_map(original, restored, reference_clean)

    # 4. Change region classification
    change_regions = classify_change_regions(original, restored, diff_map)

    # 5. Textual summary
    summary_text = generate_change_summary(diff_map, freq_analysis, change_regions)

    # 6. Overall improvement metrics
    mean_diff = float(np.mean(diff_map))
    max_diff = float(np.max(diff_map))
    mean_ssim = float(np.mean(ssim_map))

    return {
        "difference_map": diff_map,               # (H, W) float32
        "difference_map_colored": colorize_difference_map(diff_map),  # (H, W, 3) uint8
        "frequency_analysis": freq_analysis,       # dict with band info
        "ssim_map": ssim_map,                      # (H, W) float32
        "change_regions": change_regions,          # dict with masks + stats
        "summary_text": summary_text,              # string
        "metrics": {
            "mean_pixel_change": round(mean_diff, 4),
            "max_pixel_change": round(max_diff, 4),
            "mean_structural_similarity": round(mean_ssim, 4),
            "change_coverage_percent": change_regions["change_coverage_percent"],
            "noise_removal_percent": change_regions["noise_removal"]["percentage"],
            "detail_recovery_percent": change_regions["detail_recovery"]["percentage"],
            "illumination_correction_percent": change_regions["illumination_correction"]["percentage"],
            "artifact_removal_percent": change_regions["artifact_removal"]["percentage"],
        },
    }
