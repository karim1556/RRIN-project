"""
src/analysis/explainability.py
================================
Pillar 6 — Grad-CAM Explainability

Uses Gradient-weighted Class Activation Mapping to show WHERE the
U-Net generator focuses during restoration.

By hooking into the bottleneck (deepest layer) of the generator and
computing gradients of the output with respect to bottleneck activations,
we produce a heatmap showing which spatial regions most influenced
the restoration output.

This provides "explainable AI" — critical for clinical trust and funding.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


# ---- Hook-Based Feature/Gradient Capture ----------------------

class _GradCAMHooks:
    """
    Attach forward/backward hooks to a target layer to capture
    activations and gradients during a forward+backward pass.
    """

    def __init__(self) -> None:
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = None
        self._backward_handle = None

    def attach(self, target_layer: nn.Module) -> None:
        """Register hooks on the target layer."""
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Capture activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        """Remove hooks to prevent memory leaks."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
        if self._backward_handle is not None:
            self._backward_handle.remove()


# ---- Grad-CAM Computation ------------------------------------

def compute_gradcam(
    generator: nn.Module,
    input_tensor: torch.Tensor,
    target_layer_name: str = "bottleneck",
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for the generator's restoration output.

    Hooks into the specified layer (default: bottleneck), runs a
    forward pass, then backpropagates the mean output intensity to
    get spatial attention weights.

    params:
        generator — UNetGenerator (will be temporarily set to train mode for gradients)
        input_tensor — (1, 4, H, W) input tensor in [-1, 1]
        target_layer_name — which layer to hook ("bottleneck", "encoder_5", etc.)
    returns: heatmap — (H, W) float32 in [0, 1], higher = more attention
    """
    device = input_tensor.device
    _, _, H, W = input_tensor.shape

    # Find the target layer
    target_layer = _resolve_layer(generator, target_layer_name)
    if target_layer is None:
        # Fallback: return uniform heatmap
        return np.ones((H, W), dtype=np.float32) * 0.5

    # Attach hooks
    hooks = _GradCAMHooks()
    hooks.attach(target_layer)

    # Store original training state
    was_training = generator.training

    try:
        # Need gradients for Grad-CAM
        generator.eval()
        input_tensor.requires_grad_(False)

        # Create a copy that requires grad for the computation graph
        input_copy = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass
        output = generator(input_copy)

        # Use mean output intensity as the "score" to backpropagate
        # This tells us: which bottleneck features contributed most to
        # the overall brightness/content of the restored image
        score = output.mean()

        # Backward pass
        generator.zero_grad()
        score.backward(retain_graph=False)

        if hooks.activations is None or hooks.gradients is None:
            return np.ones((H, W), dtype=np.float32) * 0.5

        # Grad-CAM: weight activations by their gradient importance
        # Global average pooling of gradients → per-channel weights
        weights = hooks.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * hooks.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU (only positive contributions)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()  # (h, w)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to original image size
        heatmap = cv2.resize(cam.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    finally:
        hooks.remove()
        if was_training:
            generator.train()
        else:
            generator.eval()

    return heatmap.astype(np.float32)


def _resolve_layer(model: nn.Module, layer_name: str) -> Optional[nn.Module]:
    """
    Find a layer in the model by name.
    Supports dot-notation for nested layers (e.g., "encoder_5.conv").
    """
    try:
        parts = layer_name.split(".")
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current
    except Exception:
        return None


# ---- Per-Channel Grad-CAM ------------------------------------

def compute_per_channel_gradcam(
    generator: nn.Module,
    input_tensor: torch.Tensor,
    target_layer_name: str = "bottleneck",
) -> dict:
    """
    Compute separate Grad-CAM heatmaps for each output channel (R, G, B).

    Shows what features drove the restoration of each color channel:
    - Red channel: often highlights vessel structures
    - Green channel: most vessel contrast
    - Blue channel: background/optic disc features

    params:
        generator — UNetGenerator
        input_tensor — (1, 4, H, W)
        target_layer_name — layer to hook
    returns: dict with per-channel heatmaps and combined
    """
    device = input_tensor.device
    _, _, H, W = input_tensor.shape

    target_layer = _resolve_layer(generator, target_layer_name)
    if target_layer is None:
        uniform = np.ones((H, W), dtype=np.float32) * 0.5
        return {
            "combined": uniform,
            "red_channel": uniform,
            "green_channel": uniform,
            "blue_channel": uniform,
        }

    channel_heatmaps = {}
    channel_names = ["red_channel", "green_channel", "blue_channel"]

    for ch_idx, ch_name in enumerate(channel_names):
        hooks = _GradCAMHooks()
        hooks.attach(target_layer)

        was_training = generator.training

        try:
            generator.eval()
            input_copy = input_tensor.clone().detach().requires_grad_(True)

            output = generator(input_copy)

            # Backpropagate from specific channel
            channel_score = output[0, ch_idx, :, :].mean()
            generator.zero_grad()
            channel_score.backward(retain_graph=False)

            if hooks.activations is not None and hooks.gradients is not None:
                weights = hooks.gradients.mean(dim=[2, 3], keepdim=True)
                cam = torch.relu((weights * hooks.activations).sum(dim=1, keepdim=True))
                cam = cam.squeeze().cpu().numpy()
                if cam.max() > 0:
                    cam = cam / cam.max()
                heatmap = cv2.resize(cam.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                heatmap = np.ones((H, W), dtype=np.float32) * 0.5

            channel_heatmaps[ch_name] = heatmap

        finally:
            hooks.remove()
            if was_training:
                generator.train()
            else:
                generator.eval()

    # Combined: average of all channel heatmaps
    combined = np.mean(
        [channel_heatmaps[n] for n in channel_names],
        axis=0
    ).astype(np.float32)

    channel_heatmaps["combined"] = combined
    return channel_heatmaps


# ---- Heatmap Visualization -----------------------------------

def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a heatmap on an image with transparency blending.

    params:
        image_rgb — (H, W, 3) float32 in [0, 1]
        heatmap — (H, W) float32 in [0, 1]
        alpha — blending weight for heatmap (0 = invisible, 1 = opaque)
        colormap — OpenCV colormap to use
    returns: (H, W, 3) uint8 blended image
    """
    h, w = image_rgb.shape[:2]

    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    heatmap_uint8 = (np.clip(heatmap_resized, 0, 1) * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Blend with original
    image_uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    blended = cv2.addWeighted(image_uint8, 1 - alpha, colored_heatmap, alpha, 0)

    return blended


def generate_gradcam_insight(heatmap: np.ndarray, image_height: int, image_width: int) -> str:
    """
    Generate a textual interpretation of the Grad-CAM heatmap.

    params:
        heatmap — (H, W) float32
        image_height, image_width — original image dimensions
    returns: interpretive text string
    """
    h, w = heatmap.shape

    # Find hotspot location
    max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    # Determine quadrant
    cy, cx = h // 2, w // 2
    if max_y < cy:
        vertical = "superior"
    else:
        vertical = "inferior"
    if max_x > cx:
        horizontal = "temporal"
    else:
        horizontal = "nasal"

    quadrant = f"{vertical}-{horizontal}"

    # Coverage analysis
    high_attention = heatmap > 0.6
    attention_coverage = float(np.sum(high_attention)) / (h * w) * 100

    # Generate insight text
    sentences = [
        f"The neural network focused primarily on the {quadrant} region of the retina."
    ]

    if attention_coverage > 40:
        sentences.append(
            "Attention is broadly distributed, suggesting widespread degradation "
            "across the image that required global restoration."
        )
    elif attention_coverage > 15:
        sentences.append(
            f"Approximately {attention_coverage:.0f}% of the image received high attention, "
            "indicating localized regions of significant degradation."
        )
    else:
        sentences.append(
            f"Only {attention_coverage:.0f}% of the image received high attention, "
            "suggesting the degradation was concentrated in a small area."
        )

    # Check if center (fovea area) has high attention
    center_region = heatmap[h//3:2*h//3, w//3:2*w//3]
    center_attention = float(np.mean(center_region))
    if center_attention > 0.5:
        sentences.append(
            "Strong focus on the central/macular region indicates the AI prioritized "
            "restoring the most clinically important area for visual acuity."
        )

    return " ".join(sentences)
