"""
src/training/train.py
=====================
The core training and validation loops.

WHAT HAPPENS IN ONE TRAINING EPOCH? (for beginners)
  1. The DataLoader gives us a mini-batch of (degraded_input, clean_target) pairs
  2. The generator produces a restored image from the degraded input
  3. We update the DISCRIMINATOR:
       - Feed it the real (input, target) pair → should output ~1
       - Feed it the fake (input, generator_output) pair → should output ~0
       - Compute LSGAN loss, backpropagate, step the discriminator's optimizer
  4. We update the GENERATOR:
       - Compute all 4 loss terms (adversarial + L1 + SSIM + perceptual)
       - Backpropagate, step the generator's optimizer
  5. Log the average losses for this batch
  Repeat for every batch in the training set → one epoch complete.

VALIDATION:
  After each epoch, run the generator on the validation set with no updates.
  Compute PSNR and SSIM to measure how good the restoration is.
  If SSIM improves, save a new "best" checkpoint.
"""

import collections
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import DEVICE, USE_AMP, USE_EMA, EMA_DECAY
from src.models.losses import (
    compute_discriminator_loss,
    compute_generator_loss,
    VGGPerceptualLoss,
    SSIMLoss,
    CharbonnierLoss,
    SobelEdgeLoss,
)
from src.utils.image_utils import derive_fov_mask_from_input_tensor, tensor_to_float_array

# Mixed Precision Scalers with backward compatibility
if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    scaler_g = torch.amp.GradScaler("cuda", enabled=USE_AMP and DEVICE.type == "cuda")
    scaler_d = torch.amp.GradScaler("cuda", enabled=USE_AMP and DEVICE.type == "cuda")
else:
    scaler_g = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE.type == "cuda")
    scaler_d = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE.type == "cuda")


def get_autocast_context(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)



class ExponentialMovingAverage:
    """
    Exponential Moving Average of generator model weights.
    Provides significantly smoother, higher-quality inference outputs.
    """
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])


# ---- PSNR / SSIM metric computation -----------------------

def compute_psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    out_01    = (output + 1.0) / 2.0
    target_01 = (target + 1.0) / 2.0

    mse = torch.mean((out_01 - target_01) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * torch.log10(torch.tensor(1.0) / mse))


def compute_ssim_simple(output: torch.Tensor, target: torch.Tensor) -> float:
    out_01    = (output + 1.0) / 2.0
    target_01 = (target + 1.0) / 2.0

    mu_x   = out_01.mean()
    mu_y   = target_01.mean()
    sigma_x = out_01.var()
    sigma_y = target_01.var()
    sigma_xy = ((out_01 - mu_x) * (target_01 - mu_y)).mean()

    C1, C2 = 0.0001, 0.0009

    numerator   = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / (denominator + 1e-8))


def compute_psnr_ssim_batch(
    output_batch: torch.Tensor,
    target_batch: torch.Tensor,
) -> tuple[float, float]:
    psnr_values = []
    ssim_values = []
    B = output_batch.shape[0]

    for i in range(B):
        out  = output_batch[i:i+1]
        tgt  = target_batch[i:i+1]
        psnr_values.append(compute_psnr(out, tgt))
        ssim_values.append(compute_ssim_simple(out, tgt))

    return (
        sum(psnr_values) / len(psnr_values),
        sum(ssim_values) / len(ssim_values),
    )


# ---- One training epoch ------------------------------------

def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    perceptual_loss_module: VGGPerceptualLoss,
    ssim_loss_module: SSIMLoss,
    train_dataloader: torch.utils.data.DataLoader,
    epoch_index: int,
    logger: logging.Logger,
    charbonnier_loss_module: CharbonnierLoss = None,
    edge_loss_module: SobelEdgeLoss = None,
    ema_generator: ExponentialMovingAverage = None,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    accumulated_losses: dict = collections.defaultdict(float)
    num_batches = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch_index:03d} [train]", leave=False)
    use_amp = USE_AMP and DEVICE.type == "cuda"

    for degraded_input, real_target in progress_bar:
        try:
            degraded_input = degraded_input.to(DEVICE, non_blocking=True)
            real_target    = real_target.to(DEVICE, non_blocking=True)

            fov_mask = derive_fov_mask_from_input_tensor(degraded_input)

            # ---- Step 1: Generate fake output ----
            with get_autocast_context(use_amp):
                fake_output = generator(degraded_input)

            # ---- Step 2: Update Discriminator ----
            discriminator_optimizer.zero_grad(set_to_none=True)
            with get_autocast_context(use_amp):
                disc_loss = compute_discriminator_loss(
                    discriminator, degraded_input, real_target, fake_output.detach()
                )

            if use_amp:
                scaler_d.scale(disc_loss).backward()
                scaler_d.step(discriminator_optimizer)
                scaler_d.update()
            else:
                disc_loss.backward()
                discriminator_optimizer.step()

            # ---- Step 3: Update Generator ----
            generator_optimizer.zero_grad(set_to_none=True)
            with get_autocast_context(use_amp):
                total_gen_loss, loss_dict = compute_generator_loss(
                    discriminator, perceptual_loss_module, ssim_loss_module,
                    degraded_input, fake_output, real_target, fov_mask,
                    charbonnier_loss_module=charbonnier_loss_module,
                    edge_loss_module=edge_loss_module,
                )


            if use_amp:
                scaler_g.scale(total_gen_loss).backward()
                scaler_g.step(generator_optimizer)
                scaler_g.update()
            else:
                total_gen_loss.backward()
                generator_optimizer.step()

            if ema_generator is not None:
                ema_generator.update(generator)

            # ---- Step 4: Accumulate for logging ----
            loss_dict["discriminator"] = float(disc_loss)
            for key, val in loss_dict.items():
                accumulated_losses[key] += val
            num_batches += 1

            progress_bar.set_postfix(
                total=f"{loss_dict['total']:.3f}",
                charb=f"{loss_dict.get('charbonnier', 0.0):.3f}",
                edge=f"{loss_dict.get('edge', 0.0):.3f}",
                disc=f"{loss_dict['discriminator']:.3f}",
            )

        except RuntimeError as runtime_error:
            if "out of memory" in str(runtime_error).lower():
                logger.warning(
                    f"CUDA OOM on batch in epoch {epoch_index} — skipping batch and freeing cache."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    avg_losses = {k: v / max(1, num_batches) for k, v in accumulated_losses.items()}
    logger.info(f"Epoch {epoch_index:03d} [train] " +
                " | ".join(f"{k}={v:.4f}" for k, v in avg_losses.items()))
    return avg_losses


# ---- One validation epoch ----------------------------------

def validate_one_epoch(
    generator: nn.Module,
    validation_dataloader: torch.utils.data.DataLoader,
    epoch_index: int,
    logger: logging.Logger,
) -> dict[str, float]:
    generator.eval()
    accumulated_psnr = 0.0
    accumulated_ssim = 0.0
    num_batches = 0

    progress_bar = tqdm(validation_dataloader, desc=f"Epoch {epoch_index:03d} [val]", leave=False)
    use_amp = USE_AMP and DEVICE.type == "cuda"

    with torch.no_grad():
        for degraded_input, real_target in progress_bar:
            degraded_input = degraded_input.to(DEVICE)
            real_target    = real_target.to(DEVICE)

            with get_autocast_context(use_amp):
                restored_output = generator(degraded_input)


            restored_output_f32 = restored_output.float()
            real_target_f32     = real_target.float()

            batch_psnr, batch_ssim = compute_psnr_ssim_batch(restored_output_f32, real_target_f32)
            accumulated_psnr += batch_psnr
            accumulated_ssim += batch_ssim
            num_batches += 1

            progress_bar.set_postfix(
                psnr=f"{batch_psnr:.2f}",
                ssim=f"{batch_ssim:.4f}"
            )

    avg_metrics = {
        "psnr": accumulated_psnr / max(1, num_batches),
        "ssim": accumulated_ssim / max(1, num_batches),
    }
    logger.info(f"Epoch {epoch_index:03d} [val] psnr={avg_metrics['psnr']:.2f} ssim={avg_metrics['ssim']:.4f}")
    return avg_metrics

