"""
VITS Training Pipeline.

Stage 4.0 (core): Trains without discriminator for stable convergence.
Stage 4.1 (gan): Adds MPD/MSD discriminators after core stabilizes.

Losses:
- Core: Mel reconstruction + KL divergence + Duration prediction
- GAN: + Adversarial loss + Feature matching loss

Usage:
    koe train vits jsut --stage core
    koe train vits jsut --stage gan --resume runs/.../best.pt
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml

from modules.data_engineering.common.paths import paths
from modules.training.audio import MelConfig
from modules.training.common import create_train_val_loaders, EventLogger, ControlPlane, generate_eval_id
from modules.training.common.gan_controller import GANController, GANControllerConfig
from modules.training.common.watchdog import ThermalWatchdog, ThermalWatchdogConfig
from modules.training.dataloading import TTSBatch
from modules.training.models.vits import (
    create_vits_model,
    vits_loss_core,
)
from modules.training.models.vits.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from modules.training.models.vits.losses import (
    discriminator_loss,
    generator_adversarial_loss,
    feature_matching_loss,
)


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_loss_finite(loss: torch.Tensor, name: str = "loss") -> tuple[bool, Optional[str]]:
    """
    Check if a loss value is finite BEFORE backward.

    This catches NaN/Inf before they propagate into gradients.
    Should be called before loss.backward().

    Returns:
        (is_finite, error_detail): True if finite, else False with detail
    """
    if not torch.isfinite(loss).all():
        has_nan = torch.isnan(loss).any().item()
        has_inf = torch.isinf(loss).any().item()
        issue = "nan" if has_nan else "inf" if has_inf else "non_finite"
        return False, f"{issue}_in_{name}"
    return True, None


def check_grad_finite(model: torch.nn.Module) -> tuple[bool, Optional[str]]:
    """
    Check if gradients are finite BEFORE clip_grad_norm_.

    This catches NaN/Inf before they propagate into optimizer state.
    Must be called after backward() but before clip_grad_norm_().

    Returns:
        (is_finite, error_detail): True if all grads finite, else False with detail
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                # Find the actual bad values
                has_nan = torch.isnan(param.grad).any().item()
                has_inf = torch.isinf(param.grad).any().item()
                issue = "nan" if has_nan else "inf" if has_inf else "non_finite"
                return False, f"{issue}_in_{name}"
    return True, None


def zero_grads_safe(model: torch.nn.Module) -> None:
    """Zero gradients safely, handling None grads."""
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()


def scale_optimizer_lr(optimizer: torch.optim.Optimizer, scale: float) -> list[float]:
    """
    Temporarily scale optimizer learning rates.

    Args:
        optimizer: The optimizer to modify
        scale: Scale factor to apply (e.g., 0.5 = half LR)

    Returns:
        List of original LRs (for restoration)
    """
    original_lrs = []
    for param_group in optimizer.param_groups:
        original_lrs.append(param_group['lr'])
        param_group['lr'] *= scale
    return original_lrs


def restore_optimizer_lr(optimizer: torch.optim.Optimizer, original_lrs: list[float]) -> None:
    """Restore optimizer learning rates to original values."""
    for param_group, orig_lr in zip(optimizer.param_groups, original_lrs):
        param_group['lr'] = orig_lr


def safe_grad_norm(grad_norm: torch.Tensor | float) -> Optional[float]:
    """
    Convert grad norm to safe float value.

    Handles inf/nan from clip_grad_norm_ which can occur when:
    - AMP scaler encounters overflow
    - Gradients contain inf/nan values
    - All gradients are None

    Args:
        grad_norm: Gradient norm from clip_grad_norm_

    Returns:
        Float value, or None if invalid (NaN/Inf)
        None serializes to null in JSON, preserving alarm integrity
    """
    if isinstance(grad_norm, torch.Tensor):
        grad_norm = grad_norm.item()

    if grad_norm != grad_norm:  # NaN check
        return None
    if grad_norm == float("inf"):
        return None

    return grad_norm


def fmt_grad_norm(value: Optional[float], width: int = 6, precision: int = 2) -> str:
    """Format grad norm for display, handling None."""
    if value is None:
        return "-" * width
    return f"{value:>{width}.{precision}f}"


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    step: int,
    epoch: int,
    best_val_loss: float,
    config: dict,
    mpd: Optional[torch.nn.Module] = None,
    msd: Optional[torch.nn.Module] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None,
    controller: Optional[GANController] = None,
    speaker_list: Optional[list[str]] = None,
) -> None:
    """Save training checkpoint (generator + optional discriminators + controller)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_g_state": optimizer_g.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": config,
    }

    # Save speaker list for multi-speaker models
    if speaker_list is not None:
        checkpoint["speaker_list"] = speaker_list

    # Save discriminator states if present
    if mpd is not None:
        checkpoint["mpd_state"] = mpd.state_dict()
    if msd is not None:
        checkpoint["msd_state"] = msd.state_dict()
    if optimizer_d is not None:
        checkpoint["optimizer_d_state"] = optimizer_d.state_dict()

    # Save controller state if present
    if controller is not None:
        checkpoint["controller_state"] = controller.get_state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    config: dict,
    device: str,
    mpd: Optional[torch.nn.Module] = None,
    msd: Optional[torch.nn.Module] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None,
) -> tuple[int, int, float]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    # Handle both old (optimizer_state) and new (optimizer_g_state) formats
    if "optimizer_g_state" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state"])
    elif "optimizer_state" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_state"])

    if scaler and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])

    # Load discriminator states if present and requested
    if mpd is not None and "mpd_state" in checkpoint:
        mpd.load_state_dict(checkpoint["mpd_state"])
    if msd is not None and "msd_state" in checkpoint:
        msd.load_state_dict(checkpoint["msd_state"])
    if optimizer_d is not None and "optimizer_d_state" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state"])

    return checkpoint["step"], checkpoint["epoch"], checkpoint.get("best_val_loss", float("inf"))


def get_kl_weight(step: int, kl_anneal_config: dict) -> float:
    """Compute KL weight with optional annealing."""
    if not kl_anneal_config.get("enabled", False):
        return kl_anneal_config.get("end_weight", 1.0)

    start_weight = kl_anneal_config.get("start_weight", 0.0)
    end_weight = kl_anneal_config.get("end_weight", 1.0)
    anneal_steps = kl_anneal_config.get("steps", 20000)

    if step >= anneal_steps:
        return end_weight

    return start_weight + (end_weight - start_weight) * (step / anneal_steps)


def write_run_md(
    output_dir: Path,
    config: dict,
    dataset: str,
    cache_dir: Path,
    n_params: int,
    resume_path: Optional[Path] = None,
) -> None:
    """Write RUN.md with training metadata."""
    config_str = yaml.dump(config, default_flow_style=False)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]

    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    content = f"""# VITS Training Run

## Metadata
- **Run ID**: {output_dir.name}
- **Dataset**: {dataset}
- **Cache**: {cache_dir}
- **Git Commit**: {git_commit}
- **Config Hash**: {config_hash}
- **Parameters**: {n_params:,}
- **Created**: {datetime.now().isoformat(timespec='seconds')}
- **Resume**: {resume_path or 'N/A'}

## Config
```yaml
{config_str}```

## Checkpoints
| Checkpoint | Step | Val Loss | Notes |
|------------|------|----------|-------|
| *(updated during training)* | | | |
"""
    with open(output_dir / "RUN.md", "w") as f:
        f.write(content)


def update_run_md_checkpoint(
    output_dir: Path,
    checkpoint_name: str,
    step: int,
    val_loss: float,
    notes: str = "",
) -> None:
    """Append checkpoint to RUN.md table."""
    run_md = output_dir / "RUN.md"
    if not run_md.exists():
        return

    with open(run_md) as f:
        content = f.read()

    marker = "| *(updated during training)* | | | |"
    new_row = f"| {checkpoint_name} | {step} | {val_loss:.4f} | {notes} |"

    if marker in content:
        content = content.replace(marker, f"{new_row}\n{marker}")
        with open(run_md, "w") as f:
            f.write(content)


def train_step(
    model: torch.nn.Module,
    batch: TTSBatch,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    grad_clip: float,
    device: str,
    use_amp: bool,
    mel_config: dict,
    loss_config: dict,
) -> dict:
    """Execute single training step."""
    model.train()

    batch = batch.to(device)

    optimizer.zero_grad()

    with torch.amp.autocast('cuda', enabled=use_amp):
        outputs = model(
            phonemes=batch.phonemes,
            phoneme_mask=batch.phoneme_mask,
            mel=batch.mels,
            mel_mask=batch.mel_mask,
            speaker_idxs=batch.speaker_idxs,  # Multi-speaker support
        )

        loss, loss_dict = vits_loss_core(
            pred_audio=outputs["audio"],
            target_mel=batch.mels,
            posterior_mean=outputs["posterior_mean"],
            posterior_log_var=outputs["posterior_log_var"],
            prior_mean_aligned=outputs["prior_mean_aligned"],
            prior_log_var_aligned=outputs["prior_log_var_aligned"],
            log_dur_pred=outputs["log_dur_pred"],
            dur_target=outputs["durations"],
            mel_mask=batch.mel_mask,
            text_mask=batch.phoneme_mask,
            mel_config=mel_config,
            kl_weight=loss_config.get("kl_weight", 1.0),
            dur_weight=loss_config.get("dur_weight", 1.0),
        )

    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return {
        **loss_dict,
        "grad_norm": safe_grad_norm(grad_norm),
    }


def train_step_gan(
    model: torch.nn.Module,
    mpd: torch.nn.Module,
    msd: torch.nn.Module,
    batch: TTSBatch,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    grad_clip_g: float,
    grad_clip_d: float,
    device: str,
    use_amp: bool,
    mel_config: dict,
    loss_weights: dict,
    kl_weight: float,  # From annealing
    # Controller-driven parameters
    update_d: bool = True,
    adv_weight_scale: float = 1.0,
    mel_weight_scale: float = 1.0,
    grad_clip_scale: float = 1.0,
) -> dict:
    """
    Execute single GAN training step.

    Order:
    1. Generator forward (get fake audio)
    2. Discriminator update (detached fake) - if update_d=True
    3. Generator update (with adv + fm losses)

    Args:
        update_d: Whether to update discriminator this step (from controller)
        adv_weight_scale: Scale for adversarial weight (from controller ramp)
        mel_weight_scale: Scale for mel weight (from controller boost)
        grad_clip_scale: Scale for grad clipping (from controller tightening)
    """
    model.train()
    mpd.train()
    msd.train()

    batch = batch.to(device)

    # Track whether any scaler.step() was called this iteration.
    # CRITICAL: scaler.update() must be called once at end if any step occurred,
    # otherwise optimizer state gets stuck in STEPPED and next iter's unscale_() fails.
    did_step = False

    # Apply grad clip scaling
    effective_grad_clip_g = grad_clip_g * grad_clip_scale
    effective_grad_clip_d = grad_clip_d * grad_clip_scale

    # ===== Generator Forward =====
    with torch.amp.autocast('cuda', enabled=use_amp):
        outputs = model(
            phonemes=batch.phonemes,
            phoneme_mask=batch.phoneme_mask,
            mel=batch.mels,
            mel_mask=batch.mel_mask,
            speaker_idxs=batch.speaker_idxs,  # Multi-speaker support
        )
        fake_audio = outputs["audio"]

    # We need real audio for discriminator - reconstruct from mel target
    # For simplicity, use fake audio shape to slice real audio if available
    # In practice, the batch should contain real audio for GAN training
    # For now, we'll skip D training if no real audio (degrade gracefully)

    # ===== Discriminator Update =====
    # Skip D update if controller says so, but still compute for metrics
    skipped_d_step = not update_d
    d_real_score = 0.0
    d_fake_score = 0.0

    optimizer_d.zero_grad()

    with torch.amp.autocast('cuda', enabled=use_amp):
        # Get real audio from batch
        if batch.audio is not None:
            real_audio = batch.audio.unsqueeze(1)  # [B, 1, T]
        else:
            # No real audio - skip discriminator for this step
            real_audio = None

        if real_audio is not None:
            # Ensure same length
            min_len = min(real_audio.size(-1), fake_audio.size(-1))
            real_audio = real_audio[..., :min_len]
            fake_audio_d = fake_audio[..., :min_len].detach()

            # MPD
            y_d_rs_mpd, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = mpd(real_audio, fake_audio_d)
            loss_d_mpd, _ = discriminator_loss(y_d_rs_mpd, y_d_gs_mpd)

            # MSD
            y_d_rs_msd, y_d_gs_msd, fmap_rs_msd, fmap_gs_msd = msd(real_audio, fake_audio_d)
            loss_d_msd, _ = discriminator_loss(y_d_rs_msd, y_d_gs_msd)

            loss_d = loss_d_mpd + loss_d_msd

            # Compute D score metrics (mean logit for real/fake)
            d_real_scores = [y.mean().item() for y in y_d_rs_mpd + y_d_rs_msd]
            d_fake_scores = [y.mean().item() for y in y_d_gs_mpd + y_d_gs_msd]
            d_real_score = sum(d_real_scores) / len(d_real_scores) if d_real_scores else 0.0
            d_fake_score = sum(d_fake_scores) / len(d_fake_scores) if d_fake_scores else 0.0
        else:
            loss_d = torch.tensor(0.0, device=device)
            loss_d_mpd = torch.tensor(0.0, device=device)
            loss_d_msd = torch.tensor(0.0, device=device)

    # Only backward/step if update_d is True and we have real audio
    grad_norm_d = 0.0
    d_clip_coef = 1.0  # How much clipping reduced the gradient (1.0 = no clip)
    d_grad_non_finite = False
    d_grad_non_finite_detail = None  # Which parameter had non-finite grad
    if update_d and real_audio is not None:
        # P0: Check loss is finite BEFORE backward
        loss_d_ok, _ = check_loss_finite(loss_d, "loss_d")
        if not loss_d_ok:
            d_grad_non_finite = True  # Skip backward entirely
            d_clip_coef = 0.0  # Step skipped
        elif scaler:
            scaler.scale(loss_d).backward()
            scaler.unscale_(optimizer_d)
            # P0: Check grads BEFORE clip to prevent optimizer state corruption
            d_grads_ok_mpd, d_detail_mpd = check_grad_finite(mpd)
            d_grads_ok_msd, d_detail_msd = check_grad_finite(msd)
            if d_grads_ok_mpd and d_grads_ok_msd:
                grad_norm_d = torch.nn.utils.clip_grad_norm_(
                    list(mpd.parameters()) + list(msd.parameters()), effective_grad_clip_d
                )
                # Clip coefficient: how much the gradient was scaled down
                norm_val = grad_norm_d.item() if isinstance(grad_norm_d, torch.Tensor) else grad_norm_d
                d_clip_coef = min(1.0, effective_grad_clip_d / (norm_val + 1e-8))
                scaler.step(optimizer_d)
                did_step = True
            else:
                # Non-finite grads - zero and skip step
                zero_grads_safe(mpd)
                zero_grads_safe(msd)
                d_grad_non_finite = True
                d_grad_non_finite_detail = d_detail_mpd or d_detail_msd
                d_clip_coef = 0.0  # Step skipped
        else:
            loss_d.backward()
            d_grads_ok_mpd, d_detail_mpd = check_grad_finite(mpd)
            d_grads_ok_msd, d_detail_msd = check_grad_finite(msd)
            if d_grads_ok_mpd and d_grads_ok_msd:
                grad_norm_d = torch.nn.utils.clip_grad_norm_(
                    list(mpd.parameters()) + list(msd.parameters()), effective_grad_clip_d
                )
                norm_val = grad_norm_d.item() if isinstance(grad_norm_d, torch.Tensor) else grad_norm_d
                d_clip_coef = min(1.0, effective_grad_clip_d / (norm_val + 1e-8))
                optimizer_d.step()
            else:
                zero_grads_safe(mpd)
                zero_grads_safe(msd)
                d_grad_non_finite = True
                d_grad_non_finite_detail = d_detail_mpd or d_detail_msd
                d_clip_coef = 0.0  # Step skipped

    # ===== Generator Update =====
    optimizer_g.zero_grad()

    with torch.amp.autocast('cuda', enabled=use_amp):
        # Re-run discriminators with non-detached fake for generator gradients
        if real_audio is not None:
            fake_audio_g = fake_audio[..., :min_len]

            _, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = mpd(real_audio, fake_audio_g)
            _, y_d_gs_msd, fmap_rs_msd, fmap_gs_msd = msd(real_audio, fake_audio_g)

            # Adversarial loss
            loss_adv_mpd, _ = generator_adversarial_loss(y_d_gs_mpd)
            loss_adv_msd, _ = generator_adversarial_loss(y_d_gs_msd)
            loss_adv = loss_adv_mpd + loss_adv_msd

            # Feature matching loss
            loss_fm_mpd, _ = feature_matching_loss(fmap_rs_mpd, fmap_gs_mpd)
            loss_fm_msd, _ = feature_matching_loss(fmap_rs_msd, fmap_gs_msd)
            loss_fm = loss_fm_mpd + loss_fm_msd
        else:
            loss_adv = torch.tensor(0.0, device=device)
            loss_fm = torch.tensor(0.0, device=device)

        # Core losses
        loss_core, core_dict = vits_loss_core(
            pred_audio=outputs["audio"],
            target_mel=batch.mels,
            posterior_mean=outputs["posterior_mean"],
            posterior_log_var=outputs["posterior_log_var"],
            prior_mean_aligned=outputs["prior_mean_aligned"],
            prior_log_var_aligned=outputs["prior_log_var_aligned"],
            log_dur_pred=outputs["log_dur_pred"],
            dur_target=outputs["durations"],
            mel_mask=batch.mel_mask,
            text_mask=batch.phoneme_mask,
            mel_config=mel_config,
            kl_weight=kl_weight,
            dur_weight=loss_weights.get("dur_weight", 1.0),
        )

        # Total generator loss with controller-driven scaling
        mel_weight = loss_weights.get("mel_weight", 45.0) * mel_weight_scale
        adv_weight = loss_weights.get("adv_weight", 1.0) * adv_weight_scale
        fm_weight = loss_weights.get("fm_weight", 2.0)

        # Note: core_dict already includes mel_loss with weight=1 in vits_loss_core
        # We need to rescale here
        loss_g = (
            mel_weight * core_dict["mel_loss"]
            + kl_weight * core_dict["kl_loss"]
            + loss_weights.get("dur_weight", 1.0) * core_dict["dur_loss"]
            + adv_weight * loss_adv
            + fm_weight * loss_fm
        )

    g_grad_non_finite = False
    g_grad_non_finite_detail = None  # Which parameter had non-finite grad
    g_clip_coef = 1.0  # How much clipping reduced the gradient (1.0 = no clip)
    # P0: Check loss is finite BEFORE backward
    loss_g_ok, loss_g_detail = check_loss_finite(loss_g, "loss_g")
    if not loss_g_ok:
        g_grad_non_finite = True
        g_grad_non_finite_detail = loss_g_detail
        grad_norm_g = float('inf')  # Signal to controller
        g_clip_coef = 0.0  # Step skipped
    elif scaler:
        scaler.scale(loss_g).backward()
        scaler.unscale_(optimizer_g)
        # P0: Check grads BEFORE clip to prevent optimizer state corruption
        g_grads_ok, g_grad_issue = check_grad_finite(model)
        if g_grads_ok:
            grad_norm_g = torch.nn.utils.clip_grad_norm_(model.parameters(), effective_grad_clip_g)
            norm_val = grad_norm_g.item() if isinstance(grad_norm_g, torch.Tensor) else grad_norm_g
            g_clip_coef = min(1.0, effective_grad_clip_g / (norm_val + 1e-8))
            scaler.step(optimizer_g)
            did_step = True
        else:
            # Non-finite grads - zero and skip step
            zero_grads_safe(model)
            g_grad_non_finite = True
            g_grad_non_finite_detail = g_grad_issue
            grad_norm_g = float('inf')  # Signal to controller
            g_clip_coef = 0.0  # Step skipped
    else:
        loss_g.backward()
        g_grads_ok, g_grad_issue = check_grad_finite(model)
        if g_grads_ok:
            grad_norm_g = torch.nn.utils.clip_grad_norm_(model.parameters(), effective_grad_clip_g)
            norm_val = grad_norm_g.item() if isinstance(grad_norm_g, torch.Tensor) else grad_norm_g
            g_clip_coef = min(1.0, effective_grad_clip_g / (norm_val + 1e-8))
            optimizer_g.step()
        else:
            zero_grads_safe(model)
            g_grad_non_finite = True
            g_grad_non_finite_detail = g_grad_issue
            grad_norm_g = float('inf')
            g_clip_coef = 0.0  # Step skipped

    # CRITICAL: Always update scaler at end of iteration if any step occurred.
    # Without this, skipped steps leave optimizer in STEPPED state, causing
    # "unscale_() is being called after step()" on next iteration.
    if scaler and did_step:
        scaler.update()

    return {
        # Generator losses
        "loss_g": loss_g.item(),
        "g_loss_mel": core_dict["mel_loss"],
        "g_loss_kl": core_dict["kl_loss"],
        "g_loss_dur": core_dict["dur_loss"],
        "g_loss_adv": loss_adv.item() if isinstance(loss_adv, torch.Tensor) else loss_adv,
        "g_loss_fm": loss_fm.item() if isinstance(loss_fm, torch.Tensor) else loss_fm,
        "g_grad_norm": safe_grad_norm(grad_norm_g),
        # Discriminator losses
        "loss_d": loss_d.item() if isinstance(loss_d, torch.Tensor) else loss_d,
        "d_grad_norm": safe_grad_norm(grad_norm_d),
        "d_real_score": d_real_score,
        "d_fake_score": d_fake_score,
        # Health indicators (explicit signals for controller)
        "skipped_d_step": skipped_d_step,
        "grad_non_finite": g_grad_non_finite or d_grad_non_finite,
        "g_step_skipped": g_grad_non_finite,  # Explicit: G optimizer step was skipped
        "d_step_skipped": d_grad_non_finite,  # Explicit: D optimizer step was skipped
        "g_clip_coef": round(g_clip_coef, 4),  # How much G grads were clipped (1.0 = no clip, 0 = skipped)
        "d_clip_coef": round(d_clip_coef, 4),  # How much D grads were clipped (1.0 = no clip, 0 = skipped)
        # Non-finite gradient detail (which parameter caused the issue)
        "g_grad_non_finite_detail": g_grad_non_finite_detail,
        "d_grad_non_finite_detail": d_grad_non_finite_detail,
        # Legacy keys for backward compat
        "mel_loss": core_dict["mel_loss"],
        "kl_loss": core_dict["kl_loss"],
        "dur_loss": core_dict["dur_loss"],
        "loss_adv": loss_adv.item() if isinstance(loss_adv, torch.Tensor) else loss_adv,
        "loss_fm": loss_fm.item() if isinstance(loss_fm, torch.Tensor) else loss_fm,
        "grad_norm_g": safe_grad_norm(grad_norm_g),
        "grad_norm_d": safe_grad_norm(grad_norm_d),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    max_batches: Optional[int] = None,
    mel_config: Optional[dict] = None,
    loss_config: Optional[dict] = None,
) -> dict:
    """Run validation loop."""
    model.eval()
    mel_config = mel_config or {}
    loss_config = loss_config or {}

    totals = {"total_loss": 0, "mel_loss": 0, "kl_loss": 0, "dur_loss": 0}
    n_batches = 0

    for i, batch in enumerate(val_loader):
        if batch is None:
            continue
        if max_batches and i >= max_batches:
            break

        batch = batch.to(device)

        outputs = model(
            phonemes=batch.phonemes,
            phoneme_mask=batch.phoneme_mask,
            mel=batch.mels,
            mel_mask=batch.mel_mask,
            speaker_idxs=batch.speaker_idxs,  # Multi-speaker support
        )

        _, loss_dict = vits_loss_core(
            pred_audio=outputs["audio"],
            target_mel=batch.mels,
            posterior_mean=outputs["posterior_mean"],
            posterior_log_var=outputs["posterior_log_var"],
            prior_mean_aligned=outputs["prior_mean_aligned"],
            prior_log_var_aligned=outputs["prior_log_var_aligned"],
            log_dur_pred=outputs["log_dur_pred"],
            dur_target=outputs["durations"],
            mel_mask=batch.mel_mask,
            text_mask=batch.phoneme_mask,
            mel_config=mel_config,
            kl_weight=loss_config.get("kl_weight", 1.0),
            dur_weight=loss_config.get("dur_weight", 1.0),
        )

        for k, v in loss_dict.items():
            totals[k] += v
        n_batches += 1

    if n_batches == 0:
        return {f"val_{k}": float("inf") for k in totals}

    return {f"val_{k}": v / n_batches for k, v in totals.items()}


def find_cache_dir(dataset: str, snapshot_id: Optional[str] = None) -> Path:
    """Find cache directory for dataset."""
    from modules.data_engineering.common.paths import paths

    cache_base = paths.cache / dataset

    if snapshot_id:
        cache_dir = cache_base / snapshot_id
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache not found: {cache_dir}")
        return cache_dir

    latest = cache_base / "latest"
    if latest.exists():
        return latest.resolve()

    snapshots = sorted(cache_base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    snapshots = [s for s in snapshots if s.is_dir() and s.name != "latest"]
    if not snapshots:
        raise FileNotFoundError(f"No cache found for {dataset}. Run: koe cache create {dataset}")

    return snapshots[0]


def train(
    config: dict,
    dataset: str,
    snapshot_id: Optional[str] = None,
    resume_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Main training function."""
    run_name = config["run"]["name"]
    seed = config["run"]["seed"]
    device = config["run"]["device"]
    use_amp = config["run"]["amp"] and device == "cuda"

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = paths.runs / f"{dataset}_{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Initialize event logger
    events = EventLogger(output_dir)

    # Initialize control plane for dashboard commands
    control = ControlPlane(output_dir, events_logger=events, poll_every_steps=25, poll_every_seconds=30.0)

    # Stop control state
    stop_requested = False
    stop_ckpt_name: Optional[str] = None

    # Initialize thermal watchdog for GPU protection
    thermal_cfg = config.get("thermal", {})
    watchdog = ThermalWatchdog(
        config=ThermalWatchdogConfig(
            warn_temp=thermal_cfg.get("warn_temp", 80),
            stop_temp=thermal_cfg.get("stop_temp", 86),
            grace_seconds=thermal_cfg.get("grace_seconds", 60.0),
            check_interval_seconds=thermal_cfg.get("check_interval_seconds", 10.0),
        ),
        events_logger=events,
    )

    print("=" * 70)
    print("VITS CORE TRAINING (Stage 4.0)")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print()

    cache_dir = find_cache_dir(dataset, snapshot_id)
    print(f"Cache: {cache_dir}")

    mel_cfg = config.get("mel", {})
    mel_config = MelConfig(
        sample_rate=mel_cfg.get("sample_rate", 22050),
        n_mels=mel_cfg.get("n_mels", 80),
        n_fft=mel_cfg.get("n_fft", 1024),
        hop_length=mel_cfg.get("hop_length", 256),
        win_length=mel_cfg.get("win_length", 1024),
        f_min=mel_cfg.get("f_min", 0),
        f_max=mel_cfg.get("f_max", 8000),
    )

    # Create dataloaders using unified factory
    data_cfg = config.get("data", {})
    batch_size = data_cfg.get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 4)
    segment_seconds = data_cfg.get("segment_seconds", 3.0)
    max_audio_len = data_cfg.get("max_audio_len", None)
    speaker_balanced = data_cfg.get("speaker_balanced", None)

    train_loader, val_loader, speaker_vocab, num_speakers = create_train_val_loaders(
        cache_dir=cache_dir,
        mel_config=mel_config,
        batch_size=batch_size,
        num_workers=num_workers,
        segment_seconds=segment_seconds,
        max_audio_len=max_audio_len,
        speaker_balanced=speaker_balanced,
        seed=config.get("run", {}).get("seed", 42),
    )

    # Determine sampler mode for logging
    use_speaker_balanced = speaker_balanced if speaker_balanced is not None else (num_speakers > 1)
    sampler_mode = "speaker-balanced" if use_speaker_balanced and num_speakers > 1 else "shuffle"

    # Compute segment_samples for logging
    segment_samples = int(segment_seconds * mel_config.sample_rate) if segment_seconds else None

    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")
    if segment_samples:
        print(f"Segment: {segment_seconds}s ({segment_samples} samples)")
    if max_audio_len:
        print(f"Max audio: {max_audio_len / mel_config.sample_rate:.1f}s ({max_audio_len} samples)")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Speakers: {num_speakers}")
    print(f"Train sampler: {sampler_mode}")
    print()

    # Update config with num_speakers from data
    if "model" not in config:
        config["model"] = {}
    config["model"]["num_speakers"] = num_speakers

    # Create model
    model = create_vits_model(config)
    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")

    write_run_md(
        output_dir=output_dir,
        config=config,
        dataset=dataset,
        cache_dir=cache_dir,
        n_params=n_params,
        resume_path=resume_path,
    )

    # Optimizer
    optim_cfg = config.get("optim", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr", 2e-4),
        weight_decay=optim_cfg.get("weight_decay", 0.01),
        betas=tuple(optim_cfg.get("betas", [0.8, 0.99])),
    )

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})
    max_steps = train_cfg.get("max_steps", 100000)
    log_every = train_cfg.get("log_every_steps", 100)
    val_every = train_cfg.get("val_every_steps", 2000)
    save_every = train_cfg.get("save_every_steps", 5000)
    grad_clip = optim_cfg.get("grad_clip", 1.0)
    max_val_batches = data_cfg.get("max_val_batches", 50)

    step = 0
    epoch = 0
    best_val_loss = float("inf")
    resume_step = None
    resume_time = None
    early_save_done = False

    if resume_path and resume_path.exists():
        print(f"Resuming from: {resume_path}")
        step, epoch, best_val_loss = load_checkpoint(
            resume_path, model, optimizer, scaler, config, device
        )
        print(f"  Step: {step}, Epoch: {epoch}, Best val loss: {best_val_loss:.4f}")

        # P0: Checkpoint on resume — save immediately to new run dir
        resume_step = step
        resume_time = time.time()
        ckpt_name = f"step_{step:06d}_resume_start.pt"
        save_checkpoint(
            output_dir / "checkpoints" / ckpt_name,
            model, optimizer, scaler, step, epoch, best_val_loss, config,
            speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
        )
        update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, "resume_start")
        events.checkpoint_saved(
            f"checkpoints/{ckpt_name}", step, best_val_loss, tag="resume_start",
        )
        print(f"  Saved resume checkpoint: {ckpt_name}")
        print()

    # Log run start event
    events.run_started(
        stage="core",
        dataset=dataset,
        max_steps=max_steps,
        num_speakers=num_speakers,
        batch_size=batch_size,
        resume_from=str(resume_path) if resume_path else None,
    )

    print("=" * 70)
    print(f"Starting training (step {step} -> {max_steps})")
    print("=" * 70)
    print()
    print(f"{'Step':>8} | {'Loss':>8} | {'Mel':>8} | {'KL':>8} | {'Dur':>8} | {'GradNorm':>8}")
    print("-" * 70)

    train_iter = iter(train_loader)
    start_time = time.time()

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if batch is None:
            continue

        # Guardrail: require labeled data (phonemes/durations)
        if not batch.has_text:
            raise ValueError(
                f"This pipeline requires labeled text/phonemes, but batch has "
                f"{batch.n_labeled}/{batch.is_labeled.numel()} labeled samples. "
                "Tier 1 segment manifests are unlabeled. Use a full-utterance manifest "
                "or wait for Tier 2 labeling."
            )

        metrics = train_step(
            model, batch, optimizer, scaler, grad_clip, device, use_amp,
            mel_cfg, loss_cfg,
        )

        step += 1

        # Poll for control commands
        if control.should_poll(step):
            ctrl_request = control.poll()
            if ctrl_request:
                if ctrl_request.action == "checkpoint":
                    # Manual checkpoint request
                    tag = ctrl_request.params.get("tag", "manual")
                    ckpt_name = f"step_{step:06d}_{tag}.pt"
                    save_checkpoint(
                        output_dir / "checkpoints" / ckpt_name,
                        model, optimizer, scaler, step, epoch, best_val_loss, config,
                        speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                    )
                    update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, tag)
                    events.checkpoint_saved(
                        f"checkpoints/{ckpt_name}", step, best_val_loss, tag=tag,
                        mel_loss=metrics.get("mel_loss"),
                    )
                    control.ack(ctrl_request, success=True, result={"path": ckpt_name, "step": step})
                    print(f"         | Control: checkpoint saved ({tag})")

                elif ctrl_request.action == "eval":
                    # Extract eval params
                    eval_mode = ctrl_request.params.get("mode", "multispeaker")
                    eval_seed = ctrl_request.params.get("seed", 42)
                    eval_tag = ctrl_request.params.get("tag", "ui")
                    eval_id = generate_eval_id(eval_tag)
                    run_id = output_dir.name

                    # Ack immediately with queued status
                    control.ack(
                        ctrl_request,
                        success=True,
                        result={"status": "queued", "eval_id": eval_id, "params": ctrl_request.params},
                    )
                    print(f"         | Control: eval queued ({eval_id})")

                    # Save tagged checkpoint for reproducibility
                    eval_ckpt_name = f"step_{step:06d}_eval_{eval_tag}.pt"
                    save_checkpoint(
                        output_dir / "checkpoints" / eval_ckpt_name,
                        model, optimizer, scaler, step, epoch, best_val_loss, config,
                        speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                    )

                    # Log eval_started
                    events.eval_started(
                        eval_id=eval_id,
                        run_id=run_id,
                        step=step,
                        mode=eval_mode,
                        seed=eval_seed,
                        tag=eval_tag,
                        nonce=ctrl_request.nonce,
                    )

                    # Run eval inline (pauses training)
                    try:
                        from modules.training.pipelines.synthesize import eval_multispeaker

                        print(f"         | Running eval: {eval_id}")
                        result = eval_multispeaker(
                            run_id=run_id,
                            checkpoint=eval_ckpt_name,
                            seed=eval_seed,
                            device=str(device),
                        )

                        # Log eval_complete
                        events.eval_complete(
                            eval_id=eval_id,
                            run_id=run_id,
                            step=step,
                            artifact_dir=result["output_dir"],
                            summary={
                                "n_speakers": result["n_speakers"],
                                "n_prompts": result["n_prompts"],
                                "mean_inter_speaker_distance": result["manifest"]["separation_metrics"]["mean_inter_speaker_distance"],
                                "valid_outputs": f"{sum(s['n_valid'] for s in result['manifest']['per_speaker_summary'].values())}/{sum(s['n_samples'] for s in result['manifest']['per_speaker_summary'].values())}",
                            },
                            nonce=ctrl_request.nonce,
                        )
                        print(f"         | Eval complete: {result['output_dir']}")

                    except Exception as e:
                        events.eval_failed(
                            eval_id=eval_id,
                            run_id=run_id,
                            step=step,
                            error=str(e),
                            nonce=ctrl_request.nonce,
                        )
                        print(f"         | Eval failed: {e}")

                elif ctrl_request.action == "stop":
                    # Graceful stop request - save tagged checkpoint and exit
                    tag = ctrl_request.params.get("tag", "manual_stop")
                    stop_ckpt_name = f"step_{step:06d}_{tag}.pt"
                    save_checkpoint(
                        output_dir / "checkpoints" / stop_ckpt_name,
                        model, optimizer, scaler, step, epoch, best_val_loss, config,
                        speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                    )
                    update_run_md_checkpoint(output_dir, stop_ckpt_name, step, best_val_loss, tag)
                    events.checkpoint_saved(
                        f"checkpoints/{stop_ckpt_name}", step, best_val_loss, tag=tag,
                        mel_loss=metrics.get("mel_loss"),
                    )
                    control.ack(ctrl_request, success=True, result={
                        "path": f"checkpoints/{stop_ckpt_name}",
                        "step": step,
                        "tag": tag,
                    })
                    print(f"         | Control: stop requested, checkpoint saved ({tag})")
                    stop_requested = True

        # Check for user-requested graceful stop
        if stop_requested:
            print(f"\n{'='*70}")
            print("USER REQUESTED STOP")
            print(f"{'='*70}")
            break  # Exit loop, run normal teardown

        # Thermal watchdog check (time-based, runs independently of step cadence)
        if watchdog.should_check():
            thermal_action = watchdog.check()
            if thermal_action == "stop":
                # Emergency thermal shutdown - save checkpoint and exit
                print(f"\n{'='*70}")
                print(f"THERMAL EMERGENCY SHUTDOWN")
                print(f"GPU temperature exceeded {watchdog.config.stop_temp}°C for {watchdog.config.grace_seconds}s")
                print(f"{'='*70}")

                ckpt_name = f"step_{step:06d}_thermal_stop.pt"
                save_checkpoint(
                    output_dir / "checkpoints" / ckpt_name,
                    model, optimizer, scaler, step, epoch, best_val_loss, config,
                    speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                )
                events.checkpoint_saved(
                    f"checkpoints/{ckpt_name}", step, best_val_loss, tag="thermal_stop",
                    mel_loss=metrics.get("mel_loss"),
                )
                events.training_complete(
                    step, best_val_loss, best_val_loss, time.time() - start_time,
                    status="thermal_stop",
                    reason=f"GPU temp exceeded {watchdog.config.stop_temp}C",
                    checkpoint_path=f"checkpoints/{ckpt_name}",
                )
                print(f"Emergency checkpoint saved: {ckpt_name}")
                print(f"Exiting to protect hardware.")
                return {"status": "thermal_shutdown", "step": step, "checkpoint": ckpt_name}

        if step % log_every == 0:
            gn = fmt_grad_norm(metrics['grad_norm'], width=8, precision=4)
            print(
                f"{step:>8} | {metrics['total_loss']:>8.4f} | {metrics['mel_loss']:>8.4f} | "
                f"{metrics['kl_loss']:>8.4f} | {metrics['dur_loss']:>8.4f} | {gn}"
            )

            # Write metrics to jsonl for dashboard
            metrics_row = {
                "step": step,
                "epoch": epoch,
                "total_loss": metrics["total_loss"],
                "mel_loss": metrics["mel_loss"],
                "kl_loss": metrics["kl_loss"],
                "dur_loss": metrics["dur_loss"],
                "g_grad_norm": metrics["grad_norm"],
            }
            with open(events.metrics_file, "a") as f:
                f.write(json.dumps(metrics_row) + "\n")

        if step % val_every == 0:
            val_metrics = validate(model, val_loader, device, max_val_batches, mel_cfg, loss_cfg)
            print(f"         | Val: loss={val_metrics['val_total_loss']:.4f} "
                  f"mel={val_metrics['val_mel_loss']:.4f} "
                  f"kl={val_metrics['val_kl_loss']:.4f} "
                  f"dur={val_metrics['val_dur_loss']:.4f}")

            if val_metrics["val_total_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_total_loss"]
                save_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    model, optimizer, scaler, step, epoch, best_val_loss, config,
                    speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                )
                update_run_md_checkpoint(output_dir, "best.pt", step, best_val_loss, "new best")
                events.checkpoint_saved(
                    "checkpoints/best.pt", step, best_val_loss, tag="best", is_best=True,
                    mel_loss=val_metrics.get("val_mel_loss"),
                )
                print(f"         | New best model saved (val_loss: {best_val_loss:.4f})")

        if step % save_every == 0:
            ckpt_name = f"step_{step:06d}.pt"
            save_checkpoint(
                output_dir / "checkpoints" / ckpt_name,
                model, optimizer, scaler, step, epoch, best_val_loss, config,
                speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
            )
            update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, "periodic")
            events.checkpoint_saved(
                f"checkpoints/{ckpt_name}", step, best_val_loss, tag="periodic",
                mel_loss=metrics.get("mel_loss"),
            )
            early_save_done = True  # Periodic save counts as early save

        # P0: Early-first save for resumed runs (5 min OR 500 steps, whichever first)
        if resume_step is not None and not early_save_done:
            elapsed_since_resume = time.time() - resume_time
            steps_since_resume = step - resume_step
            if elapsed_since_resume >= 300 or steps_since_resume >= 500:
                ckpt_name = f"step_{step:06d}_early.pt"
                save_checkpoint(
                    output_dir / "checkpoints" / ckpt_name,
                    model, optimizer, scaler, step, epoch, best_val_loss, config,
                    speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                )
                update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, "early")
                events.checkpoint_saved(
                    f"checkpoints/{ckpt_name}", step, best_val_loss, tag="early",
                    mel_loss=metrics.get("mel_loss"),
                )
                print(f"         | Early checkpoint saved: {ckpt_name}")
                early_save_done = True

    # Save final checkpoint (skip if stop was requested - we already saved stop checkpoint)
    if not stop_requested:
        save_checkpoint(
            output_dir / "checkpoints" / "final.pt",
            model, optimizer, scaler, step, epoch, best_val_loss, config,
            speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
        )
        update_run_md_checkpoint(output_dir, "final.pt", step, best_val_loss, "final")
        events.checkpoint_saved("checkpoints/final.pt", step, best_val_loss, tag="final")

    # Final drain: handle any pending control requests before completing
    def handle_final_control(req):
        if req.action == "checkpoint":
            tag = req.params.get("tag", "final_drain")
            ckpt_name = f"step_{step:06d}_{tag}.pt"
            save_checkpoint(
                output_dir / "checkpoints" / ckpt_name,
                model, optimizer, scaler, step, epoch, best_val_loss, config,
                speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
            )
            update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, tag)
            events.checkpoint_saved(f"checkpoints/{ckpt_name}", step, best_val_loss, tag=tag)
            control.ack(req, success=True, result={"path": ckpt_name, "step": step})
            print(f"Final drain: checkpoint saved ({tag})")
        elif req.action == "stop":
            # Execute stop even in final drain (user wants tagged checkpoint)
            tag = req.params.get("tag", "final_stop")
            ckpt_name = f"step_{step:06d}_{tag}.pt"
            save_checkpoint(
                output_dir / "checkpoints" / ckpt_name,
                model, optimizer, scaler, step, epoch, best_val_loss, config,
                speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
            )
            update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, tag)
            events.checkpoint_saved(f"checkpoints/{ckpt_name}", step, best_val_loss, tag=tag)
            control.ack(req, success=True, result={
                "path": f"checkpoints/{ckpt_name}",
                "step": step,
                "tag": tag,
                "note": "final_drain",
            })
            print(f"Final drain: stop executed, checkpoint saved ({tag})")
        else:
            # Eval can't run at training end
            control.ack(req, success=False, error=f"Training complete, cannot execute {req.action}")
            print(f"Final drain: rejected {req.action} (training complete)")

    control.drain(handle_final_control)

    # Final validation (skip if stopped to exit faster)
    if not stop_requested:
        final_val = validate(model, val_loader, device, max_val_batches, mel_cfg, loss_cfg)
        final_val_loss = final_val["val_total_loss"]
    else:
        final_val_loss = best_val_loss  # Use last known

    total_time = time.time() - start_time

    # Emit training_complete with appropriate status
    if stop_requested:
        events.training_complete(
            step, best_val_loss, final_val_loss, total_time,
            status="user_stopped",
            reason="user_requested",
            checkpoint_path=f"checkpoints/{stop_ckpt_name}",
        )
    else:
        events.training_complete(step, best_val_loss, final_val_loss, total_time)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE" if not stop_requested else "TRAINING STOPPED")
    print("=" * 70)
    print(f"Total steps: {step}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Final val loss: {final_val_loss:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir / 'checkpoints'}")
    print("=" * 70)

    return {
        "step": step,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "output_dir": str(output_dir),
        "status": "user_stopped" if stop_requested else "success",
    }


def train_gan(
    config: dict,
    dataset: str,
    snapshot_id: Optional[str] = None,
    resume_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Main GAN training function (Stage 4.1)."""
    run_name = config["run"]["name"]
    seed = config["run"]["seed"]
    device = config["run"]["device"]
    use_amp = config["run"]["amp"] and device == "cuda"

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = paths.runs / f"{dataset}_{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Initialize event logger
    events = EventLogger(output_dir)

    # Initialize control plane for dashboard commands
    control = ControlPlane(output_dir, events_logger=events, poll_every_steps=25, poll_every_seconds=30.0)

    # Stop control state
    stop_requested = False
    stop_ckpt_name: Optional[str] = None

    # Initialize thermal watchdog for GPU protection
    thermal_cfg = config.get("thermal", {})
    watchdog = ThermalWatchdog(
        config=ThermalWatchdogConfig(
            warn_temp=thermal_cfg.get("warn_temp", 80),
            stop_temp=thermal_cfg.get("stop_temp", 86),
            grace_seconds=thermal_cfg.get("grace_seconds", 60.0),
            check_interval_seconds=thermal_cfg.get("check_interval_seconds", 10.0),
        ),
        events_logger=events,
    )

    # Get discriminator config
    disc_cfg = config.get("discriminator", {})
    disc_start_step = disc_cfg.get("disc_start_step", 10000)

    print("=" * 70)
    print("VITS GAN TRAINING (Stage 4.1)")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Discriminator starts at step: {disc_start_step}")
    print()

    cache_dir = find_cache_dir(dataset, snapshot_id)
    print(f"Cache: {cache_dir}")

    mel_cfg = config.get("mel", {})
    mel_config = MelConfig(
        sample_rate=mel_cfg.get("sample_rate", 22050),
        n_mels=mel_cfg.get("n_mels", 80),
        n_fft=mel_cfg.get("n_fft", 1024),
        hop_length=mel_cfg.get("hop_length", 256),
        win_length=mel_cfg.get("win_length", 1024),
        f_min=mel_cfg.get("f_min", 0),
        f_max=mel_cfg.get("f_max", 8000),
    )

    # Create dataloaders using unified factory
    data_cfg = config.get("data", {})
    batch_size = data_cfg.get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 4)
    segment_seconds = data_cfg.get("segment_seconds", 3.0)
    max_audio_len = data_cfg.get("max_audio_len", None)
    speaker_balanced = data_cfg.get("speaker_balanced", None)

    train_loader, val_loader, speaker_vocab, num_speakers = create_train_val_loaders(
        cache_dir=cache_dir,
        mel_config=mel_config,
        batch_size=batch_size,
        num_workers=num_workers,
        segment_seconds=segment_seconds,
        max_audio_len=max_audio_len,
        speaker_balanced=speaker_balanced,
        seed=config.get("run", {}).get("seed", 42),
    )

    # Determine sampler mode for logging
    use_speaker_balanced = speaker_balanced if speaker_balanced is not None else (num_speakers > 1)
    sampler_mode = "speaker-balanced" if use_speaker_balanced and num_speakers > 1 else "shuffle"

    # Compute segment_samples for logging
    segment_samples = int(segment_seconds * mel_config.sample_rate) if segment_seconds else None

    print(f"Batch size: {batch_size}")
    if segment_samples:
        print(f"Segment: {segment_seconds}s ({segment_samples} samples)")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Speakers: {num_speakers}")
    print(f"Train sampler: {sampler_mode}")
    print()

    # Update config with num_speakers from data
    if "model" not in config:
        config["model"] = {}
    config["model"]["num_speakers"] = num_speakers

    # Create generator model
    model = create_vits_model(config)
    model = model.to(device)
    n_params_g = model.count_parameters()
    print(f"Generator parameters: {n_params_g:,}")

    # Create discriminators
    mpd = MultiPeriodDiscriminator(
        periods=tuple(disc_cfg.get("mpd_periods", [2, 3, 5, 7, 11]))
    ).to(device)
    msd = MultiScaleDiscriminator().to(device)

    n_params_d = sum(p.numel() for p in mpd.parameters()) + sum(p.numel() for p in msd.parameters())
    print(f"Discriminator parameters: {n_params_d:,}")
    print(f"Total parameters: {n_params_g + n_params_d:,}")

    write_run_md(
        output_dir=output_dir,
        config=config,
        dataset=dataset,
        cache_dir=cache_dir,
        n_params=n_params_g + n_params_d,
        resume_path=resume_path,
    )

    # Separate optimizers
    optim_cfg = config.get("optim", {})
    optimizer_g = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr_g", 2e-4),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
        betas=tuple(optim_cfg.get("betas", [0.8, 0.99])),
    )
    optimizer_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=optim_cfg.get("lr_d", 1e-4),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
        betas=tuple(optim_cfg.get("betas", [0.8, 0.99])),
    )

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})
    kl_anneal_cfg = config.get("kl_anneal", {})
    max_steps = train_cfg.get("max_steps", 200000)
    log_every = train_cfg.get("log_every_steps", 100)
    val_every = train_cfg.get("val_every_steps", 2000)
    save_every = train_cfg.get("save_every_steps", 5000)
    grad_clip_g = optim_cfg.get("grad_clip_g", 1.0)
    grad_clip_d = optim_cfg.get("grad_clip_d", 1.0)
    max_val_batches = data_cfg.get("max_val_batches", 50)

    # Create GAN controller for health monitoring + auto-mitigations
    controller_cfg = config.get("controller", {})
    gan_controller = GANController(
        config=GANControllerConfig(
            window_size=controller_cfg.get("window_size", 200),
            d_loss_low_threshold=controller_cfg.get("d_loss_low_threshold", 0.15),
            g_loss_adv_high_threshold=controller_cfg.get("g_loss_adv_high_threshold", 8.0),
            d_throttle_every=controller_cfg.get("d_throttle_every", 2),
            adv_ramp_steps=controller_cfg.get("adv_ramp_steps", 3000),
            # Hard ceiling (emergency stop)
            absolute_grad_limit_g=controller_cfg.get("absolute_grad_limit_g", 3000.0),
            absolute_grad_limit_d=controller_cfg.get("absolute_grad_limit_d", 3000.0),
            # Consecutive spikes detection
            consecutive_spikes_for_unstable=controller_cfg.get("consecutive_spikes_for_unstable", 3),
            # Spike density detection
            spike_density_window=controller_cfg.get("spike_density_window", 20),
            spike_density_threshold=controller_cfg.get("spike_density_threshold", 3),
            # EMA early warning
            ema_elevated_limit_g=controller_cfg.get("ema_elevated_limit_g", 500.0),
            ema_elevated_limit_d=controller_cfg.get("ema_elevated_limit_d", 500.0),
            ema_elevated_steps_threshold=controller_cfg.get("ema_elevated_steps_threshold", 50),
            # Clip coefficient detection
            clip_coef_hard_threshold=controller_cfg.get("clip_coef_hard_threshold", 0.05),
            clip_coef_hard_steps=controller_cfg.get("clip_coef_hard_steps", 30),
            clip_coef_median_threshold=controller_cfg.get("clip_coef_median_threshold", 0.1),
            clip_coef_median_window=controller_cfg.get("clip_coef_median_window", 50),
            # Level-based stability thresholds (exponential decay)
            stability_required_steps_l1=controller_cfg.get("stability_required_steps_l1", 200),
            stability_required_steps_l2=controller_cfg.get("stability_required_steps_l2", 500),
            stability_required_steps_l3=controller_cfg.get("stability_required_steps_l3", 1000),
            # Soft gates for de-escalation (block while grads elevated)
            soft_grad_limit_g=controller_cfg.get("soft_grad_limit_g", 2000.0),
            soft_grad_limit_d=controller_cfg.get("soft_grad_limit_d", 2000.0),
            # D-real velocity tracking (replaces threshold-based triggers)
            d_real_velocity_window=controller_cfg.get("d_real_velocity_window", 20),
            d_real_velocity_warning=controller_cfg.get("d_real_velocity_warning", 0.01),
            d_real_velocity_critical=controller_cfg.get("d_real_velocity_critical", 0.02),
            d_real_velocity_emergency=controller_cfg.get("d_real_velocity_emergency", 0.05),
            d_real_velocity_lr_min=controller_cfg.get("d_real_velocity_lr_min", 0.1),
            # D-freeze configuration
            d_freeze_start_level=controller_cfg.get("d_freeze_start_level", 3),
            d_unfreeze_warmup_steps=controller_cfg.get("d_unfreeze_warmup_steps", 50),
            # D-freeze probes
            d_freeze_probe_forward_only=controller_cfg.get("d_freeze_probe_forward_only", True),
            d_freeze_probe_interval=controller_cfg.get("d_freeze_probe_interval", 100),
            d_freeze_probe_duration=controller_cfg.get("d_freeze_probe_duration", 10),
            # Observation mode: disable escalation but keep event logging
            escalation_enabled=controller_cfg.get("escalation_enabled", True),
            hard_ceiling_enabled=controller_cfg.get("hard_ceiling_enabled", True),
            # Escalation mitigation settings (grad clip scales, LR reduction)
            grad_clip_scales=tuple(controller_cfg.get("grad_clip_scales", [0.5, 0.25, 0.1])),
            lr_scale_factor=controller_cfg.get("lr_scale_factor", 0.5),
        ),
        disc_start_step=disc_start_step,
    )
    # Print observation mode status prominently
    if not gan_controller.config.escalation_enabled:
        print("GAN Controller: OBSERVATION MODE - events logged but no interventions")
    else:
        print(f"GAN Controller: ADV ramp over {gan_controller.config.adv_ramp_steps} steps after disc start")
    print(f"  Hard ceiling: G={gan_controller.config.absolute_grad_limit_g:.0f}, D={gan_controller.config.absolute_grad_limit_d:.0f}" + (" (disabled)" if not gan_controller.config.hard_ceiling_enabled else ""))
    print(f"  Spike density: {gan_controller.config.spike_density_threshold} in {gan_controller.config.spike_density_window} steps")
    print(f"  EMA warning: >{gan_controller.config.ema_elevated_limit_g:.0f} for {gan_controller.config.ema_elevated_steps_threshold} steps")
    print(f"  De-escalation: L1={gan_controller.config.stability_required_steps_l1}, L2={gan_controller.config.stability_required_steps_l2}, L3={gan_controller.config.stability_required_steps_l3} steps")
    print(f"  Soft gates: EMA_G<{gan_controller.config.soft_grad_limit_g:.0f}")
    print(f"  D-real velocity: warn={gan_controller.config.d_real_velocity_warning}, crit={gan_controller.config.d_real_velocity_critical}, emerg={gan_controller.config.d_real_velocity_emergency}")
    print(f"  D-freeze: starts at L{gan_controller.config.d_freeze_start_level}, probes={'forward-only' if gan_controller.config.d_freeze_probe_forward_only else 'update D'}")
    print(f"Session: {events.session_id}")

    # Metrics file is now managed by EventLogger (session-based)
    metrics_file = events.metrics_file

    step = 0
    epoch = 0
    best_val_loss = float("inf")
    resume_step = None
    resume_time = None
    early_save_done = False

    if resume_path and resume_path.exists():
        print(f"Resuming from: {resume_path}")
        step, epoch, best_val_loss = load_checkpoint(
            resume_path, model, optimizer_g, scaler, config, device,
            mpd=mpd, msd=msd, optimizer_d=optimizer_d
        )
        # Load controller state if present
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        if "controller_state" in ckpt:
            gan_controller.load_state_dict(ckpt["controller_state"])
            print(f"  Controller state restored")
        print(f"  Step: {step}, Epoch: {epoch}, Best val loss: {best_val_loss:.4f}")

        # P0: Checkpoint on resume — save immediately to new run dir
        resume_step = step
        resume_time = time.time()
        ckpt_name = f"step_{step:06d}_resume_start.pt"
        save_checkpoint(
            output_dir / "checkpoints" / ckpt_name,
            model, optimizer_g, scaler, step, epoch, best_val_loss, config,
            mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
            speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
        )
        update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, "resume_start")
        events.checkpoint_saved(
            f"checkpoints/{ckpt_name}", step, best_val_loss, tag="resume_start",
            alarm_state=gan_controller.state.alarm_state.value,
        )
        print(f"  Saved resume checkpoint: {ckpt_name}")
        print()

    # Log run start event
    events.run_started(
        stage="gan",
        dataset=dataset,
        max_steps=max_steps,
        num_speakers=num_speakers,
        batch_size=batch_size,
        resume_from=str(resume_path) if resume_path else None,
    )

    print("=" * 70)
    print(f"Starting training (step {step} -> {max_steps})")
    print("=" * 70)
    print()
    print(f"{'Step':>8} | {'G':>8} | {'D':>8} | {'Mel':>8} | {'Adv':>8} | {'FM':>8} | {'GN_G':>6} | {'GN_D':>6} | Status")
    print("-" * 95)

    train_iter = iter(train_loader)
    start_time = time.time()

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if batch is None:
            continue

        # Guardrail: require labeled data (phonemes/durations)
        if not batch.has_text:
            raise ValueError(
                f"This pipeline requires labeled text/phonemes, but batch has "
                f"{batch.n_labeled}/{batch.is_labeled.numel()} labeled samples. "
                "Tier 1 segment manifests are unlabeled. Use a full-utterance manifest "
                "or wait for Tier 2 labeling."
            )

        # Get current KL weight (with annealing)
        kl_weight = get_kl_weight(step, kl_anneal_cfg)

        # Decide whether to use GAN or core-only
        use_gan = step >= disc_start_step

        if use_gan:
            # Get controller decisions
            update_d = gan_controller.should_update_d(step)
            adv_weight_scale = gan_controller.get_adv_weight_scale(step)
            mel_weight_scale = gan_controller.get_mel_weight_scale(step)
            grad_clip_scale = gan_controller.get_grad_clip_scale(step)
            lr_scale = gan_controller.get_lr_scale(step)

            # P1: Apply LR scaling if active
            if lr_scale < 1.0:
                orig_lr_g = scale_optimizer_lr(optimizer_g, lr_scale)
                orig_lr_d = scale_optimizer_lr(optimizer_d, lr_scale)

            metrics = train_step_gan(
                model, mpd, msd, batch,
                optimizer_g, optimizer_d, scaler,
                grad_clip_g, grad_clip_d, device, use_amp,
                mel_cfg, loss_cfg, kl_weight,
                update_d=update_d,
                adv_weight_scale=adv_weight_scale,
                mel_weight_scale=mel_weight_scale,
                grad_clip_scale=grad_clip_scale,
            )

            # P1: Restore original LRs
            if lr_scale < 1.0:
                restore_optimizer_lr(optimizer_g, orig_lr_g)
                restore_optimizer_lr(optimizer_d, orig_lr_d)

            # Record step with controller and track alarm changes
            prev_alarm = gan_controller.state.alarm_state.value
            prev_escalation = gan_controller.state.escalation_level
            prev_d_freeze = gan_controller.state.d_freeze_active
            controller_state = gan_controller.record_step(
                step=step,
                d_loss=metrics["loss_d"],
                g_loss_adv=metrics.get("g_loss_adv", metrics.get("loss_adv", 0.0)),
                grad_norm_g=metrics.get("g_grad_norm", metrics.get("grad_norm_g", 0.0)),
                grad_norm_d=metrics.get("d_grad_norm", metrics.get("grad_norm_d", 0.0)),
                g_step_skipped=metrics.get("g_step_skipped", False),
                d_step_skipped=metrics.get("d_step_skipped", False),
                g_clip_coef=metrics.get("g_clip_coef"),
                d_clip_coef=metrics.get("d_clip_coef"),
                d_real_score=metrics.get("d_real_score"),
                d_fake_score=metrics.get("d_fake_score"),
                mel_loss=metrics.get("g_loss_mel"),
            )
            curr_alarm = gan_controller.state.alarm_state.value
            curr_escalation = gan_controller.state.escalation_level
            curr_d_freeze = gan_controller.state.d_freeze_active
            state_reason = gan_controller.state.last_alarm_reason or "unknown"
            if curr_alarm != prev_alarm:
                events.alarm_state_change(
                    prev_alarm, curr_alarm, step,
                    reason=state_reason
                )
                print(f"         | Step {step}: Alarm {prev_alarm} → {curr_alarm} ({state_reason})")
            if curr_escalation != prev_escalation:
                events.escalation_level_change(
                    prev_escalation, curr_escalation, step,
                    reason=state_reason
                )
                print(f"         | Step {step}: Escalation L{prev_escalation} → L{curr_escalation} ({state_reason})")
            metrics["controller"] = controller_state

            # Save "last known good" checkpoint when D-freeze activates (regardless of level)
            # This gives a clean recovery point before discriminator/G dynamics drift further.
            if curr_d_freeze and not prev_d_freeze:
                lkg_name = f"step_{step:06d}_last_known_good.pt"
                save_checkpoint(
                    output_dir / "checkpoints" / lkg_name,
                    model, optimizer_g, scaler, step, epoch, best_val_loss, config,
                    mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
                    speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                )
                events.checkpoint_saved(
                    f"checkpoints/{lkg_name}", step, best_val_loss, tag="last_known_good",
                    mel_loss=metrics.get("mel_loss"),
                    alarm_state=curr_alarm,
                )
                print(f"         | D-freeze activated: saved last_known_good checkpoint")

            # P0: Emergency stop check - absolute grad limit or consecutive NaN/Inf
            if gan_controller.requires_emergency_stop():
                emergency_reason = gan_controller.state.emergency_reason
                print(f"\n{'='*70}")
                print("EMERGENCY STOP - GRADIENT CATASTROPHE")
                print(f"Reason: {emergency_reason}")
                print(f"Step: {step}")
                print(f"Metrics at failure:")
                print(f"  G loss: {metrics.get('g_loss', 'N/A'):.4f}" if isinstance(metrics.get('g_loss'), (int, float)) else f"  G loss: {metrics.get('g_loss', 'N/A')}")
                print(f"  D loss: {metrics.get('d_loss', 'N/A'):.4f}" if isinstance(metrics.get('d_loss'), (int, float)) else f"  D loss: {metrics.get('d_loss', 'N/A')}")
                print(f"  Mel loss: {metrics.get('mel_loss', 'N/A'):.4f}" if isinstance(metrics.get('mel_loss'), (int, float)) else f"  Mel loss: {metrics.get('mel_loss', 'N/A')}")
                print(f"  KL loss: {metrics.get('kl_loss', 'N/A'):.4f}" if isinstance(metrics.get('kl_loss'), (int, float)) else f"  KL loss: {metrics.get('kl_loss', 'N/A')}")
                print(f"  G grad norm: {metrics.get('g_grad_norm', 'N/A')}")
                print(f"  D grad norm: {metrics.get('d_grad_norm', 'N/A')}")
                print(f"  Escalation level: L{gan_controller.state.escalation_level}")
                print(f"  Consecutive NaN/Inf: {gan_controller.state.consecutive_nan_inf}")
                print(f"{'='*70}")

                # Save emergency checkpoint
                ckpt_name = f"step_{step:06d}_emergency.pt"
                save_checkpoint(
                    output_dir / "checkpoints" / ckpt_name,
                    model, optimizer_g, scaler, step, epoch, best_val_loss, config,
                    mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
                    speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                )
                events.checkpoint_saved(
                    f"checkpoints/{ckpt_name}", step, best_val_loss, tag="emergency",
                    mel_loss=metrics.get("mel_loss"),
                    alarm_state=gan_controller.state.alarm_state.value,
                )
                events.training_complete(
                    step, best_val_loss, best_val_loss, time.time() - start_time,
                    status="emergency_stop",
                    reason=emergency_reason,
                    checkpoint_path=f"checkpoints/{ckpt_name}",
                )
                print(f"Emergency checkpoint saved: {ckpt_name}")
                print("Training halted to prevent further damage.")
                return {"status": "emergency_stop", "step": step, "checkpoint": ckpt_name, "reason": emergency_reason}
        else:
            # Core-only training before disc_start_step
            metrics = train_step(
                model, batch, optimizer_g, scaler,
                grad_clip_g, device, use_amp,
                mel_cfg, {"kl_weight": kl_weight, "dur_weight": loss_cfg.get("dur_weight", 1.0)},
            )
            # Add placeholders for GAN metrics
            metrics["loss_g"] = metrics["total_loss"]
            metrics["loss_d"] = 0.0
            metrics["loss_adv"] = 0.0
            metrics["loss_fm"] = 0.0
            metrics["grad_norm_g"] = metrics["grad_norm"]
            metrics["grad_norm_d"] = 0.0
            metrics["g_loss_adv"] = 0.0
            metrics["g_grad_norm"] = metrics["grad_norm"]
            metrics["d_grad_norm"] = 0.0
            metrics["d_real_score"] = 0.0
            metrics["d_fake_score"] = 0.0
            metrics["skipped_d_step"] = True
            metrics["g_step_skipped"] = False  # Core-only path never skips G
            metrics["d_step_skipped"] = True   # D not active yet
            metrics["g_clip_coef"] = 1.0
            metrics["d_clip_coef"] = 1.0
            metrics["controller"] = {"controller_alarm": "healthy", "adv_weight_scale": 0.0}

        step += 1

        # Write metrics to JSONL
        metrics_row = {
            "step": step,
            "epoch": epoch,
            "loss_g": metrics["loss_g"],
            "loss_d": metrics["loss_d"],
            "g_loss_mel": metrics.get("g_loss_mel", metrics.get("mel_loss", 0.0)),
            "g_loss_kl": metrics.get("g_loss_kl", metrics.get("kl_loss", 0.0)),
            "g_loss_dur": metrics.get("g_loss_dur", metrics.get("dur_loss", 0.0)),
            "g_loss_adv": metrics.get("g_loss_adv", metrics.get("loss_adv", 0.0)),
            "g_loss_fm": metrics.get("g_loss_fm", metrics.get("loss_fm", 0.0)),
            "g_grad_norm": metrics.get("g_grad_norm", metrics.get("grad_norm_g", 0.0)),
            "d_grad_norm": metrics.get("d_grad_norm", metrics.get("grad_norm_d", 0.0)),
            "d_real_score": metrics.get("d_real_score", 0.0),
            "d_fake_score": metrics.get("d_fake_score", 0.0),
            "kl_weight": kl_weight,
            "adv_active": use_gan,
            "skipped_d_step": metrics.get("skipped_d_step", False),
            # Explicit step skip + clip coefficient signals for diagnostics
            "g_step_skipped": metrics.get("g_step_skipped", False),
            "d_step_skipped": metrics.get("d_step_skipped", False),
            "g_clip_coef": metrics.get("g_clip_coef", 1.0),
            "d_clip_coef": metrics.get("d_clip_coef", 1.0),
            **{f"ctrl_{k}": v for k, v in metrics.get("controller", {}).items()},
        }
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics_row) + "\n")

        # Poll for control commands
        if control.should_poll(step):
            ctrl_request = control.poll()
            if ctrl_request:
                if ctrl_request.action == "checkpoint":
                    # Manual checkpoint request
                    tag = ctrl_request.params.get("tag", "manual")
                    ckpt_name = f"step_{step:06d}_{tag}.pt"
                    save_checkpoint(
                        output_dir / "checkpoints" / ckpt_name,
                        model, optimizer_g, scaler, step, epoch, best_val_loss, config,
                        mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
                        speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                    )
                    update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, tag)
                    events.checkpoint_saved(
                        f"checkpoints/{ckpt_name}", step, best_val_loss, tag=tag,
                        mel_loss=metrics.get("mel_loss"),
                        alarm_state=gan_controller.state.alarm_state.value,
                    )
                    control.ack(ctrl_request, success=True, result={"path": ckpt_name, "step": step})
                    print(f"         | Control: checkpoint saved ({tag})")

                elif ctrl_request.action == "eval":
                    # Extract eval params
                    eval_mode = ctrl_request.params.get("mode", "multispeaker")
                    eval_seed = ctrl_request.params.get("seed", 42)
                    eval_tag = ctrl_request.params.get("tag", "ui")
                    eval_id = generate_eval_id(eval_tag)
                    run_id = output_dir.name

                    # Ack immediately with queued status
                    control.ack(
                        ctrl_request,
                        success=True,
                        result={"status": "queued", "eval_id": eval_id, "params": ctrl_request.params},
                    )
                    print(f"         | Control: eval queued ({eval_id})")

                    # Save tagged checkpoint for reproducibility
                    eval_ckpt_name = f"step_{step:06d}_eval_{eval_tag}.pt"
                    save_checkpoint(
                        output_dir / "checkpoints" / eval_ckpt_name,
                        model, optimizer_g, scaler, step, epoch, best_val_loss, config,
                        mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
                        speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                    )

                    # Log eval_started
                    events.eval_started(
                        eval_id=eval_id,
                        run_id=run_id,
                        step=step,
                        mode=eval_mode,
                        seed=eval_seed,
                        tag=eval_tag,
                        nonce=ctrl_request.nonce,
                    )

                    # Run eval inline (pauses training)
                    try:
                        from modules.training.pipelines.synthesize import eval_multispeaker

                        print(f"         | Running eval: {eval_id}")
                        result = eval_multispeaker(
                            run_id=run_id,
                            checkpoint=eval_ckpt_name,
                            seed=eval_seed,
                            device=str(device),
                        )

                        # Log eval_complete
                        events.eval_complete(
                            eval_id=eval_id,
                            run_id=run_id,
                            step=step,
                            artifact_dir=result["output_dir"],
                            summary={
                                "n_speakers": result["n_speakers"],
                                "n_prompts": result["n_prompts"],
                                "mean_inter_speaker_distance": result["manifest"]["separation_metrics"]["mean_inter_speaker_distance"],
                                "valid_outputs": f"{sum(s['n_valid'] for s in result['manifest']['per_speaker_summary'].values())}/{sum(s['n_samples'] for s in result['manifest']['per_speaker_summary'].values())}",
                            },
                            nonce=ctrl_request.nonce,
                        )
                        print(f"         | Eval complete: {result['output_dir']}")

                    except Exception as e:
                        events.eval_failed(
                            eval_id=eval_id,
                            run_id=run_id,
                            step=step,
                            error=str(e),
                            nonce=ctrl_request.nonce,
                        )
                        print(f"         | Eval failed: {e}")

                elif ctrl_request.action == "stop":
                    # Graceful stop request - save tagged checkpoint and exit
                    tag = ctrl_request.params.get("tag", "manual_stop")
                    stop_ckpt_name = f"step_{step:06d}_{tag}.pt"
                    save_checkpoint(
                        output_dir / "checkpoints" / stop_ckpt_name,
                        model, optimizer_g, scaler, step, epoch, best_val_loss, config,
                        mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
                        speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                    )
                    update_run_md_checkpoint(output_dir, stop_ckpt_name, step, best_val_loss, tag)
                    events.checkpoint_saved(
                        f"checkpoints/{stop_ckpt_name}", step, best_val_loss, tag=tag,
                        mel_loss=metrics.get("mel_loss"),
                        alarm_state=gan_controller.state.alarm_state.value,
                    )
                    control.ack(ctrl_request, success=True, result={
                        "path": f"checkpoints/{stop_ckpt_name}",
                        "step": step,
                        "tag": tag,
                    })
                    print(f"         | Control: stop requested, checkpoint saved ({tag})")
                    stop_requested = True

        # Check for user-requested graceful stop
        if stop_requested:
            print(f"\n{'='*70}")
            print("USER REQUESTED STOP")
            print(f"{'='*70}")
            break  # Exit loop, run normal teardown

        # Thermal watchdog check (time-based, runs independently of step cadence)
        if watchdog.should_check():
            thermal_action = watchdog.check()
            if thermal_action == "stop":
                # Emergency thermal shutdown - save checkpoint and exit
                print(f"\n{'='*70}")
                print(f"THERMAL EMERGENCY SHUTDOWN")
                print(f"GPU temperature exceeded {watchdog.config.stop_temp}°C for {watchdog.config.grace_seconds}s")
                print(f"{'='*70}")

                ckpt_name = f"step_{step:06d}_thermal_stop.pt"
                save_checkpoint(
                    output_dir / "checkpoints" / ckpt_name,
                    model, optimizer_g, scaler, step, epoch, best_val_loss, config,
                    mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
                    speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                )
                events.checkpoint_saved(
                    f"checkpoints/{ckpt_name}", step, best_val_loss, tag="thermal_stop",
                    mel_loss=metrics.get("mel_loss"),
                    alarm_state=gan_controller.state.alarm_state.value,
                )
                events.training_complete(
                    step, best_val_loss, best_val_loss, time.time() - start_time,
                    status="thermal_stop",
                    reason=f"GPU temp exceeded {watchdog.config.stop_temp}C",
                    checkpoint_path=f"checkpoints/{ckpt_name}",
                )
                print(f"Emergency checkpoint saved: {ckpt_name}")
                print(f"Exiting to protect hardware.")
                return {"status": "thermal_shutdown", "step": step, "checkpoint": ckpt_name}

        if step % log_every == 0:
            # Build status string
            ctrl = metrics.get("controller", {})
            alarm = ctrl.get("controller_alarm", "healthy")
            adv_scale = ctrl.get("adv_weight_scale", 0.0)
            esc_level = ctrl.get("escalation_level", 0)
            stable_steps = ctrl.get("stable_steps_at_level", 0)
            stability_req = ctrl.get("stability_threshold", 200)

            if not use_gan:
                status = "CORE"
            elif esc_level > 0:
                # Show escalation level and steps to decay
                steps_to_decay = max(0, stability_req - stable_steps)
                status = f"GAN/L{esc_level} {steps_to_decay:>3d}"
            elif alarm != "healthy":
                status = f"GAN/{alarm.upper()}"
            elif adv_scale < 1.0:
                status = f"GAN/ramp:{adv_scale:.1%}"
            else:
                status = "GAN"

            gn_g = fmt_grad_norm(metrics['grad_norm_g'])
            gn_d = fmt_grad_norm(metrics['grad_norm_d'])
            print(
                f"{step:>8} | {metrics['loss_g']:>8.4f} | {metrics['loss_d']:>8.4f} | "
                f"{metrics['mel_loss']:>8.4f} | {metrics['loss_adv']:>8.4f} | {metrics['loss_fm']:>8.4f} | "
                f"{gn_g} | {gn_d} | {status}"
            )

        if step % val_every == 0:
            val_metrics = validate(model, val_loader, device, max_val_batches, mel_cfg, loss_cfg)
            print(f"         | Val: loss={val_metrics['val_total_loss']:.4f} "
                  f"mel={val_metrics['val_mel_loss']:.4f}")

            if val_metrics["val_total_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_total_loss"]
                save_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    model, optimizer_g, scaler, step, epoch, best_val_loss, config,
                    mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
                    speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                )
                update_run_md_checkpoint(output_dir, "best.pt", step, best_val_loss, "new best")
                events.checkpoint_saved(
                    "checkpoints/best.pt", step, best_val_loss, tag="best", is_best=True,
                    mel_loss=val_metrics.get("val_mel_loss"),
                    alarm_state=gan_controller.state.alarm_state.value,
                )
                print(f"         | New best model saved")

        if step % save_every == 0:
            ckpt_name = f"step_{step:06d}.pt"
            save_checkpoint(
                output_dir / "checkpoints" / ckpt_name,
                model, optimizer_g, scaler, step, epoch, best_val_loss, config,
                mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
                speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
            )
            update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, "periodic")
            events.checkpoint_saved(
                f"checkpoints/{ckpt_name}", step, best_val_loss, tag="periodic",
                mel_loss=metrics.get("mel_loss"),
                alarm_state=gan_controller.state.alarm_state.value,
            )
            early_save_done = True  # Periodic save counts as early save

        # P0: Early-first save for resumed runs (5 min OR 500 steps, whichever first)
        if resume_step is not None and not early_save_done:
            elapsed_since_resume = time.time() - resume_time
            steps_since_resume = step - resume_step
            if elapsed_since_resume >= 300 or steps_since_resume >= 500:
                ckpt_name = f"step_{step:06d}_early.pt"
                save_checkpoint(
                    output_dir / "checkpoints" / ckpt_name,
                    model, optimizer_g, scaler, step, epoch, best_val_loss, config,
                    mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
                    speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
                )
                update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, "early")
                events.checkpoint_saved(
                    f"checkpoints/{ckpt_name}", step, best_val_loss, tag="early",
                    mel_loss=metrics.get("mel_loss"),
                    alarm_state=gan_controller.state.alarm_state.value,
                )
                print(f"         | Early checkpoint saved: {ckpt_name}")
                early_save_done = True

    # Save final checkpoint (skip if stop was requested - we already saved stop checkpoint)
    if not stop_requested:
        save_checkpoint(
            output_dir / "checkpoints" / "final.pt",
            model, optimizer_g, scaler, step, epoch, best_val_loss, config,
            mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
            speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
        )
        update_run_md_checkpoint(output_dir, "final.pt", step, best_val_loss, "final")
        events.checkpoint_saved(
            "checkpoints/final.pt", step, best_val_loss, tag="final",
            alarm_state=gan_controller.state.alarm_state.value,
        )

    # Final drain: handle any pending control requests before completing
    def handle_final_control(req):
        if req.action == "checkpoint":
            tag = req.params.get("tag", "final_drain")
            ckpt_name = f"step_{step:06d}_{tag}.pt"
            save_checkpoint(
                output_dir / "checkpoints" / ckpt_name,
                model, optimizer_g, scaler, step, epoch, best_val_loss, config,
                mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
                speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
            )
            update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, tag)
            events.checkpoint_saved(
                f"checkpoints/{ckpt_name}", step, best_val_loss, tag=tag,
                alarm_state=gan_controller.state.alarm_state.value,
            )
            control.ack(req, success=True, result={"path": ckpt_name, "step": step})
            print(f"Final drain: checkpoint saved ({tag})")
        elif req.action == "stop":
            # Execute stop even in final drain (user wants tagged checkpoint)
            tag = req.params.get("tag", "final_stop")
            ckpt_name = f"step_{step:06d}_{tag}.pt"
            save_checkpoint(
                output_dir / "checkpoints" / ckpt_name,
                model, optimizer_g, scaler, step, epoch, best_val_loss, config,
                mpd=mpd, msd=msd, optimizer_d=optimizer_d, controller=gan_controller,
                speaker_list=speaker_vocab.speakers if num_speakers > 1 else None,
            )
            update_run_md_checkpoint(output_dir, ckpt_name, step, best_val_loss, tag)
            events.checkpoint_saved(
                f"checkpoints/{ckpt_name}", step, best_val_loss, tag=tag,
                alarm_state=gan_controller.state.alarm_state.value,
            )
            control.ack(req, success=True, result={
                "path": f"checkpoints/{ckpt_name}",
                "step": step,
                "tag": tag,
                "note": "final_drain",
            })
            print(f"Final drain: stop executed, checkpoint saved ({tag})")
        else:
            # Eval can't run at training end
            control.ack(req, success=False, error=f"Training complete, cannot execute {req.action}")
            print(f"Final drain: rejected {req.action} (training complete)")

    control.drain(handle_final_control)

    # Final validation (skip if stopped to exit faster)
    if not stop_requested:
        final_val = validate(model, val_loader, device, max_val_batches, mel_cfg, loss_cfg)
        final_val_loss = final_val["val_total_loss"]
    else:
        final_val_loss = best_val_loss  # Use last known

    total_time = time.time() - start_time

    # Emit training_complete with appropriate status
    if stop_requested:
        events.training_complete(
            step, best_val_loss, final_val_loss, total_time,
            status="user_stopped",
            reason="user_requested",
            checkpoint_path=f"checkpoints/{stop_ckpt_name}",
        )
    else:
        events.training_complete(step, best_val_loss, final_val_loss, total_time)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE" if not stop_requested else "TRAINING STOPPED")
    print("=" * 70)
    print(f"Total steps: {step}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Final val loss: {final_val_loss:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir / 'checkpoints'}")
    print("=" * 70)

    return {
        "step": step,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "output_dir": str(output_dir),
        "status": "user_stopped" if stop_requested else "success",
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train VITS model")
    parser.add_argument("dataset", help="Dataset name (jsut, jvs)")
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/training/vits_core.yaml"))
    parser.add_argument("--stage", choices=["core", "gan"], default="core", help="Training stage")
    parser.add_argument("--snapshot", "-s", help="Cache snapshot ID")
    parser.add_argument("--resume", "-r", type=Path, help="Resume from checkpoint")
    parser.add_argument("--output-dir", "-o", type=Path, help="Output directory")
    parser.add_argument("--max-steps", type=int, help="Override max_steps")
    parser.add_argument("--batch-size", type=int, help="Override batch_size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")

    args = parser.parse_args()

    config_path = args.config

    # Select default config based on stage
    if not config_path.exists():
        if args.stage == "core":
            config_path = Path("configs/training/vits_core.yaml")
        else:
            config_path = Path("configs/training/vits_gan.yaml")

    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        if args.stage == "core":
            config = {
                "run": {"name": "vits_core", "seed": 42, "device": "cuda", "amp": True},
                "data": {"batch_size": 16, "num_workers": 4, "max_val_batches": 50},
                "mel": {"sample_rate": 22050, "n_mels": 80, "hop_length": 256},
                "model": {"latent_dim": 192},
                "loss": {"kl_weight": 1.0, "dur_weight": 1.0},
                "optim": {"lr": 2e-4, "weight_decay": 0.01, "grad_clip": 1.0},
                "train": {"max_steps": 100000, "log_every_steps": 100, "val_every_steps": 2000, "save_every_steps": 5000},
            }
        else:
            config = {
                "run": {"name": "vits_gan", "seed": 42, "device": "cuda", "amp": True},
                "data": {"batch_size": 16, "num_workers": 4, "max_val_batches": 50},
                "mel": {"sample_rate": 22050, "n_mels": 80, "hop_length": 256},
                "model": {"latent_dim": 192},
                "discriminator": {"enabled": True, "disc_start_step": 10000},
                "loss": {"mel_weight": 45.0, "kl_weight": 1.0, "dur_weight": 1.0, "adv_weight": 1.0, "fm_weight": 2.0},
                "kl_anneal": {"enabled": True, "start_weight": 0.0, "end_weight": 1.0, "steps": 20000},
                "optim": {"lr_g": 2e-4, "lr_d": 1e-4, "weight_decay": 0.0, "grad_clip_g": 1.0, "grad_clip_d": 1.0},
                "train": {"max_steps": 200000, "log_every_steps": 100, "val_every_steps": 2000, "save_every_steps": 5000},
            }

    # Apply overrides
    if args.max_steps:
        config["train"]["max_steps"] = args.max_steps
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        if args.stage == "core":
            config["optim"]["lr"] = args.lr
        else:
            config["optim"]["lr_g"] = args.lr
    if args.no_amp:
        config["run"]["amp"] = False

    # Select training function based on stage
    if args.stage == "core":
        train(
            config=config,
            dataset=args.dataset,
            snapshot_id=args.snapshot,
            resume_path=args.resume,
            output_dir=args.output_dir,
        )
    else:
        train_gan(
            config=config,
            dataset=args.dataset,
            snapshot_id=args.snapshot,
            resume_path=args.resume,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
