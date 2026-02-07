"""
Audio DSP helpers for segmentation and trim detection.

Provides:
- Pause detection using adaptive RMS thresholding (internal silences)
- Trim detection for content onset/offset boundaries

Usage:
    from modules.data_engineering.common.audio import (
        PauseDetectionConfig,
        detect_silence_regions,
        TrimDetectionConfig,
        detect_trim_region,
    )

    config = PauseDetectionConfig()
    regions, debug_info = detect_silence_regions(waveform, sr, config)

    trim_config = TrimDetectionConfig()
    trim_start, trim_end, trim_info = detect_trim_region(waveform, sr, trim_config)
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class PauseDetectionConfig:
    """Configuration for pause/silence detection."""

    # RMS computation
    window_ms: int = 20
    hop_ms: int = 10

    # Adaptive threshold (default mode)
    adaptive: bool = True
    floor_db: float = -60.0  # absolute floor
    margin_db: float = 8.0  # below p10
    percentile: float = 10.0  # use 10th percentile

    # Manual override (if adaptive=False)
    silence_threshold_db: float = -40.0

    # Safety
    min_silence_db_clip: float = -80.0  # avoid log(0)
    eps: float = 1e-8

    # Region filtering
    min_pause_ms: int = 150  # minimum pause to consider
    merge_gap_ms: int = 80  # merge regions closer than this
    pad_ms: int = 30  # pad regions (clamped to [0, duration_ms])


@dataclass
class SilenceRegion:
    """A detected silence/pause region in audio."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def midpoint_ms(self) -> int:
        return (self.start_ms + self.end_ms) // 2

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {"start_ms": self.start_ms, "end_ms": self.end_ms}

    @classmethod
    def from_dict(cls, d: dict) -> "SilenceRegion":
        """Create from dict."""
        return cls(start_ms=d["start_ms"], end_ms=d["end_ms"])


def compute_rms_db(
    waveform: torch.Tensor,
    sr: int,
    window_ms: int = 20,
    hop_ms: int = 10,
    eps: float = 1e-8,
    min_db: float = -80.0,
) -> tuple[torch.Tensor, int]:
    """
    Compute RMS energy in dB over sliding windows.

    Args:
        waveform: Audio tensor [T]
        sr: Sample rate
        window_ms: Window size in milliseconds
        hop_ms: Hop size in milliseconds
        eps: Small value to avoid log(0)
        min_db: Minimum dB value (clips below this)

    Returns:
        Tuple of (rms_db tensor [N_frames], hop_samples)
    """
    window_samples = int(window_ms * sr / 1000)
    hop_samples = int(hop_ms * sr / 1000)

    if window_samples < 1:
        window_samples = 1
    if hop_samples < 1:
        hop_samples = 1

    # Compute number of frames
    n_samples = len(waveform)
    n_frames = max(1, (n_samples - window_samples) // hop_samples + 1)

    rms_values = []
    for i in range(n_frames):
        start = i * hop_samples
        end = start + window_samples
        if end > n_samples:
            end = n_samples
        if end <= start:
            rms_values.append(eps)
            continue

        frame = waveform[start:end]
        rms = torch.sqrt(torch.mean(frame**2) + eps)
        rms_values.append(rms.item())

    rms = torch.tensor(rms_values, dtype=torch.float32)

    # Convert to dB
    rms_db = 20 * torch.log10(torch.clamp(rms, min=eps))

    # Clip to minimum
    rms_db = torch.clamp(rms_db, min=min_db)

    return rms_db, hop_samples


def compute_adaptive_threshold(
    rms_db: torch.Tensor,
    floor_db: float = -60.0,
    margin_db: float = 8.0,
    percentile: float = 10.0,
) -> tuple[float, dict]:
    """
    Compute adaptive silence threshold based on RMS distribution.

    The threshold is: max(floor_db, percentile(rms_db, p) - margin_db)

    This adapts to the noise floor of each recording while maintaining
    a reasonable minimum floor.

    Args:
        rms_db: RMS energy in dB [N_frames]
        floor_db: Absolute floor threshold
        margin_db: Margin below percentile
        percentile: Which percentile to use (typically 10)

    Returns:
        Tuple of (threshold_db, debug_info dict)
    """
    # Compute percentile
    p = percentile / 100.0
    sorted_vals = torch.sort(rms_db).values
    idx = int(len(sorted_vals) * p)
    idx = max(0, min(idx, len(sorted_vals) - 1))
    rms_db_p = sorted_vals[idx].item()

    # Adaptive threshold
    threshold_db = max(floor_db, rms_db_p - margin_db)

    debug_info = {
        "rms_db_p10": round(rms_db_p, 2),
        "threshold_db_used": round(threshold_db, 2),
        "thr_formula": f"max({floor_db}, {rms_db_p:.1f} - {margin_db})",
    }

    return threshold_db, debug_info


def find_silent_runs(
    rms_db: torch.Tensor,
    threshold_db: float,
    hop_ms: int,
    min_pause_ms: int,
) -> list[tuple[int, int]]:
    """
    Find contiguous runs of silent frames.

    Args:
        rms_db: RMS energy in dB [N_frames]
        threshold_db: Silence threshold in dB
        hop_ms: Frame hop in milliseconds
        min_pause_ms: Minimum pause duration to keep

    Returns:
        List of (start_ms, end_ms) tuples
    """
    is_silent = rms_db < threshold_db

    regions = []
    in_region = False
    start_frame = 0

    for i, silent in enumerate(is_silent.tolist()):
        if silent and not in_region:
            # Start of silent region
            in_region = True
            start_frame = i
        elif not silent and in_region:
            # End of silent region
            in_region = False
            end_frame = i
            start_ms = start_frame * hop_ms
            end_ms = end_frame * hop_ms
            if end_ms - start_ms >= min_pause_ms:
                regions.append((start_ms, end_ms))

    # Handle region at end
    if in_region:
        end_frame = len(is_silent)
        start_ms = start_frame * hop_ms
        end_ms = end_frame * hop_ms
        if end_ms - start_ms >= min_pause_ms:
            regions.append((start_ms, end_ms))

    return regions


def merge_close_regions(
    regions: list[tuple[int, int]],
    merge_gap_ms: int,
) -> list[tuple[int, int]]:
    """
    Merge regions that are closer than merge_gap_ms.

    Args:
        regions: List of (start_ms, end_ms) tuples, sorted by start
        merge_gap_ms: Maximum gap to merge

    Returns:
        Merged regions
    """
    if not regions:
        return []

    merged = [regions[0]]

    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]

        if start - prev_end <= merge_gap_ms:
            # Merge with previous
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def pad_regions(
    regions: list[tuple[int, int]],
    pad_ms: int,
    duration_ms: int,
) -> list[tuple[int, int]]:
    """
    Pad region boundaries, clamped to valid range.

    Args:
        regions: List of (start_ms, end_ms) tuples
        pad_ms: Padding amount
        duration_ms: Total audio duration

    Returns:
        Padded regions
    """
    padded = []
    for start, end in regions:
        new_start = max(0, start - pad_ms)
        new_end = min(duration_ms, end + pad_ms)
        padded.append((new_start, new_end))
    return padded


def detect_silence_regions(
    waveform: torch.Tensor,
    sr: int,
    config: PauseDetectionConfig | None = None,
) -> tuple[list[SilenceRegion], dict]:
    """
    Detect silence/pause regions in audio using adaptive RMS thresholding.

    Algorithm:
    1. Compute RMS over sliding windows
    2. Convert to dB: rms_db = 20 * log10(max(rms, eps))
    3. Clip to min_silence_db_clip
    4. Compute threshold:
       - If adaptive: thr = max(floor_db, percentile(rms_db, p10) - margin_db)
       - Else: thr = silence_threshold_db
    5. Mark frames silent where rms_db < thr
    6. Find contiguous silent runs >= min_pause_ms
    7. Merge regions separated by < merge_gap_ms
    8. Pad regions by pad_ms (clamped to valid bounds)
    9. Return SilenceRegion objects + debug_info

    Args:
        waveform: Audio tensor [T] (mono, any sample rate)
        sr: Sample rate
        config: PauseDetectionConfig (uses defaults if None)

    Returns:
        Tuple of:
            - List of SilenceRegion objects
            - Debug info dict with threshold computation details
    """
    if config is None:
        config = PauseDetectionConfig()

    duration_ms = int(len(waveform) * 1000 / sr)

    # Handle very short audio
    if duration_ms < config.min_pause_ms:
        return [], {
            "rms_db_p10": 0.0,
            "threshold_db_used": config.floor_db,
            "thr_formula": "audio_too_short",
            "silence_pct": 0.0,
            "duration_ms": duration_ms,
        }

    # Step 1-3: Compute RMS in dB
    rms_db, hop_samples = compute_rms_db(
        waveform,
        sr,
        window_ms=config.window_ms,
        hop_ms=config.hop_ms,
        eps=config.eps,
        min_db=config.min_silence_db_clip,
    )

    hop_ms = config.hop_ms

    # Step 4: Compute threshold
    if config.adaptive:
        threshold_db, debug_info = compute_adaptive_threshold(
            rms_db,
            floor_db=config.floor_db,
            margin_db=config.margin_db,
            percentile=config.percentile,
        )
    else:
        threshold_db = config.silence_threshold_db
        debug_info = {
            "rms_db_p10": 0.0,
            "threshold_db_used": threshold_db,
            "thr_formula": f"manual({threshold_db})",
        }

    # Compute silence ratio (fraction of frames below threshold)
    is_silent = rms_db < threshold_db
    silence_pct = is_silent.float().mean().item() * 100.0
    debug_info["silence_pct"] = round(silence_pct, 2)
    debug_info["duration_ms"] = duration_ms

    # Step 5-6: Find silent runs
    raw_regions = find_silent_runs(
        rms_db,
        threshold_db,
        hop_ms=hop_ms,
        min_pause_ms=config.min_pause_ms,
    )

    # Step 7: Merge close regions
    merged_regions = merge_close_regions(raw_regions, config.merge_gap_ms)

    # Step 8: Pad regions
    padded_regions = pad_regions(merged_regions, config.pad_ms, duration_ms)

    # Step 9: Convert to SilenceRegion objects
    regions = [
        SilenceRegion(start_ms=start, end_ms=end)
        for start, end in padded_regions
    ]

    return regions, debug_info


def regions_to_breakpoints(
    regions: list[SilenceRegion],
    duration_ms: int,
    min_lead_ms: int = 250,
    min_tail_ms: int = 250,
) -> list[int]:
    """
    Convert silence regions to breakpoint positions (midpoints).

    Filters out breakpoints that are too close to audio boundaries.

    Args:
        regions: List of SilenceRegion objects
        duration_ms: Total audio duration
        min_lead_ms: Minimum distance from start
        min_tail_ms: Minimum distance from end

    Returns:
        List of breakpoint positions in ms (sorted)
    """
    breakpoints = []

    for region in regions:
        midpoint = region.midpoint_ms

        # Filter by position
        if midpoint < min_lead_ms:
            continue
        if midpoint > duration_ms - min_tail_ms:
            continue

        breakpoints.append(midpoint)

    return sorted(breakpoints)


@dataclass
class TrimDetectionConfig:
    """Configuration for content trim detection (onset/offset boundaries)."""

    # RMS computation
    window_ms: int = 20
    hop_ms: int = 10

    # Adaptive threshold (asymmetric margins)
    onset_margin_db: float = 8.0    # margin below percentile for speech onset
    offset_margin_db: float = 10.0  # margin below percentile for speech offset
    floor_db: float = -60.0         # absolute floor
    percentile: float = 10.0        # noise floor percentile

    # Content validity
    min_content_ms: int = 500       # minimum duration; below → fallback to full audio

    # Outward padding
    pad_start_ms: int = 30          # padding before detected onset
    pad_end_ms: int = 50            # padding after detected offset

    # Safety
    min_db_clip: float = -80.0      # avoid log(0)
    eps: float = 1e-8


def _compute_boundary_confidence(
    rms_db: torch.Tensor,
    boundary_frame: int,
    direction: str,
    context_frames: int = 5,
    k: float = 0.5,
    b: float = 6.0,
) -> float:
    """
    Compute logistic confidence score for a trim boundary.

    Compares mean RMS energy inside (speech side) vs outside (silence side)
    of the boundary. Sharp energy transitions yield high confidence.

    Args:
        rms_db: RMS energy in dB [N_frames]
        boundary_frame: Frame index of the detected boundary
        direction: "onset" (speech is right) or "offset" (speech is left)
        context_frames: Frames on each side for comparison window
        k: Logistic steepness parameter
        b: Logistic midpoint (dB delta where confidence = 0.5)

    Returns:
        Confidence in [0, 1]
    """
    if direction not in {"onset", "offset"}:
        msg = f"direction must be 'onset' or 'offset', got {direction!r}"
        raise ValueError(msg)

    n = len(rms_db)
    if n < 2:
        return 0.0

    if direction == "onset":
        # Inside = right of boundary (speech), outside = left (silence)
        outside_start = max(0, boundary_frame - context_frames)
        outside_end = boundary_frame
        inside_start = boundary_frame
        inside_end = min(n, boundary_frame + context_frames)
    else:
        # Inside = left of boundary (speech), outside = right (silence)
        inside_start = max(0, boundary_frame - context_frames)
        inside_end = boundary_frame
        outside_start = boundary_frame
        outside_end = min(n, boundary_frame + context_frames)

    if inside_end <= inside_start or outside_end <= outside_start:
        return 0.0

    mean_inside = rms_db[inside_start:inside_end].mean().item()
    mean_outside = rms_db[outside_start:outside_end].mean().item()

    delta_db = mean_inside - mean_outside
    confidence = 1.0 / (1.0 + math.exp(-k * (delta_db - b)))

    return round(confidence, 4)


def _trim_config_to_debug(config: TrimDetectionConfig) -> dict:
    """Extract config knobs for debug provenance."""
    return {
        "onset_margin_db": config.onset_margin_db,
        "offset_margin_db": config.offset_margin_db,
        "floor_db": config.floor_db,
        "percentile": config.percentile,
        "pad_start_ms": config.pad_start_ms,
        "pad_end_ms": config.pad_end_ms,
        "min_content_ms": config.min_content_ms,
        "window_ms": config.window_ms,
        "hop_ms": config.hop_ms,
        "min_db_clip": config.min_db_clip,
        "eps": config.eps,
    }


def detect_trim_region(
    waveform: torch.Tensor,
    sr: int,
    config: TrimDetectionConfig | None = None,
) -> tuple[int, int, dict]:
    """
    Detect content onset/offset boundaries for trimming.

    Algorithm:
    1. Compute RMS energy over sliding windows → dB
    2. Compute noise floor percentile from RMS distribution
    3. Derive asymmetric thresholds (onset_margin_db, offset_margin_db)
    4. Scan forward for first frame above onset threshold → trim_start
    5. Scan backward for last frame above offset threshold → trim_end
    6. Apply asymmetric outward padding (pad_start_ms, pad_end_ms)
    7. Clamp to [0, duration_ms]
    8. Validity check: if content < min_content_ms → fallback to full audio
    9. Compute logistic confidence scores at each boundary

    Args:
        waveform: Audio tensor [T] (mono, any sample rate)
        sr: Sample rate
        config: TrimDetectionConfig (uses defaults if None)

    Returns:
        Tuple of:
            - trim_start_ms: Content onset in ms (absolute from file start)
            - trim_end_ms: Content offset in ms (absolute from file start)
            - debug_info: Dict with thresholds, confidence, fallback details,
              and config provenance
    """
    if config is None:
        config = TrimDetectionConfig()

    duration_ms = int(len(waveform) * 1000 / sr)

    # Effective percentile after clamping to valid quantile range
    p = max(0.0, min(float(config.percentile), 100.0))
    q = p / 100.0
    q = max(1e-6, min(q, 1.0 - 1e-6))
    percentile_effective = round(q * 100.0, 6)

    # Handle very short audio
    if duration_ms < config.min_content_ms:
        return 0, duration_ms, {
            "onset_threshold_db": config.floor_db,
            "offset_threshold_db": config.floor_db,
            "rms_db_at_percentile": 0.0,
            "percentile_effective": percentile_effective,
            "confidence_start": 0.0,
            "confidence_end": 0.0,
            "found_onset": False,
            "found_offset": False,
            "fallback_to_full_audio": True,
            "fallback_reason": "audio_too_short",
            "duration_ms": duration_ms,
            **_trim_config_to_debug(config),
        }

    # Step 1: Compute RMS in dB
    rms_db, _hop_samples = compute_rms_db(
        waveform,
        sr,
        window_ms=config.window_ms,
        hop_ms=config.hop_ms,
        eps=config.eps,
        min_db=config.min_db_clip,
    )

    hop_ms = config.hop_ms
    n_frames = len(rms_db)

    # Step 2: Compute noise floor percentile (local — no coupling to
    # compute_adaptive_threshold key names or Delta schema)
    rms_db_p = torch.quantile(rms_db.float(), q).item()

    # Step 3: Asymmetric thresholds from shared percentile
    onset_threshold = max(config.floor_db, rms_db_p - config.onset_margin_db)
    offset_threshold = max(config.floor_db, rms_db_p - config.offset_margin_db)

    # Step 4: Scan forward for content onset
    found_onset = False
    onset_frame = 0
    for i in range(n_frames):
        if rms_db[i].item() >= onset_threshold:
            onset_frame = i
            found_onset = True
            break

    # Step 5: Scan backward for content offset
    found_offset = False
    offset_frame = n_frames - 1
    for i in range(n_frames - 1, -1, -1):
        if rms_db[i].item() >= offset_threshold:
            offset_frame = i
            found_offset = True
            break

    # No content detected at all → fallback immediately
    if not found_onset or not found_offset:
        return 0, duration_ms, {
            "onset_threshold_db": round(onset_threshold, 2),
            "offset_threshold_db": round(offset_threshold, 2),
            "rms_db_at_percentile": round(rms_db_p, 2),
            "percentile_effective": percentile_effective,
            "confidence_start": 0.0,
            "confidence_end": 0.0,
            "found_onset": found_onset,
            "found_offset": found_offset,
            "onset_frame": onset_frame,
            "offset_frame": offset_frame,
            "n_frames": n_frames,
            "fallback_to_full_audio": True,
            "fallback_reason": "no_content_detected",
            "duration_ms": duration_ms,
            **_trim_config_to_debug(config),
        }

    # Convert frames to ms (left-edge timing, consistent with pause detection)
    trim_start = onset_frame * hop_ms
    trim_end = (offset_frame + 1) * hop_ms

    # Step 6: Apply asymmetric outward padding
    trim_start -= config.pad_start_ms
    trim_end += config.pad_end_ms

    # Raw values before clamp/validity — useful for diagnosing near-misses
    raw_trim_start = max(0, min(trim_start, duration_ms))
    raw_trim_end = max(0, min(trim_end, duration_ms))

    # Step 7: Clamp to valid range
    trim_start = raw_trim_start
    trim_end = raw_trim_end

    # Step 8: Validity check
    fallback_to_full_audio = False
    fallback_reason = None

    if trim_end <= trim_start:
        fallback_to_full_audio = True
        fallback_reason = "inverted"
        trim_start, trim_end = 0, duration_ms
    elif trim_end - trim_start < config.min_content_ms:
        fallback_to_full_audio = True
        fallback_reason = "min_content"
        trim_start, trim_end = 0, duration_ms

    # Step 9: Confidence scores — zero on fallback (boundary is meaningless)
    if fallback_to_full_audio:
        confidence_start = 0.0
        confidence_end = 0.0
    else:
        confidence_start = _compute_boundary_confidence(
            rms_db, onset_frame, direction="onset"
        )
        confidence_end = _compute_boundary_confidence(
            rms_db, offset_frame, direction="offset"
        )

    debug_info = {
        "onset_threshold_db": round(onset_threshold, 2),
        "offset_threshold_db": round(offset_threshold, 2),
        "rms_db_at_percentile": round(rms_db_p, 2),
        "percentile_effective": percentile_effective,
        "confidence_start": confidence_start,
        "confidence_end": confidence_end,
        "found_onset": found_onset,
        "found_offset": found_offset,
        "onset_frame": onset_frame,
        "offset_frame": offset_frame,
        "n_frames": n_frames,
        "fallback_to_full_audio": fallback_to_full_audio,
        "fallback_reason": fallback_reason,
        "trim_start_ms_raw": raw_trim_start,
        "trim_end_ms_raw": raw_trim_end,
        "duration_ms": duration_ms,
        **_trim_config_to_debug(config),
    }

    return trim_start, trim_end, debug_info
