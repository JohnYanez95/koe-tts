"""
Monotonic Alignment Search (MAS) for VITS.

Finds the optimal monotonic alignment path between text and mel frames
using dynamic programming. This is the "hard" alignment used in VITS
that replaces external aligners.

Reference: Kim et al., VITS (2021), Section 3.2
"""

import torch
import numpy as np
from numba import jit


@jit(nopython=True)
def _mas_width_1_numpy(neg_log_probs: np.ndarray) -> np.ndarray:
    """
    Monotonic alignment search with path width 1 (Numba-accelerated).

    This finds the optimal path through the negative log probability matrix
    where each text token must be aligned to at least one mel frame.

    Args:
        neg_log_probs: [T_text, T_mel] negative log probabilities

    Returns:
        path: [T_text, T_mel] binary alignment matrix
    """
    T_text, T_mel = neg_log_probs.shape

    # DP table: Q[i, j] = best score to align text[:i+1] to mel[:j+1]
    Q = np.full((T_text + 1, T_mel + 1), np.inf, dtype=np.float32)
    Q[0, 0] = 0.0

    # Fill DP table
    for i in range(T_text):
        for j in range(T_mel):
            if j >= i:  # Monotonicity constraint: j >= i
                # Either extend from (i, j-1) or start new alignment from (i-1, j-1)
                if j > 0:
                    Q[i + 1, j + 1] = min(
                        Q[i + 1, j] + neg_log_probs[i, j],  # Extend
                        Q[i, j] + neg_log_probs[i, j],      # New
                    )
                else:
                    Q[i + 1, j + 1] = Q[i, j] + neg_log_probs[i, j]

    # Backtrack to find path
    path = np.zeros((T_text, T_mel), dtype=np.float32)
    i, j = T_text - 1, T_mel - 1

    while i >= 0 and j >= 0:
        path[i, j] = 1.0

        if i == 0:
            # Fill remaining mel frames for first text token
            for k in range(j):
                path[i, k] = 1.0
            break

        if j == 0:
            break

        # Check which direction we came from
        if Q[i + 1, j] < Q[i, j]:
            # Came from extending (i, j-1)
            j -= 1
        else:
            # Came from new alignment (i-1, j-1)
            i -= 1
            j -= 1

    return path


def monotonic_alignment_search(
    neg_log_probs: torch.Tensor,
    text_mask: torch.Tensor,
    mel_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Batch monotonic alignment search.

    Finds optimal monotonic alignment between text and mel sequences
    for each item in the batch.

    Args:
        neg_log_probs: [B, T_text, T_mel] negative log probabilities
            Typically computed as: -log(p(z_mel | text_hidden))
        text_mask: [B, T_text] True for valid text positions
        mel_mask: [B, T_mel] True for valid mel positions

    Returns:
        path: [B, T_text, T_mel] binary alignment matrix
            path[b, i, j] = 1 if text token i aligns to mel frame j
    """
    B = neg_log_probs.size(0)
    device = neg_log_probs.device

    # Get sequence lengths
    text_lens = text_mask.sum(dim=1).cpu().numpy().astype(int)
    mel_lens = mel_mask.sum(dim=1).cpu().numpy().astype(int)

    # Convert to numpy for Numba
    neg_log_probs_np = neg_log_probs.cpu().numpy()

    # Run MAS for each batch item
    paths = []
    for b in range(B):
        t_len = text_lens[b]
        m_len = mel_lens[b]

        # Extract valid region
        nlp = neg_log_probs_np[b, :t_len, :m_len]

        # Run MAS
        path = _mas_width_1_numpy(nlp)

        # Pad back to full size
        full_path = np.zeros_like(neg_log_probs_np[b])
        full_path[:t_len, :m_len] = path
        paths.append(full_path)

    paths = np.stack(paths, axis=0)
    return torch.from_numpy(paths).to(device)


def generate_path(
    duration: torch.Tensor,
    text_mask: torch.Tensor,
    mel_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Generate alignment path from durations.

    This is used during inference when we have predicted durations
    instead of running MAS.

    Args:
        duration: [B, T_text] duration (frames) per text token
        text_mask: [B, T_text] valid text positions
        mel_mask: [B, T_mel] valid mel positions

    Returns:
        path: [B, T_text, T_mel] binary alignment matrix
    """
    B, T_text = duration.shape
    T_mel = mel_mask.size(1)
    device = duration.device

    path = torch.zeros(B, T_text, T_mel, device=device)

    # Cumulative duration gives mel frame positions
    cum_dur = torch.cumsum(duration, dim=1)  # [B, T_text]

    for b in range(B):
        start = 0
        for t in range(T_text):
            if not text_mask[b, t]:
                break
            end = min(cum_dur[b, t].item(), T_mel)
            if start < end:
                path[b, t, int(start):int(end)] = 1.0
            start = end

    return path


def compute_log_probs_for_mas(
    prior_mean: torch.Tensor,
    prior_log_var: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probabilities for MAS.

    Computes log p(z[j] | prior[i]) for all (i, j) pairs,
    which is used to find the optimal alignment.

    Args:
        prior_mean: [B, latent_dim, T_text] prior mean from text encoder
        prior_log_var: [B, latent_dim, T_text] prior log variance
        z: [B, latent_dim, T_mel] latent from posterior encoder

    Returns:
        log_probs: [B, T_text, T_mel] log probabilities
    """
    # Reshape for broadcasting
    # prior: [B, C, T_text, 1]
    # z: [B, C, 1, T_mel]
    prior_mean = prior_mean.unsqueeze(-1)
    prior_log_var = prior_log_var.unsqueeze(-1)
    z = z.unsqueeze(2)

    # Gaussian log probability
    # log p(z | mean, var) = -0.5 * (log(2pi) + log_var + (z - mean)^2 / var)
    log_2pi = 1.8378770664093453
    var = torch.exp(prior_log_var)

    log_probs = -0.5 * (log_2pi + prior_log_var + (z - prior_mean) ** 2 / var)

    # Sum over latent dimension
    log_probs = log_probs.sum(dim=1)  # [B, T_text, T_mel]

    return log_probs
