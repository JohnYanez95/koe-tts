# Postmortem: NaN Collapse in GAN Training

**Run ID:** `multi_vits_gan_20260126_041738`
**Date:** 2026-01-26
**Status:** Resolved

## Summary

Training experienced catastrophic gradient explosion at step 36,255, entering an irrecoverable NaN state by step 36,737. The run continued for 13,144 more steps producing garbage outputs. The GAN controller detected instability but could not prevent or recover from the collapse.

## Impact

- **Wasted compute:** ~8 hours of GPU time on unusable training
- **Polluted artifacts:** 13k steps of NaN metrics and invalid checkpoints
- **No usable recovery point:** Emergency checkpoint was already poisoned

## Timeline

| Time | Step | Event |
|------|------|-------|
| T+0h | 27,400 | Run resumed from checkpoint |
| T+1h | 29,070-33,239 | 4 episodic instabilities, each ~500 steps, recovered |
| T+3h | 33,240-36,254 | 3,000 steps stable, mitigations fully decayed |
| T+4h | 36,255 | g_grad=3,813, no protection active (grad_clip_scale=1.0) |
| T+4h | 36,256 | **EXPLOSION**: g_grad=NaN, loss_g=NaN, alarm triggered |
| T+4h | 36,256-36,737 | Chaotic recovery attempts, gradients spike to 94,868 |
| T+4h | 36,737 | Persistent NaN state begins |
| T+12h | 49,399 | Run completes (12,805 alarms triggered, no recovery) |

## Root Cause

### Primary: Detection was reactive, not preventive

NaN is detected AFTER `clip_grad_norm_` produces it. Once generated, NaN poisons all downstream computations.

### Contributing Factors

1. **Mitigation too weak**: Grad clip scale reduced from 1.0 → 0.5 (only halved). At step 36,255, grad was 3,813 — a 0.5x clip still allows high effective gradients.

2. **No escalation**: Only mitigation available was grad clip tightening. Missing: LR reduction, discriminator disabling, AMP overflow handling, emergency stop.

3. **Rapid mitigation decay**: Mitigations expired after 500 steps, resetting to full power without gradual recovery. Episode 4 ended at 33,239 → full reset by 33,739 → explosion at 36,255 hit with no protection.

4. **Relative thresholds adapted to instability**: Spike detection used p99 × 2.0 threshold. After 1000+ norm steps, p99 ≈ 3000. Grad of 3,813 barely triggered alarm.

### Key Metrics at Collapse

| Metric | Step 36,255 (last healthy) | Step 36,256 (explosion) |
|--------|---------------------------|------------------------|
| g_grad_norm | 3,813 | NaN |
| loss_g | 80.74 | NaN |
| g_loss_mel | 1.19 | NaN |
| g_loss_adv | 12.44 | NaN |
| grad_clip_scale | 1.0 (no protection) | 0.5 (too late) |
| alarm_state | healthy | unstable |

## Detection & Response

### What worked
- Controller detected instability and triggered UNSTABLE alarm
- Grad clip tightening activated (0.5x scale)
- Alarm state changes logged to events.jsonl

### What failed
- Detection happened AFTER NaN was generated
- Mitigation was insufficient to recover from explosion
- No escalation path beyond grad clip tightening
- No emergency stop mechanism
- Training continued for 13k steps producing garbage

## Fixes Shipped

### P0: Emergency Stop (commit `9d06c9d`)
- Absolute grad limit (default 5000): Hard ceiling triggers immediate emergency
- Consecutive NaN/Inf detection (default 3): Emergency after N bad steps
- Pre-clip grad check: Detects NaN/Inf before optimizer state corruption
- Emergency checkpoint saved before exit
- Clean termination with `training_complete(status="emergency_stop")`

### P1: Escalating Mitigation Ladder (commit `a1387f8`)
- Level 1: Grad clip 1.0 → 0.5
- Level 2: Grad clip → 0.25, LR → 50%
- Level 3: Grad clip → 0.1, LR → 50%, D freeze
- Level 4: Emergency stop (fail closed)
- Escalation memory: 2000 steps — repeated instability escalates rather than restarts
- Fix: Spike detection compares against window BEFORE adding current value

### Refinements (commit `d48627a`)
- Separate absolute limits for G and D
- D freeze probe windows (unfreeze every 100 steps to test stability)
- Loss value checking before backward (catch NaN before grads)
- "Last known good" checkpoint at level 3 entry

## Follow-ups

| Priority | Item | Status |
|----------|------|--------|
| P0 | Emergency stop on NaN/Inf | ✅ Shipped |
| P0 | Absolute grad ceiling | ✅ Shipped |
| P1 | Escalating mitigation ladder | ✅ Shipped |
| P1 | Sticky mitigations | ✅ Shipped |
| P2 | Gradual decay with stability validation | Open |
| P2 | Dashboard: 4-graph layout + alarm timeline | Open |

## Hypothetical Outcome with Fixes

With the shipped fixes, this run would have:

1. **Step 36,255**: Absolute grad limit (5000) — close call but safe (grad=3,813)
2. **Step 36,256**: NaN detected → consecutive counter = 1
3. **Step 36,257-36,258**: More NaN → consecutive counter = 2, 3 → **EMERGENCY STOP**
4. **Result**: Clean exit with `step_036258_emergency.pt` and `step_036255_last_known_good.pt` instead of 13k wasted steps

## Links

- **Events log:** `runs/multi_vits_gan_20260126_041738/events.jsonl`
- **Metrics:** `runs/multi_vits_gan_20260126_041738/train/metrics.jsonl`
- **Config:** `runs/multi_vits_gan_20260126_041738/config.yaml`
- **Commits:**
  - P0 emergency stop: `9d06c9d`
  - P1 escalation ladder: `a1387f8`
  - Refinements: `d48627a`
