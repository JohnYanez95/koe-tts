# Postmortem: multi_vits_gan_20260127_resume27k

## Incident Summary

| Field | Value |
|-------|-------|
| Run ID | `multi_vits_gan_20260127_resume27k` |
| Status | **Lost** |
| Detection | Dashboard "Lost" status / stale metrics heuristic |
| Duration | ~13 minutes |
| Steps completed | 842 (27,400 → 28,242) |
| Data lost | 842 steps (0 checkpoints) |
| Impact | Low (recoverable from parent checkpoint) |

---

## Confirmed Facts

- Metrics stopped at step 28,242; machine rebooted later
- No `training_complete` event logged
- No emergency stop triggered (`ctrl_emergency_stop: false`)
- No NaN/Inf detected at any logged step
- No checkpoints written during this run
- System journal marked as "corrupted or uncleanly shut down"
- No Windows shutdown events (IDs 1074, 6006, 6008) logged

---

## Timeline (Local Time)

| Time | Event |
|------|-------|
| 04:41 | Training started, resumed from `step_027400_stop.pt` |
| 04:48 | Alarm state → `unstable` at step 27,807 |
| ~04:54 | Last metrics logged (step 28,242) |
| ~04:54 | **Crash**: Generator grad norm missing, consistent with interruption during/around G backward or logging |
| ~04:54–05:47 | System down (~1 hour) |
| 05:47 | Windows boot sequence begins |
| 09:32 | WSL2 instance starts |

---

## Evidence

**Training state at last logged step (28,242):**
- `g_grad_norm: None` — missing (D completed normally with 27.0)
- `ctrl_nan_inf_detected: false`
- `ctrl_escalation_level: 2` — triggered at final step
- `ctrl_emergency_stop: false`
- Losses in normal range (loss_g ~66, loss_d ~15)

**System indicators:**
- `dmesg`: `"system.journal corrupted or uncleanly shut down"`
- `last -x`: reboot with no preceding shutdown record

**Windows Event Log (Event ID 109, Kernel-Power):**
- Action: `Power Action Reboot`
- Reason: `Kernel API` (not User API)
- Event Code: `0x0`
- Interpretation: Kernel-initiated reboot — consistent with BSOD + auto-restart

**Ruled out:**
- Power loss (Event ID 109 confirms kernel-initiated reboot)
- Training numerical instability (no NaN/Inf)
- Scheduled Windows Update (no shutdown events)
- User-initiated shutdown (Kernel API, not User API)
- Control plane emergency stop
- WSL2 crash (init errors in dmesg appear post-boot, not causal)
- NoiseCancelingEngine.exe crash at 09:32 (unrelated, hours after incident)

---

## Root Cause Analysis

**Root cause:** BSOD (driver/kernel crash) — no minidump written

| Hypothesis | Evidence For | Evidence Against | Status |
|------------|--------------|------------------|--------|
| BSOD (driver/kernel) | Event ID 109 Kernel API reboot, crash mid-GPU-op, no minidump | — | **Most likely** |
| GPU driver crash | Crash during G backward pass, no dump (GPU crashes often can't write dumps) | No LiveKernelEvent found | Likely (subset of BSOD) |
| Power loss / PSU trip | — | Event ID 109 confirms kernel-initiated reboot | **Ruled out** |
| Thermal shutdown | RTX 3090 under sustained GAN load | No temp logs, but would likely leave events | Low |

**Crash signature:** Generator grad norm missing at last logged step (`g_grad_norm=None`), consistent with interruption during/around G backward or logging. Discriminator completed normally. No numerical instability. Points to external termination (host/GPU/driver) rather than training logic failure.

**No minidump:** The absence of `.dmp` files despite a kernel-initiated reboot suggests either:
1. Crash too severe to write dump (common with GPU/driver hangs)
2. Minidumps disabled in system settings
3. Disk write failed during crash

---

## Observability Gaps

1. No GPU telemetry during training (temp, power, utilization)
2. No checkpoint-on-resume (first save would've been step 30,000)
3. No host heartbeat file for distinguishing "training hung" vs "host died"
4. Windows crash diagnostics not yet checked

---

## Actions

### Immediate (before next overnight run)

| Priority | Action | Status |
|----------|--------|--------|
| P0 | Check Reliability Monitor for Critical events ~05:47 | ✅ Done — no BSOD event, only unrelated app crash |
| P0 | Check for BSOD minidump files | ✅ Done — no dumps found |
| P0 | Check Event Viewer: Kernel-Power 41, WHEA-Logger, LiveKernelEvent | ✅ Done — Event ID 109 found (Kernel API reboot) |
| P0 | Confirm checkpoint cadence: save on resume + early first save | ✅ Done — `resume_start` + `early` (5min/500 steps) |
| P1 | Reduce `save_every_steps` to 1000–2500 for stress testing | Pending |

### If GPU TDR confirmed (LiveKernelEvent 141/117)
- Evaluate TDR delay registry adjustment (details pending confirmation)

### Longer-term improvements

| Area | Action |
|------|--------|
| Checkpointing | ✅ Save immediately on resume (`resume_start` tag) |
| Checkpointing | ✅ Early first save for resumed runs (500 steps OR 5 min, whichever first) |
| Telemetry | Log GPU stats to `runs/<run_id>/telemetry/gpu.csv` |
| Telemetry | Add host heartbeat file (`runs/<run_id>/heartbeat.json`) for stale detection |
| Host config | Verify: Sleep=Never, Hibernate=Never, UPS health |

---

## Recovery Path

```bash
koe train vits multi --stage gan \
  --resume runs/multi_vits_gan_20260126_034507/checkpoints/step_027400_stop.pt
```

Net loss: 842 steps (~13 min training time)

---

## Diagnostics Summary

**Completed:**
- ✅ Reliability Monitor: No critical events at crash time (only unrelated NoiseCancelingEngine.exe at 09:32)
- ✅ Event Viewer: Event ID 109 (Kernel-Power) — Kernel API initiated reboot
- ✅ Minidump check: No `.dmp` files found

**Conclusion:** Kernel-initiated reboot (BSOD) during GPU operation, no crash dump preserved. Most likely a GPU driver crash during generator backward pass. Power loss ruled out.
