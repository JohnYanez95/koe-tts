# Postmortem: CUDA Unknown Error During KVM Switch

## Incident Summary

| Field | Value |
|-------|-------|
| Date | 2026-01-29 |
| Run ID | `multi_vits_gan_20260129_calibrated` |
| Step Range | 15,000 → 19,652 |
| Outcome | **CUDA crash** — GPU became inaccessible |
| Incident Type | `infra_crash` |
| Training Status | Healthy (zero alarms) before crash |

---

## Context

Run was testing calibrated controller thresholds (fix for config bug). Training was healthy with zero alarms for 4,652 steps when the crash occurred.

User noted they switched a KVM to observe progress, then saw the crash.

---

## Error

```
RuntimeError: CUDA error: unknown error
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
```

GPU state became completely inaccessible — even printing model/optimizer state triggered CUDA errors.

---

## Hardware/Software

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 3090 (24GB) |
| Driver (Windows) | 576.28 |
| Driver (Linux/WSL) | 575.51.03 |
| CUDA | 12.9 |
| Platform | WSL2 on Windows |
| Temp at crash | Unknown (39C when checked after restart) |
| Thermal watchdog | `thermal_action = 'ok'` before crash |

---

## Possible Causes

| Cause | Likelihood | Notes |
|-------|------------|-------|
| KVM switch disrupting GPU | **High** | User was switching KVM when crash occurred |
| WSL2 GPU passthrough instability | Medium | Known issue with WSL2 + CUDA |
| Outdated driver | Medium | 576.28 is old; latest is 591.86 WHQL |
| Memory corruption from earlier OOM | Low | Previous OOM crash mentioned in history |
| Thermal event | Low | Watchdog showed 'ok' before crash |

---

## Related Incidents

- Previous OOM crash mentioned during original training
- BSOD during earlier run (2026-01-27, documented in separate postmortem)

---

## Action Items

| Priority | Action | Status |
|----------|--------|--------|
| P1 | Update NVIDIA driver to latest (591.86 WHQL) | **Pending** |
| P2 | Avoid KVM switching during training | Workaround |
| P2 | Consider adding GPU health check before resume | Future |
| P3 | Test with `CUDA_LAUNCH_BLOCKING=1` if recurs | Future |

---

## Recovery

Checkpoint available at step 18,000. Training was healthy before crash — can resume:

```bash
koe train vits multi --stage gan \
  --resume runs/multi_vits_gan_20260129_calibrated/checkpoints/step_018000.pt \
  --output-dir runs/multi_vits_gan_20260129_calibrated \
  --max-steps 35000 --save-every-steps 2000
```

---

## Driver Update Instructions

Current: 576.28
Latest: 591.86 WHQL

1. Download from [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/)
2. Install on Windows (WSL uses Windows driver)
3. Restart WSL: `wsl --shutdown` then reopen

---

## Lessons Learned

1. **KVM switches can crash GPU** — Avoid switching display during training
2. **Keep drivers updated** — 576.28 is multiple versions behind
3. **Training state was healthy** — This was infra, not model instability
4. **Checkpoints saved progress** — Lost only ~1,652 steps (18k → 19.6k)
