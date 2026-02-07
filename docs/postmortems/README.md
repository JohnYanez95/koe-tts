# Postmortems

Incident reports for training failures and infrastructure issues.

## Incident Index

| Date | Run ID | Type | Incident | Outcome |
|------|--------|------|----------|---------|
| 2026-01-29 | `multi_vits_gan_20260129_calibrated` | `infra_crash` | [CUDA Unknown Error (KVM)](./2026-01-29_cuda_unknown_error_kvm.md) | GPU crash during KVM switch; driver outdated |
| 2026-01-29 | multiple | `trigger_miscalibration` | [Clip-Coef Trigger Too Aggressive](./2026-01-29_poisoned_checkpoint_resumes.md) | 6/6 runs failed — triggers kill healthy training |
| 2026-01-29 | `multi_vits_gan_20260128_210613` | `numeric_instability` | [Grad Ceiling Emergency](./2026-01-29_multi_vits_gan_20260128_210613_grad_ceiling_emergency.md) | Spike density + EMA triggers shipped |
| 2026-01-27 | `multi_vits_gan_20260127_resume27k` | `infra_crash` | [Host Crash / Lost Run](./2026-01-27_multi_vits_gan_resume27k_bsod_lost.md) | BSOD (driver crash), checkpoint-on-resume shipped |
| 2026-01-26 | `multi_vits_gan_20260126_041738` | `numeric_instability` | [NaN Collapse](./2026-01-26_multi_vits_gan_20260126_041738_nan_collapse.md) | P0/P1 fixes shipped: emergency stop, escalation ladder |

## Rolling Log

See [gan_stability_log.md](./gan_stability_log.md) for cross-incident trends and mitigation tracking.

## File Naming Convention

```
YYYY-MM-DD_<run_id>_<short_slug>.md
```

Example: `2026-01-26_multi_vits_gan_20260126_041738_nan_collapse.md`

## Template

Each postmortem should include:

1. **Title + Run ID** — Clear identification
2. **Summary** — 2-3 sentence overview
3. **Impact** — What was lost (compute, time, data)
4. **Timeline** — Step-by-step breakdown
5. **Root Cause** — Technical analysis
6. **Detection/Response** — What worked, what failed
7. **Fixes Shipped** — Commits with SHAs
8. **Follow-ups** — P0/P1/P2 items
9. **Links** — Artifacts, dashboards, PRs
