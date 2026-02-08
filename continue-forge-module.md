# Continuation: Forge Module Build

**Primary plan:** `forge-module-plan.md` (project root — 655 lines, read first)
**Claude plan:** `~/.claude/plans/sparkling-floating-fern.md`
**Branch:** `project/refactor`
**Date:** 2026-02-06
**Status:** Phase 1 not started — directory structure created, no code written yet

---

## Parallel Workstreams

| Workstream | File | Focus | Status |
|---|---|---|---|
| **Repo Restructuring** | `CONTINUATION.md` | Documentation, presentation, algorithm docs | ~40% |
| **This doc** — Forge Module | `continue-forge-module.md` | Infrastructure code (`modules/forge/`) | Phase 1 not started |

**Touch points (handle when you get there):**
- Repo restructuring may update `docs/architecture/README.md` to reference forge
- README.md repo map may include `modules/forge/` if code exists by then
- Weekly reports may cover forge progress

These are not blockers — the two workstreams can proceed independently.

---

## What's Done

1. **Plan finalized** — Full specification in `forge-module-plan.md` (root)
2. **Directory structure created** — `modules/forge/{sql,archive,storage,query,secrets,models,cache}/`
3. **Documentation routed** — User has already routed files to proper locations

---

## What's Next: Phase 1 Foundation

### 1.1 Copy filters.py (Zero Changes)

```bash
# Source file is already read and ready to copy
cp modules/data_engineering/common/filters.py modules/forge/sql/filters.py
```

Or write it directly — the content is 494 lines, fully functional, no modifications needed.

### 1.2 Create archive/safety.py (NEW - Critical Security Control)

This is the gatekeeper function for all archive extraction. Must implement:

**Security checks (all required)**:
- Path traversal (`../`) — `os.path.normpath()` + `is_relative_to()`
- Absolute paths — Reject leading `/`
- Symlinks — **Reject by default** (`allow_symlinks=False`)
- Hardlinks — **Reject by default** (`allow_hardlinks=False`)
- Device files — Reject `tarfile.CHRTYPE`, `tarfile.BLKTYPE`
- Size bombs — Cumulative + per-file limits during stream-copy
- Path length — Limit 1024 chars
- Permissions — Reset to `0o755` (dirs) / `0o644` (files)
- Windows paths — Reject `\` and `:` in member names
- Unicode — `unicodedata.normalize('NFC', name)` before validation

**Public API**:
```python
class ExtractionError(Exception): ...

@dataclass
class ExtractionLimits:
    max_file_size: int = 500_000_000        # 500 MB per file
    max_total_size: int = 10_000_000_000    # 10 GB total
    max_files: int = 100_000
    max_path_length: int = 1024
    allowed_extensions: set[str] | None = None

def validate_archive_member(
    name: str,
    size: int,
    member_type: str,  # 'file', 'dir', 'symlink', 'hardlink', 'device'
    link_target: str | None,
    extraction_root: Path,
    limits: ExtractionLimits,
    *,
    allow_symlinks: bool = False,
    allow_hardlinks: bool = False,
) -> Path: ...

def is_path_safe(path: str, root: Path) -> bool: ...
```

### 1.3 Create `__init__.py` files

Each submodule needs an `__init__.py` that exports the public API.

---

## Build Order (Remaining)

```
Phase 1: Foundation (CURRENT)
├── forge.sql.filters           ← COPY from filters.py
└── forge.archive.safety        ← NEW

Phase 2: Archive Handlers
├── forge.archive.tar           ← NEW (manual loop, no extractall)
└── forge.archive.zip           ← NEW (manual loop, no extractall)

Phase 3: Storage
├── forge.storage.protocols     ← NEW (StorageBackend protocol)
└── forge.storage.s3            ← COPY+MODIFY from s3.py

Phase 4: Query Layer
├── forge.query.spark           ← COPY+HARDEN from spark.py
└── forge.query.duckdb          ← REFACTOR from duckdb_client.py

Phase 5: Integration
├── forge.secrets.vault         ← NEW (pure HTTP, no subprocess)
├── forge.models.mlflow         ← NEW
└── forge.cache.manager         ← NEW (atomic extraction)
```

---

## Key Source Files

| Component | Source | Action |
|-----------|--------|--------|
| `forge/sql/filters.py` | `modules/data_engineering/common/filters.py` | Copy as-is |
| `forge/storage/s3.py` | `modules/data_engineering/common/s3.py` | Copy + add timeouts, S3StorageBackend |
| `forge/query/spark.py` | `modules/data_engineering/common/spark.py` | Copy + URI validation |
| `forge/query/duckdb.py` | `modules/data_engineering/common/duckdb_client.py` | Refactor (remove paths dep, rename query→query_raw) |

---

## Security Playbook Alignment

Every component follows: **safe default → explicit escape hatch → regression test**

- Archive handlers: symlinks/hardlinks rejected by default
- DuckDB: `query_raw()` is explicit escape hatch (structured methods preferred)
- URI schemes: fail closed with `ValueError` if wrong (no auto-conversion)
- Timeouts: env-overridable defaults on all external calls

---

## Test Files to Create

1. `tests/forge/test_filters.py` — Copy from `tests/test_filters.py`
2. `tests/forge/test_archive_safety.py` — Programmatically generate malicious archives
3. `tests/forge/test_safe_defaults.py` — Prove unsafe behavior requires explicit opt-in

---

## Reference Documents

### Primary Plan (READ FIRST)
- **`forge-module-plan.md`** — Full specification (655 lines) at project root
  - Module structure, public APIs, security checks table
  - Build order with dependencies
  - Migration path for koe-tts
  - Verification plan with test file list

### Security Playbook
- **`~/.claude/projects/-home-john-Repos-koe-tts/memory/secure-by-default-playbook.md`** — Security patterns
  - Section 7: Archive Extraction Safety (critical for Phase 1-2)
  - Section 8: Shell, Eval, Subprocess Safety (critical for Phase 5 vault.py)
  - Quick Ship Checklist at the end

### Infrastructure Context
- **`/home/john/Repos/forge-infra/ROADMAP.md`** — Infrastructure roadmap
  - What's running (MinIO, Vault, HMS, MLflow)
  - What's blocked (needs forge module)
  - Playbook-aligned decisions table

- **`/home/john/Repos/forge-infra/ARCHITECTURE.md`** — North Star v5.2 architecture
  - Storage layer design
  - Catalog integration patterns
  - Bootstrap flow

### Claude Session Plan
- **`~/.claude/plans/sparkling-floating-fern.md`** — Claude plan file (mirrors forge-module-plan.md)

---

## Source Files to Copy/Modify

These existing files will be copied or refactored into forge:

| Destination | Source | Lines | Notes |
|-------------|--------|-------|-------|
| `modules/forge/sql/filters.py` | `modules/data_engineering/common/filters.py` | 494 | Copy as-is |
| `modules/forge/storage/s3.py` | `modules/data_engineering/common/s3.py` | 425 | Add S3StorageBackend, timeouts |
| `modules/forge/query/spark.py` | `modules/data_engineering/common/spark.py` | 151 | Add URI scheme validation |
| `modules/forge/query/duckdb.py` | `modules/data_engineering/common/duckdb_client.py` | 251 | Remove paths dep, rename query→query_raw |

Tests to copy:
| Destination | Source | Lines |
|-------------|--------|-------|
| `tests/forge/test_filters.py` | `tests/test_filters.py` | 396 |

---

## Quick Start Commands

```bash
# Verify directory structure exists
ls -la modules/forge/

# Copy filters.py as first step
cp modules/data_engineering/common/filters.py modules/forge/sql/filters.py

# After building, verify imports work
python -c "from modules.forge.sql.filters import build_where; print('OK')"
```

---

## Notes

- Python version floor: 3.12+ (for tarfile data filter as defense-in-depth)
- Clean break migration: no backward-compat shims
- All credentials set at connection time, never persisted
