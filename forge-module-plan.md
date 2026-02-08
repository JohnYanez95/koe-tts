# Forge Module Plan

## Overview

Build `modules/forge/` as an extraction-ready Python package providing reusable data platform utilities. This module is the client-side SDK for the North Star infrastructure (MinIO, HMS, MLflow, Vault).

**Goal**: Portable toolset across projects with security baked in.

**Python version floor**: 3.12+ required.

---

## Module Structure

```
modules/forge/
├── __init__.py                 # Package exports
├── sql/
│   ├── __init__.py
│   └── filters.py              # Parameterized SQL, injection prevention
├── storage/
│   ├── __init__.py
│   ├── protocols.py            # StorageBackend protocol + error types
│   └── s3.py                   # S3/MinIO client (implements StorageBackend)
├── archive/
│   ├── __init__.py
│   ├── safety.py               # Path validation, security checks (primary control)
│   ├── tar.py                  # Safe tar extraction (manual loop, no extractall)
│   └── zip.py                  # Safe zip extraction (manual loop, no extractall)
├── query/
│   ├── __init__.py
│   ├── duckdb.py               # DuckDB + Delta client (parameterized)
│   └── spark.py                # Spark session factory (single source of truth)
├── secrets/
│   ├── __init__.py
│   └── vault.py                # HashiCorp Vault client (with timeouts)
├── models/
│   ├── __init__.py
│   └── mlflow.py               # MLflow wrapper with forge:// URIs
└── cache/
    ├── __init__.py
    ├── manager.py              # Cache pull/push orchestration (atomic extraction)
    └── manifest.py             # Local cache state tracking
```

---

## Decisions (Playbook-Aligned)

Every decision below follows the secure-by-default playbook pattern: **safe default, explicit escape hatch, regression test**.

| Decision | Default | Escape Hatch | Test |
|---|---|---|---|
| Tar extraction method | Manual loop: `validate_archive_member()` → `tar.extractfile()` → stream-copy to fd we control. Stdlib `tarfile` filters (`filter='data'`) are **defense-in-depth only**, never the primary control. | None — this is the only extraction path | Programmatically generate malicious tar: traversal (`../`), absolute paths, oversized members, device/hardlink entries; assert rejection |
| ZIP extraction method | Manual loop: validate each member → stream-copy. No `extractall()`. Check Unix symlink bits in external attributes. | None | Zip Slip traversal, oversized members, symlink-via-external-attributes; assert rejection |
| Symlinks & hardlinks | Rejected by default (`allow_symlinks=False`, `allow_hardlinks=False` on handlers) | Opt-in via handler constructor flags | Archives containing symlink/hardlink escaping extraction root are rejected unless explicitly enabled |
| Path validation extras | Reject `\` and `:` in member names (Windows-ish paths). Normalize Unicode (NFC) before validation. | None — these are cheap 1-line checks | Member names with backslashes, drive letters, and NFC/NFKC-collapsible sequences; assert rejection |
| Cache extraction | Atomic: extract to `.tmp/<uuid>` under `cache_root`, validate, then `os.rename()` to final path | None | Simulate failure mid-extraction; assert no partial final cache directory exists |
| Cache manifest | Record archive SHA-256, extracted file list + sizes, timestamp, forge version | None | Pull same dataset twice; verify manifest consistency and `status()` uses manifest |
| Spark factory/config | `get_spark()` is the **only** factory. Owns Delta + `spark_catalog` wiring, `spark.jars.packages` concatenation, `spark.extraListeners`, and any OpenLineage hooks. | None | Assert required Spark conf keys are present (Delta on `spark_catalog`, single jars string, listeners set once) |
| Lake URI schemes | Spark requires `s3a://…`; DuckDB requires `s3://…`. **Fail closed** with `ValueError` and actionable message naming the correct env var. No auto-conversion. | None | Pass wrong scheme to each engine; assert `ValueError` |
| DuckDB raw SQL | `query_raw(sql)` is the explicitly unvalidated path. Structured methods (`query_table`, `scan_delta`) are preferred and parameterized. | `query_raw` exists but is clearly labeled in name + docstring | Unit tests cover structured paths; `query_raw` docstring warns about injection risk |
| DuckDB credentials | Set at connection time only. Never baked into persistent `.duckdb` files. | None | N/A (design constraint) |
| Extraction limits | Defaults: **10 GB total**, **500 MB per file**, **100k files**, **1024 char path length** | Explicit `ExtractionLimits(...)` constructor override for known-large corpora | Archive declares 1 TB or exceeds file count; assert rejection |
| External-call timeouts | Vault: `connect=5s, read=30s`. S3: botocore `connect_timeout=5s, read_timeout=30s` + max attempts. Env-overridable. | Env vars (`FORGE_VAULT_CONNECT_TIMEOUT`, `FORGE_S3_READ_TIMEOUT`, etc.) | Mock slow/unreachable Vault/MinIO; assert timeout fires cleanly |
| StorageBackend protocol | Minimal, disk-first: `put()`, `get_to_path()`, `exists()`, `list()`. Raise `NotFoundError` / `StorageError`. | Add `stat()` later when manifest comparison needs it | MinIO round-trip: `put` then `get_to_path`; verify checksum. Missing key returns `NotFoundError`. |

---

## Build Order & Dependencies

```
Phase 1: Foundation (no external deps beyond stdlib)
├── forge.sql.filters           ← COPY from filters.py (zero changes)
└── forge.archive.safety        ← NEW (primary security control)

Phase 2: Archive Handlers (depends on Phase 1)
├── forge.archive.tar           ← NEW (manual extraction loop using safety.py)
└── forge.archive.zip           ← NEW (manual extraction loop using safety.py)

Phase 3: Storage (depends on Phase 1)
├── forge.storage.protocols     ← NEW (StorageBackend protocol + error types)
└── forge.storage.s3            ← COPY+MODIFY from s3.py (parameterize domain, add timeouts)

Phase 4: Query Layer (depends on Phases 1, 3)
├── forge.query.spark           ← COPY+HARDEN from spark.py (URI validation, config assertions)
└── forge.query.duckdb          ← REFACTOR from duckdb_client.py (remove paths dep, add query_raw)

Phase 5: Integration (depends on Phases 1–4)
├── forge.secrets.vault         ← NEW (with timeouts, no shell=True)
├── forge.models.mlflow         ← NEW
└── forge.cache.manager         ← NEW (atomic extraction + manifest)
```

---

## Phase 1: Foundation

### 1.1 forge/sql/filters.py

**Source**: `modules/data_engineering/common/filters.py` (493 lines)
**Action**: Copy as-is (zero modifications needed)
**Tests**: Copy `tests/test_filters.py` (65+ tests)

**Public API**:
```python
class FilterParseError(ValueError)
def validate_ident(name, kind, allowed) -> str
def quote_ident(name) -> str                    # DuckDB double-quote
def quote_spark_ident(name) -> str              # Spark backtick-quote
def safe_sql_string(value) -> str               # Escape for literals
def parse_filter(expr, allowed_cols, schema_map) -> ParsedFilter
def build_where(filters, any_filters, ...) -> (str, list)
def parse_columns(columns, schema_map, allowed_cols) -> str
def schema_map_from_columns(columns) -> dict
# Constants: MAX_IN_LIST, MAX_FILTERS, MAX_FILTER_LENGTH, MAX_COLUMNS
```

### 1.2 forge/archive/safety.py

**Source**: NEW
**Critical**: Primary security control for corpus ingestion

**Security stance**: We do **not** trust stdlib tar filters as the primary control. `tarfile` filters (`filter='data'`) are applied as defense-in-depth only. The primary control is our manual extraction loop: validate every member → open destination fd ourselves → stream-copy with size enforcement. This protects us even if Python has a filter bypass bug (see CVE-2025-4138).

**Default policy**: Reject all symlinks and hardlinks unless explicitly opted in.

**Security checks**:

| Check | Threat | Implementation |
|---|---|---|
| Path traversal | `../` sequences | `os.path.normpath()` + containment check via `is_relative_to()` |
| Absolute paths | `/etc/passwd` | Reject leading `/` |
| Symlink escape | Links outside extraction dir | **Reject by default**. If opted in: resolve + check `is_relative_to()` |
| Hardlink attack | Hard links bypass checks | **Reject by default**. If opted in: validate target is within root |
| Device files | `/dev/zero` DoS | Reject `tarfile.CHRTYPE`, `tarfile.BLKTYPE` |
| Size bombs | Extraction bombs | Cumulative + per-file limits, enforced during stream-copy |
| Path length | Symlink expansion overflow | Limit to 1024 chars |
| Permissions | Metadata attacks | Reset to `0o755` (dirs) / `0o644` (files) |
| Windows paths | `C:\...`, `..\` | Reject `\` and `:` in member names |
| Unicode traps | NFC/NFKC collisions | `unicodedata.normalize('NFC', name)` before validation |

**Public API**:
```python
class ExtractionError(Exception):
    """Raised when archive member fails security validation."""

@dataclass
class ExtractionLimits:
    max_file_size: int = 500_000_000        # 500 MB per file
    max_total_size: int = 10_000_000_000    # 10 GB total
    max_files: int = 100_000
    max_path_length: int = 1024
    allowed_extensions: set[str] | None = None  # None = allow all

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
) -> Path:
    """Validate member, return safe extraction path or raise ExtractionError.

    This is the gatekeeper function called for every archive member before
    any bytes are written to disk. It enforces all checks in the security
    table above.
    """

def is_path_safe(path: str, root: Path) -> bool:
    """Check if path stays within root after resolution."""
```

---

## Phase 2: Archive Handlers

### 2.1 forge/archive/tar.py

**Extraction method**: Manual loop. We never call `extractall()`. For each member:
1. Call `validate_archive_member()` — reject or get safe destination path
2. For regular files: `tar.extractfile(member)` → stream-copy to a new fd we open ourselves
3. For directories: `os.makedirs()` with controlled permissions
4. Apply `filter='data'` as defense-in-depth (not the primary control)
5. Enforce per-file size cap during stream-copy (read in chunks, track bytes)

**Public API**:
```python
class TarHandler:
    def __init__(
        self,
        limits: ExtractionLimits | None = None,
        *,
        allow_symlinks: bool = False,
        allow_hardlinks: bool = False,
    ):
        ...

    def extract(self, archive: Path, dest: Path) -> list[Path]:
        """Safely extract tar archive, return list of extracted paths."""

    def list_members(self, archive: Path) -> list[ArchiveMember]:
        """List contents without extracting (for preview/validation)."""

    def create(self, source: Path, archive: Path, compression: str = 'gz') -> None:
        """Create tar archive from directory."""

@dataclass
class ArchiveMember:
    name: str
    size: int
    is_file: bool
    is_dir: bool
    is_symlink: bool
    link_target: str | None = None
```

### 2.2 forge/archive/zip.py

**Same extraction method as tar**: manual loop, no `extractall()`. ZIP can carry Unix symlink bits in external attributes — treat these as "link" type and reject by default.

**Public API**: Same shape as `TarHandler` but for ZIP files (`ZipHandler`).

**Note**: Current koe-tts uses `zipfile.extractall()` without safety checks in:
- `modules/data_engineering/ingest/jsut/extract.py:70`
- `modules/data_engineering/ingest/jvs/extract.py:90`

These must be updated to use `forge.archive.zip.ZipHandler`.

---

## Phase 3: Storage

### 3.1 forge/storage/protocols.py

**Source**: NEW
**Purpose**: Load-bearing interface — `CacheManager` depends on this.

```python
from typing import Protocol, Iterator
from pathlib import Path

class StorageError(Exception):
    """Base error for storage operations."""

class NotFoundError(StorageError):
    """Raised when a requested key does not exist."""

class StorageBackend(Protocol):
    """Minimal, disk-first storage protocol.

    Designed for large objects (multi-GB corpus tarballs). All reads
    go to disk, not memory. Add stat() when manifest comparison needs it.
    """

    def put(self, key: str, source: Path) -> str:
        """Upload file to storage. Returns the storage key."""
        ...

    def get_to_path(self, key: str, dest: Path) -> Path:
        """Download object directly to disk. Returns dest path.

        Raises NotFoundError if key does not exist.
        """
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists in storage."""
        ...

    def list(self, prefix: str) -> Iterator[str]:
        """List keys matching prefix."""
        ...
```

### 3.2 forge/storage/s3.py

**Source**: `modules/data_engineering/common/s3.py` (425 lines)
**Action**: Copy with modifications

**Changes from original**:
1. Make `build_raw_zone_prefix()` fully parameterized (remove default `domain="koe"`)
2. Add `S3StorageBackend` class implementing `StorageBackend` protocol
3. Add botocore timeout configuration: `connect_timeout`, `read_timeout`, `max_attempts` (env-overridable)
4. `get_to_path()` uses boto3 `download_file()` for multipart downloads + automatic retries

**Public API**:
```python
class S3ConfigError(Exception)
class S3UploadError(Exception)

# Timeout defaults (env-overridable)
S3_CONNECT_TIMEOUT = int(os.getenv("FORGE_S3_CONNECT_TIMEOUT", "5"))
S3_READ_TIMEOUT = int(os.getenv("FORGE_S3_READ_TIMEOUT", "30"))
S3_MAX_ATTEMPTS = int(os.getenv("FORGE_S3_MAX_ATTEMPTS", "3"))

def validate_path_component(value, name) -> str
def get_s3_config() -> dict | None
def get_s3_client() -> S3Client | None
def is_s3_available() -> bool
def ensure_bucket(bucket, client) -> bool
def upload_file(local_path, bucket, key, ...) -> dict
def upload_directory(local_path, bucket, s3_prefix, ...) -> list[dict]
def collect_upload_files(local_path, file_filter, max_files) -> list[Path]

class S3StorageBackend:
    """StorageBackend implementation for S3/MinIO."""
    def __init__(self, bucket: str, prefix: str = "")
    def put(self, key: str, source: Path) -> str
    def get_to_path(self, key: str, dest: Path) -> Path
    def exists(self, key: str) -> bool
    def list(self, prefix: str) -> Iterator[str]
```

---

## Phase 4: Query Layer

### 4.1 forge/query/spark.py

**Source**: `modules/data_engineering/common/spark.py` (151 lines)
**Action**: Copy + harden

**Changes from original**:
1. Validate `FORGE_LAKE_ROOT_S3A` starts with `s3a://` — fail closed with `ValueError` and actionable message if wrong scheme is passed
2. `get_spark()` remains the **only** public factory
3. Delta + `spark_catalog` config is hard-coded as single source of truth
4. `spark.jars.packages` built from one concatenated string
5. `spark.extraListeners` set only here, never elsewhere

**Public API**:
```python
def get_spark(app_name: str = "forge") -> SparkSession:
    """Single source of truth for Spark configuration.

    Owns: Delta + spark_catalog, S3/MinIO access, OpenLineage listener,
    and all spark.jars.packages in one concatenated string.

    Validates FORGE_LAKE_ROOT_S3A starts with s3a://.
    Raises ValueError if URI scheme is wrong.
    """

def stop_spark() -> None

# Reads from env vars:
# - METASTORE_URI (default: thrift://localhost:9083)
# - MINIO_ENDPOINT (default: http://localhost:9000)
# - MINIO_ROOT_USER, MINIO_ROOT_PASSWORD
# - FORGE_LAKE_ROOT_S3A (default: s3a://forge/lake)
# - OPENLINEAGE_URL (default: http://localhost:5000)
```

### 4.2 forge/query/duckdb.py

**Source**: `modules/data_engineering/common/duckdb_client.py` (251 lines)
**Action**: Refactor to remove `paths.py` dependency, rename `query()` → `query_raw()`

**Changes from original**:
1. Remove `from .paths import paths` dependency — accept `lake_root` as constructor arg
2. Rename `query(sql)` → `query_raw(sql)` with docstring warning about injection risk
3. Validate `lake_root` starts with `s3://` (for S3 clients) — fail closed if wrong scheme
4. Credentials set at connection time only, never persisted

**Public API**:
```python
class DuckDBClient:
    """DuckDB client with parameterized lake root."""

    def __init__(self, lake_root: Path | str):
        self.lake_root = Path(lake_root)
        self._conn = None

    def get_connection(self) -> duckdb.DuckDBPyConnection
    def query_raw(self, sql: str) -> pd.DataFrame:
        """Execute raw SQL. No validation or parameterization.

        WARNING: This method does not validate or sanitize input.
        Prefer query_table() or scan_delta() for safe, structured access.
        """
    def query_table(self, layer, dataset, table, ...) -> pd.DataFrame
    def scan_delta(self, path, columns, limit) -> pd.DataFrame
    def list_tables(self) -> list[dict]
    def table_info(self, layer, dataset, table) -> pd.DataFrame

# Factory functions
def create_duckdb_client(lake_root: Path | str) -> DuckDBClient

def create_s3_duckdb_client(
    lake_root: str,  # Must start with s3://
    endpoint: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
) -> DuckDBClient:
    """Create DuckDB client for S3-backed lake.

    Validates lake_root starts with s3://. Raises ValueError if
    s3a:// is passed (that's for Spark — use FORGE_LAKE_ROOT_S3).
    """
```

**koe-tts integration** (after refactor):
```python
from modules.forge.query.duckdb import create_duckdb_client
from modules.data_engineering.common.paths import paths

# Bind to repo's lake root
client = create_duckdb_client(paths.lake)
df = client.query_table("silver", "jsut", "utterances")
```

---

## Phase 5: Integration

### 5.1 forge/secrets/vault.py

**Source**: NEW
**Critical**: No `shell=True`, no `subprocess`. Pure HTTP client with timeouts.

```python
class VaultClient:
    def __init__(
        self,
        addr: str | None = None,
        token: str | None = None,
        timeout: tuple[int, int] = (
            int(os.getenv("FORGE_VAULT_CONNECT_TIMEOUT", "5")),
            int(os.getenv("FORGE_VAULT_READ_TIMEOUT", "30")),
        ),
    ):
        self.addr = addr or os.getenv("VAULT_ADDR", "http://localhost:8200")
        self.token = token or os.getenv("VAULT_TOKEN")
        self.timeout = timeout  # (connect, read)

    def get_secret(self, path: str) -> dict
    def get_field(self, path: str, field: str) -> str:
        """Get a single field from a Vault secret.

        Designed to power a future `forge bootstrap` safely:
        pure Python, no eval, no shell=True, no subprocess.
        """
    def is_available(self) -> bool

    @classmethod
    def from_env(cls) -> "VaultClient"
```

### 5.2 forge/models/mlflow.py

```python
class ModelRegistry:
    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5002"
        )

    def load_model(self, uri: str):
        """Load model with forge:// URI support.

        forge://vits@prod   → models:/vits/Production
        forge://vits@v3     → models:/vits/3
        models:/...         → passthrough
        """

    def register(self, model, name: str, stage: str = "None")
    def list_models(self) -> list[dict]

    @classmethod
    def from_env(cls) -> "ModelRegistry"
```

### 5.3 forge/cache/manager.py

```python
class CacheManager:
    """Atomic tarball → local cache flow.

    Extraction is atomic: extract to .tmp/<uuid>, validate, then
    os.rename() to final path. If anything fails, no partial cache
    directory exists at the final path.
    """

    def __init__(
        self,
        storage: StorageBackend,
        archive_handler: TarHandler | ZipHandler,
        cache_root: Path,
    ):
        ...

    def pull(self, dataset: str, version: str = "latest") -> Path:
        """Download + extract to cache atomically, return cache path.

        Flow:
        1. storage.get_to_path() → temp archive file
        2. Extract to .tmp/<uuid> under cache_root
        3. Validate extraction (file count, sizes)
        4. Write manifest (SHA-256, file list, sizes, timestamp, forge version)
        5. os.rename() to final path
        """

    def status(self, dataset: str) -> dict:
        """What's cached locally. Uses manifest for checksums and freshness."""

    def invalidate(self, dataset: str) -> None:
        """Remove from local cache."""

    def list_cached(self) -> list[str]:
        """List all cached datasets."""
```

### 5.4 forge/cache/manifest.py

```python
@dataclass
class CacheManifest:
    archive_sha256: str
    files: list[ManifestEntry]
    total_size: int
    extracted_at: datetime
    forge_version: str

@dataclass
class ManifestEntry:
    path: str
    size: int
```

---

## Migration Path for koe-tts

Clean break, no shims. Update all imports at once.

**Before** (scattered imports):
```python
from modules.data_engineering.common.filters import build_where
from modules.data_engineering.common.s3 import get_s3_client
from modules.data_engineering.common.duckdb_client import query_table
from modules.data_engineering.common.spark import get_spark
```

**After** (forge imports):
```python
from modules.forge.sql.filters import build_where
from modules.forge.storage.s3 import get_s3_client
from modules.forge.query.duckdb import create_duckdb_client
from modules.forge.query.spark import get_spark
```

**Migration grep** (includes tests/):
```bash
grep -r "from modules.data_engineering.common.filters" modules/ tests/
grep -r "from modules.data_engineering.common.s3" modules/ tests/
grep -r "from modules.data_engineering.common.duckdb_client" modules/ tests/
grep -r "from modules.data_engineering.common.spark" modules/ tests/
grep -r "modules.data_engineering.common" tests/
```

**Files to delete** (after migration complete):
- `modules/data_engineering/common/filters.py` → moved to forge
- `modules/data_engineering/common/s3.py` → moved to forge
- `modules/data_engineering/common/spark.py` → moved to forge
- `modules/data_engineering/common/duckdb_client.py` → moved to forge

**Unsafe extraction sites to update**:
- `modules/data_engineering/ingest/jsut/extract.py:70` → use `forge.archive.zip.ZipHandler`
- `modules/data_engineering/ingest/jvs/extract.py:90` → use `forge.archive.zip.ZipHandler`

---

## Verification Plan

### Unit Tests

1. `tests/forge/test_filters.py` — Copy existing 65+ tests, update import paths
2. `tests/forge/test_archive_safety.py` — Test with programmatically generated malicious archives:
   - Path traversal (`../../../etc/passwd`)
   - Absolute paths (`/etc/passwd`)
   - Symlink escape (`evil -> /etc/passwd`)
   - Hardlink attack
   - Size bomb (1 TB declared size)
   - Device file creation
   - Windows-style paths (`C:\Windows\System32`)
   - Backslash traversal (`..\..\..\etc\passwd`)
   - Unicode normalization bypass attempts
3. `tests/forge/test_archive_tar.py` — Manual extraction loop produces correct output; malicious members rejected
4. `tests/forge/test_archive_zip.py` — Same coverage as tar; ZIP symlink-via-external-attributes rejected
5. `tests/forge/test_duckdb.py` — Parameterized client; `query_table` uses safe path; `query_raw` exists but is unvalidated
6. `tests/forge/test_spark_config.py` — Assert `spark_catalog`, Delta config, jars concatenation, listeners set once
7. `tests/forge/test_uri_schemes.py` — Wrong URI scheme to Spark/DuckDB raises `ValueError`
8. `tests/forge/test_timeouts.py` — Mock slow Vault/S3; assert timeout fires
9. `tests/forge/test_safe_defaults.py` — Prove unsafe behavior (symlinks, raw SQL) requires explicit opt-in

### Integration Tests

1. Archive extraction: Create test tarball with edge cases, verify safe extraction
2. S3 round-trip: `put` + `get_to_path`, verify checksum and `NotFoundError` on missing key
3. DuckDB + Delta: Create Delta table, query via DuckDB client
4. Spark + HMS: Register table in HMS, resolve via Spark
5. Cache atomic extraction: Kill mid-extraction, verify no partial cache at final path

### End-to-End

```python
python -c "
from modules.forge.sql.filters import build_where
from modules.forge.archive.tar import TarHandler
from modules.forge.query.spark import get_spark

# Verify imports work
print('Forge module loaded successfully')
"
```

---

## Files to Create

| File | Lines (est.) | Priority |
|---|---|---|
| `modules/forge/__init__.py` | 50 | P1 |
| `modules/forge/sql/__init__.py` | 10 | P1 |
| `modules/forge/sql/filters.py` | 493 (copy) | P1 |
| `modules/forge/archive/__init__.py` | 10 | P1 |
| `modules/forge/archive/safety.py` | 250 | P1 |
| `modules/forge/archive/tar.py` | 180 | P2 |
| `modules/forge/archive/zip.py` | 130 | P2 |
| `modules/forge/storage/__init__.py` | 10 | P3 |
| `modules/forge/storage/protocols.py` | 60 | P3 |
| `modules/forge/storage/s3.py` | 480 (copy+modify) | P3 |
| `modules/forge/query/__init__.py` | 10 | P4 |
| `modules/forge/query/spark.py` | 170 (copy+harden) | P4 |
| `modules/forge/query/duckdb.py` | 320 (refactor) | P4 |
| `modules/forge/secrets/__init__.py` | 10 | P5 |
| `modules/forge/secrets/vault.py` | 100 | P5 |
| `modules/forge/models/__init__.py` | 10 | P5 |
| `modules/forge/models/mlflow.py` | 100 | P5 |
| `modules/forge/cache/__init__.py` | 10 | P5 |
| `modules/forge/cache/manager.py` | 180 | P5 |
| `modules/forge/cache/manifest.py` | 60 | P5 |
| `tests/forge/test_filters.py` | 396 (copy) | P1 |
| `tests/forge/test_archive_safety.py` | 250 | P1 |
| `tests/forge/test_archive_tar.py` | 150 | P2 |
| `tests/forge/test_archive_zip.py` | 120 | P2 |
| `tests/forge/test_spark_config.py` | 60 | P4 |
| `tests/forge/test_uri_schemes.py` | 40 | P4 |
| `tests/forge/test_safe_defaults.py` | 80 | P1 |

**Total**: ~3,400 lines (including copies and tests)
