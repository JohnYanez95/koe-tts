# DuckDB vs Unity Catalog: Evaluation for koe-tts

## Current State

**Stack**: PySpark 4.x + Delta Lake 4.x + delta-rs (deltalake) for writes
**Pain Points**:
- Spark JVM startup: ~5-8s cold start for simple queries
- Heavy memory footprint for ad-hoc analysis
- No unified query interface for traceability/exploration
- Catalog is a simple Delta table (`lake/_catalog/tables`) - no lineage, no governance

**What works well**:
- Delta Lake ACID guarantees for writes
- Time travel for debugging pipeline issues
- Schema enforcement on medallion layers

---

## Option A: DuckDB (Query Layer)

### Architecture

```
Write Path (unchanged):
  PySpark + delta-spark → Delta tables (Bronze/Silver/Gold)

Read/Analysis Path (new):
  DuckDB + delta extension → query Delta tables directly
  └── Labeler data.py: replace JSONL reads with SQL
  └── Ad-hoc analysis: CLI or Python
  └── Dashboard metrics: fast aggregations
```

### Pros

| Benefit | Impact |
|---------|--------|
| **Zero JVM** | Instant startup (~100ms vs 5-8s), no Java dependency for reads |
| **Native Delta reads** | [delta extension](https://duckdb.org/docs/stable/core_extensions/delta) uses delta-kernel-rs, reads Parquet + log natively |
| **Arrow interop** | Direct handoff to Pandas/Polars without serialization |
| **SQL-first exploration** | Ad-hoc queries on any layer without Spark notebooks |
| **Already compatible** | delta-rs (deltalake) already in deps, Arrow ecosystem aligned |
| **Low integration cost** | Add `duckdb` dep, write thin wrappers, incremental adoption |
| **Performance** | [March 2025 benchmarks](https://duckdb.org/2025/03/21/maximizing-your-delta-scan-performance) show competitive TPC-H performance |

### Cons

| Limitation | Mitigation |
|------------|------------|
| **Read-only for Delta** | Keep PySpark/delta-rs for writes (already works) |
| **No write support** | DuckDB roadmap includes it; workaround via delta-rs `write_deltalake()` |
| **Single-node only** | Fine for our scale (<100k utterances) |
| **Metadata freshness** | Delta extension caches metadata; may need explicit refresh after writes |

### Integration Points

1. **Labeler data layer** (`modules/labeler/app/data.py`):
   ```python
   # Replace: load_manifest() reading JSONL line-by-line
   # With: DuckDB query on gold utterances Delta table
   import duckdb
   conn = duckdb.connect()
   df = conn.execute("""
       SELECT utterance_id, text, phonemes, audio_relpath, duration_sec, speaker_id
       FROM delta_scan('lake/gold/jsut/utterances')
       WHERE split = 'train'
   """).fetchdf()
   ```

2. **Label coverage dashboard**:
   ```sql
   SELECT g.split, COUNT(*) as total,
          COUNT(l.utterance_id) as labeled
   FROM delta_scan('lake/gold/jsut/utterances') g
   LEFT JOIN read_json_auto('runs/labeling/published/jsut/labels.jsonl') l
       USING (utterance_id)
   GROUP BY g.split
   ```

3. **Optimizer analysis** (heuristic.py): fast aggregations on segment_breaks

4. **CLI exploration**: `duckdb` shell for ad-hoc queries

### Migration Path

```
Phase 1: Add duckdb dependency, use for read-only analysis (no changes to writes)
Phase 2: Replace labeler load_manifest() / load_auto_breakpoints() with SQL
Phase 3: Add DuckDB CLI commands to koe for exploration (koe query "SELECT ...")
Phase 4: Evaluate replacing PySpark reads with DuckDB where beneficial
```

---

## Option B: Unity Catalog OSS

### Architecture

```
┌─────────────────────────────────────┐
│     Unity Catalog Server (Java)     │
│  - Metadata store (PostgreSQL/MySQL)│
│  - REST API for catalog operations  │
│  - Lineage, access control          │
└──────────────┬──────────────────────┘
               │ REST API
    ┌──────────┴──────────┐
    │                     │
  Spark              DuckDB/CLI
  (writes)           (reads)
    │                     │
    └──────────┬──────────┘
               ▼
         Delta Tables
```

### Pros

| Benefit | Impact |
|---------|--------|
| **Industry standard** | [Open sourced by Databricks](https://www.databricks.com/blog/open-sourcing-unity-catalog), LF AI Foundation project |
| **Unified governance** | Single source of truth for catalog, lineage, access control |
| **Multi-format** | Supports Delta, Iceberg, Hudi, Parquet via UniForm |
| **Multi-engine** | [DuckDB, Spark, Trino, etc.](https://www.unitycatalog.io/) can all query same catalog |
| **AI asset support** | Can catalog models, functions, not just tables |
| **Future-proof** | If we scale to multi-user or cloud, governance is built-in |

### Cons

| Limitation | Severity |
|------------|----------|
| **Java 17 required** | Adds JVM dependency for catalog server |
| **Server process** | Must run continuously (or Docker container) |
| **Backend DB** | [Requires PostgreSQL or MySQL](https://docs.unitycatalog.io/server/configuration/) for metadata |
| **Operational overhead** | More infrastructure to manage vs file-based catalog |
| **Overkill for single-user** | Governance features not needed for solo/small team |
| **Migration effort** | Must register all existing tables, update all readers |
| **Complexity** | sbt build, Java ecosystem, more moving parts |

### When Unity Catalog Makes Sense

- Multi-user data platform with access control needs
- Lineage tracking required for compliance/auditing
- Multiple compute engines need unified view
- Planning to deploy to cloud (Databricks, AWS, Azure)
- AI model registry needed alongside data catalog

### When Unity Catalog is Overkill

- Single developer / small team
- Local-first development
- No compliance requirements
- Simple medallion architecture with clear lineage in code
- Don't need model registry (using MLflow separately)

---

## Recommendation

### For koe-tts: **DuckDB (Option A)**

**Rationale**:

1. **Immediate value**: 5-8s Spark startup → 100ms DuckDB for reads
2. **Zero new infrastructure**: No server, no database, no Java beyond what Spark already needs
3. **Incremental adoption**: Start with labeler reads, expand as needed
4. **Already aligned**: delta-rs + Arrow already in stack
5. **Right-sized**: Single-user, local-first, <100k utterances

**Unity Catalog is premature** for current scale. If we later need:
- Multi-user governance → revisit
- Cloud deployment → consider managed Databricks Unity Catalog
- Cross-engine lineage → evaluate then

### Implementation Plan

```
Week 1: Add DuckDB
├── Add duckdb to pyproject.toml
├── Create modules/data_engineering/common/duckdb.py (thin wrapper)
├── Add koe query CLI command for ad-hoc exploration
└── Verify Delta reads work on all layers

Week 2: Labeler Integration
├── Replace load_manifest() with DuckDB query
├── Replace load_auto_breakpoints() with SQL join
├── Add label coverage metrics via SQL
└── Profile performance vs current JSONL reads

Week 3: Optimizer Integration
├── Use DuckDB for loading published labels
├── SQL aggregations for heuristic metrics
└── Document query patterns

Future: Evaluate expanding to training pipeline reads
```

---

## Appendix: Key Links

### DuckDB + Delta
- [DuckDB Delta Extension Docs](https://duckdb.org/docs/stable/core_extensions/delta)
- [Native Delta Lake Support (2024 announcement)](https://duckdb.org/2024/06/10/delta)
- [Delta Scan Performance (March 2025)](https://duckdb.org/2025/03/21/maximizing-your-delta-scan-performance)
- [DuckDB Roadmap](https://duckdb.org/roadmap)
- [duckdb-delta GitHub](https://github.com/duckdb/duckdb-delta)

### Unity Catalog OSS
- [Unity Catalog Docs](https://docs.unitycatalog.io/)
- [GitHub: unitycatalog/unitycatalog](https://github.com/unitycatalog/unitycatalog)
- [Quickstart Guide](https://docs.unitycatalog.io/quickstart/)
- [Open Source Announcement](https://www.databricks.com/blog/open-sourcing-unity-catalog)
- [UC with DuckDB integration](https://medium.com/@kywe665/unity-catalog-oss-with-hudi-delta-iceberg-and-emr-duckdb-710ab8f8a7dc)

### delta-rs (already in stack)
- [delta-rs Docs](https://delta-io.github.io/delta-rs/)
- [Delta Lake without Spark](https://delta.io/blog/delta-lake-without-spark/)
