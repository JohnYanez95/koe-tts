# NAS Migration Plan

> Move `data/`, `lake/`, `models/` to NAS without changing code.

## Current State (Workstation)

Storage lives on a local drive, accessed via symlinks:

```
~/Repos/koe-tts/
├── data -> $KOE_DATA_ROOT/data      # Audio, archives
├── lake -> $KOE_DATA_ROOT/lake      # Delta tables
└── models -> $KOE_DATA_ROOT/models  # Checkpoints, exports
```

Local SSD used for hot cache:
```
data/cache/      # Training cache snapshots (fast reads)
```

## Target State (NAS)

NAS device for cold storage.

Mount point: `$KOE_DATA_ROOT` (set via env var)

```
~/Repos/koe-tts/
├── data -> $KOE_DATA_ROOT/data         # NAS (cold storage)
├── lake -> $KOE_DATA_ROOT/lake         # NAS (Delta tables)
├── models -> $KOE_DATA_ROOT/models     # NAS (checkpoints)
└── data/cache/ -> $KOE_LOCAL_ROOT/     # Local SSD (hot cache)
```

---

## Path Contract Invariants

These must remain true after migration:

| Invariant | Expression |
|-----------|------------|
| Audio resolution | `audio_abspath = DATA_ROOT / "data" / audio_relpath` |
| Delta table paths | `lake/{layer}/{dataset}/{table}/` unchanged |
| Manifest paths | `lake/gold/{dataset}/manifests/{snapshot_id}.jsonl` |
| Catalog fast reads | `koe catalog list` uses delta-rs (no Spark boot) |

The `DATA_ROOT` environment variable (or `~/.koe/config.yaml`) is the only thing that changes.

---

## Services (Optional, Later)

Once NAS is stable, consider running always-on services via Docker Compose:

### Tier 1: Nice to Have
| Service | Purpose | Image |
|---------|---------|-------|
| MLflow | Experiment tracking, model registry | `ghcr.io/mlflow/mlflow` |
| Labeler App | Web UI for phoneme/QC labeling | Custom |

### Tier 2: Only If Needed
| Service | Purpose | Image |
|---------|---------|-------|
| Postgres | Metastore backend, MLflow backend | `postgres:16` |
| Hive Metastore | Spark SQL table discovery | `apache/hive:4.0.0` |

The metastore is only needed if you want `spark.table("gold_jsut")` syntax instead of path-based reads. Current `koe catalog` approach is sufficient for single-user workloads.

### Compose Sketch (Future)

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow
    ports:
      - "5000:5000"
    volumes:
      - /volume1/koe/mlruns:/mlflow/mlruns
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow/mlflow.db

  # Optional: Hive Metastore (only if Spark SQL discovery needed)
  metastore-db:
    image: postgres:16
    environment:
      POSTGRES_USER: hive
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: metastore
    volumes:
      - /volume1/koe/metastore_pg:/var/lib/postgresql/data

  hive-metastore:
    image: apache/hive:4.0.0
    environment:
      SERVICE_NAME: metastore
    ports:
      - "9083:9083"
    depends_on:
      - metastore-db
```

Spark would then connect via:
```python
spark.conf.set("spark.hadoop.hive.metastore.uris", "thrift://NAS_IP:9083")
```

---

## Cutover Plan

### Pre-Migration Checklist

- [ ] NAS mounted at `$KOE_DATA_ROOT`
- [ ] Sufficient space (~50GB for current data + growth)
- [ ] Network performance acceptable (1GbE minimum, 2.5GbE preferred)

### Migration Steps

```bash
# 1. Stop any running pipelines
# 2. Sync data to NAS
rsync -avP --progress $OLD_DATA_ROOT/ $KOE_DATA_ROOT/

# 3. Verify sync
diff -rq $OLD_DATA_ROOT/lake $KOE_DATA_ROOT/lake

# 4. Update symlinks
cd ~/Repos/koe-tts
rm data lake models
ln -s $KOE_DATA_ROOT/data data
ln -s $KOE_DATA_ROOT/lake lake
ln -s $KOE_DATA_ROOT/models models

# 5. Smoke tests
koe catalog list
koe catalog describe gold.jsut.utterances
koe gold jsut --no-write-delta  # Dry run to verify reads
```

### Rollback

If issues arise:
```bash
# Point symlinks back to local drive
rm data lake models
ln -s $OLD_DATA_ROOT/data data
ln -s $OLD_DATA_ROOT/lake lake
ln -s $OLD_DATA_ROOT/models models
```

---

## Performance Considerations

| Operation | Expected Impact |
|-----------|-----------------|
| `koe catalog list` | No change (delta-rs, small metadata) |
| `koe bronze/silver/gold` | Slightly slower writes (network) |
| Training data loading | Use local SSD cache for hot data |
| Large file reads | Depends on NAS network (2.5GbE recommended) |

### Mitigation: Local Cache

Keep training cache on local SSD for fast iteration:

```bash
# In ~/.koe/config.yaml or environment
CACHE_ROOT=/local/ssd/koe-cache
```

Training reads from cache; only cold data hits NAS.

---

## Timeline

| Date | Milestone |
|------|-----------|
| Jan 25, 2026 | Current: workstation storage working |
| Jan 30, 2026 | NAS arrives |
| Feb 1, 2026 | Target: migration complete |
| Feb 15, 2026 | Optional: Docker services on NAS |
