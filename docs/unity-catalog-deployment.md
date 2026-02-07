# Unity Catalog OSS Deployment Guide

Deploy Unity Catalog OSS as a centralized data catalog for Delta Lake tables.

## Security Notice

**This guide starts with authentication disabled for initial setup.** Before exposing UC to any network:

1. Enable authentication (`server.authorization=enable`)
2. Configure token-based access
3. Never bind to `0.0.0.0` with auth disabled

## Prerequisites

- Docker + Docker Compose
- PostgreSQL (runs in container)
- NFS or local storage for Delta tables

## Quick Start (Localhost Only)

### 1. Directory Structure

```bash
mkdir -p unity-catalog/{etc/conf,etc/data,pgdata,backups}
mkdir -p /path/to/lake/{bronze,silver,gold}
cd unity-catalog
```

### 2. Configuration Files

**`etc/conf/hibernate.properties`**
```properties
hibernate.connection.driver_class=org.postgresql.Driver
hibernate.connection.url=jdbc:postgresql://postgres:5432/unity_catalog
hibernate.connection.username=uc_admin
hibernate.connection.password=${UC_DB_PASSWORD}
hibernate.hbm2ddl.auto=update
hibernate.dialect=org.hibernate.dialect.PostgreSQLDialect
```

> **Note:** Use environment variable substitution or copy from `hibernate.properties.example` and set the password.

**`etc/conf/server.properties`**
```properties
server.port=8080
server.env=dev
# SECURITY: Enable auth before exposing to network
server.authorization=disable
```

### 3. Docker Compose

**`docker-compose.yml`**
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16.4-alpine
    container_name: uc-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: unity_catalog
      POSTGRES_USER: uc_admin
      POSTGRES_PASSWORD: ${UC_DB_PASSWORD}
    volumes:
      - ./pgdata:/var/lib/postgresql/data
    expose:
      - "5432"
    networks:
      - uc-internal
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U uc_admin -d unity_catalog"]
      interval: 10s
      timeout: 5s
      retries: 5

  unity-catalog:
    image: unitycatalog/unitycatalog:v0.3.1
    container_name: unity-catalog
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./etc/conf:/opt/unitycatalog/etc/conf:ro
      - ./etc/data:/opt/unitycatalog/etc/data
      - ${LAKE_PATH:-/path/to/lake}:/lake:rw
    ports:
      # SECURITY: Localhost only by default
      # Change to "<your-ip>:8080:8080" only after enabling auth
      - "127.0.0.1:8080:8080"
    networks:
      - uc-internal
      - uc-external
    healthcheck:
      test: ["CMD-SHELL", "busybox wget -q --spider http://localhost:8080/api/2.1/unity-catalog/catalogs || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  uc-internal:
    internal: true
  uc-external:
```

**`.env`** (create from `.env.example`, never commit)
```bash
UC_DB_PASSWORD=your-secure-password-here
LAKE_PATH=/path/to/your/lake
```

### 4. Start Services

```bash
# Set password (don't use shell history)
read -s UC_DB_PASSWORD && export UC_DB_PASSWORD

docker compose up -d
docker compose logs -f unity-catalog
```

### 5. Verify

```bash
curl http://127.0.0.1:8080/api/2.1/unity-catalog/catalogs
# Should return: {"catalogs": []}
```

---

## Client Integration

### Environment Variables

| Variable | Default | Used By |
|----------|---------|---------|
| `UC_SERVER_URL` | `http://127.0.0.1:8080` | Spark, DuckDB |
| `UC_API_URL` | `http://127.0.0.1:8080/api/2.1/unity-catalog` | Python REST client |
| `UC_TOKEN` | (none) | All clients (when auth enabled) |
| `UC_ENABLED` | `false` | Application toggle |

**Important:** Spark and DuckDB use the base URL. The Python REST client uses the `/api/2.1/unity-catalog` path.

### Spark Configuration

```python
# Known-good baseline: Spark 3.5.3 + Delta 3.2.1
# Experimental: Spark 4.x + Delta 4.x (works but less tested with UC)

builder = (
    SparkSession.builder
    .config("spark.jars.packages",
            "io.delta:delta-spark_2.13:4.0.0,"
            "io.unitycatalog:unitycatalog-spark_2.13:0.3.1")
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.catalog.unity",
            "io.unitycatalog.spark.UCSingleCatalog")
    .config("spark.sql.catalog.unity.uri",
            os.getenv("UC_SERVER_URL", "http://127.0.0.1:8080"))
)
```

### DuckDB Configuration

> **Security Note:** The `uc_catalog` extension is installed from DuckDB's `core_nightly` channel at runtime. Review [DuckDB's extension security guidance](https://duckdb.org/docs/extensions/overview) before use in production.

```python
import duckdb

conn = duckdb.connect()

# Install extensions (downloads at runtime)
conn.execute("INSTALL uc_catalog FROM core_nightly;")
conn.execute("LOAD uc_catalog;")
conn.execute("INSTALL delta;")
conn.execute("LOAD delta;")

# Create secret with base URL (not /api/... path)
conn.execute("""
    CREATE SECRET uc (
        TYPE UC,
        TOKEN 'not-used',
        ENDPOINT 'http://127.0.0.1:8080',
        AWS_REGION 'us-east-2'
    );
""")

# Attach catalog
conn.execute("ATTACH 'my_catalog' AS my_catalog (TYPE UC_CATALOG);")

# Query
conn.execute("SELECT * FROM my_catalog.my_schema.my_table").fetchdf()
```

---

## Enabling Authentication

Before exposing UC to a network:

1. **Edit `server.properties`:**
   ```properties
   server.authorization=enable
   ```

2. **Change conf mount to writable** (UC writes `token.txt`):
   ```yaml
   - ./etc/conf:/opt/unitycatalog/etc/conf:rw
   ```

3. **Restart UC:**
   ```bash
   docker compose restart unity-catalog
   ```

4. **Retrieve admin token:**
   ```bash
   cat etc/conf/token.txt
   ```

5. **Configure clients:**
   ```bash
   export UC_TOKEN=$(cat etc/conf/token.txt)
   ```

6. **Now safe to bind to network:**
   ```yaml
   ports:
     - "<YOUR_LAN_IP>:8080:8080"  # Your LAN IP
   ```

---

## NFS Mount (Remote Storage)

If your Delta tables are on NAS/NFS:

### Security Checklist

- [ ] Export to specific client IPs only (not entire subnet)
- [ ] Enable root squashing (`root_squash` not `no_root_squash`)
- [ ] Firewall NFS ports (2049, 111) to trusted hosts only
- [ ] Use NFSv4.1 with hard mounts

### Mount Options

```bash
# Recommended mount options for Delta Lake
sudo mount -t nfs -o vers=4.1,hard,timeo=600,retrans=2 \
    nas-ip:/volume/lake /mnt/lake
```

### Concurrency Rules

- **Single writer:** Treat Spark as the only writer
- **DuckDB:** Read-only via UC or `delta_scan()`
- **No concurrent writes:** Avoid two Spark jobs writing the same table

---

## Backup & Recovery

### Automated Backup Script

```bash
#!/bin/sh
set -eu
BACKUP_DIR="./backups"
RETENTION_DAYS=14

mkdir -p "$BACKUP_DIR"

# Dump and compress
docker exec -e PGPASSWORD="$UC_DB_PASSWORD" uc-postgres \
  pg_dump -U uc_admin unity_catalog \
  | gzip > "${BACKUP_DIR}/uc_$(date +%Y%m%d).sql.gz"

# Prune old backups
find "$BACKUP_DIR" -name "uc_*.sql.gz" -mtime +$RETENTION_DAYS -delete
```

### Recovery

```bash
gunzip -c backups/uc_YYYYMMDD.sql.gz | \
  docker exec -i -e PGPASSWORD="$UC_DB_PASSWORD" uc-postgres \
  psql -U uc_admin unity_catalog
```

---

## Troubleshooting

### UC using embedded H2 instead of Postgres

**Symptom:** Tables disappear after container restart

**Check:**
```bash
docker logs unity-catalog | grep -i "datasource\|h2\|postgres"
```

**Fix:** Ensure `hibernate.properties` is mounted correctly at `/opt/unitycatalog/etc/conf/`

### DuckDB extension install fails

**Symptom:** `INSTALL uc_catalog FROM core_nightly` errors

**Options:**
1. Pin DuckDB version in your dependencies
2. Use direct `delta_scan('/path/to/table')` as fallback
3. Check network access to DuckDB extension servers

### Healthcheck fails

**Symptom:** Container marked unhealthy

**Note:** Some UC images don't have `busybox`. Remove the healthcheck if needed:
```yaml
healthcheck:
  disable: true
```

---

## Files to Never Commit

Add to `.gitignore`:

```gitignore
# Secrets
.env
*.env
token.txt
**/token.txt
hibernate.properties

# Runtime data
pgdata/
backups/
etc/data/
```

Commit only `*.example` templates.

---

## Version Compatibility

| Component | Tested Version | Notes |
|-----------|----------------|-------|
| Unity Catalog | v0.3.1 | Pinned for stability |
| PostgreSQL | 16.4-alpine | Metadata store |
| PySpark | 3.5.3+ or 4.0.x | 3.5.3 is baseline |
| Delta Spark | 3.2.1 or 4.0.x | Match Spark version |
| DuckDB | 1.0+ | Pin in production |

---

## References

- [Unity Catalog OSS Docs](https://docs.unitycatalog.io/)
- [Unity Catalog GitHub](https://github.com/unitycatalog/unitycatalog)
- [DuckDB UC Integration](https://duckdb.org/docs/extensions/uc_catalog)
- [Delta Lake](https://delta.io/)
