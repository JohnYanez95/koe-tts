/**
 * Artifacts list component.
 */

import { useState, useEffect, useCallback } from 'react';
import { getArtifacts, type ArtifactInfo } from '../api/client';

interface ArtifactsListProps {
  runId: string | null;
}

export function ArtifactsList({ runId }: ArtifactsListProps) {
  const [artifacts, setArtifacts] = useState<ArtifactInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchArtifacts = useCallback(async () => {
    if (!runId) {
      setArtifacts([]);
      return;
    }
    setIsLoading(true);
    try {
      const data = await getArtifacts(runId);
      setArtifacts(data.eval);
    } catch {
      setArtifacts([]);
    } finally {
      setIsLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    fetchArtifacts();
    const interval = setInterval(fetchArtifacts, 15000); // Refresh every 15s
    return () => clearInterval(interval);
  }, [fetchArtifacts]);

  const formatTime = (ts: string) => {
    const date = new Date(ts);
    return date.toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const typeColors: Record<string, string> = {
    multispeaker: '#8b5cf6',
    eval: '#3b82f6',
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Eval Artifacts</h3>
        <button
          onClick={fetchArtifacts}
          disabled={isLoading}
          style={styles.refreshButton}
          title="Refresh artifacts list"
        >
          {isLoading ? '...' : '↻'}
        </button>
      </div>

      {isLoading && artifacts.length === 0 ? (
        <div style={styles.loading}>Loading...</div>
      ) : artifacts.length === 0 ? (
        <div style={styles.empty}>No artifacts yet</div>
      ) : (
        <div style={styles.list}>
          {artifacts.map((artifact) => (
            <a
              key={artifact.path}
              href={`/runs/${artifact.path}`}
              target="_blank"
              rel="noopener noreferrer"
              style={styles.item}
            >
              <div style={styles.itemHeader}>
                <span style={styles.itemName}>{artifact.name}</span>
                <span
                  style={{
                    ...styles.itemType,
                    backgroundColor: typeColors[artifact.type] || '#6b7280',
                  }}
                >
                  {artifact.type}
                </span>
              </div>
              <div style={styles.itemTime}>{formatTime(artifact.updated_at)}</div>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    padding: 16,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  title: {
    margin: 0,
    fontSize: 14,
    fontWeight: 600,
    color: '#9ca3af',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  refreshButton: {
    background: 'none',
    border: '1px solid #404040',
    borderRadius: 4,
    color: '#9ca3af',
    fontSize: 14,
    padding: '2px 8px',
    cursor: 'pointer',
  },
  loading: {
    color: '#6b7280',
    fontSize: 13,
  },
  empty: {
    color: '#6b7280',
    fontSize: 13,
    fontStyle: 'italic',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    maxHeight: 200,
    overflowY: 'auto',
  },
  item: {
    display: 'block',
    backgroundColor: '#262626',
    borderRadius: 6,
    padding: 10,
    textDecoration: 'none',
    color: 'inherit',
    border: '1px solid transparent',
    transition: 'all 0.15s ease',
  },
  itemHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  itemName: {
    fontSize: 12,
    fontWeight: 500,
    color: '#e0e0e0',
    fontFamily: 'monospace',
  },
  itemType: {
    fontSize: 9,
    fontWeight: 600,
    padding: '2px 5px',
    borderRadius: 3,
    color: '#fff',
    textTransform: 'uppercase',
  },
  itemTime: {
    fontSize: 10,
    color: '#6b7280',
  },
};
