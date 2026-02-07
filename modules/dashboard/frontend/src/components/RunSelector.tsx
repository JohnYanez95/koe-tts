/**
 * Run selector component.
 *
 * Badge ordering (left → right): Stage → Alarm → ⚡ → Status (INIT/DONE/STOPPED/EMERGENCY/FAILED)
 */

import { useState, useEffect } from 'react';
import { listRuns, type RunSummary, isTerminalStatus } from '../api/client';

interface RunSelectorProps {
  selectedRunId: string | null;
  onSelectRun: (runId: string) => void;
}

interface Badge {
  key: string;
  label: string;
  color: string;
  textColor: string;
  title: string;
  dimmed?: boolean;
}

/**
 * Get ordered badges for a run following canonical rules:
 * 1. Stage (always shown)
 * 2. Alarm (GAN + has_metrics only, UNKNOWN if missing)
 * 3. ⚡ (supports_control, dimmed if terminal)
 * 4. Status: INIT | DONE | STOPPED | EMERGENCY | FAILED
 */
function getBadges(run: RunSummary): Badge[] {
  const badges: Badge[] = [];
  const isTerminal = isTerminalStatus(run.status);

  // 1. Stage badge (always)
  const stageColors: Record<string, string> = {
    gan: '#a855f7',
    core: '#3b82f6',
    baseline: '#6b7280',
    duration: '#06b6d4',
  };
  badges.push({
    key: 'stage',
    label: run.stage.toUpperCase(),
    color: stageColors[run.stage] || '#6b7280',
    textColor: '#fff',
    title: `Stage: ${run.stage}`,
  });

  // 2. Alarm badge (GAN + has_metrics only)
  if (run.stage === 'gan' && run.has_metrics) {
    const alarmColors: Record<string, string> = {
      healthy: '#4ade80',
      unstable: '#fbbf24',
      d_dominant: '#f97316',
      g_collapse: '#ef4444',
      emergency: '#dc2626',
    };
    const alarm = run.alarm_state || 'unknown';
    const alarmColor = alarmColors[alarm] || '#6b7280';
    const isHealthy = alarm === 'healthy';
    const isUnknown = alarm === 'unknown' || !alarmColors[alarm];

    badges.push({
      key: 'alarm',
      label: isHealthy ? '●' : isUnknown ? 'UNKNOWN' : alarm.replace('_', ' '),
      color: alarmColor,
      textColor: isHealthy ? '#000' : '#fff',
      title: `Alarm: ${alarm}`,
    });
  }

  // 3. Control plane badge (only for active runs)
  if (run.supports_control && !isTerminal) {
    badges.push({
      key: 'control',
      label: '⚡',
      color: 'transparent',
      textColor: '#fbbf24',
      title: 'Control plane enabled',
    });
  }

  // 4. Status badge
  if (run.status === 'running' && !run.has_metrics) {
    // INIT: currently running but no metrics yet
    badges.push({
      key: 'status',
      label: 'INIT',
      color: '#404040',
      textColor: '#9ca3af',
      title: 'Initializing - waiting for first metrics',
    });
  } else if (run.status === 'emergency') {
    badges.push({
      key: 'status',
      label: 'EMERGENCY',
      color: '#dc2626',
      textColor: '#fff',
      title: 'Emergency stop triggered',
    });
  } else if (isTerminal && !run.has_metrics && run.checkpoint_count === 0) {
    // FAILED: terminal, no metrics, no checkpoints
    badges.push({
      key: 'status',
      label: 'FAILED',
      color: '#7f1d1d',
      textColor: '#fca5a5',
      title: 'Run failed - no metrics or checkpoints',
    });
  } else if (run.status === 'stopped') {
    badges.push({
      key: 'status',
      label: 'STOPPED',
      color: '#78350f',
      textColor: '#fcd34d',
      title: 'Run stopped (user or thermal)',
    });
  } else if (run.status === 'completed') {
    badges.push({
      key: 'status',
      label: 'DONE',
      color: '#374151',
      textColor: '#9ca3af',
      title: 'Run completed',
    });
  }

  return badges;
}

/**
 * Format relative time (e.g., "2h ago", "3d ago")
 */
function formatRelativeTime(ts: string): string {
  if (!ts) return '';
  const now = Date.now();
  const then = new Date(ts).getTime();
  const diffMs = now - then;

  const minutes = Math.floor(diffMs / 60000);
  const hours = Math.floor(diffMs / 3600000);
  const days = Math.floor(diffMs / 86400000);

  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'just now';
}

export function RunSelector({ selectedRunId, onSelectRun }: RunSelectorProps) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function fetchRuns() {
      try {
        const data = await listRuns();
        setRuns(data);
        setError(null);

        // Auto-select latest if none selected
        if (!selectedRunId && data.length > 0) {
          onSelectRun(data[0].run_id);
        }
      } catch (e) {
        setError(e instanceof Error ? e : new Error('Failed to fetch runs'));
      } finally {
        setIsLoading(false);
      }
    }

    fetchRuns();
    const interval = setInterval(fetchRuns, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [selectedRunId, onSelectRun]);

  if (isLoading) {
    return <div style={styles.container}>Loading runs...</div>;
  }

  if (error) {
    return <div style={styles.container}>Error: {error.message}</div>;
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Training Runs</h3>
      <div style={styles.list}>
        {runs.map((run) => {
          const badges = getBadges(run);
          const isTerminal = isTerminalStatus(run.status);

          // Build meta line: step 25,300 • 9h ago • 3 ckpts
          const metaParts: string[] = [];
          metaParts.push(`step ${run.step.toLocaleString()}`);
          if (run.updated_at) {
            metaParts.push(formatRelativeTime(run.updated_at));
          }
          if (run.checkpoint_count > 0) {
            metaParts.push(`${run.checkpoint_count} ckpt${run.checkpoint_count !== 1 ? 's' : ''}`);
          }

          return (
            <div
              key={run.run_id}
              style={{
                ...styles.item,
                ...(run.run_id === selectedRunId ? styles.itemSelected : {}),
                ...(isTerminal ? styles.itemDimmed : {}),
              }}
              onClick={() => onSelectRun(run.run_id)}
            >
              <div style={styles.runHeader}>
                <span style={styles.runId}>{run.run_id}</span>
              </div>
              <div style={styles.badgeRow}>
                {badges.map((badge) => (
                  <span
                    key={badge.key}
                    style={{
                      ...styles.pillBadge,
                      backgroundColor: badge.color,
                      color: badge.textColor,
                      opacity: badge.dimmed ? 0.5 : 1,
                    }}
                    title={badge.title}
                  >
                    {badge.label}
                  </span>
                ))}
              </div>
              <div style={styles.runMeta}>
                {metaParts.join(' • ')}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    padding: 16,
    height: '100%',
    overflow: 'auto',
  },
  title: {
    margin: '0 0 12px 0',
    fontSize: 14,
    fontWeight: 600,
    color: '#9ca3af',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  item: {
    backgroundColor: '#262626',
    borderRadius: 6,
    padding: 12,
    cursor: 'pointer',
    border: '1px solid transparent',
    transition: 'all 0.15s ease',
  },
  itemSelected: {
    borderColor: '#3b82f6',
    backgroundColor: '#1e3a5f',
  },
  itemDimmed: {
    opacity: 0.7,
  },
  runHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  runId: {
    fontSize: 12,
    fontWeight: 500,
    color: '#e0e0e0',
    fontFamily: 'monospace',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    flex: 1,
  },
  badgeRow: {
    display: 'flex',
    gap: 4,
    marginBottom: 6,
    flexWrap: 'wrap',
    alignItems: 'center',
  },
  pillBadge: {
    fontSize: 9,
    fontWeight: 600,
    padding: '2px 6px',
    borderRadius: 10,
  },
  runMeta: {
    fontSize: 11,
    color: '#6b7280',
  },
};
