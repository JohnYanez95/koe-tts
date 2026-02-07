/**
 * Main run view page.
 */

import { useState, useEffect, useMemo, useCallback } from 'react';
import { getRunMeta, getMetrics, type RunMeta, type MetricsPoint, type TrainingEvent, isTerminalStatus } from '../api/client';
import { useSSE } from '../hooks/useSSE';
import { useGpuPolling, useEventsPolling } from '../hooks/usePolling';
import { RunSelector } from '../components/RunSelector';
import { StabilityStrip } from '../components/StabilityStrip';
import { LossChart } from '../components/LossChart';
import { GradNormChart } from '../components/GradNormChart';
import { GanBalanceChart } from '../components/GanBalanceChart';
import { QualityChart } from '../components/QualityChart';
import { DScoreChart } from '../components/DScoreChart';
import { EventsTimeline } from '../components/EventsTimeline';
import { ArtifactsList } from '../components/ArtifactsList';
import { ControlPanel } from '../components/ControlPanel';
import { EmergencyBanner } from '../components/EmergencyBanner';

// Zoom range for synchronized chart zooming (manual brush selection)
export interface ZoomRange {
  startStep: number;
  endStep: number;
}

// How many trailing points to fetch from backend
const TRAILING_POINTS = 500;

export function RunView() {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runMeta, setRunMeta] = useState<RunMeta | null>(null);
  const [historicalMetrics, setHistoricalMetrics] = useState<MetricsPoint[]>([]);
  const [isLoadingMeta, setIsLoadingMeta] = useState(false);
  const [highlightedEventTs] = useState<string | null>(null);
  const [zoomRange, setZoomRange] = useState<ZoomRange | null>(null);

  // Handler for synchronized brush zoom across all charts
  const handleZoomChange = useCallback((range: ZoomRange | null) => {
    setZoomRange(range);
  }, []);

  // Reset zoom on 'R' key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'r' || e.key === 'R') {
        // Don't reset if user is typing in an input
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
          return;
        }
        setZoomRange(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Determine if run is terminal (use status from RunMeta)
  const isTerminal = runMeta ? isTerminalStatus(runMeta.status) : false;

  // SSE for live metrics - disabled for terminal runs
  const { metrics: liveMetrics, streamState, clear: clearLive } = useSSE({
    runId: selectedRunId,
    enabled: !isTerminal,
  });

  // Polling for GPU and events
  const { gpu } = useGpuPolling({ enabled: true });
  const { events, isLoading: eventsLoading } = useEventsPolling({
    runId: selectedRunId,
    enabled: true,
    limit: 200,  // Higher limit to capture evals within metrics window
  });

  // Fetch run metadata when selection changes
  useEffect(() => {
    if (!selectedRunId) {
      setRunMeta(null);
      setHistoricalMetrics([]);
      return;
    }

    // Capture non-null runId for closure
    const currentRunId = selectedRunId;

    async function fetchMeta() {
      setIsLoadingMeta(true);
      try {
        const meta = await getRunMeta(currentRunId);
        setRunMeta(meta);

        // Fetch historical metrics separately (may 404 for empty runs)
        // Only fetch trailing window size to save memory
        try {
          const metrics = await getMetrics(currentRunId, 0, TRAILING_POINTS, true);  // tail=true for latest data
          setHistoricalMetrics(metrics.metrics);
        } catch (metricsErr) {
          // Metrics not available (empty run or different structure)
          console.warn('No metrics available for run:', currentRunId);
          setHistoricalMetrics([]);
        }
        clearLive();
      } catch (e) {
        console.error('Failed to fetch run meta:', e);
      } finally {
        setIsLoadingMeta(false);
      }
    }

    fetchMeta();
  }, [selectedRunId, clearLive]);

  // Combine historical + live metrics into a sliding window
  const allMetrics = useMemo(() => {
    if (liveMetrics.length === 0) return historicalMetrics;
    // Find where live metrics start
    const lastHistorical = historicalMetrics[historicalMetrics.length - 1]?.step ?? 0;
    const newLive = liveMetrics.filter((m) => m.step > lastHistorical);
    const combined = [...historicalMetrics, ...newLive];
    // Keep only trailing window to prevent unbounded growth
    if (combined.length > TRAILING_POINTS) {
      return combined.slice(-TRAILING_POINTS);
    }
    return combined;
  }, [historicalMetrics, liveMetrics]);

  // Periodically refetch historical to keep data fresh and fill gaps
  useEffect(() => {
    if (!selectedRunId || liveMetrics.length === 0) return;

    // Refetch every 100 live data points to keep historical fresh
    if (liveMetrics.length % 100 === 0) {
      console.log(`Periodic refetch at ${liveMetrics.length} live points`);
      getMetrics(selectedRunId, 0, TRAILING_POINTS, true)
        .then((result) => {
          setHistoricalMetrics(result.metrics);
        })
        .catch((err) => {
          console.warn('Failed to refetch metrics:', err);
        });
    }
  }, [liveMetrics.length, selectedRunId]);

  // Current metric (latest point)
  const currentMetric = useMemo(() => {
    return allMetrics[allMetrics.length - 1] ?? null;
  }, [allMetrics]);

  // Last finite metric (for NaN-aware display)
  // A metric is "finite" if its key losses are valid numbers
  const lastFiniteMetric = useMemo(() => {
    for (let i = allMetrics.length - 1; i >= 0; i--) {
      const m = allMetrics[i];
      // Check if key losses are finite
      const hasFiniteLoss =
        (m.g_loss_mel !== null && Number.isFinite(m.g_loss_mel)) ||
        (m.loss_g !== null && Number.isFinite(m.loss_g));
      if (hasFiniteLoss) return m;
    }
    return null;
  }, [allMetrics]);

  // Historical KL values for spike detection (finite values only)
  const historicalKl = useMemo(() => {
    return allMetrics
      .map((m) => m.g_loss_kl)
      .filter((v): v is number => v !== null && Number.isFinite(v));
  }, [allMetrics]);

  // Compute emergency state from metrics and events
  const emergencyState = useMemo(() => {
    const latest = allMetrics[allMetrics.length - 1];

    // Check metrics for emergency indicators
    const metricEmergency = latest?.ctrl_emergency_stop === true;
    const metricNan = latest?.ctrl_nan_inf_detected === true;
    const metricReason = latest?.ctrl_emergency_reason;

    // Check events for training_complete with emergency_stop status
    const emergencyEvent = events.find(
      (e: TrainingEvent) =>
        e.event === 'training_complete' && e.status === 'emergency_stop'
    );

    const active = metricEmergency || metricNan || !!emergencyEvent;

    // Determine title and detail
    let title = 'Emergency Stop';
    let detail = metricReason ?? (emergencyEvent?.reason as string) ?? undefined;
    let step = latest?.step ?? (emergencyEvent?.step as number);
    let checkpointPath = emergencyEvent?.checkpoint_path as string | undefined;

    if (metricNan && !metricEmergency) {
      title = 'NaN/Inf Detected';
      detail = detail ?? 'Non-finite values in gradients or loss';
    }

    return { active, title, detail, step, checkpointPath };
  }, [allMetrics, events]);

  const alarmColor = (alarm: string) => {
    switch (alarm) {
      case 'healthy':
        return '#4ade80';
      case 'unstable':
        return '#fbbf24';
      case 'd_dominant':
        return '#f97316';
      case 'g_collapse':
        return '#ef4444';
      default:
        return '#9ca3af';
    }
  };

  return (
    <div style={styles.container}>
      {/* Sidebar */}
      <aside style={styles.sidebar}>
        <RunSelector selectedRunId={selectedRunId} onSelectRun={setSelectedRunId} />
      </aside>

      {/* Main content */}
      <main style={styles.main}>
        {!selectedRunId ? (
          <div style={styles.placeholder}>Select a run to view details</div>
        ) : isLoadingMeta ? (
          <div style={styles.placeholder}>Loading...</div>
        ) : runMeta ? (
          <>
            {/* Emergency banner (if triggered) */}
            <EmergencyBanner
              active={emergencyState.active}
              title={emergencyState.title}
              detail={emergencyState.detail}
              step={emergencyState.step}
              checkpointPath={emergencyState.checkpointPath}
            />

            {/* Stability strip (for GAN stage or when we have controller data) */}
            {(runMeta.stage === 'gan' || currentMetric?.ctrl_controller_alarm) && (
              <StabilityStrip
                current={currentMetric}
                lastFinite={lastFiniteMetric}
                historicalKl={historicalKl}
                gpu={gpu}
              />
            )}

            {/* Header */}
            <header style={styles.header}>
              <div style={styles.headerLeft}>
                <h1 style={styles.runTitle}>{runMeta.run_id}</h1>
                <div style={styles.runMeta}>
                  <span style={styles.badge}>{runMeta.stage.toUpperCase()}</span>
                  <span style={styles.metaItem}>
                    Step {runMeta.current_step.toLocaleString()}
                  </span>
                  <span style={styles.metaItem}>
                    {runMeta.checkpoints.length} checkpoints
                  </span>
                </div>
              </div>
              <div style={styles.headerRight}>
                <div
                  style={{
                    ...styles.alarmBadge,
                    backgroundColor: alarmColor(runMeta.alarm_state),
                  }}
                >
                  {runMeta.alarm_state}
                </div>
                <div style={styles.connectionStatus}>
                  {isTerminal ? (
                    <span
                      style={runMeta.status === 'lost' ? styles.lost : styles.ended}
                      title={
                        runMeta.status_source === 'event'
                          ? `${runMeta.status}${runMeta.status_reason ? ` (${runMeta.status_reason})` : ''}`
                          : runMeta.status_source === 'mtime_heuristic'
                          ? `No metrics since ${runMeta.last_updated_at ? new Date(runMeta.last_updated_at).toLocaleString() : 'unknown'} (>${runMeta.metrics_stale_seconds}s)`
                          : 'Legacy run (no metrics)'
                      }
                    >
                      ■ {runMeta.status === 'lost' ? 'Lost' : 'Ended'}
                    </span>
                  ) : streamState === 'connected' ? (
                    <span style={styles.connected}>● Live</span>
                  ) : streamState === 'not_available' ? (
                    <span style={styles.notAvailable} title="No metrics stream for this run">○ No stream</span>
                  ) : streamState === 'connecting' ? (
                    <span style={styles.connecting}>◌ Connecting</span>
                  ) : (
                    <span style={styles.disconnected}>○ Offline</span>
                  )}
                </div>
              </div>
            </header>

            {/* Zoom controls - only show when manually zoomed (not in follow mode) */}
            {zoomRange && (
              <div style={styles.zoomControls}>
                <span style={styles.zoomLabel}>
                  Zoomed: {zoomRange.startStep.toLocaleString()} - {zoomRange.endStep.toLocaleString()}
                </span>
                <button style={styles.resetButton} onClick={() => setZoomRange(null)}>
                  Reset (R)
                </button>
              </div>
            )}

            {/* Charts section - all charts share synchronized zoom */}
            {runMeta.stage === 'gan' ? (
              /* 4-chart GAN layout */
              <div style={styles.ganChartsGrid}>
                <div style={styles.chartCell}>
                  <GradNormChart points={allMetrics} hasMetrics={runMeta.has_metrics} zoomRange={zoomRange} onZoomChange={handleZoomChange} />
                </div>
                <div style={styles.chartCell}>
                  <GanBalanceChart points={allMetrics} hasMetrics={runMeta.has_metrics} zoomRange={zoomRange} onZoomChange={handleZoomChange} />
                </div>
                <div style={styles.chartCell}>
                  <QualityChart points={allMetrics} events={events} hasMetrics={runMeta.has_metrics} zoomRange={zoomRange} onZoomChange={handleZoomChange} />
                </div>
                <div style={styles.chartCell}>
                  <DScoreChart points={allMetrics} hasMetrics={runMeta.has_metrics} zoomRange={zoomRange} onZoomChange={handleZoomChange} />
                </div>
              </div>
            ) : (
              /* Non-GAN: loss chart with optional grad norm */
              <div style={styles.chartContainer}>
                <LossChart metrics={allMetrics} stage={runMeta.stage} events={events} hasMetrics={runMeta.has_metrics} zoomRange={zoomRange} />
                {/* Grad norm chart if we have grad data */}
                {allMetrics.some((m) => m.g_grad_norm !== null) && (
                  <div style={{ marginTop: 16 }}>
                    <GradNormChart points={allMetrics} hasMetrics={runMeta.has_metrics} zoomRange={zoomRange} />
                  </div>
                )}
              </div>
            )}

            {/* Bottom row */}
            <div style={styles.bottomRow}>
              <div style={styles.eventsContainer}>
                <EventsTimeline events={events} isLoading={eventsLoading} highlightedEventTs={highlightedEventTs} />
              </div>
              <div style={styles.rightColumn}>
                <div style={styles.artifactsContainer}>
                  <ArtifactsList runId={selectedRunId} />
                </div>
                <div style={styles.controlContainer}>
                  <ControlPanel
                    runId={selectedRunId}
                    supportsControl={runMeta.supports_control}
                    alarmState={runMeta.alarm_state}
                    events={events}
                    stage={runMeta.stage}
                  />
                </div>
              </div>
            </div>
          </>
        ) : (
          <div style={styles.placeholder}>Failed to load run</div>
        )}
      </main>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    height: '100vh',
    backgroundColor: '#0f0f0f',
  },
  sidebar: {
    width: 320,
    flexShrink: 0,
    padding: 16,
    borderRight: '1px solid #262626',
    overflow: 'auto',
  },
  main: {
    flex: 1,
    padding: 24,
    overflow: 'auto',
  },
  placeholder: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    color: '#6b7280',
    fontSize: 16,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 24,
  },
  headerLeft: {},
  runTitle: {
    margin: 0,
    fontSize: 20,
    fontWeight: 600,
    color: '#e0e0e0',
    fontFamily: 'monospace',
  },
  runMeta: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    marginTop: 8,
  },
  badge: {
    fontSize: 10,
    fontWeight: 600,
    padding: '3px 8px',
    borderRadius: 4,
    backgroundColor: '#3b82f6',
    color: '#fff',
  },
  metaItem: {
    fontSize: 13,
    color: '#9ca3af',
  },
  headerRight: {
    display: 'flex',
    alignItems: 'center',
    gap: 16,
  },
  alarmBadge: {
    fontSize: 11,
    fontWeight: 600,
    padding: '4px 10px',
    borderRadius: 4,
    color: '#000',
    textTransform: 'uppercase',
  },
  connectionStatus: {
    fontSize: 12,
  },
  connected: {
    color: '#4ade80',
  },
  disconnected: {
    color: '#6b7280',
  },
  ended: {
    color: '#9ca3af',
  },
  lost: {
    color: '#f97316',
  },
  notAvailable: {
    color: '#6b7280',
  },
  connecting: {
    color: '#fbbf24',
  },
  chartContainer: {
    marginBottom: 16,
    minWidth: 0,
  },
  ganChartsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: 16,
    marginBottom: 16,
  },
  chartCell: {
    minWidth: 0,
  },
  bottomRow: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 16,
  },
  eventsContainer: {},
  rightColumn: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  artifactsContainer: {},
  controlContainer: {},
  zoomControls: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
    padding: '8px 12px',
    backgroundColor: '#1a1a1a',
    borderRadius: 6,
    border: '1px solid #3b82f6',
  },
  zoomLabel: {
    fontSize: 12,
    color: '#9ca3af',
    fontFamily: 'monospace',
  },
  resetButton: {
    fontSize: 11,
    padding: '4px 10px',
    backgroundColor: '#3b82f6',
    color: '#fff',
    border: 'none',
    borderRadius: 4,
    cursor: 'pointer',
  },
};
