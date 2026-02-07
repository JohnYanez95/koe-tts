/**
 * Stability Strip - "Now" readout for instant training status.
 * Shows current values for all key metrics with NaN-aware display.
 */

import { useMemo } from 'react';
import type { MetricsPoint, GpuResponse } from '../api/client';

interface StabilityStripProps {
  current: MetricsPoint | null;
  lastFinite: MetricsPoint | null;
  historicalKl: number[]; // For KL spike detection
  gpu?: GpuResponse | null;
}

// Format number with fallback for null/NaN
function fmt(v: number | null, decimals = 2): string {
  if (v === null || !Number.isFinite(v)) return '—';
  return v.toFixed(decimals);
}

// Format with explicit NaN indicator when current differs from lastFinite
function fmtWithNan(
  current: number | null,
  lastFinite: number | null,
  decimals = 2
): { display: string; isNan: boolean } {
  const isNan = current === null || !Number.isFinite(current);
  if (isNan && lastFinite !== null && Number.isFinite(lastFinite)) {
    return { display: lastFinite.toFixed(decimals), isNan: true };
  }
  return { display: fmt(current, decimals), isNan };
}

// Compute rolling median for KL spike detection
function rollingMedian(values: number[], window: number): number {
  if (values.length === 0) return 0;
  const slice = values.slice(-window);
  const sorted = [...slice].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

export function StabilityStrip({
  current,
  lastFinite,
  historicalKl,
  gpu,
}: StabilityStripProps) {
  // Derived signals
  const derived = useMemo(() => {
    if (!current) return null;

    // D confusion gap - use normalized probs if available, fallback to raw
    const realProb = current.ctrl_d_real_prob ?? null;
    const fakeProb = current.ctrl_d_fake_prob ?? null;
    // With normalized probs: both near 0.5 = confused
    // Gap from 0.5 matters more than gap between them
    const confusionGap = realProb !== null && fakeProb !== null
      ? Math.max(Math.abs(realProb - 0.5), Math.abs(fakeProb - 0.5))
      : Math.abs((current.d_real_score ?? 0) - (current.d_fake_score ?? 0));

    // KL spike detection (5× rolling median)
    const klMedian = rollingMedian(historicalKl, 200);
    const klCurrent = current.g_loss_kl;
    const klSpike =
      klCurrent !== null &&
      Number.isFinite(klCurrent) &&
      klMedian > 0 &&
      klCurrent / klMedian > 5;

    // Controller active
    const controllerActive =
      (current.ctrl_grad_clip_scale !== null && current.ctrl_grad_clip_scale < 1.0) ||
      (current.ctrl_d_lr_scale !== null && current.ctrl_d_lr_scale < 1.0) ||
      current.ctrl_d_throttle_active === true;

    return { confusionGap, klSpike, controllerActive };
  }, [current, historicalKl]);

  if (!current) {
    return (
      <div style={styles.container}>
        <div style={styles.placeholder}>No metrics data</div>
      </div>
    );
  }

  const nanDetected = current.ctrl_nan_inf_detected === true;
  const alarm = current.ctrl_controller_alarm ?? 'unknown';
  const escalation = current.ctrl_escalation_level;

  // Use lastFinite for display when NaN detected
  const src = nanDetected && lastFinite ? lastFinite : current;

  return (
    <div style={styles.container}>
      {/* Status section */}
      <div style={styles.section}>
        <div style={styles.sectionLabel}>Status</div>
        <div style={styles.row}>
          <span
            style={{
              ...styles.badge,
              backgroundColor: alarmColor(alarm),
              color: alarm === 'healthy' ? '#000' : '#fff',
            }}
          >
            {alarm.toUpperCase()}
          </span>
          {nanDetected && (
            <span style={styles.nanDot} title="NaN/Inf detected">
              ●
            </span>
          )}
          {escalation !== null && escalation > 0 && (
            <span style={styles.escalationBadge}>L{escalation}</span>
          )}
          {derived?.controllerActive && (
            <span style={styles.controllerActiveBadge} title="Controller mitigations active">
              ⚡
            </span>
          )}
        </div>
      </div>

      {/* Core quality section */}
      <div style={styles.section}>
        <div style={styles.sectionLabel}>Quality</div>
        <div style={styles.metricsRow}>
          <MetricValue
            label="mel"
            current={current.g_loss_mel}
            lastFinite={lastFinite?.g_loss_mel ?? null}
            nanDetected={nanDetected}
          />
          <MetricValue
            label="fm"
            current={current.g_loss_fm}
            lastFinite={lastFinite?.g_loss_fm ?? null}
            nanDetected={nanDetected}
          />
          <MetricValue
            label="kl"
            current={current.g_loss_kl}
            lastFinite={lastFinite?.g_loss_kl ?? null}
            nanDetected={nanDetected}
            highlight={derived?.klSpike}
          />
        </div>
      </div>

      {/* Adversarial section */}
      <div style={styles.section}>
        <div style={styles.sectionLabel}>Adversarial</div>
        <div style={styles.metricsRow}>
          <MetricValue
            label="D"
            current={current.loss_d}
            lastFinite={lastFinite?.loss_d ?? null}
            nanDetected={nanDetected}
          />
          <MetricValue
            label="adv"
            current={current.g_loss_adv}
            lastFinite={lastFinite?.g_loss_adv ?? null}
            nanDetected={nanDetected}
          />
          <div style={styles.metric}>
            <span style={styles.metricLabel}>r/f</span>
            <span style={styles.metricValue}>
              {fmt(src.ctrl_d_real_prob ?? src.d_real_score, 2)} / {fmt(src.ctrl_d_fake_prob ?? src.d_fake_score, 2)}
            </span>
            {derived && (
              <span
                style={{
                  ...styles.gapIndicator,
                  color: derived.confusionGap < 0.15 ? '#f97316' : '#6b7280',
                }}
                title={`Distance from 0.5: ${derived.confusionGap.toFixed(3)} (low = confused)`}
              >
                Δ{derived.confusionGap.toFixed(2)}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Controls section */}
      <div style={styles.section}>
        <div style={styles.sectionLabel}>Controls</div>
        <div style={styles.metricsRow}>
          <div style={styles.metric}>
            <span style={styles.metricLabel}>clip</span>
            <span
              style={{
                ...styles.metricValue,
                color: (current.ctrl_grad_clip_scale ?? 1) < 1 ? '#fbbf24' : '#9ca3af',
              }}
            >
              {fmt(current.ctrl_grad_clip_scale, 2)}×
            </span>
          </div>
          <div style={styles.metric}>
            <span style={styles.metricLabel}>D lr</span>
            <span
              style={{
                ...styles.metricValue,
                color: (current.ctrl_d_lr_scale ?? 1) < 1 ? '#fbbf24' : '#9ca3af',
              }}
            >
              {fmt(current.ctrl_d_lr_scale, 2)}×
            </span>
          </div>
          {current.ctrl_d_throttle_active && (
            <div style={styles.metric}>
              <span style={styles.metricLabel}>D skip</span>
              <span style={{ ...styles.metricValue, color: '#fbbf24' }}>
                1/{current.ctrl_d_throttle_every ?? '?'}
              </span>
            </div>
          )}
          <div style={styles.metric}>
            <span style={styles.metricLabel}>alarms</span>
            <span style={styles.metricValueSmall}>
              {current.ctrl_alarms_triggered_total?.toLocaleString() ?? '—'}
            </span>
          </div>
        </div>
      </div>

      {/* Gradients section */}
      <div style={styles.section}>
        <div style={styles.sectionLabel}>Gradients</div>
        <div style={styles.metricsRow}>
          {/* Generator gradients - green */}
          <div style={styles.metric}>
            <span style={{ ...styles.metricLabel, color: '#4ade80' }}>G raw</span>
            <span style={{ ...styles.metricValue, color: '#4ade80' }}>
              {fmt(current.g_grad_norm, 1)}
            </span>
          </div>
          <div style={styles.metric}>
            <span style={{ ...styles.metricLabel, color: '#4ade80' }}>G ema</span>
            <span style={{ ...styles.metricValue, color: '#22c55e' }}>
              {fmt(current.ctrl_ema_grad_g, 1)}
            </span>
          </div>
          <div style={styles.metric}>
            <span style={{ ...styles.metricLabel, color: '#4ade80' }}>G clip</span>
            <span style={{
              ...styles.metricValue,
              color: (current.g_clip_coef ?? 1) < 0.1 ? '#fbbf24' : '#4ade80',
            }}>
              {fmt(current.g_clip_coef, 2)}
            </span>
          </div>
          {/* Discriminator gradients - red */}
          <div style={styles.metric}>
            <span style={{ ...styles.metricLabel, color: '#f87171' }}>D raw</span>
            <span style={{ ...styles.metricValue, color: '#f87171' }}>
              {fmt(current.d_grad_norm, 1)}
            </span>
          </div>
          <div style={styles.metric}>
            <span style={{ ...styles.metricLabel, color: '#f87171' }}>D ema</span>
            <span style={{ ...styles.metricValue, color: '#ef4444' }}>
              {fmt(current.ctrl_ema_grad_d, 1)}
            </span>
          </div>
          <div style={styles.metric}>
            <span style={{ ...styles.metricLabel, color: '#f87171' }}>D clip</span>
            <span style={{
              ...styles.metricValue,
              color: (current.d_clip_coef ?? 1) < 0.1 ? '#fbbf24' : '#f87171',
            }}>
              {fmt(current.d_clip_coef, 2)}
            </span>
          </div>
        </div>
      </div>

      {/* GPU section */}
      {gpu?.available && gpu.gpus.length > 0 && (
        <div style={styles.section}>
          <div style={styles.sectionLabel}>GPU</div>
          <div style={styles.gpuRow}>
            {gpu.gpus.slice(0, 1).map((g) => {
              const memPct = Math.round((g.mem_used_mb / g.mem_total_mb) * 100);
              const tempColor = g.temp_c > 80 ? '#ef4444' : g.temp_c > 70 ? '#fbbf24' : '#4ade80';
              return (
                <div key={g.index} style={styles.gpuMetrics}>
                  <div style={styles.gpuMetric}>
                    <span style={styles.metricLabel}>temp</span>
                    <span style={{ ...styles.metricValue, color: tempColor }}>{g.temp_c}°</span>
                  </div>
                  <div style={styles.gpuMetric}>
                    <span style={styles.metricLabel}>util</span>
                    <div style={styles.miniBarContainer}>
                      <div style={{ ...styles.miniBar, width: `${g.util_pct}%`, backgroundColor: '#22c55e' }} />
                    </div>
                    <span style={styles.metricValueSmall}>{g.util_pct}%</span>
                  </div>
                  <div style={styles.gpuMetric}>
                    <span style={styles.metricLabel}>mem</span>
                    <div style={styles.miniBarContainer}>
                      <div style={{ ...styles.miniBar, width: `${memPct}%`, backgroundColor: memPct > 90 ? '#ef4444' : '#3b82f6' }} />
                    </div>
                    <span style={styles.metricValueSmall}>{memPct}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Step/Epoch */}
      <div style={styles.stepSection}>
        <span style={styles.stepValue}>Step {current.step.toLocaleString()}</span>
        {current.epoch !== null && (
          <span style={styles.epochValue}>Epoch {current.epoch}</span>
        )}
      </div>
    </div>
  );
}

// Individual metric display component
function MetricValue({
  label,
  current,
  lastFinite,
  nanDetected,
  decimals = 2,
  highlight = false,
}: {
  label: string;
  current: number | null;
  lastFinite: number | null;
  nanDetected: boolean;
  decimals?: number;
  highlight?: boolean;
}) {
  const { display, isNan } = fmtWithNan(current, lastFinite, decimals);
  const showNanIndicator = nanDetected && isNan && lastFinite !== null;

  return (
    <div style={styles.metric}>
      <span style={styles.metricLabel}>{label}</span>
      <span
        style={{
          ...styles.metricValue,
          color: highlight ? '#ef4444' : showNanIndicator ? '#6b7280' : '#e0e0e0',
        }}
      >
        {display}
        {highlight && <span style={styles.spikeBadge}>!</span>}
      </span>
      {showNanIndicator && <span style={styles.nanIndicator}>NaN</span>}
    </div>
  );
}

function alarmColor(alarm: string): string {
  switch (alarm) {
    case 'healthy':
      return '#4ade80';
    case 'unstable':
      return '#fbbf24';
    case 'd_dominant':
      return '#f97316';
    case 'g_collapse':
      return '#ef4444';
    case 'emergency':
      return '#dc2626';
    default:
      return '#6b7280';
  }
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: 24,
    padding: '12px 16px',
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    marginBottom: 16,
    flexWrap: 'wrap',
  },
  placeholder: {
    color: '#6b7280',
    fontSize: 13,
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  sectionLabel: {
    fontSize: 10,
    fontWeight: 600,
    color: '#6b7280',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  metricsRow: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 12,
  },
  badge: {
    fontSize: 10,
    fontWeight: 600,
    padding: '2px 8px',
    borderRadius: 4,
  },
  nanDot: {
    color: '#ef4444',
    fontSize: 16,
    lineHeight: 1,
  },
  escalationBadge: {
    fontSize: 10,
    fontWeight: 600,
    padding: '2px 6px',
    borderRadius: 4,
    backgroundColor: '#fbbf24',
    color: '#000',
  },
  controllerActiveBadge: {
    fontSize: 14,
    color: '#fbbf24',
    lineHeight: 1,
  },
  metric: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: 1,
  },
  metricLabel: {
    fontSize: 10,
    color: '#6b7280',
  },
  metricValue: {
    fontSize: 13,
    fontFamily: 'monospace',
    color: '#e0e0e0',
    display: 'flex',
    alignItems: 'center',
    gap: 4,
  },
  metricValueSmall: {
    fontSize: 11,
    fontFamily: 'monospace',
    color: '#9ca3af',
  },
  nanIndicator: {
    fontSize: 9,
    color: '#ef4444',
    fontWeight: 600,
  },
  spikeBadge: {
    fontSize: 10,
    fontWeight: 700,
    color: '#ef4444',
    marginLeft: 2,
  },
  gapIndicator: {
    fontSize: 10,
    fontFamily: 'monospace',
  },
  stepSection: {
    marginLeft: 'auto',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end',
    gap: 2,
  },
  stepValue: {
    fontSize: 14,
    fontWeight: 600,
    color: '#e0e0e0',
    fontFamily: 'monospace',
  },
  epochValue: {
    fontSize: 11,
    color: '#6b7280',
  },
  gpuRow: {
    display: 'flex',
    gap: 8,
  },
  gpuMetrics: {
    display: 'flex',
    gap: 12,
  },
  gpuMetric: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: 2,
  },
  miniBarContainer: {
    width: 40,
    height: 4,
    backgroundColor: '#404040',
    borderRadius: 2,
    overflow: 'hidden',
  },
  miniBar: {
    height: '100%',
    borderRadius: 2,
    transition: 'width 0.3s ease',
  },
};
