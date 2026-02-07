/**
 * GAN Balance chart.
 * Shows adversarial losses and discriminator scores.
 * Answers: "Is D overpowering G, or is the game balanced?"
 */

import { useMemo, useState, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceArea,
} from 'recharts';
import type { MetricsPoint } from '../api/client';
import type { ZoomRange } from '../pages/RunView';

interface GanBalanceChartProps {
  hasMetrics?: boolean;
  points: MetricsPoint[];
  zoomRange?: ZoomRange | null;
  onZoomChange?: (range: ZoomRange | null) => void;
}

// Downsample data for chart performance

// Clamp non-finite values to null
function safeVal(v: number | null): number | null {
  if (v === null || !Number.isFinite(v)) return null;
  return v;
}

export function GanBalanceChart({ points, hasMetrics = true, zoomRange, onZoomChange }: GanBalanceChartProps) {
  const [showScores, setShowScores] = useState(false);

  // Brush selection state for drag-to-zoom
  const [refAreaLeft, setRefAreaLeft] = useState<number | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<number | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);

  const handleMouseDown = useCallback((e: { activeLabel?: string }) => {
    if (e.activeLabel && onZoomChange) {
      setRefAreaLeft(parseInt(e.activeLabel));
      setIsSelecting(true);
    }
  }, [onZoomChange]);

  const handleMouseMove = useCallback((e: { activeLabel?: string }) => {
    if (isSelecting && e.activeLabel) {
      setRefAreaRight(parseInt(e.activeLabel));
    }
  }, [isSelecting]);

  const handleMouseUp = useCallback(() => {
    if (refAreaLeft !== null && refAreaRight !== null && refAreaLeft !== refAreaRight && onZoomChange) {
      const startStep = Math.min(refAreaLeft, refAreaRight);
      const endStep = Math.max(refAreaLeft, refAreaRight);
      onZoomChange({ startStep, endStep });
    }
    setRefAreaLeft(null);
    setRefAreaRight(null);
    setIsSelecting(false);
  }, [refAreaLeft, refAreaRight, onZoomChange]);

  // Transform data with optional zoom filtering
  // Use normalized probabilities (0-1) if available, fallback to raw scores
  const chartData = useMemo(() => {
    let data = points.map((p) => ({
      step: p.step,
      loss_d: safeVal(p.loss_d),
      g_loss_adv: safeVal(p.g_loss_adv),
      // Prefer normalized probs (0-1), fallback to raw scores for old data
      d_real: safeVal(p.ctrl_d_real_prob) ?? safeVal(p.d_real_score),
      d_fake: safeVal(p.ctrl_d_fake_prob) ?? safeVal(p.d_fake_score),
    }));

    // Filter by zoom range if set
    if (zoomRange) {
      data = data.filter((d) => d.step >= zoomRange.startStep && d.step <= zoomRange.endStep);
    }
    return data;
  }, [points, zoomRange]);

  // Check if we have any valid data
  const hasData = useMemo(() => {
    return chartData.some((d) => d.loss_d !== null || d.g_loss_adv !== null);
  }, [chartData]);

  // Check if we have score data
  const hasScores = useMemo(() => {
    return chartData.some((d) => d.d_real !== null || d.d_fake !== null);
  }, [chartData]);

  // Compute balance indicator from latest point
  const balanceState = useMemo(() => {
    const latest = chartData[chartData.length - 1];
    if (!latest) return null;

    const lossD = latest.loss_d;
    const advG = latest.g_loss_adv;

    if (lossD === null || advG === null) return null;

    // Heuristic interpretation:
    // - D loss very low + adv high → D crushing G
    // - D loss high + adv low → D weak, G may be cheating
    // - Both moderate and stable → balanced
    if (lossD < 5 && advG > 20) return { state: 'd_strong', label: 'D dominant' };
    if (lossD > 30 && advG < 5) return { state: 'g_strong', label: 'G dominant' };
    return { state: 'balanced', label: 'Balanced' };
  }, [chartData]);

  if (chartData.length === 0 || !hasData) {
    return (
      <div style={styles.container}>
        <h3 style={styles.title}>GAN Balance</h3>
        <div style={styles.empty}>
          {hasMetrics === false
            ? 'Legacy run - no metrics recorded'
            : 'No adversarial data'}
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>
        <span>Is D overpowering G?</span>
        {balanceState && (
          <span
            style={{
              ...styles.balanceBadge,
              backgroundColor: balanceColor(balanceState.state),
            }}
          >
            {balanceState.label}
          </span>
        )}
      </h3>
      <div style={{ ...styles.chartWrapper, userSelect: isSelecting ? 'none' : undefined }}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, bottom: 5, left: 0 }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis
              dataKey="step"
              type="number"
              domain={['dataMin', 'dataMax']}
              allowDataOverflow
              stroke="#666"
              fontSize={11}
              tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
            />
            <YAxis
              yAxisId="left"
              stroke="#666"
              fontSize={11}
              domain={[0, 'auto']}
            />
            {showScores && (
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="#9ca3af"
                fontSize={11}
                domain={[0, 1]}
                tickFormatter={(v) => v.toFixed(1)}
              />
            )}
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1a1a',
                border: '1px solid #333',
                borderRadius: 4,
                fontSize: 12,
              }}
              labelStyle={{ color: '#9ca3af' }}
              formatter={(value, name) => {
                if (typeof value === 'number' && Number.isFinite(value)) {
                  return [value.toFixed(2), name];
                }
                return ['N/A', name];
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />

            {/* Brush selection overlay */}
            {isSelecting && refAreaLeft !== null && refAreaRight !== null && (
              <ReferenceArea
                yAxisId="left"
                x1={refAreaLeft}
                x2={refAreaRight}
                strokeOpacity={0.3}
                fill="#3b82f6"
                fillOpacity={0.3}
              />
            )}

            {/* Discriminator loss - orange */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="loss_d"
              stroke="#f97316"
              strokeWidth={2}
              dot={false}
              name="D loss"
              connectNulls={false}
            />

            {/* Generator adversarial loss - blue */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="g_loss_adv"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="G adv"
              connectNulls={false}
            />

            {/* D scores (optional, on right axis) */}
            {showScores && (
              <>
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="d_real"
                  stroke="#4ade80"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  dot={false}
                  name="D(real)"
                  connectNulls={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="d_fake"
                  stroke="#f472b6"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  dot={false}
                  name="D(fake)"
                  connectNulls={false}
                />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div style={styles.footer}>
        <div style={styles.legend}>
          <span style={styles.legendItem}>
            <span style={{ ...styles.legendLine, backgroundColor: '#f97316' }} />
            D loss
          </span>
          <span style={styles.legendItem}>
            <span style={{ ...styles.legendLine, backgroundColor: '#3b82f6' }} />
            G adversarial
          </span>
        </div>
        {hasScores && (
          <button
            onClick={() => setShowScores(!showScores)}
            style={styles.toggleButton}
          >
            {showScores ? 'Hide' : 'Show'} D scores
          </button>
        )}
      </div>
    </div>
  );
}

function balanceColor(state: string): string {
  switch (state) {
    case 'd_strong':
      return '#f97316';
    case 'g_strong':
      return '#3b82f6';
    case 'balanced':
      return '#4ade80';
    default:
      return '#6b7280';
  }
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    padding: 16,
  },
  title: {
    margin: '0 0 8px 0',
    fontSize: 13,
    fontWeight: 500,
    color: '#9ca3af',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  balanceBadge: {
    fontSize: 10,
    fontWeight: 600,
    padding: '2px 6px',
    borderRadius: 4,
    color: '#000',
  },
  chartWrapper: {
    marginTop: 4,
  },
  empty: {
    color: '#6b7280',
    fontSize: 13,
    textAlign: 'center',
    padding: 40,
  },
  footer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
  },
  legend: {
    display: 'flex',
    gap: 16,
    fontSize: 11,
    color: '#6b7280',
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
  },
  legendLine: {
    width: 16,
    height: 2,
  },
  toggleButton: {
    fontSize: 10,
    padding: '3px 8px',
    borderRadius: 4,
    border: '1px solid #333',
    backgroundColor: 'transparent',
    color: '#9ca3af',
    cursor: 'pointer',
  },
};
