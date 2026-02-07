/**
 * D-Score Chart - Discriminator real vs fake probabilities over time.
 * Shows normalized (0-1) probabilities with solid green (real) and pink (fake).
 * Answers: "Is D distinguishing real from fake, or confused?"
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
  ReferenceLine,
  ReferenceArea,
} from 'recharts';
import type { MetricsPoint } from '../api/client';
import type { ZoomRange } from '../pages/RunView';

interface DScoreChartProps {
  points: MetricsPoint[];
  hasMetrics?: boolean;
  zoomRange?: ZoomRange | null;
  onZoomChange?: (range: ZoomRange | null) => void;
}

// Clamp non-finite values to null
function safeVal(v: number | null): number | null {
  if (v === null || !Number.isFinite(v)) return null;
  return v;
}

export function DScoreChart({ points, hasMetrics = true, zoomRange, onZoomChange }: DScoreChartProps) {
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

  // Transform data - use normalized probs if available, fallback to raw scores
  const chartData = useMemo(() => {
    let data = points.map((p) => ({
      step: p.step,
      // Prefer normalized probs (0-1), fallback to raw scores
      d_real: safeVal(p.ctrl_d_real_prob) ?? safeVal(p.d_real_score),
      d_fake: safeVal(p.ctrl_d_fake_prob) ?? safeVal(p.d_fake_score),
      confusion: p.ctrl_d_confusion_active,
    }));

    // Filter by zoom range if set
    if (zoomRange) {
      data = data.filter((d) => d.step >= zoomRange.startStep && d.step <= zoomRange.endStep);
    }
    return data;
  }, [points, zoomRange]);

  // Check if we have any valid score data
  const hasData = useMemo(() => {
    return chartData.some((d) => d.d_real !== null || d.d_fake !== null);
  }, [chartData]);

  // Compute Y-axis domain centered at 0.5 with symmetric bounds
  const { yAxisDomain, yAxisTicks } = useMemo(() => {
    let maxDeviation = 0.1; // Minimum epsilon
    for (const d of chartData) {
      if (d.d_real !== null) {
        maxDeviation = Math.max(maxDeviation, Math.abs(d.d_real - 0.5));
      }
      if (d.d_fake !== null) {
        maxDeviation = Math.max(maxDeviation, Math.abs(d.d_fake - 0.5));
      }
    }
    // Add 10% padding and ensure we don't exceed [0, 1]
    const epsilon = Math.min(0.5, maxDeviation * 1.1);
    const lower = Math.max(0, 0.5 - epsilon);
    const upper = Math.min(1, 0.5 + epsilon);

    // Generate explicit tick values centered at 0.5
    const ticks = [lower, 0.5 - epsilon / 2, 0.5, 0.5 + epsilon / 2, upper];

    return {
      yAxisDomain: [lower, upper] as [number, number],
      yAxisTicks: ticks.map(t => Math.round(t * 100) / 100), // Round to 2 decimal places
    };
  }, [chartData]);

  // Get latest values for status display
  const latest = useMemo(() => {
    const last = chartData[chartData.length - 1];
    if (!last) return null;

    const real = last.d_real;
    const fake = last.d_fake;

    if (real === null || fake === null) return null;

    // Separation: how well D distinguishes (higher = better)
    const separation = Math.abs(real - fake);
    // Confusion: both near 0.5
    const realConfused = Math.abs(real - 0.5) < 0.15;
    const fakeConfused = Math.abs(fake - 0.5) < 0.15;
    const confused = realConfused && fakeConfused;

    return {
      real,
      fake,
      separation,
      confused,
      status: confused ? 'confused' : separation > 0.3 ? 'separating' : 'learning',
    };
  }, [chartData]);

  if (chartData.length === 0 || !hasData) {
    return (
      <div style={styles.container}>
        <h3 style={styles.title}>D Scores</h3>
        <div style={styles.empty}>
          {hasMetrics === false
            ? 'Legacy run - no metrics recorded'
            : 'No discriminator data (pre-GAN stage?)'}
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>
        <span>Can D tell real from fake?</span>
        {latest && (
          <span
            style={{
              ...styles.statusBadge,
              backgroundColor: statusColor(latest.status),
              color: latest.status === 'separating' ? '#000' : '#fff',
            }}
          >
            {latest.status === 'confused' && 'CONFUSED'}
            {latest.status === 'separating' && `Δ${latest.separation.toFixed(2)}`}
            {latest.status === 'learning' && 'learning'}
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
              stroke="#666"
              fontSize={11}
              tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
            />
            <YAxis
              stroke="#666"
              fontSize={11}
              domain={yAxisDomain}
              ticks={yAxisTicks}
              allowDataOverflow={false}
              tickFormatter={(v) => v.toFixed(2)}
            />
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
                  return [value.toFixed(3), name];
                }
                return ['N/A', name];
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />

            {/* Brush selection overlay */}
            {isSelecting && refAreaLeft !== null && refAreaRight !== null && (
              <ReferenceArea
                x1={refAreaLeft}
                x2={refAreaRight}
                strokeOpacity={0.3}
                fill="#3b82f6"
                fillOpacity={0.3}
              />
            )}

            {/* Reference line at 0.5 (confusion zone) */}
            <ReferenceLine
              y={0.5}
              stroke="#fbbf24"
              strokeDasharray="4 4"
              strokeWidth={1}
              strokeOpacity={0.5}
            />

            {/* D(real) - solid green */}
            <Line
              type="monotone"
              dataKey="d_real"
              stroke="#4ade80"
              strokeWidth={2}
              dot={false}
              name="D(real)"
              connectNulls={false}
            />

            {/* D(fake) - solid pink */}
            <Line
              type="monotone"
              dataKey="d_fake"
              stroke="#f472b6"
              strokeWidth={2}
              dot={false}
              name="D(fake)"
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div style={styles.footer}>
        <span style={styles.hint}>
          Ideal: real → 1.0, fake → 0.0 | Confused: both → 0.5
        </span>
        {latest && (
          <span style={styles.values}>
            <span style={{ color: '#4ade80' }}>R: {latest.real.toFixed(2)}</span>
            {' / '}
            <span style={{ color: '#f472b6' }}>F: {latest.fake.toFixed(2)}</span>
          </span>
        )}
      </div>
    </div>
  );
}

function statusColor(status: string): string {
  switch (status) {
    case 'separating':
      return '#4ade80';
    case 'learning':
      return '#3b82f6';
    case 'confused':
      return '#f97316';
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
  statusBadge: {
    fontSize: 10,
    fontWeight: 600,
    padding: '2px 6px',
    borderRadius: 4,
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
    fontSize: 11,
  },
  hint: {
    color: '#6b7280',
  },
  values: {
    fontFamily: 'monospace',
    color: '#9ca3af',
  },
};
