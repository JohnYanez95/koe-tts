/**
 * Gradient norms chart with log scale and threshold reference lines.
 * Provides early warning for training instability.
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

interface GradNormChartProps {
  points: MetricsPoint[];
  absLimitG?: number;
  absLimitD?: number;
  softLimitG?: number;
  softLimitD?: number;
  hasMetrics?: boolean;
  zoomRange?: ZoomRange | null;
  onZoomChange?: (range: ZoomRange | null) => void;
}

// Default thresholds matching gan_controller.py defaults
const DEFAULT_ABS_LIMIT = 5000;
const DEFAULT_SOFT_LIMIT = 2000;
const EPS = 0.01;

// Clamp for log scale: null/NaN/<=0 -> null (skip point)
function clampLog(v: number | null): number | null {
  if (v === null) return null;
  if (!Number.isFinite(v)) return null;
  if (v <= 0) return null;
  return Math.max(v, EPS);
}


export function GradNormChart({
  points,
  absLimitG = DEFAULT_ABS_LIMIT,
  // absLimitD and softLimitD available for future use with separate D thresholds
  absLimitD: _absLimitD = DEFAULT_ABS_LIMIT,
  softLimitG = DEFAULT_SOFT_LIMIT,
  softLimitD: _softLimitD = DEFAULT_SOFT_LIMIT,
  hasMetrics = true,
  zoomRange,
  onZoomChange,
}: GradNormChartProps) {
  // Suppress unused variable warnings (reserved for separate G/D threshold lines)
  void _absLimitD;
  void _softLimitD;

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
  // Layered approach: raw is base, clipped/EMA overlay when active
  const chartData = useMemo(() => {
    let data = points.map((p) => {
      const gClipCoef = p.g_clip_coef ?? 1;
      const dClipCoef = p.d_clip_coef ?? 1;
      const gRaw = p.g_grad_norm;
      const dRaw = p.d_grad_norm;
      const escalation = p.ctrl_escalation_level ?? 0;

      // Effective gradient = raw * clip_coef (what actually updates weights)
      const gEffective = gRaw !== null ? gRaw * gClipCoef : null;
      const dEffective = dRaw !== null ? dRaw * dClipCoef : null;

      // State flags for this point
      const gClipping = gClipCoef < 0.95; // Clipping active when <95%
      const dClipping = dClipCoef < 0.95;
      const emaActive = escalation > 0;

      const gRawClamped = clampLog(gRaw);
      const dRawClamped = clampLog(dRaw);
      const gEffClamped = clampLog(gEffective);
      const dEffClamped = clampLog(dEffective);
      const emaGClamped = clampLog(p.ctrl_ema_grad_g);
      const emaDClamped = clampLog(p.ctrl_ema_grad_d);

      return {
        step: p.step,
        // Raw gradients - ALWAYS shown as base layer
        g_raw: gRawClamped,
        d_raw: dRawClamped,
        // Clipped gradients - only shown when clipping is active (overlays raw)
        g_clipped_overlay: gClipping ? gEffClamped : null,
        d_clipped_overlay: dClipping ? dEffClamped : null,
        // EMA - only shown when escalation active (overlays raw)
        ema_g_overlay: emaActive ? emaGClamped : null,
        ema_d_overlay: emaActive ? emaDClamped : null,
        // Keep full values for domain calculation
        g_eff: gEffClamped,
        d_eff: dEffClamped,
        ema_g: emaGClamped,
        ema_d: emaDClamped,
        // Metadata for tooltip
        g_clip: gClipCoef,
        d_clip: dClipCoef,
        g_clipping: gClipping,
        d_clipping: dClipping,
        alarm: p.ctrl_controller_alarm,
        escalation,
      };
    });

    // Filter by zoom range if set
    if (zoomRange) {
      data = data.filter((d) => d.step >= zoomRange.startStep && d.step <= zoomRange.endStep);
    }
    return data;
  }, [points, zoomRange]);

  // Check if we have any valid grad data
  const hasGradData = useMemo(() => {
    return chartData.some((d) => d.g_raw !== null || d.d_raw !== null);
  }, [chartData]);

  // Check if we have D grad data (GAN stage)
  const hasDGrad = useMemo(() => {
    return chartData.some((d) => d.d_raw !== null);
  }, [chartData]);

  // Compute Y-axis domain from all visible gradient data
  const yAxisDomain = useMemo(() => {
    const values: number[] = [];
    for (const d of chartData) {
      // Include all gradient values for proper scaling
      if (d.g_raw !== null) values.push(d.g_raw);
      if (d.d_raw !== null) values.push(d.d_raw);
      if (d.g_eff !== null) values.push(d.g_eff);
      if (d.d_eff !== null) values.push(d.d_eff);
      if (d.ema_g !== null) values.push(d.ema_g);
      if (d.ema_d !== null) values.push(d.ema_d);
    }
    if (values.length === 0) return [EPS, 10000];
    const min = Math.max(EPS, Math.min(...values) * 0.8); // 20% padding below
    const max = Math.max(...values) * 1.2; // 20% padding above
    return [min, max];
  }, [chartData]);

  // Get current state for legend display (lines are state-aware per-point)
  const currentState = useMemo(() => {
    const latest = chartData[chartData.length - 1];
    return {
      gClipping: latest?.g_clipping ?? false,
      dClipping: latest?.d_clipping ?? false,
      escalation: latest?.escalation ?? 0,
    };
  }, [chartData]);

  // Get latest alarm state for display
  const latestAlarm = chartData[chartData.length - 1]?.alarm ?? 'unknown';
  const latestEscalation = chartData[chartData.length - 1]?.escalation ?? 0;

  if (chartData.length === 0 || !hasGradData) {
    return (
      <div style={styles.container}>
        <h3 style={styles.title}>Gradient Norms</h3>
        <div style={styles.empty}>
          {hasMetrics === false
            ? 'Legacy run - no metrics recorded'
            : 'No gradient data'}
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>
        <span style={styles.titleLeft}>
          Are gradients under control?
          {latestEscalation !== null && latestEscalation > 0 && (
            <span style={styles.escalationBadge}>L{latestEscalation}</span>
          )}
        </span>
        <span style={styles.alarmIndicator}>
          <span
            style={{
              ...styles.alarmDot,
              backgroundColor: alarmColor(latestAlarm),
            }}
          />
          {latestAlarm}
        </span>
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
              allowDataOverflow
            />
            {/* Single Y-axis for all gradients (log scale) */}
            <YAxis
              yAxisId="left"
              stroke="#666"
              fontSize={11}
              scale="log"
              domain={yAxisDomain}
              allowDataOverflow
              tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0))}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1a1a',
                border: '1px solid #333',
                borderRadius: 4,
                fontSize: 12,
              }}
              labelStyle={{ color: '#9ca3af' }}
              formatter={(value, name, props) => {
                if (typeof value === 'number' && Number.isFinite(value)) {
                  const formatted = value.toFixed(1);
                  // Add clip coefficient info for clipped values
                  if (name === 'G clipped' && props.payload?.g_clip < 0.99) {
                    return [`${formatted} (×${props.payload.g_clip.toFixed(2)})`, name];
                  }
                  if (name === 'D clipped' && props.payload?.d_clip < 0.99) {
                    return [`${formatted} (×${props.payload.d_clip.toFixed(2)})`, name];
                  }
                  return [formatted, name];
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

            {/* Absolute limit (emergency threshold) - gray dashed, on left axis */}
            <ReferenceLine
              yAxisId="left"
              y={absLimitG}
              stroke="#6b7280"
              strokeDasharray="4 4"
              strokeWidth={2}
            />

            {/* Soft limit (decay threshold) - yellow dashed, on left axis */}
            <ReferenceLine
              yAxisId="left"
              y={softLimitG}
              stroke="#fbbf24"
              strokeDasharray="2 2"
              strokeWidth={1}
            />

            {/* Generator - BLUE family (layered: raw base, clipped/EMA overlay) */}
            {/* Layer 1: G raw - always visible as dim base */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="g_raw"
              stroke="#93c5fd"
              strokeWidth={1.5}
              strokeOpacity={0.6}
              dot={false}
              name="G raw"
              connectNulls={false}
            />
            {/* Layer 2: G clipped - bright overlay when clipping active */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="g_clipped_overlay"
              stroke="#3b82f6"
              strokeWidth={2.5}
              dot={false}
              name="G clipped"
              connectNulls={false}
            />
            {/* Layer 3: G EMA - dashed overlay when escalation active */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="ema_g_overlay"
              stroke="#60a5fa"
              strokeWidth={2.5}
              strokeDasharray="6 3"
              dot={false}
              name="G EMA"
              connectNulls={false}
            />

            {/* Discriminator - ORANGE family (layered: raw base, clipped/EMA overlay) */}
            {hasDGrad && (
              <>
                {/* Layer 1: D raw - always visible as dim base */}
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="d_raw"
                  stroke="#fdba74"
                  strokeWidth={1.5}
                  strokeOpacity={0.6}
                  dot={false}
                  name="D raw"
                  connectNulls={false}
                />
                {/* Layer 2: D clipped - bright overlay when clipping active */}
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="d_clipped_overlay"
                  stroke="#f97316"
                  strokeWidth={2.5}
                  dot={false}
                  name="D clipped"
                  connectNulls={false}
                />
                {/* Layer 3: D EMA - dashed overlay when escalation active */}
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="ema_d_overlay"
                  stroke="#fb923c"
                  strokeWidth={2.5}
                  strokeDasharray="6 3"
                  dot={false}
                  name="D EMA"
                  connectNulls={false}
                />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {/* Legend: dim=raw base, bright=clipped overlay, dashed=EMA overlay */}
      <div style={styles.thresholdLegend}>
        <span style={styles.legendItem}>
          <span style={{ ...styles.legendLine, backgroundColor: '#93c5fd', opacity: 0.6 }} />
          raw (base)
        </span>
        <span style={styles.legendItem}>
          <span style={{ ...styles.legendLine, backgroundColor: '#3b82f6' }} />
          clipped {currentState.gClipping ? '(active)' : ''}
        </span>
        {currentState.escalation > 0 && (
          <span style={styles.legendItem}>
            <span style={{ ...styles.legendLine, backgroundColor: '#60a5fa', borderStyle: 'dashed' }} />
            EMA (L{currentState.escalation})
          </span>
        )}
        <span style={{ ...styles.legendItem, marginLeft: 'auto', color: '#6b7280', fontSize: 10 }}>
          soft: {softLimitG.toLocaleString()} | abs: {absLimitG.toLocaleString()}
        </span>
      </div>
    </div>
  );
}

function alarmColor(alarm: string | null): string {
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
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  titleLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  escalationBadge: {
    fontSize: 10,
    fontWeight: 600,
    padding: '2px 6px',
    borderRadius: 4,
    backgroundColor: '#fbbf24',
    color: '#000',
    textTransform: 'none',
    letterSpacing: 0,
  },
  alarmIndicator: {
    fontSize: 12,
    fontWeight: 400,
    color: '#6b7280',
    textTransform: 'none',
    letterSpacing: 0,
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  alarmDot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    display: 'inline-block',
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
  thresholdLegend: {
    display: 'flex',
    gap: 16,
    marginTop: 8,
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
};
