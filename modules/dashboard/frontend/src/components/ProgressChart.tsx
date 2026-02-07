/**
 * Progress chart with event markers.
 * Shows main loss trend with eval/checkpoint markers overlaid.
 * Answers: "Is the run progressing, and did evals improve outcomes?"
 */

import { useMemo, useState, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from 'recharts';
import type { MetricsPoint, TrainingEvent } from '../api/client';
import type { ZoomRange } from '../pages/RunView';

interface ProgressChartProps {
  points: MetricsPoint[];
  events: TrainingEvent[];
  onMarkerClick?: (eventTs: string) => void;
  hasMetrics?: boolean;
  zoomRange?: ZoomRange | null;
  onZoomChange?: (range: ZoomRange | null) => void;
}

// Downsample data for chart performance

// Clamp non-finite values to null
function safeVal(v: number | null): number | null {
  if (v === null || !Number.isFinite(v)) return null;
  return v;
}

// Tags that indicate meaningful checkpoints (not periodic saves)
const MEANINGFUL_CHECKPOINT_TAGS = [
  'manual',
  'eval',
  'last_known_good',
  'emergency',
  'best',
  'final',
  'thermal',
  'stop',
];

function isMeaningfulCheckpoint(event: TrainingEvent): boolean {
  const tag = event.tag as string | undefined;
  if (!tag) return false;
  return MEANINGFUL_CHECKPOINT_TAGS.some((t) => tag.toLowerCase().includes(t));
}

interface EventMarker {
  step: number;
  type: 'eval_complete' | 'eval_failed' | 'checkpoint' | 'emergency';
  label: string;
  color: string;
  ts: string; // Event timestamp for linking to EventsTimeline
  details?: Record<string, unknown>;
}

export function ProgressChart({ points, events, onMarkerClick, hasMetrics = true, zoomRange, onZoomChange }: ProgressChartProps) {
  // Brush selection state (for drag-to-zoom)
  const [refAreaLeft, setRefAreaLeft] = useState<number | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<number | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);

  // Compute metrics step range for filtering markers
  const metricsRange = useMemo(() => {
    if (points.length === 0) return { min: 0, max: 0 };
    const steps = points.map((p) => p.step);
    return { min: Math.min(...steps), max: Math.max(...steps) };
  }, [points]);

  // Transform data (update frequency controlled by polling interval)
  const chartData = useMemo(() => {
    const data = points.map((p) => ({
      step: p.step,
      loss: safeVal(p.loss_g) ?? safeVal(p.g_loss_mel) ?? safeVal(p.mel_loss),
    }));

    // Filter by zoom range if set
    if (zoomRange) {
      return data.filter((d) => d.step >= zoomRange.startStep && d.step <= zoomRange.endStep);
    }
    return data;
  }, [points, zoomRange]);

  // Handle brush selection start
  const handleMouseDown = useCallback((e: { activeLabel?: string }) => {
    if (e.activeLabel) {
      setRefAreaLeft(parseInt(e.activeLabel));
      setIsSelecting(true);
    }
  }, []);

  // Handle brush selection move
  const handleMouseMove = useCallback((e: { activeLabel?: string }) => {
    if (isSelecting && e.activeLabel) {
      setRefAreaRight(parseInt(e.activeLabel));
    }
  }, [isSelecting]);

  // Handle brush selection end
  const handleMouseUp = useCallback(() => {
    if (refAreaLeft !== null && refAreaRight !== null && refAreaLeft !== refAreaRight) {
      const startStep = Math.min(refAreaLeft, refAreaRight);
      const endStep = Math.max(refAreaLeft, refAreaRight);
      onZoomChange?.({ startStep, endStep });
    }
    setRefAreaLeft(null);
    setRefAreaRight(null);
    setIsSelecting(false);
  }, [refAreaLeft, refAreaRight, onZoomChange]);

  // Extract event markers (filtered to metrics range)
  const markers = useMemo((): EventMarker[] => {
    const result: EventMarker[] = [];

    for (const e of events) {
      const step = e.step as number | undefined;
      if (step === undefined) continue;
      // Skip events outside the metrics range
      if (step < metricsRange.min || step > metricsRange.max) continue;

      if (e.event === 'eval_complete') {
        result.push({
          step,
          type: 'eval_complete',
          label: `Eval ${e.eval_id ?? ''}`,
          color: '#22c55e',
          ts: e.ts,
          details: {
            eval_id: e.eval_id,
            inter_speaker: e.inter_speaker,
            silence_pct: e.silence_pct,
            valid_count: e.valid_count,
          },
        });
      } else if (e.event === 'eval_failed') {
        result.push({
          step,
          type: 'eval_failed',
          label: `Eval failed`,
          color: '#ef4444',
          ts: e.ts,
          details: { error: e.error },
        });
      } else if (e.event === 'checkpoint_saved' && isMeaningfulCheckpoint(e)) {
        result.push({
          step,
          type: 'checkpoint',
          label: `Ckpt: ${e.tag ?? 'saved'}`,
          color: '#6b7280',
          ts: e.ts,
          details: { tag: e.tag, path: e.path },
        });
      } else if (
        e.event === 'training_complete' &&
        e.status === 'emergency_stop'
      ) {
        result.push({
          step,
          type: 'emergency',
          label: 'Emergency Stop',
          color: '#dc2626',
          ts: e.ts,
          details: { reason: e.reason, checkpoint_path: e.checkpoint_path },
        });
      }
    }

    return result;
  }, [events, metricsRange]);

  // Check if we have any valid data
  const hasData = useMemo(() => {
    return chartData.some((d) => d.loss !== null);
  }, [chartData]);

  // Get step range for marker display
  const stepRange = useMemo(() => {
    if (chartData.length === 0) return { min: 0, max: 100000 };
    return {
      min: chartData[0].step,
      max: chartData[chartData.length - 1].step,
    };
  }, [chartData]);

  // Filter markers to visible range
  const visibleMarkers = useMemo(() => {
    return markers.filter(
      (m) => m.step >= stepRange.min && m.step <= stepRange.max
    );
  }, [markers, stepRange]);

  if (chartData.length === 0 || !hasData) {
    return (
      <div style={styles.container}>
        <h3 style={styles.title}>Training Progress</h3>
        <div style={styles.empty}>
          {hasMetrics === false
            ? 'Legacy run - no metrics recorded'
            : 'No loss data'}
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>
        Training Progress
        {visibleMarkers.length > 0 && (
          <span style={styles.markerCount}>
            {visibleMarkers.filter((m) => m.type === 'eval_complete').length} evals
          </span>
        )}
      </h3>
      <div style={styles.chartWrapper}>
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
            <YAxis
              stroke="#666"
              fontSize={11}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1a1a',
                border: '1px solid #333',
                borderRadius: 4,
                fontSize: 12,
              }}
              labelStyle={{ color: '#9ca3af' }}
              formatter={(value) => {
                if (typeof value === 'number' && Number.isFinite(value)) {
                  return [value.toFixed(2), 'Loss'];
                }
                return ['N/A', 'Loss'];
              }}
            />

            {/* Selection overlay for brush zoom */}
            {isSelecting && refAreaLeft !== null && refAreaRight !== null && (
              <ReferenceArea
                x1={refAreaLeft}
                x2={refAreaRight}
                strokeOpacity={0.3}
                fill="#3b82f6"
                fillOpacity={0.3}
              />
            )}

            {/* Event markers as reference lines */}
            {visibleMarkers.map((marker, i) => (
              <ReferenceLine
                key={`${marker.type}-${marker.step}-${i}`}
                x={marker.step}
                stroke={marker.color}
                strokeWidth={marker.type === 'emergency' ? 2 : 1}
                strokeDasharray={marker.type === 'checkpoint' ? '2 2' : undefined}
              />
            ))}

            {/* Main loss line */}
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="Loss"
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Event markers legend */}
      {visibleMarkers.length > 0 && (
        <div style={styles.markersLegend}>
          {visibleMarkers.slice(-5).map((marker, i) => (
            <MarkerBadge
              key={i}
              marker={marker}
              onClick={onMarkerClick ? () => onMarkerClick(marker.ts) : undefined}
            />
          ))}
          {visibleMarkers.length > 5 && (
            <span style={styles.moreMarkers}>
              +{visibleMarkers.length - 5} more
            </span>
          )}
        </div>
      )}
    </div>
  );
}

interface MarkerBadgeProps {
  marker: EventMarker;
  onClick?: () => void;
}

function MarkerBadge({ marker, onClick }: MarkerBadgeProps) {
  const stepK = (marker.step / 1000).toFixed(1);

  // Build tooltip content
  let tooltip = `Step ${marker.step.toLocaleString()}`;
  if (marker.details) {
    if (marker.details.eval_id) tooltip += `\nEval: ${marker.details.eval_id}`;
    if (marker.details.inter_speaker !== undefined)
      tooltip += `\nInter-speaker: ${marker.details.inter_speaker}`;
    if (marker.details.silence_pct !== undefined)
      tooltip += `\nSilence: ${marker.details.silence_pct}%`;
    if (marker.details.tag) tooltip += `\nTag: ${marker.details.tag}`;
    if (marker.details.reason) tooltip += `\nReason: ${marker.details.reason}`;
  }
  if (onClick) tooltip += '\nClick to jump to event';

  return (
    <span
      style={{
        ...styles.markerBadge,
        borderColor: marker.color,
        color: marker.color,
        cursor: onClick ? 'pointer' : 'default',
      }}
      title={tooltip}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => e.key === 'Enter' && onClick() : undefined}
    >
      {marker.type === 'eval_complete' && '✓'}
      {marker.type === 'eval_failed' && '✗'}
      {marker.type === 'checkpoint' && '⬤'}
      {marker.type === 'emergency' && '!'}
      {' '}
      {stepK}k
    </span>
  );
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
  markerCount: {
    fontSize: 10,
    color: '#6b7280',
    fontWeight: 400,
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
  markersLegend: {
    display: 'flex',
    gap: 8,
    marginTop: 8,
    flexWrap: 'wrap',
    alignItems: 'center',
  },
  markerBadge: {
    fontSize: 10,
    padding: '2px 6px',
    borderRadius: 4,
    border: '1px solid',
    backgroundColor: 'transparent',
    fontFamily: 'monospace',
  },
  moreMarkers: {
    fontSize: 10,
    color: '#6b7280',
  },
};
