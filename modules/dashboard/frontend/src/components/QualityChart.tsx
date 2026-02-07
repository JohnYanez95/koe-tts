/**
 * Quality / Reconstruction chart.
 * Shows mel on secondary axis (primary quality metric), fm+kl on primary axis.
 * Answers: "Is audio fidelity improving?"
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
import type { MetricsPoint, TrainingEvent } from '../api/client';
import type { ZoomRange } from '../pages/RunView';

interface QualityChartProps {
  points: MetricsPoint[];
  events?: TrainingEvent[];
  hasMetrics?: boolean;
  zoomRange?: ZoomRange | null;
  onZoomChange?: (range: ZoomRange | null) => void;
}



// Clamp non-finite values to null
function safeVal(v: number | null): number | null {
  if (v === null || !Number.isFinite(v)) return null;
  return v;
}

export function QualityChart({ points, events, hasMetrics = true, zoomRange, onZoomChange }: QualityChartProps) {
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

  // Compute metrics step range for filtering evals
  const metricsRange = useMemo(() => {
    if (points.length === 0) return { min: 0, max: 0 };
    const steps = points.map((p) => p.step);
    return { min: Math.min(...steps), max: Math.max(...steps) };
  }, [points]);

  // Use all data (update frequency controlled by polling interval)
  const chartData = useMemo(() => {
    let data = points.map((p) => ({
      step: p.step,
      mel: safeVal(p.g_loss_mel) ?? safeVal(p.mel_loss),
      fm: safeVal(p.g_loss_fm),
      kl: safeVal(p.g_loss_kl) ?? safeVal(p.kl_loss),
    }));

    // Filter by zoom range if set
    if (zoomRange) {
      data = data.filter((d) => d.step >= zoomRange.startStep && d.step <= zoomRange.endStep);
    }
    return data;
  }, [points, zoomRange]);

  // Extract checkpoint_saved events with val mel_loss for validation line (filtered to metrics range)
  const valCheckpoints = useMemo(() => {
    if (!events) return [];
    return events
      .filter((e) => e.event === 'checkpoint_saved' && e.mel_loss !== undefined)
      .map((e) => ({
        step: e.step as number,
        val_mel: e.mel_loss as number,
      }))
      .filter((v) => v.step >= metricsRange.min && v.step <= metricsRange.max)
      .sort((a, b) => a.step - b.step);
  }, [events, metricsRange]);

  // Merge validation checkpoints and eval points into chart data for connected lines
  const { chartDataWithVal, evalSteps } = useMemo(() => {
    // Create maps of step -> values
    const valMap = new Map(valCheckpoints.map((v) => [v.step, v.val_mel]));
    const evalMelMap = new Map<number, number>();
    const evalKlMap = new Map<number, number>();

    if (events) {
      events
        .filter((e) => e.event === 'eval_complete' && e.losses)
        .forEach((e) => {
          const losses = e.losses as Record<string, number>;
          const step = e.step as number;
          // Only include evals within metrics range
          if (step >= metricsRange.min && step <= metricsRange.max) {
            if (losses?.mel_loss != null) evalMelMap.set(step, losses.mel_loss);
            if (losses?.kl_loss != null) evalKlMap.set(step, losses.kl_loss);
          }
        });
    }

    const existingSteps = new Set(chartData.map((d) => d.step));

    // Add val_mel and eval losses to chart data points
    const data = chartData.map((d) => ({
      ...d,
      val_mel: valMap.get(d.step) ?? null,
      eval_mel: evalMelMap.get(d.step) ?? null,
      eval_kl: evalKlMap.get(d.step) ?? null,
    }));

    // Add eval points that don't exist in chartData (already filtered to metrics range)
    for (const [step, melLoss] of evalMelMap) {
      if (!existingSteps.has(step)) {
        data.push({
          step,
          mel: null,
          fm: null,
          kl: null,
          val_mel: valMap.get(step) ?? null,
          eval_mel: melLoss,
          eval_kl: evalKlMap.get(step) ?? null,
        });
      }
    }

    // Sort by step
    data.sort((a, b) => a.step - b.step);

    return {
      chartDataWithVal: data,
      evalSteps: Array.from(evalMelMap.keys()),
    };
  }, [chartData, valCheckpoints, events, metricsRange]);

  // Check if we have eval data
  const hasEvalData = evalSteps.length > 0;

  // Check if we have any valid data
  const hasData = useMemo(() => {
    return chartDataWithVal.some((d) => d.mel !== null || d.fm !== null || d.kl !== null);
  }, [chartDataWithVal]);

  // Check if we have validation data
  const hasValData = valCheckpoints.length > 0;

  // Compute mel domain and ticks for right axis (primary quality metric)
  // Tight range: floor(min) to ceil(max) at 0.5 increments
  const { melDomain, melTicks } = useMemo(() => {
    const melValues = chartDataWithVal
      .flatMap((d) => [d.mel, d.val_mel, d.eval_mel])
      .filter((v): v is number => v !== null);
    if (melValues.length === 0) return { melDomain: [0, 2] as [number, number], melTicks: [0, 0.5, 1.0, 1.5, 2.0] };

    const min = Math.min(...melValues);
    const max = Math.max(...melValues);

    // Floor min to nearest 0.5, ceil max to nearest 0.5
    const niceMin = Math.floor(min * 2) / 2;
    const niceMax = Math.ceil(max * 2) / 2;

    // Generate ticks at 0.5 intervals
    const ticks: number[] = [];
    for (let t = niceMin; t <= niceMax; t += 0.5) {
      ticks.push(Math.round(t * 10) / 10); // Avoid float precision issues
    }

    return { melDomain: [niceMin, niceMax] as [number, number], melTicks: ticks };
  }, [chartDataWithVal]);

  // Compute fm+kl domain for left axis (lumped together)
  const fmKlDomain = useMemo(() => {
    const values = chartDataWithVal
      .flatMap((d) => [d.fm, d.kl, d.eval_kl])
      .filter((v): v is number => v !== null);
    if (values.length === 0) return [0, 10];
    const max = Math.max(...values);
    // Cap at reasonable max for display, but allow expansion
    return [0, Math.min(max * 1.1, 100)];
  }, [chartDataWithVal]);

  // Check if KL has exploded (max > 50)
  const klExploded = useMemo(() => {
    const klValues = chartDataWithVal.map((d) => d.kl).filter((v): v is number => v !== null);
    if (klValues.length === 0) return false;
    return Math.max(...klValues) > 50;
  }, [chartDataWithVal]);

  if (chartDataWithVal.length === 0 || !hasData) {
    return (
      <div style={styles.container}>
        <h3 style={styles.title}>Quality Losses</h3>
        <div style={styles.empty}>
          {hasMetrics === false
            ? 'Legacy run - no metrics recorded'
            : 'No quality data'}
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>
        Is audio fidelity improving?
        {klExploded && <span style={styles.warningBadge}>KL elevated</span>}
      </h3>
      <div style={{ ...styles.chartWrapper, userSelect: isSelecting ? 'none' : undefined }}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart
            data={chartDataWithVal}
            margin={{ top: 5, right: 50, bottom: 5, left: 0 }}
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
            {/* Primary Y axis for fm+kl (lumped together) */}
            <YAxis
              yAxisId="left"
              stroke="#666"
              fontSize={11}
              domain={fmKlDomain}
              tickFormatter={(v) => v.toFixed(1)}
            />
            {/* Secondary Y axis for mel (primary quality metric) */}
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#3b82f6"
              fontSize={11}
              domain={melDomain}
              ticks={melTicks}
              tickFormatter={(v) => v.toFixed(1)}
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
                yAxisId="left"
                x1={refAreaLeft}
                x2={refAreaRight}
                strokeOpacity={0.3}
                fill="#3b82f6"
                fillOpacity={0.3}
              />
            )}

            {/* Mel loss - primary quality signal (own axis) */}
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="mel"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="mel"
              connectNulls={false}
            />

            {/* Feature matching loss */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="fm"
              stroke="#22c55e"
              strokeWidth={1.5}
              dot={false}
              name="fm"
              connectNulls={false}
            />

            {/* Validation mel loss - from checkpoint events, connected dots */}
            {hasValData && (
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="val_mel"
                stroke="#f97316"
                strokeWidth={2}
                strokeDasharray="4 2"
                dot={{ r: 4, fill: '#f97316', stroke: '#fff', strokeWidth: 1 }}
                name="val_mel"
                connectNulls={true}
              />
            )}

            {/* KL loss - on primary axis with fm */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="kl"
              stroke="#a855f7"
              strokeWidth={1.5}
              strokeDasharray={klExploded ? '4 2' : undefined}
              dot={false}
              name="kl"
              connectNulls={false}
            />

            {/* Vertical reference lines at eval steps */}
            {evalSteps.map((step) => (
              <ReferenceLine
                key={`eval-ref-${step}`}
                yAxisId="left"
                x={step}
                stroke="#10b981"
                strokeWidth={2}
                strokeDasharray="6 4"
              />
            ))}

            {/* Eval mel loss - large prominent markers on mel axis */}
            {hasEvalData && (
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="eval_mel"
                stroke="#10b981"
                strokeWidth={0}
                dot={{ r: 10, fill: '#10b981', stroke: '#fff', strokeWidth: 3 }}
                activeDot={{ r: 14, fill: '#10b981', stroke: '#fff', strokeWidth: 3 }}
                name="eval_mel"
                connectNulls={false}
                isAnimationActive={false}
              />
            )}

            {/* Eval KL loss - large prominent markers on fm+kl axis */}
            {hasEvalData && (
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="eval_kl"
                stroke="#14b8a6"
                strokeWidth={0}
                dot={{ r: 10, fill: '#14b8a6', stroke: '#fff', strokeWidth: 3 }}
                activeDot={{ r: 14, fill: '#14b8a6', stroke: '#fff', strokeWidth: 3 }}
                name="eval_kl"
                connectNulls={false}
                isAnimationActive={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div style={styles.legend}>
        <span style={styles.legendItem}>
          <span style={{ ...styles.legendLine, backgroundColor: '#3b82f6' }} />
          mel (train, right axis)
        </span>
        {hasValData && (
          <span style={styles.legendItem}>
            <span style={{ ...styles.legendLine, backgroundColor: '#f97316', borderStyle: 'dashed' }} />
            mel (val)
          </span>
        )}
        {hasEvalData && (
          <span style={styles.legendItem}>
            <span style={{ ...styles.legendLine, backgroundColor: '#10b981', borderStyle: 'dashed' }} />
            mel (eval)
          </span>
        )}
        <span style={styles.legendItem}>
          <span style={{ ...styles.legendLine, backgroundColor: '#22c55e' }} />
          fm
        </span>
        <span style={styles.legendItem}>
          <span style={{ ...styles.legendLine, backgroundColor: '#a855f7' }} />
          kl
        </span>
        {hasEvalData && (
          <span style={styles.legendItem}>
            <span style={{ ...styles.legendLine, backgroundColor: '#14b8a6', borderStyle: 'dashed' }} />
            kl (eval)
          </span>
        )}
      </div>
    </div>
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
  warningBadge: {
    fontSize: 10,
    fontWeight: 600,
    padding: '2px 6px',
    borderRadius: 4,
    backgroundColor: '#a855f7',
    color: '#fff',
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
  legend: {
    display: 'flex',
    gap: 16,
    marginTop: 8,
    fontSize: 11,
    color: '#6b7280',
    flexWrap: 'wrap',
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
