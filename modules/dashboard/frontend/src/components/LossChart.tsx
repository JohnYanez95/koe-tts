/**
 * Loss chart component using Recharts.
 */

import { useMemo } from 'react';
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
} from 'recharts';
import type { MetricsPoint, TrainingEvent } from '../api/client';
import type { ZoomRange } from '../pages/RunView';

interface LossChartProps {
  metrics: MetricsPoint[];
  stage: string;
  events?: TrainingEvent[];
  hasMetrics?: boolean; // False for legacy runs without metrics.jsonl
  zoomRange?: ZoomRange | null;
}


export function LossChart({ metrics, stage, events, hasMetrics = true, zoomRange }: LossChartProps) {
  // Extract eval losses map for merging
  const evalMaps = useMemo(() => {
    const melMap = new Map<number, number>();
    const klMap = new Map<number, number>();
    const durMap = new Map<number, number>();

    if (events) {
      events
        .filter((e) => e.event === 'eval_complete' && e.losses)
        .forEach((e) => {
          const losses = e.losses as Record<string, number>;
          const step = e.step as number;
          if (losses?.mel_loss != null) melMap.set(step, losses.mel_loss);
          if (losses?.kl_loss != null) klMap.set(step, losses.kl_loss);
          if (losses?.dur_loss != null) durMap.set(step, losses.dur_loss);
        });
    }

    return { melMap, klMap, durMap };
  }, [events]);

  // Compute metrics step range for filtering evals
  const metricsRange = useMemo(() => {
    if (metrics.length === 0) return { min: 0, max: 0 };
    const steps = metrics.map((m) => m.step);
    return { min: Math.min(...steps), max: Math.max(...steps) };
  }, [metrics]);

  // Use all metrics data (update frequency controlled by polling interval)
  const chartData = useMemo(() => {
    const existingSteps = new Set(metrics.map((m) => m.step));

    // Build base data from metrics
    const data = metrics.map((m) => ({
      step: m.step,
      mel: m.g_loss_mel ?? m.mel_loss ?? null,
      kl: m.g_loss_kl ?? m.kl_loss ?? null,
      dur: m.g_loss_dur ?? m.dur_loss ?? null,
      adv: m.g_loss_adv ?? null,
      fm: m.g_loss_fm ?? null,
      d_loss: m.loss_d ?? null,
      eval_mel: evalMaps.melMap.get(m.step) ?? null,
      eval_kl: evalMaps.klMap.get(m.step) ?? null,
      eval_dur: evalMaps.durMap.get(m.step) ?? null,
    }));

    // Add eval points that weren't in the metrics data (only if within metrics range)
    for (const [step, melLoss] of evalMaps.melMap) {
      if (!existingSteps.has(step) && step >= metricsRange.min && step <= metricsRange.max) {
        data.push({
          step,
          mel: null,
          kl: null,
          dur: null,
          adv: null,
          fm: null,
          d_loss: null,
          eval_mel: melLoss,
          eval_kl: evalMaps.klMap.get(step) ?? null,
          eval_dur: evalMaps.durMap.get(step) ?? null,
        });
      }
    }

    // Sort by step to maintain order
    data.sort((a, b) => a.step - b.step);

    // Filter by zoom range if set
    if (zoomRange) {
      return data.filter((d) => d.step >= zoomRange.startStep && d.step <= zoomRange.endStep);
    }
    return data;
  }, [metrics, evalMaps, zoomRange, metricsRange]);

  // Check if we have eval data and get eval steps for reference lines (filtered to metrics range)
  const { hasEvalData, evalSteps } = useMemo(() => {
    if (!events) return { hasEvalData: false, evalSteps: [] as number[] };
    const steps = events
      .filter((e) => e.event === 'eval_complete' && e.losses)
      .map((e) => e.step as number)
      .filter((step) => step >= metricsRange.min && step <= metricsRange.max);
    return { hasEvalData: steps.length > 0, evalSteps: steps };
  }, [events, metricsRange]);

  const isGan = stage === 'gan';

  if (chartData.length === 0) {
    return (
      <div style={styles.container}>
        <h3 style={styles.title}>Loss</h3>
        <div style={styles.empty}>
          {hasMetrics === false
            ? 'Legacy run - no metrics recorded'
            : 'No metrics data'}
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>
        Loss
        <span style={styles.subtitle}>
          Step {chartData[chartData.length - 1]?.step.toLocaleString()}
        </span>
      </h3>
      <div style={styles.chartWrapper}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis
              dataKey="step"
              type="number"
              domain={['dataMin', 'dataMax']}
              stroke="#666"
              fontSize={11}
              tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
            />
            <YAxis stroke="#666" fontSize={11} domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1a1a',
                border: '1px solid #333',
                borderRadius: 4,
                fontSize: 12,
              }}
              labelStyle={{ color: '#9ca3af' }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />

            {/* Always show mel */}
            <Line
              type="monotone"
              dataKey="mel"
              stroke="#f97316"
              strokeWidth={2}
              dot={false}
              name="Mel"
            />

            {/* KL/Dur shown for all stages */}
            <Line
              type="monotone"
              dataKey="kl"
              stroke="#8b5cf6"
              strokeWidth={1.5}
              dot={false}
              name="KL"
            />
            <Line
              type="monotone"
              dataKey="dur"
              stroke="#06b6d4"
              strokeWidth={1.5}
              dot={false}
              name="Dur"
            />

            {/* GAN-specific losses */}
            {isGan && (
              <>
                <Line
                  type="monotone"
                  dataKey="adv"
                  stroke="#ef4444"
                  strokeWidth={1.5}
                  dot={false}
                  name="Adv"
                />
                <Line
                  type="monotone"
                  dataKey="fm"
                  stroke="#22c55e"
                  strokeWidth={1.5}
                  dot={false}
                  name="FM"
                />
                <Line
                  type="monotone"
                  dataKey="d_loss"
                  stroke="#3b82f6"
                  strokeWidth={1.5}
                  dot={false}
                  name="D Loss"
                />
              </>
            )}

            {/* Vertical reference lines at eval steps */}
            {evalSteps.map((step) => (
              <ReferenceLine
                key={`eval-ref-${step}`}
                x={step}
                stroke="#10b981"
                strokeWidth={2}
                strokeDasharray="4 4"
                label={{
                  value: '📊',
                  position: 'top',
                  fontSize: 16,
                }}
              />
            ))}

            {/* Eval losses - large prominent markers */}
            {hasEvalData && (
              <>
                <Line
                  type="monotone"
                  dataKey="eval_mel"
                  stroke="#10b981"
                  strokeWidth={0}
                  dot={{ r: 10, fill: '#10b981', stroke: '#fff', strokeWidth: 3 }}
                  activeDot={{ r: 14, fill: '#10b981', stroke: '#fff', strokeWidth: 3 }}
                  name="Eval Mel"
                  connectNulls={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="eval_kl"
                  stroke="#14b8a6"
                  strokeWidth={0}
                  dot={{ r: 10, fill: '#14b8a6', stroke: '#fff', strokeWidth: 3 }}
                  activeDot={{ r: 14, fill: '#14b8a6', stroke: '#fff', strokeWidth: 3 }}
                  name="Eval KL"
                  connectNulls={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="eval_dur"
                  stroke="#0d9488"
                  strokeWidth={0}
                  dot={{ r: 10, fill: '#0d9488', stroke: '#fff', strokeWidth: 3 }}
                  activeDot={{ r: 14, fill: '#0d9488', stroke: '#fff', strokeWidth: 3 }}
                  name="Eval Dur"
                  connectNulls={false}
                  isAnimationActive={false}
                />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
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
    margin: '0 0 12px 0',
    fontSize: 14,
    fontWeight: 600,
    color: '#9ca3af',
    textTransform: 'uppercase',
    letterSpacing: 1,
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  subtitle: {
    fontSize: 12,
    fontWeight: 400,
    color: '#6b7280',
    textTransform: 'none',
    letterSpacing: 0,
  },
  chartWrapper: {
    marginTop: 8,
  },
  empty: {
    color: '#6b7280',
    fontSize: 13,
    textAlign: 'center',
    padding: 40,
  },
};
