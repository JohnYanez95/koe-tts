/**
 * SSE hook for streaming metrics.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { decodeMetric, type MetricsPoint } from '../api/client';

const MAX_POINTS = 600; // Rolling window size (slightly larger than display window for smooth transitions)
const UPDATE_INTERVAL = 10; // Only update UI every N steps

export interface UseSSEOptions {
  runId: string | null;
  enabled?: boolean;
  updateInterval?: number; // Update UI every N steps (default: 10)
}

export type StreamState = 'connecting' | 'connected' | 'disconnected' | 'not_available' | 'disabled';

export interface UseSSEResult {
  metrics: MetricsPoint[];
  isConnected: boolean;
  streamState: StreamState;
  error: Error | null;
  clear: () => void;
}

export function useSSE({ runId, enabled = true, updateInterval = UPDATE_INTERVAL }: UseSSEOptions): UseSSEResult {
  const [metrics, setMetrics] = useState<MetricsPoint[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [streamState, setStreamState] = useState<StreamState>('disabled');
  const [error, setError] = useState<Error | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const pendingMetricsRef = useRef<MetricsPoint[]>([]); // Buffer for batching updates

  const clear = useCallback(() => {
    setMetrics([]);
  }, []);

  useEffect(() => {
    if (!runId || !enabled) {
      setStreamState('disabled');
      return;
    }

    setStreamState('connecting');
    const eventSource = new EventSource(`/api/runs/${runId}/stream`);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setIsConnected(true);
      setStreamState('connected');
      setError(null);
    };

    eventSource.addEventListener('metrics', (event) => {
      try {
        const raw = JSON.parse(event.data);
        const data = decodeMetric(raw);
        if (!data) {
          console.warn('Malformed metric payload, skipping:', raw);
          return;
        }

        // Buffer the metric
        pendingMetricsRef.current.push(data);

        // Only flush to state every updateInterval steps
        if (data.step % updateInterval === 0) {
          const pending = pendingMetricsRef.current;
          pendingMetricsRef.current = [];

          setMetrics((prev) => {
            const next = [...prev, ...pending];
            // Keep rolling window
            if (next.length > MAX_POINTS) {
              return next.slice(-MAX_POINTS);
            }
            return next;
          });
        }
      } catch (e) {
        console.error('Failed to parse SSE data:', e);
      }
    });

    // Handle "complete" event from legacy/empty runs
    eventSource.addEventListener('complete', (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.reason === 'no_metrics') {
          setStreamState('not_available');
          eventSource.close();
        }
      } catch (e) {
        console.error('Failed to parse complete event:', e);
      }
    });

    eventSource.onerror = () => {
      setIsConnected(false);
      setStreamState('disconnected');
      setError(new Error('SSE connection lost'));
    };

    return () => {
      eventSource.close();
      eventSourceRef.current = null;
      setIsConnected(false);
      setStreamState('disabled');
    };
  }, [runId, enabled]);

  return { metrics, isConnected, streamState, error, clear };
}
