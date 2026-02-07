/**
 * Polling hooks for GPU telemetry and events.
 */

import { useState, useEffect, useCallback } from 'react';
import { getGpuInfo, getEvents, type GpuResponse, type TrainingEvent } from '../api/client';

// =============================================================================
// GPU Polling
// =============================================================================

export interface UseGpuPollingOptions {
  enabled?: boolean;
  intervalMs?: number;
}

export interface UseGpuPollingResult {
  gpu: GpuResponse | null;
  isLoading: boolean;
  error: Error | null;
  refresh: () => void;
}

export function useGpuPolling({
  enabled = true,
  intervalMs = 5000,
}: UseGpuPollingOptions = {}): UseGpuPollingResult {
  const [gpu, setGpu] = useState<GpuResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await getGpuInfo();
      setGpu(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e : new Error('Failed to fetch GPU info'));
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    refresh();
    const interval = setInterval(refresh, intervalMs);
    return () => clearInterval(interval);
  }, [enabled, intervalMs, refresh]);

  return { gpu, isLoading, error, refresh };
}

// =============================================================================
// Events Polling
// =============================================================================

export interface UseEventsPollingOptions {
  runId: string | null;
  enabled?: boolean;
  intervalMs?: number;
  limit?: number;
}

export interface UseEventsPollingResult {
  events: TrainingEvent[];
  isLoading: boolean;
  error: Error | null;
  refresh: () => void;
}

export function useEventsPolling({
  runId,
  enabled = true,
  intervalMs = 3000,
  limit = 50,
}: UseEventsPollingOptions): UseEventsPollingResult {
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const refresh = useCallback(async () => {
    if (!runId) return;

    setIsLoading(true);
    try {
      const data = await getEvents(runId, true, limit);
      setEvents(data.events);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e : new Error('Failed to fetch events'));
    } finally {
      setIsLoading(false);
    }
  }, [runId, limit]);

  useEffect(() => {
    if (!enabled || !runId) return;

    refresh();
    const interval = setInterval(refresh, intervalMs);
    return () => clearInterval(interval);
  }, [enabled, runId, intervalMs, refresh]);

  return { events, isLoading, error, refresh };
}
