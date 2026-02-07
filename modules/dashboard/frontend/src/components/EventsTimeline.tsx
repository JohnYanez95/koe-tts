/**
 * Events timeline component.
 */

import { useEffect, useRef } from 'react';
import type { TrainingEvent } from '../api/client';

interface EventsTimelineProps {
  events: TrainingEvent[];
  isLoading: boolean;
  highlightedEventTs?: string | null;
}

const eventIcons: Record<string, string> = {
  run_started: '🚀',
  resume_from: '↩️',
  checkpoint_saved: '💾',
  alarm_state_change: '⚠️',
  alarm_state_change_recovery: '🟢',
  escalation_level_change: '🔺',
  escalation_level_change_up: '🔺',
  escalation_level_change_down: '🔻',
  training_complete: '✅',
  exception: '❌',
  control_ack: '📨',
  eval_started: '🔍',
  eval_complete: '📊',
  eval_failed: '💥',
};

const eventColors: Record<string, string> = {
  run_started: '#22c55e',
  resume_from: '#3b82f6',
  checkpoint_saved: '#8b5cf6',
  alarm_state_change: '#fbbf24',
  alarm_state_change_recovery: '#22c55e',
  escalation_level_change: '#f97316',
  escalation_level_change_up: '#f97316',    // Orange for escalation
  escalation_level_change_down: '#22c55e',  // Green for de-escalation
  training_complete: '#22c55e',
  exception: '#ef4444',
  control_ack: '#06b6d4',
  eval_started: '#f59e0b',
  eval_complete: '#22c55e',
  eval_failed: '#ef4444',
};

/** Get the effective event type for styling (handles alarm recovery and escalation direction) */
function getEffectiveEventType(event: TrainingEvent): string {
  if (event.event === 'alarm_state_change' && event.current === 'healthy') {
    return 'alarm_state_change_recovery';
  }
  if (event.event === 'escalation_level_change') {
    const prev = event.previous_level as number;
    const curr = event.current_level as number;
    return curr > prev ? 'escalation_level_change_up' : 'escalation_level_change_down';
  }
  return event.event;
}

function formatEventTime(ts: string): string {
  const date = new Date(ts);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatEventDetails(event: TrainingEvent): string {
  const { ts, event: eventType, ...rest } = event;

  switch (eventType) {
    case 'run_started':
      return `Stage: ${rest.stage}, Dataset: ${rest.dataset}`;
    case 'resume_from':
      return `From: ${rest.checkpoint}`;
    case 'checkpoint_saved':
      return `Step ${rest.step} → ${rest.path}${rest.is_best ? ' (best)' : ''}`;
    case 'alarm_state_change':
      return `${rest.previous} → ${rest.current}${rest.reason ? ` (${rest.reason})` : ''}`;
    case 'escalation_level_change': {
      const prev = rest.previous_level as number;
      const curr = rest.current_level as number;
      const direction = curr > prev ? '↑' : '↓';
      return `L${prev} → L${curr} ${direction}${rest.reason ? ` (${rest.reason})` : ''}`;
    }
    case 'training_complete':
      return `Step ${rest.step}, Val loss: ${rest.final_val_loss}`;
    case 'exception':
      return `${rest.error_type}: ${rest.message}`;
    case 'control_ack': {
      const result = rest.result as Record<string, unknown> | undefined;
      return `${rest.action}: ${rest.success ? 'ok' : 'failed'}${result?.eval_id ? ` (${result.eval_id})` : ''}`;
    }
    case 'eval_started':
      return `${rest.eval_id} @ step ${rest.step}`;
    case 'eval_complete': {
      const summary = rest.summary as Record<string, unknown> | undefined;
      const losses = rest.losses as Record<string, number> | undefined;
      // Show training losses if available (VITS eval), otherwise show multispeaker summary
      if (losses?.mel_loss !== undefined) {
        return `${rest.eval_id} @ ${rest.step}: mel=${losses.mel_loss.toFixed(3)} kl=${losses.kl_loss?.toFixed(3) ?? 'N/A'} dur=${losses.dur_loss?.toFixed(3) ?? 'N/A'}`;
      }
      return `${rest.eval_id}: ${summary?.valid_outputs || 'done'}`;
    }
    case 'eval_failed':
      return `${rest.eval_id}: ${rest.error}`;
    default:
      return JSON.stringify(rest);
  }
}

export function EventsTimeline({ events, isLoading, highlightedEventTs }: EventsTimelineProps) {
  // Show events in reverse chronological order
  const sortedEvents = [...events].reverse();
  const containerRef = useRef<HTMLDivElement>(null);
  const eventRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  // Scroll to highlighted event when it changes
  useEffect(() => {
    if (!highlightedEventTs) return;

    const eventEl = eventRefs.current.get(highlightedEventTs);
    if (eventEl && containerRef.current) {
      eventEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [highlightedEventTs]);

  return (
    <div style={styles.container} ref={containerRef}>
      <h3 style={styles.title}>
        Events
        {isLoading && <span style={styles.loadingDot} />}
      </h3>

      {sortedEvents.length === 0 ? (
        <div style={styles.empty}>No events yet</div>
      ) : (
        <div style={styles.timeline}>
          {sortedEvents.map((event, i) => {
            const isHighlighted = event.ts === highlightedEventTs;
            const effectiveType = getEffectiveEventType(event);
            return (
              <div
                key={`${event.ts}-${i}`}
                ref={(el) => {
                  if (el) eventRefs.current.set(event.ts, el);
                }}
                style={{
                  ...styles.event,
                  ...(isHighlighted ? styles.eventHighlighted : {}),
                }}
              >
                <div style={styles.eventIcon}>{eventIcons[effectiveType] || '📌'}</div>
                <div style={styles.eventContent}>
                  <div style={styles.eventHeader}>
                    <span
                      style={{
                        ...styles.eventType,
                        color: eventColors[effectiveType] || '#9ca3af',
                      }}
                    >
                      {event.event.replace(/_/g, ' ')}
                    </span>
                    <span style={styles.eventTime}>{formatEventTime(event.ts)}</span>
                  </div>
                  <div style={styles.eventDetails}>{formatEventDetails(event)}</div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    padding: 16,
    maxHeight: 400,
    overflow: 'auto',
  },
  title: {
    margin: '0 0 12px 0',
    fontSize: 14,
    fontWeight: 600,
    color: '#9ca3af',
    textTransform: 'uppercase',
    letterSpacing: 1,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  loadingDot: {
    display: 'inline-block',
    width: 8,
    height: 8,
    borderRadius: '50%',
    backgroundColor: '#3b82f6',
    animation: 'pulse 1s infinite',
  },
  empty: {
    color: '#6b7280',
    fontSize: 13,
    fontStyle: 'italic',
  },
  timeline: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  event: {
    display: 'flex',
    gap: 10,
    padding: 8,
    backgroundColor: '#262626',
    borderRadius: 6,
    transition: 'background-color 0.3s, box-shadow 0.3s',
  },
  eventHighlighted: {
    backgroundColor: '#1e3a5f',
    boxShadow: '0 0 0 2px #3b82f6',
  },
  eventIcon: {
    fontSize: 14,
    width: 20,
    textAlign: 'center',
  },
  eventContent: {
    flex: 1,
    minWidth: 0,
  },
  eventHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 2,
  },
  eventType: {
    fontSize: 12,
    fontWeight: 500,
    textTransform: 'capitalize',
  },
  eventTime: {
    fontSize: 10,
    color: '#6b7280',
    fontFamily: 'monospace',
  },
  eventDetails: {
    fontSize: 11,
    color: '#9ca3af',
    fontFamily: 'monospace',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
};
