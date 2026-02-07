/**
 * Control panel for sending checkpoint/eval requests to training.
 */

import { useState, useCallback, useEffect, useMemo } from 'react';
import {
  sendControlRequest,
  triggerBackendEval,
  getEvents,
  getCheckpoints,
  type ControlResponse,
  type BackendEvalResponse,
  type TrainingEvent,
  type CheckpointInfo,
} from '../api/client';

interface ControlPanelProps {
  runId: string;
  supportsControl: boolean;
  alarmState?: string;
  events?: TrainingEvent[];
  stage?: string;
}

interface RequestStatus {
  type: 'success' | 'error' | 'pending';
  message: string;
  nonce?: string;
  evalId?: string;
  artifactDir?: string;
  timestamp: Date;
}

type StopState = 'idle' | 'requesting' | 'acked' | 'stopped';

export function ControlPanel({ runId, supportsControl, alarmState, events = [], stage }: ControlPanelProps) {
  // Determine eval mode based on stage
  // VITS stages use teacher mode (training-comparable losses)
  // Multi-speaker runs use multispeaker mode (speaker separation)
  // GAN stage uses multispeaker for regression testing with audio grid
  // Core stage uses teacher mode for training-comparable loss metrics
  const evalMode = stage === 'core' ? 'teacher' : 'multispeaker';
  const [checkpointTag, setCheckpointTag] = useState('manual');
  const [evalSeed, setEvalSeed] = useState(42);
  const [isLoading, setIsLoading] = useState(false);
  const [lastStatus, setLastStatus] = useState<RequestStatus | null>(null);
  const [stopState, setStopState] = useState<StopState>('idle');

  // Checkpoint selection state
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('best');
  const [isLoadingCheckpoints, setIsLoadingCheckpoints] = useState(false);

  // Detect terminal state: run is completed or stopped
  const isTerminal = useMemo(() => {
    // Check alarm_state
    const terminalStates = ['stopped', 'complete', 'finished'];
    if (alarmState && terminalStates.includes(alarmState.toLowerCase())) {
      return true;
    }
    // Check events for training_complete or training_stopped
    return events.some(
      (e) => e.event === 'training_complete' || e.event === 'training_stopped'
    );
  }, [alarmState, events]);

  // Can control only if supports_control AND not terminal
  const canControl = supportsControl && !isTerminal;

  // Load checkpoints when runId changes
  useEffect(() => {
    async function loadCheckpoints() {
      setIsLoadingCheckpoints(true);
      try {
        const response = await getCheckpoints(runId);
        setCheckpoints(response.checkpoints);
        // Default to best if available, otherwise first checkpoint
        if (response.checkpoints.length > 0) {
          const best = response.checkpoints.find((c) => c.name === 'best.pt');
          setSelectedCheckpoint(best ? 'best.pt' : response.checkpoints[0].name);
        }
      } catch (e) {
        console.error('Failed to load checkpoints:', e);
        setCheckpoints([]);
      } finally {
        setIsLoadingCheckpoints(false);
      }
    }
    loadCheckpoints();
  }, [runId]);

  const handleRequest = useCallback(
    async (action: 'checkpoint' | 'eval') => {
      setIsLoading(true);
      setLastStatus({
        type: 'pending',
        message: `Sending ${action} request...`,
        timestamp: new Date(),
      });

      try {
        const params =
          action === 'checkpoint'
            ? { tag: checkpointTag }
            : { mode: evalMode, seed: evalSeed, tag: 'ui' };
        const response: ControlResponse = await sendControlRequest(runId, {
          action,
          params,
        });

        setLastStatus({
          type: 'success',
          message: response.message,
          nonce: response.nonce,
          timestamp: new Date(),
        });
      } catch (e) {
        setLastStatus({
          type: 'error',
          message: e instanceof Error ? e.message : 'Request failed',
          timestamp: new Date(),
        });
      } finally {
        setIsLoading(false);
      }
    },
    [runId, checkpointTag, evalSeed, evalMode]
  );

  const handleStop = useCallback(async () => {
    setIsLoading(true);
    setStopState('requesting');
    setLastStatus({
      type: 'pending',
      message: 'Requesting stop...',
      timestamp: new Date(),
    });

    try {
      const response: ControlResponse = await sendControlRequest(runId, {
        action: 'stop',
        params: { tag: 'manual_stop' },
      });
      setStopState('acked');
      setLastStatus({
        type: 'pending',
        message: 'Stop acknowledged - waiting for exit...',
        nonce: response.nonce,
        timestamp: new Date(),
      });
    } catch (e) {
      setStopState('idle');
      setLastStatus({
        type: 'error',
        message: e instanceof Error ? e.message : 'Stop request failed',
        timestamp: new Date(),
      });
    } finally {
      setIsLoading(false);
    }
  }, [runId]);

  // Poll for training_complete event when stop is acked
  useEffect(() => {
    if (stopState !== 'acked') return;

    const pollInterval = setInterval(async () => {
      try {
        const response = await getEvents(runId, true, 10);
        const completeEvent = response.events.find(
          (e) => e.event === 'training_complete' && e.status === 'user_stopped'
        );
        if (completeEvent) {
          setStopState('stopped');
          setLastStatus({
            type: 'success',
            message: 'Stopped cleanly',
            timestamp: new Date(),
          });
        }
      } catch {
        // Ignore poll errors
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [stopState, runId]);

  const handleBackendEval = useCallback(async (forceRerun: boolean = false) => {
    setIsLoading(true);
    setLastStatus({
      type: 'pending',
      message: forceRerun ? 'Running eval (this may take a minute)...' : 'Checking for cached eval...',
      timestamp: new Date(),
    });

    try {
      const response: BackendEvalResponse = await triggerBackendEval(runId, {
        seed: evalSeed,
        mode: evalMode,
        tag: 'ui',
        checkpoint: selectedCheckpoint,
        force_rerun: forceRerun,
      });

      if (response.success) {
        setLastStatus({
          type: 'success',
          message: response.cached ? 'Loaded cached eval' : 'Eval complete!',
          evalId: response.eval_id,
          artifactDir: response.artifact_dir || undefined,
          timestamp: new Date(),
        });
        // Refresh checkpoints to update eval status
        if (!response.cached) {
          const updated = await getCheckpoints(runId);
          setCheckpoints(updated.checkpoints);
        }
      } else {
        setLastStatus({
          type: 'error',
          message: response.error || 'Eval failed',
          evalId: response.eval_id,
          timestamp: new Date(),
        });
      }
    } catch (e) {
      setLastStatus({
        type: 'error',
        message: e instanceof Error ? e.message : 'Request failed',
        timestamp: new Date(),
      });
    } finally {
      setIsLoading(false);
    }
  }, [runId, evalSeed, evalMode, selectedCheckpoint]);

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Control</h3>

      {canControl ? (
        <>
          {/* Checkpoint controls - only for control-capable runs */}
          <div style={styles.section}>
            <label style={styles.label}>Checkpoint tag</label>
            <input
              type="text"
              value={checkpointTag}
              onChange={(e) => setCheckpointTag(e.target.value)}
              style={styles.input}
              placeholder="manual"
              disabled={isLoading}
            />
            <button
              onClick={() => handleRequest('checkpoint')}
              disabled={isLoading}
              style={styles.button}
            >
              Request Checkpoint
            </button>
          </div>

          {/* Eval controls - runs on CPU, doesn't pause training */}
          <div style={styles.section}>
            <label style={styles.label}>Checkpoint</label>
            <select
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              style={styles.select}
              disabled={isLoading || isLoadingCheckpoints}
            >
              {checkpoints.map((ckpt) => (
                <option key={ckpt.name} value={ckpt.name}>
                  {ckpt.name.replace('.pt', '')}
                  {ckpt.step ? ` (step ${ckpt.step.toLocaleString()})` : ''}
                  {ckpt.eval_status === 'complete' ? ' ✓' : ''}
                </option>
              ))}
            </select>
            {checkpoints.find((c) => c.name === selectedCheckpoint)?.eval_status === 'complete' && (
              <span style={styles.cachedNote}>Eval cached - will load instantly</span>
            )}

            <label style={{ ...styles.label, marginTop: 8 }}>Eval seed</label>
            <input
              type="number"
              value={evalSeed}
              onChange={(e) => setEvalSeed(parseInt(e.target.value) || 42)}
              style={{ ...styles.input, marginBottom: 8 }}
              disabled={isLoading}
            />
            <div style={styles.evalButtons}>
              <button
                onClick={() => handleBackendEval(false)}
                disabled={isLoading}
                style={{ ...styles.button, ...styles.evalButton }}
              >
                {isLoading ? 'Running...' : 'Run Eval'}
              </button>
              {checkpoints.find((c) => c.name === selectedCheckpoint)?.eval_status === 'complete' && (
                <button
                  onClick={() => handleBackendEval(true)}
                  disabled={isLoading}
                  style={{ ...styles.button, ...styles.rerunButton }}
                  title="Force re-run even if cached"
                >
                  Re-run
                </button>
              )}
            </div>
            <span style={styles.evalNote}>Runs on CPU - training continues</span>
          </div>

          {/* Stop control */}
          <div style={{ ...styles.section, marginTop: 8, borderTop: '1px solid #333', paddingTop: 12 }}>
            <button
              onClick={handleStop}
              disabled={isLoading || stopState !== 'idle'}
              style={{ ...styles.button, ...styles.stopButton }}
              title="If an eval is running, stop will execute after eval completes."
            >
              {stopState === 'idle' && 'Save & Stop'}
              {stopState === 'requesting' && 'Requesting...'}
              {stopState === 'acked' && 'Stopping...'}
              {stopState === 'stopped' && 'Stopped'}
            </button>
            <span style={styles.evalNote}>Saves checkpoint and exits cleanly</span>
          </div>
        </>
      ) : supportsControl && isTerminal ? (
        <>
          {/* Run is complete - show disabled state */}
          <div style={styles.terminalNotice}>
            Run completed. Controls disabled.
          </div>

          {/* Eval controls - backend-triggered for completed runs */}
          <div style={styles.section}>
            <label style={styles.label}>Checkpoint</label>
            <select
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              style={styles.select}
              disabled={isLoading || isLoadingCheckpoints}
            >
              {checkpoints.map((ckpt) => (
                <option key={ckpt.name} value={ckpt.name}>
                  {ckpt.name.replace('.pt', '')}
                  {ckpt.step ? ` (step ${ckpt.step.toLocaleString()})` : ''}
                  {ckpt.eval_status === 'complete' ? ' ✓' : ''}
                </option>
              ))}
            </select>
            {checkpoints.find((c) => c.name === selectedCheckpoint)?.eval_status === 'complete' && (
              <span style={styles.cachedNote}>Eval cached - will load instantly</span>
            )}

            <label style={{ ...styles.label, marginTop: 8 }}>Eval seed</label>
            <input
              type="number"
              value={evalSeed}
              onChange={(e) => setEvalSeed(parseInt(e.target.value) || 42)}
              style={{ ...styles.input, marginBottom: 8 }}
              disabled={isLoading}
            />
            <div style={styles.evalButtons}>
              <button
                onClick={() => handleBackendEval(false)}
                disabled={isLoading}
                style={{ ...styles.button, ...styles.evalButton }}
              >
                {isLoading ? 'Running...' : 'Run Eval'}
              </button>
              {checkpoints.find((c) => c.name === selectedCheckpoint)?.eval_status === 'complete' && (
                <button
                  onClick={() => handleBackendEval(true)}
                  disabled={isLoading}
                  style={{ ...styles.button, ...styles.rerunButton }}
                  title="Force re-run even if cached"
                >
                  Re-run
                </button>
              )}
            </div>
          </div>
        </>
      ) : (
        <>
          {/* Legacy run warning */}
          <div style={styles.legacyWarning}>
            Legacy run (no live control). Checkpoint requests require control plane.
          </div>

          {/* Eval controls - backend-triggered for legacy runs */}
          <div style={styles.section}>
            <label style={styles.label}>Checkpoint</label>
            <select
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              style={styles.select}
              disabled={isLoading || isLoadingCheckpoints}
            >
              {checkpoints.map((ckpt) => (
                <option key={ckpt.name} value={ckpt.name}>
                  {ckpt.name.replace('.pt', '')}
                  {ckpt.step ? ` (step ${ckpt.step.toLocaleString()})` : ''}
                  {ckpt.eval_status === 'complete' ? ' ✓' : ''}
                </option>
              ))}
            </select>
            {checkpoints.find((c) => c.name === selectedCheckpoint)?.eval_status === 'complete' && (
              <span style={styles.cachedNote}>Eval cached - will load instantly</span>
            )}

            <label style={{ ...styles.label, marginTop: 8 }}>Eval seed</label>
            <input
              type="number"
              value={evalSeed}
              onChange={(e) => setEvalSeed(parseInt(e.target.value) || 42)}
              style={{ ...styles.input, marginBottom: 8 }}
              disabled={isLoading}
            />
            <div style={styles.evalButtons}>
              <button
                onClick={() => handleBackendEval(false)}
                disabled={isLoading}
                style={{ ...styles.button, ...styles.evalButton }}
              >
                {isLoading ? 'Running...' : 'Run Eval'}
              </button>
              {checkpoints.find((c) => c.name === selectedCheckpoint)?.eval_status === 'complete' && (
                <button
                  onClick={() => handleBackendEval(true)}
                  disabled={isLoading}
                  style={{ ...styles.button, ...styles.rerunButton }}
                  title="Force re-run even if cached"
                >
                  Re-run
                </button>
              )}
            </div>
          </div>
        </>
      )}

      {/* Status area */}
      {lastStatus && (
        <div
          style={{
            ...styles.status,
            ...(lastStatus.type === 'success'
              ? styles.statusSuccess
              : lastStatus.type === 'error'
              ? styles.statusError
              : styles.statusPending),
          }}
        >
          <div style={styles.statusMessage}>{lastStatus.message}</div>
          {lastStatus.nonce && (
            <div style={styles.statusNonce}>Nonce: {lastStatus.nonce}</div>
          )}
          {lastStatus.evalId && (
            <div style={styles.statusNonce}>Eval: {lastStatus.evalId}</div>
          )}
          {lastStatus.artifactDir && (
            <div style={styles.statusNonce}>Output: {lastStatus.artifactDir}</div>
          )}
          <div style={styles.statusTime}>
            {lastStatus.timestamp.toLocaleTimeString()}
          </div>
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
    border: '1px solid #262626',
  },
  title: {
    margin: '0 0 12px 0',
    fontSize: 14,
    fontWeight: 600,
    color: '#9ca3af',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  unsupportedWarning: {
    padding: 12,
    backgroundColor: '#2a2a2a',
    borderRadius: 6,
    color: '#9ca3af',
    fontSize: 12,
    lineHeight: 1.5,
    border: '1px solid #404040',
  },
  legacyWarning: {
    padding: 8,
    marginBottom: 12,
    backgroundColor: 'rgba(251, 191, 36, 0.1)',
    borderRadius: 4,
    color: '#fbbf24',
    fontSize: 11,
    border: '1px solid rgba(251, 191, 36, 0.2)',
  },
  terminalNotice: {
    padding: 8,
    marginBottom: 12,
    backgroundColor: 'rgba(74, 222, 128, 0.1)',
    borderRadius: 4,
    color: '#4ade80',
    fontSize: 11,
    border: '1px solid rgba(74, 222, 128, 0.2)',
  },
  section: {
    marginBottom: 12,
  },
  label: {
    display: 'block',
    fontSize: 11,
    color: '#6b7280',
    marginBottom: 4,
    textTransform: 'uppercase',
  },
  input: {
    width: '100%',
    padding: '8px 10px',
    fontSize: 13,
    backgroundColor: '#0f0f0f',
    border: '1px solid #404040',
    borderRadius: 4,
    color: '#e0e0e0',
    marginBottom: 8,
    boxSizing: 'border-box',
    fontFamily: 'monospace',
  },
  button: {
    width: '100%',
    padding: '8px 12px',
    fontSize: 12,
    fontWeight: 500,
    backgroundColor: '#3b82f6',
    border: 'none',
    borderRadius: 4,
    color: '#fff',
    cursor: 'pointer',
    transition: 'background-color 0.15s',
  },
  evalButton: {
    backgroundColor: '#6366f1',
    marginBottom: 4,
    flex: 1,
  },
  stopButton: {
    backgroundColor: '#ef4444',
    marginBottom: 4,
  },
  evalNote: {
    display: 'block',
    fontSize: 10,
    color: '#6b7280',
    textAlign: 'center',
  },
  status: {
    marginTop: 12,
    padding: 10,
    borderRadius: 6,
    fontSize: 12,
  },
  statusSuccess: {
    backgroundColor: 'rgba(74, 222, 128, 0.1)',
    border: '1px solid rgba(74, 222, 128, 0.3)',
    color: '#4ade80',
  },
  statusError: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    border: '1px solid rgba(239, 68, 68, 0.3)',
    color: '#ef4444',
  },
  statusPending: {
    backgroundColor: 'rgba(251, 191, 36, 0.1)',
    border: '1px solid rgba(251, 191, 36, 0.3)',
    color: '#fbbf24',
  },
  statusMessage: {
    marginBottom: 4,
  },
  statusNonce: {
    fontFamily: 'monospace',
    fontSize: 11,
    opacity: 0.8,
  },
  statusTime: {
    fontSize: 10,
    opacity: 0.6,
    marginTop: 4,
  },
  select: {
    width: '100%',
    padding: '8px 10px',
    fontSize: 13,
    backgroundColor: '#0f0f0f',
    border: '1px solid #404040',
    borderRadius: 4,
    color: '#e0e0e0',
    marginBottom: 8,
    boxSizing: 'border-box',
    fontFamily: 'monospace',
    cursor: 'pointer',
  },
  cachedNote: {
    display: 'block',
    fontSize: 10,
    color: '#4ade80',
    marginBottom: 8,
  },
  evalButtons: {
    display: 'flex',
    gap: 8,
  },
  rerunButton: {
    backgroundColor: '#404040',
    flex: '0 0 auto',
    width: 'auto',
    padding: '8px 12px',
  },
};
