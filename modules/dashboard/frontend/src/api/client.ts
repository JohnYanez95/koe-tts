/**
 * API client for KOE Dashboard backend.
 */

// =============================================================================
// Types
// =============================================================================

export interface RunSummary {
  run_id: string;
  dataset: string;
  stage: string;
  step: number;
  updated_at: string;
  started_at: string; // Parsed from run_id datetime suffix
  status: string; // running, completed, stopped, emergency
  alarm_state: string; // healthy, unstable, d_dominant, g_collapse, stopped
  supports_control: boolean;
  has_metrics: boolean;
  checkpoint_count: number;
}

/**
 * Check if a RunSummary is in a terminal state.
 * Convenience wrapper around isTerminalStatus.
 */
export function isTerminalRun(run: RunSummary): boolean {
  return isTerminalStatus(run.status);
}

export interface RunMeta {
  run_id: string;
  config: Record<string, unknown>;
  run_md: string | null;
  checkpoints: string[];
  last_step: number;
  last_timestamp: string | null;
  created_at: string | null;
  current_step: number;
  last_updated_at: string | null;
  alarm_state: string;
  stage: string;
  status: string; // running, stopped, completed, emergency, lost
  status_source: string; // "event" | "mtime_heuristic" | "legacy"
  status_reason: string | null; // "user_requested" | "thermal" | "mtime_timeout" | etc
  metrics_stale_seconds: number; // Threshold for mtime heuristic
  has_metrics: boolean; // Whether metrics.jsonl exists
  supports_control: boolean;
}

/**
 * Check if a run status is terminal (not actively running).
 * Works with both RunSummary.status and RunMeta.status.
 */
export function isTerminalStatus(status: string): boolean {
  return ['completed', 'stopped', 'emergency', 'lost'].includes(status);
}

export interface ControlRequest {
  action: 'checkpoint' | 'eval' | 'stop';
  params?: Record<string, unknown>;
}

export interface ControlResponse {
  success: boolean;
  nonce: string;
  message: string;
}

export interface MetricsPoint {
  step: number;
  epoch: number | null;
  // Generator losses
  loss_g: number | null;
  g_loss_mel: number | null;
  g_loss_kl: number | null;
  g_loss_dur: number | null;
  g_loss_adv: number | null;
  g_loss_fm: number | null;
  g_grad_norm: number | null;
  // Discriminator losses
  loss_d: number | null;
  d_grad_norm: number | null;
  d_real_score: number | null;
  d_fake_score: number | null;
  // Normalized D scores (0-1 probability)
  ctrl_d_real_prob: number | null;
  ctrl_d_fake_prob: number | null;
  ctrl_d_real_velocity: number | null;
  ctrl_d_confusion_active: boolean | null;
  // Control/state
  kl_weight: number | null;
  adv_active: boolean | null;
  ctrl_controller_alarm: string | null;
  ctrl_adv_weight_scale: number | null;
  ctrl_d_lr_scale: number | null;
  ctrl_d_throttle_active: boolean | null;
  ctrl_d_throttle_every: number | null;
  ctrl_alarms_triggered_total: number | null;
  // P1/P2: Stability controller state
  ctrl_escalation_level: number | null;
  ctrl_grad_clip_scale: number | null;
  ctrl_nan_inf_detected: boolean | null;
  ctrl_emergency_stop: boolean | null;
  ctrl_emergency_reason: string | null;
  ctrl_ema_grad_g: number | null;
  ctrl_ema_grad_d: number | null;
  ctrl_stable_steps_at_level: number | null;
  // Clip coefficients (1.0 = no clip, lower = more clipping)
  g_clip_coef: number | null;
  d_clip_coef: number | null;
  // Legacy fields for core training
  total_loss: number | null;
  mel_loss: number | null;
  kl_loss: number | null;
  dur_loss: number | null;
}

// Helper: pick first defined value from multiple possible keys (for legacy compat)
function pick<T>(raw: Record<string, unknown>, keys: string[]): T | null {
  for (const k of keys) {
    if (raw[k] !== undefined && raw[k] !== null) return raw[k] as T;
  }
  return null;
}

function safeNumber(v: unknown): number | null {
  if (typeof v !== 'number') return null;
  return Number.isFinite(v) ? v : null;
}

function safeBool(v: unknown): boolean | null {
  if (typeof v !== 'boolean') return null;
  return v;
}

function safeString(v: unknown): string | null {
  if (typeof v !== 'string') return null;
  return v;
}

/**
 * Decode raw metric JSON to MetricsPoint, coercing non-finite values to null.
 * Handles both ctrl_* prefixed fields and legacy unprefixed fields.
 * Returns null if the payload is malformed (missing step).
 */
export function decodeMetric(raw: unknown): MetricsPoint | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const obj = raw as Record<string, unknown>;

  // Step is required
  if (typeof obj.step !== 'number') return null;

  // Controller fields may be ctrl_* or legacy unprefixed
  const alarm = pick<string>(obj, ['ctrl_controller_alarm', 'controller_alarm']);
  const nanInf = pick<boolean>(obj, ['ctrl_nan_inf_detected', 'nan_inf_detected']);
  const escLevel = pick<number>(obj, ['ctrl_escalation_level', 'escalation_level']);
  const clipScale = pick<number>(obj, ['ctrl_grad_clip_scale', 'grad_clip_scale']);
  const emergencyStop = pick<boolean>(obj, ['ctrl_emergency_stop', 'emergency_stop']);
  const emergencyReason = pick<string>(obj, ['ctrl_emergency_reason', 'emergency_reason']);
  const emaG = pick<number>(obj, ['ctrl_ema_grad_g', 'ema_grad_g']);
  const emaD = pick<number>(obj, ['ctrl_ema_grad_d', 'ema_grad_d']);
  const stableSteps = pick<number>(obj, ['ctrl_stable_steps_at_level', 'stable_steps_at_level']);

  return {
    step: obj.step as number,
    epoch: safeNumber(obj.epoch),
    loss_g: safeNumber(obj.loss_g),
    g_loss_mel: safeNumber(obj.g_loss_mel),
    g_loss_kl: safeNumber(obj.g_loss_kl),
    g_loss_dur: safeNumber(obj.g_loss_dur),
    g_loss_adv: safeNumber(obj.g_loss_adv),
    g_loss_fm: safeNumber(obj.g_loss_fm),
    g_grad_norm: safeNumber(obj.g_grad_norm),
    loss_d: safeNumber(obj.loss_d),
    d_grad_norm: safeNumber(obj.d_grad_norm),
    d_real_score: safeNumber(obj.d_real_score),
    d_fake_score: safeNumber(obj.d_fake_score),
    ctrl_d_real_prob: safeNumber(obj.ctrl_d_real_prob),
    ctrl_d_fake_prob: safeNumber(obj.ctrl_d_fake_prob),
    ctrl_d_real_velocity: safeNumber(obj.ctrl_d_real_velocity),
    ctrl_d_confusion_active: safeBool(obj.ctrl_d_confusion_active),
    kl_weight: safeNumber(obj.kl_weight),
    adv_active: safeBool(obj.adv_active),
    ctrl_controller_alarm: safeString(alarm),
    ctrl_adv_weight_scale: safeNumber(obj.ctrl_adv_weight_scale),
    ctrl_d_lr_scale: safeNumber(obj.ctrl_d_lr_scale),
    ctrl_d_throttle_active: safeBool(obj.ctrl_d_throttle_active),
    ctrl_d_throttle_every: safeNumber(obj.ctrl_d_throttle_every),
    ctrl_alarms_triggered_total: safeNumber(obj.ctrl_alarms_triggered_total),
    // P1/P2: Stability controller state (normalized from ctrl_* or legacy)
    ctrl_escalation_level: safeNumber(escLevel),
    ctrl_grad_clip_scale: safeNumber(clipScale),
    ctrl_nan_inf_detected: safeBool(nanInf),
    ctrl_emergency_stop: safeBool(emergencyStop),
    ctrl_emergency_reason: safeString(emergencyReason),
    ctrl_ema_grad_g: safeNumber(emaG),
    ctrl_ema_grad_d: safeNumber(emaD),
    ctrl_stable_steps_at_level: safeNumber(stableSteps),
    // Clip coefficients
    g_clip_coef: safeNumber(obj.g_clip_coef),
    d_clip_coef: safeNumber(obj.d_clip_coef),
    // Legacy fields for core training
    total_loss: safeNumber(obj.total_loss),
    mel_loss: safeNumber(obj.mel_loss),
    kl_loss: safeNumber(obj.kl_loss),
    dur_loss: safeNumber(obj.dur_loss),
  };
}

export interface MetricsResponse {
  metrics: MetricsPoint[];
  total_lines: number;
  cursor: number;
}

export interface GpuInfo {
  index: number;
  name: string;
  temp_c: number;
  util_pct: number;
  mem_used_mb: number;
  mem_total_mb: number;
  power_w: number | null;
}

export interface GpuResponse {
  available: boolean;
  timestamp: string;
  gpus: GpuInfo[];
}

export interface TrainingEvent {
  ts: string;
  event: string;
  [key: string]: unknown;
}

export interface EventsResponse {
  events: TrainingEvent[];
  total_lines: number;
  cursor: number;
}

export interface ArtifactInfo {
  name: string;
  path: string;
  type: string;
  updated_at: string;
}

export interface ArtifactsResponse {
  eval: ArtifactInfo[];
}

// =============================================================================
// API Functions
// =============================================================================

const API_BASE = '/api';

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function listRuns(): Promise<RunSummary[]> {
  return fetchJson<RunSummary[]>(`${API_BASE}/runs`);
}

export async function getRunMeta(runId: string): Promise<RunMeta> {
  return fetchJson<RunMeta>(`${API_BASE}/runs/${runId}/meta`);
}

export async function getMetrics(
  runId: string,
  after: number = 0,
  limit: number = 1000,
  tail: boolean = false
): Promise<MetricsResponse> {
  const raw = await fetchJson<{ metrics: unknown[]; total_lines: number; cursor: number }>(
    `${API_BASE}/runs/${runId}/metrics?after=${after}&limit=${limit}&tail=${tail}`
  );
  // Decode and filter out malformed entries
  const metrics = raw.metrics
    .map((m) => decodeMetric(m))
    .filter((m): m is MetricsPoint => m !== null);
  return {
    metrics,
    total_lines: raw.total_lines,
    cursor: raw.cursor,
  };
}

export async function getGpuInfo(): Promise<GpuResponse> {
  return fetchJson<GpuResponse>(`${API_BASE}/gpu`);
}

export async function getEvents(
  runId: string,
  tail: boolean = true,
  limit: number = 50
): Promise<EventsResponse> {
  return fetchJson<EventsResponse>(
    `${API_BASE}/runs/${runId}/events?tail=${tail}&limit=${limit}`
  );
}

export async function getArtifacts(runId: string): Promise<ArtifactsResponse> {
  return fetchJson<ArtifactsResponse>(`${API_BASE}/runs/${runId}/artifacts`);
}

export function getHealthCheck(): Promise<{ status: string; timestamp: string }> {
  return fetchJson(`${API_BASE}/health`);
}

export async function sendControlRequest(
  runId: string,
  request: ControlRequest
): Promise<ControlResponse> {
  const response = await fetch(`${API_BASE}/runs/${runId}/control`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

export interface BackendEvalRequest {
  seed?: number;
  mode?: string;
  speakers?: string[];
  prompts_file?: string;
  tag?: string;
  checkpoint?: string;  // Specific checkpoint name (e.g. "step_025000.pt")
  force_rerun?: boolean;  // Force re-run even if eval exists
}

export interface BackendEvalResponse {
  success: boolean;
  eval_id: string;
  artifact_dir: string | null;
  error: string | null;
  cached?: boolean;  // True if result was from existing eval
}

export interface CheckpointInfo {
  name: string;  // e.g. "step_025000.pt", "best.pt"
  step: number | null;  // Extracted from name, null for best/final
  tag: string | null;  // e.g. "emergency", "manual", null for periodic
  created_at: string;  // ISO timestamp
  size_mb: number;
  eval_status: 'none' | 'complete' | 'running';
  eval_dir: string | null;  // Path to eval results if complete
}

export interface CheckpointsResponse {
  checkpoints: CheckpointInfo[];
  total: number;
}

/**
 * Trigger backend eval for any run (including legacy runs without control plane).
 * This runs eval as a synchronous backend operation.
 */
export async function triggerBackendEval(
  runId: string,
  request: BackendEvalRequest = {}
): Promise<BackendEvalResponse> {
  const response = await fetch(`${API_BASE}/runs/${runId}/eval`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get list of checkpoints for a run with their eval status.
 */
export async function getCheckpoints(runId: string): Promise<CheckpointsResponse> {
  const response = await fetch(`${API_BASE}/runs/${runId}/checkpoints`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}
