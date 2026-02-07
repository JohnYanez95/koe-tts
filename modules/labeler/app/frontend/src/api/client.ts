const BASE = ''

export interface DatasetInfo {
  name: string
  manifest_count: number
}

export interface StratumInfo {
  stratum: number
  count: number
  heuristic_hits: number
  label: string
}

export interface SessionInfo {
  session_id: string
  dataset: string
  created_at: string
  batch_size: number
  stratum: number | null
  total: number
  labeled: number
  skipped: number
  remaining: number
  pct: number
  published: boolean
}

export interface HeuristicRunInfo {
  params_hash: string
  name: string
  method: string
  processed: number
  with_breaks: number
  elapsed_s: number
  created_at: string
}

export interface DatasetDetail {
  name: string
  total_utterances: number
  strata: StratumInfo[]
  sessions: SessionInfo[]
  heuristic_runs: HeuristicRunInfo[]
}

export interface PauBreak {
  pau_idx: number
  token_position: number
  ms: number | null
  ms_proposed: number | null
  use_break: boolean
  noise_zone_ms: number | null  // (b) marker position — noise zone boundary
}

export interface UtteranceItem {
  utterance_id: string
  text: string
  phonemes: string
  audio_url: string
  duration_sec: number
  speaker_id: string | null
  sample_rate: number
  pau_count: number
  pau_breaks: PauBreak[]
  trim_start_ms: number | null
  trim_end_ms: number | null
  status: string
}

export interface BatchResponse {
  session_id: string
  heuristic_version: string
  heuristic_params_hash: string
  items: UtteranceItem[]
  total: number
  labeled: number
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + url, init)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

export async function listDatasets(): Promise<DatasetInfo[]> {
  return fetchJSON('/api/datasets')
}

export async function getDatasetDetail(dataset: string): Promise<DatasetDetail> {
  return fetchJSON(`/api/datasets/${dataset}`)
}

export async function createSession(
  dataset: string,
  batchSize: number,
  stratum: number | null,
  heuristicOnly: boolean = false,
  heuristicParamsHash: string | null = null,
): Promise<SessionInfo> {
  return fetchJSON('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset,
      batch_size: batchSize,
      stratum,
      heuristic_only: heuristicOnly,
      heuristic_params_hash: heuristicParamsHash,
    }),
  })
}

export async function getSession(sessionId: string): Promise<SessionInfo> {
  return fetchJSON(`/api/sessions/${sessionId}`)
}

export async function getBatch(sessionId: string): Promise<BatchResponse> {
  return fetchJSON(`/api/sessions/${sessionId}/batch`)
}

export interface PauBreakSavePayload {
  pau_idx: number
  token_position: number
  ms_proposed: number | null
  ms: number | null
  delta_ms: number | null
  use_break: boolean
  noise_zone_ms: number | null  // (b) marker position — noise zone boundary
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetchJSON(`/api/sessions/${sessionId}`, { method: 'DELETE' })
}

export async function deleteAllSessions(dataset: string): Promise<void> {
  await fetchJSON(`/api/datasets/${dataset}/sessions`, { method: 'DELETE' })
}

export async function publishSession(sessionId: string): Promise<{ labels_published: number; total_published: number }> {
  return fetchJSON(`/api/sessions/${sessionId}/publish`, { method: 'POST' })
}

export async function saveLabels(
  sessionId: string,
  idx: number,
  breaks: PauBreakSavePayload[],
  status: 'labeled' | 'skipped' = 'labeled',
  trimStartMs?: number,
  trimEndMs?: number,
): Promise<void> {
  await fetchJSON(`/api/sessions/${sessionId}/item/${idx}/labels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      breaks,
      status,
      trim_start_ms: trimStartMs ?? null,
      trim_end_ms: trimEndMs ?? null,
    }),
  })
}
