import { useCallback, useEffect, useRef, useState } from 'react'
import {
  type BatchResponse,
  type DatasetInfo,
  type DatasetDetail,
  type PauBreak,
  type PauBreakSavePayload,
  createSession,
  deleteSession,
  deleteAllSessions,
  publishSession,
  getBatch,
  listDatasets,
  getDatasetDetail,
  saveLabels,
} from '../api/client'
import { BatchInfo } from '../components/BatchInfo'
import { HotkeyHelp } from '../components/HotkeyHelp'
import { NavControls } from '../components/NavControls'
import { PhonemePills } from '../components/PhonemePills'
import { TextDisplay } from '../components/TextDisplay'
import { Waveform, type WaveformHandle, type WaveformMarker, type TrimState } from '../components/Waveform'
import { useSessionStorage } from '../hooks/useSessionStorage'
import { useUndoStack } from '../hooks/useUndoStack'
import { useToast } from '../contexts/ToastContext'

type ViewState =
  | { kind: 'loading' }
  | { kind: 'pick_dataset'; datasets: DatasetInfo[] }
  | { kind: 'pick_session'; dataset: DatasetDetail }
  | { kind: 'labeling'; batch: BatchResponse; currentIdx: number }
  | { kind: 'error'; message: string }

export function LabelView() {
  const [state, setState] = useState<ViewState>({ kind: 'loading' })
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [datasetName, setDatasetName] = useState<string | null>(null)
  // Per-item pau breaks state: Map<itemIdx, PauBreak[]>
  const [pauBreaksMap, setPauBreaksMap] = useState<Map<number, PauBreak[]>>(new Map())
  // Per-item trim state: Map<itemIdx, TrimState>
  const [trimMap, setTrimMap] = useState<Map<number, TrimState>>(new Map())
  const [savedIndices, setSavedIndices] = useState<Set<number>>(new Set())
  const [skippedIndices, setSkippedIndices] = useState<Set<number>>(new Set())
  const [dirty, setDirty] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [lastClickedMs, setLastClickedMs] = useState<number | null>(null)
  const [lastClickedPauIdx, setLastClickedPauIdx] = useState<number | null>(null)
  const [heuristicOnly, setHeuristicOnly] = useState(true)
  const [batchSize, setBatchSize] = useState(25)
  const [samplingMode, setSamplingMode] = useState<'random' | 'stratum'>('random')
  const [selectedStratum, setSelectedStratum] = useState<number>(1)
  const [selectedHeuristic, setSelectedHeuristic] = useState<string | null>(null)
  const [confirmModal, setConfirmModal] = useState<{
    message: string
    onConfirm: () => void
  } | null>(null)
  const [showHotkeyHelp, setShowHotkeyHelp] = useState(false)
  const waveformRef = useRef<WaveformHandle>(null)

  // Session storage for crash recovery
  const { saveState } = useSessionStorage(sessionId)

  // Toast notifications
  const { showToast } = useToast()

  // Undo stack for current item (tracks pauBreaks and trim for current index)
  type UndoState = { pauBreaks: PauBreak[]; trim: TrimState }
  const undoStack = useUndoStack<UndoState>()
  const lastUndoIdxRef = useRef<number | null>(null)

  // Helper to push current state to undo stack before modification
  const pushUndo = useCallback(() => {
    if (state.kind !== 'labeling') return
    const idx = state.currentIdx
    // Clear undo stack when switching items
    if (lastUndoIdxRef.current !== idx) {
      undoStack.clear()
      lastUndoIdxRef.current = idx
    }
    const pauBreaks = pauBreaksMap.get(idx) ?? []
    const durationMs = Math.round(state.batch.items[idx].duration_sec * 1000)
    const trim = trimMap.get(idx) ?? { startMs: Math.min(100, durationMs), endMs: Math.max(durationMs - 100, 0) }
    undoStack.push({ pauBreaks, trim })
  }, [state, pauBreaksMap, trimMap, undoStack])

  // Undo handler
  const handleUndo = useCallback(() => {
    if (state.kind !== 'labeling') return
    const idx = state.currentIdx
    const prev = undoStack.undo()
    if (!prev) return
    setPauBreaksMap((m) => {
      const next = new Map(m)
      next.set(idx, prev.pauBreaks)
      return next
    })
    setTrimMap((m) => {
      const next = new Map(m)
      next.set(idx, prev.trim)
      return next
    })
    setDirty(true)
  }, [state, undoStack])

  // Redo handler
  const handleRedo = useCallback(() => {
    if (state.kind !== 'labeling') return
    const idx = state.currentIdx
    const next = undoStack.redo()
    if (!next) return
    setPauBreaksMap((m) => {
      const nm = new Map(m)
      nm.set(idx, next.pauBreaks)
      return nm
    })
    setTrimMap((m) => {
      const nm = new Map(m)
      nm.set(idx, next.trim)
      return nm
    })
    setDirty(true)
  }, [state, undoStack])

  // Hotkey help toggle (? key)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      // ? requires shift on US keyboard, but some keyboards type ? directly
      if (e.key === '?' || (e.key === '/' && e.shiftKey)) {
        e.preventDefault()
        setShowHotkeyHelp((prev) => !prev)
      }
      // Esc closes help modal
      if (e.key === 'Escape' && showHotkeyHelp) {
        e.preventDefault()
        e.stopPropagation()
        setShowHotkeyHelp(false)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [showHotkeyHelp])

  // Load datasets on mount
  useEffect(() => {
    listDatasets()
      .then((datasets) => {
        if (datasets.length === 1) {
          getDatasetDetail(datasets[0].name).then((detail) =>
            setState({ kind: 'pick_session', dataset: detail }),
          )
        } else {
          setState({ kind: 'pick_dataset', datasets })
        }
      })
      .catch((err) => setState({ kind: 'error', message: String(err) }))
  }, [])

  const selectDataset = useCallback((name: string) => {
    setState({ kind: 'loading' })
    getDatasetDetail(name)
      .then((detail) => setState({ kind: 'pick_session', dataset: detail }))
      .catch((err) => setState({ kind: 'error', message: String(err) }))
  }, [])

  const startSession = useCallback(
    (dataset: string, bs: number, stratum: number | null, hOnly: boolean, heuristicHash: string | null) => {
      setState({ kind: 'loading' })
      setDatasetName(dataset)
      createSession(dataset, bs, stratum, hOnly, heuristicHash)
        .then((session) => {
          setSessionId(session.session_id)
          return loadBatch(session.session_id)
        })
        .catch((err) => setState({ kind: 'error', message: String(err) }))
    },
    [],
  )

  const resumeSession = useCallback((sid: string, dataset: string) => {
    setState({ kind: 'loading' })
    setSessionId(sid)
    setDatasetName(dataset)
    loadBatch(sid).catch((err) => setState({ kind: 'error', message: String(err) }))
  }, [])

  const handleDeleteSession = useCallback(
    (sid: string, datasetName: string) => {
      setConfirmModal({
        message: `Delete session ${sid.slice(0, 15)}...?`,
        onConfirm: () => {
          setConfirmModal(null)
          deleteSession(sid)
            .then(() => {
              // Clear localStorage for this session on delete
              localStorage.removeItem(`koe-labeler:${sid}`)
              return getDatasetDetail(datasetName)
            })
            .then((detail) => setState({ kind: 'pick_session', dataset: detail }))
            .catch((err) => setState({ kind: 'error', message: String(err) }))
        },
      })
    },
    [],
  )

  const handlePublish = useCallback(
    (sid: string, dsName: string) => {
      setConfirmModal({
        message: `Publish labels from ${sid.slice(0, 15)}... to the labels table?`,
        onConfirm: () => {
          setConfirmModal(null)
          publishSession(sid)
            .then(() => {
              // Clear localStorage for this session on successful publish
              localStorage.removeItem(`koe-labeler:${sid}`)
              showToast('Labels published', 'success')
              return getDatasetDetail(dsName)
            })
            .then((detail) => setState({ kind: 'pick_session', dataset: detail }))
            .catch((err) => setState({ kind: 'error', message: String(err) }))
        },
      })
    },
    [showToast],
  )

  const handleDeleteAll = useCallback(
    (datasetName: string) => {
      setConfirmModal({
        message: `Delete ALL sessions for ${datasetName}?`,
        onConfirm: () => {
          setConfirmModal(null)
          deleteAllSessions(datasetName)
            .then(() => getDatasetDetail(datasetName))
            .then((detail) => setState({ kind: 'pick_session', dataset: detail }))
            .catch((err) => setState({ kind: 'error', message: String(err) }))
        },
      })
    },
    [],
  )

  // Modal keyboard: Y to confirm, N/Esc to cancel
  useEffect(() => {
    if (!confirmModal) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'y' || e.key === 'Y') {
        confirmModal.onConfirm()
      } else if (e.key === 'n' || e.key === 'N' || e.key === 'Escape') {
        setConfirmModal(null)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [confirmModal])

  const loadBatch = async (sid: string) => {
    const batch = await getBatch(sid)
    const pbMap = new Map<number, PauBreak[]>()
    const tMap = new Map<number, TrimState>()
    batch.items.forEach((item, idx) => {
      pbMap.set(idx, item.pau_breaks.map((pb) => ({
        ...pb,
        noise_zone_ms: pb.noise_zone_ms ?? null,
      })))
      const dMs = Math.round(item.duration_sec * 1000)
      tMap.set(idx, {
        startMs: item.trim_start_ms ?? Math.min(100, dMs),
        endMs: item.trim_end_ms ?? Math.max(dMs - 100, 0),
      })
    })
    const initialSaved = new Set<number>()
    const initialSkipped = new Set<number>()
    batch.items.forEach((item, idx) => {
      if (item.status === 'labeled' || item.status === 'skipped') {
        initialSaved.add(idx)
      }
      if (item.status === 'skipped') {
        initialSkipped.add(idx)
      }
    })

    // Always use server's saved/skipped status (source of truth)
    setSavedIndices(initialSaved)
    setSkippedIndices(initialSkipped)

    // Try to restore UI state from localStorage (position, markers, trims)
    const storedRaw = localStorage.getItem(`koe-labeler:${sid}`)
    if (storedRaw) {
      try {
        const stored = JSON.parse(storedRaw) as {
          currentIdx: number
          pauBreaksMap: [number, PauBreak[]][]
          trimMap: [number, { startMs: number; endMs: number }][]
          lastClickedPauIdx: number | null
        }
        // Restore UI state from localStorage
        setPauBreaksMap(new Map(stored.pauBreaksMap))
        setTrimMap(new Map(stored.trimMap))
        setLastClickedPauIdx(stored.lastClickedPauIdx)
        setDirty(false)
        setIsPlaying(false)
        setLastClickedMs(null)
        setState({ kind: 'labeling', batch, currentIdx: stored.currentIdx })
        // Show toast after state update (slight delay for render)
        setTimeout(() => showToast('Session restored from browser cache', 'info'), 100)
        return
      } catch {
        // Corrupted data, continue with fresh state
        localStorage.removeItem(`koe-labeler:${sid}`)
      }
    }
    setPauBreaksMap(pbMap)
    setTrimMap(tMap)
    setDirty(false)
    setIsPlaying(false)
    setLastClickedMs(null)
    setLastClickedPauIdx(null)
    setState({ kind: 'labeling', batch, currentIdx: 0 })
  }

  // Auto-save state to localStorage on changes
  useEffect(() => {
    if (state.kind !== 'labeling' || !sessionId) return
    saveState({
      currentIdx: state.currentIdx,
      pauBreaksMap,
      trimMap,
      savedIndices,
      skippedIndices,
      lastClickedPauIdx,
    })
  }, [state, sessionId, pauBreaksMap, trimMap, savedIndices, skippedIndices, lastClickedPauIdx, saveState])

  /** Build the save payload — ALL pau breaks, including use_break=false */
  const buildSavePayload = (pauBreaks: PauBreak[]): PauBreakSavePayload[] => {
    return pauBreaks.map((pb) => {
      const deltaMs = pb.ms != null && pb.ms_proposed != null
        ? pb.ms - pb.ms_proposed
        : null
      return {
        pau_idx: pb.pau_idx,
        token_position: pb.token_position,
        ms_proposed: pb.ms_proposed,
        ms: pb.ms,
        delta_ms: deltaMs,
        use_break: pb.use_break,
        noise_zone_ms: pb.noise_zone_ms ?? null,
      }
    })
  }

  const saveCurrentItem = async (idx: number) => {
    if (!sessionId) return
    const pauBreaks = pauBreaksMap.get(idx) ?? []
    const trim = trimMap.get(idx)
    await saveLabels(
      sessionId, idx, buildSavePayload(pauBreaks), 'labeled',
      trim?.startMs, trim?.endMs,
    )
    setSavedIndices((prev) => new Set(prev).add(idx))
    // Remove from skipped if it was previously skipped
    setSkippedIndices((prev) => {
      if (!prev.has(idx)) return prev
      const next = new Set(prev)
      next.delete(idx)
      return next
    })
  }

  const saveCurrent = useCallback(async () => {
    if (!sessionId || state.kind !== 'labeling' || !dirty) return
    await saveCurrentItem(state.currentIdx)
    setDirty(false)
    showToast('Saved', 'success')
  }, [sessionId, state, pauBreaksMap, trimMap, dirty, showToast])

  const exitToSessionList = useCallback(() => {
    if (datasetName) {
      setState({ kind: 'loading' })
      getDatasetDetail(datasetName)
        .then((detail) => setState({ kind: 'pick_session', dataset: detail }))
        .catch((err) => setState({ kind: 'error', message: String(err) }))
    }
  }, [datasetName])

  const navigate = useCallback(
    async (direction: 'prev' | 'next') => {
      if (state.kind !== 'labeling') return
      // Auto-save before navigating
      if (dirty && sessionId) {
        await saveCurrentItem(state.currentIdx)
        setDirty(false)
      }
      // End of batch — check for unlabeled items
      if (direction === 'next' && state.currentIdx >= state.batch.items.length - 1) {
        waveformRef.current?.pause()
        setIsPlaying(false)

        // Find first unlabeled item
        let firstUnlabeled: number | null = null
        for (let i = 0; i < state.batch.items.length; i++) {
          if (!savedIndices.has(i)) {
            firstUnlabeled = i
            break
          }
        }

        if (firstUnlabeled !== null) {
          // Offer to jump to unlabeled
          setConfirmModal({
            message: `${state.batch.items.length - savedIndices.size} item(s) unlabeled. Jump to first unlabeled?`,
            onConfirm: () => {
              setConfirmModal(null)
              setState({ ...state, currentIdx: firstUnlabeled! })
            },
          })
        } else {
          // All labeled, exit
          const labeledCount = savedIndices.size - skippedIndices.size
          const skippedCount = skippedIndices.size
          const summary = skippedCount > 0
            ? `${labeledCount} labeled, ${skippedCount} skipped`
            : `${labeledCount} labeled`
          setConfirmModal({
            message: `Batch complete (${summary}). Return to session list?`,
            onConfirm: () => {
              setConfirmModal(null)
              exitToSessionList()
            },
          })
        }
        return
      }
      waveformRef.current?.pause()
      setIsPlaying(false)
      setLastClickedMs(null)
      const newIdx =
        direction === 'next'
          ? Math.min(state.currentIdx + 1, state.batch.items.length - 1)
          : Math.max(state.currentIdx - 1, 0)
      setState({ ...state, currentIdx: newIdx })
    },
    [state, pauBreaksMap, sessionId, dirty, savedIndices, skippedIndices, exitToSessionList],
  )

  const skipCurrent = useCallback(async () => {
    if (!sessionId || state.kind !== 'labeling') return
    const pauBreaks = pauBreaksMap.get(state.currentIdx) ?? []
    const trim = trimMap.get(state.currentIdx)
    await saveLabels(sessionId, state.currentIdx, buildSavePayload(pauBreaks), 'skipped', trim?.startMs, trim?.endMs)
    setSavedIndices((prev) => new Set(prev).add(state.currentIdx))
    setSkippedIndices((prev) => new Set(prev).add(state.currentIdx))
    setDirty(false)
    // Auto-advance to next
    waveformRef.current?.pause()
    setIsPlaying(false)
    const newIdx = Math.min(state.currentIdx + 1, state.batch.items.length - 1)
    if (newIdx !== state.currentIdx) {
      setState({ ...state, currentIdx: newIdx })
    }
  }, [sessionId, state, pauBreaksMap])

  const undoSkip = useCallback(async () => {
    if (!sessionId || state.kind !== 'labeling') return
    const pauBreaks = pauBreaksMap.get(state.currentIdx) ?? []
    const trim = trimMap.get(state.currentIdx)
    // Save as 'labeled' instead of 'skipped'
    await saveLabels(sessionId, state.currentIdx, buildSavePayload(pauBreaks), 'labeled', trim?.startMs, trim?.endMs)
    // Keep in savedIndices, remove from skippedIndices
    setSkippedIndices((prev) => {
      const next = new Set(prev)
      next.delete(state.currentIdx)
      return next
    })
    setDirty(false)
    showToast('Marked as labeled', 'success')
  }, [sessionId, state, pauBreaksMap, trimMap, showToast])

  const onTogglePau = useCallback(
    (pauIdx: number) => {
      if (state.kind !== 'labeling') return
      pushUndo()
      setPauBreaksMap((prev) => {
        const next = new Map(prev)
        const breaks = [...(next.get(state.currentIdx) ?? [])]
        const i = breaks.findIndex((pb) => pb.pau_idx === pauIdx)
        if (i >= 0) {
          breaks[i] = { ...breaks[i], use_break: !breaks[i].use_break }
          if (breaks[i].ms != null) setLastClickedMs(breaks[i].ms)
        }
        next.set(state.currentIdx, breaks)
        return next
      })
      setDirty(true)
    },
    [state, pushUndo],
  )

  const onMarkerDrag = useCallback(
    (pauIdx: number, newMs: number) => {
      if (state.kind !== 'labeling') return
      pushUndo()
      setLastClickedMs(newMs)
      setLastClickedPauIdx(pauIdx)
      setPauBreaksMap((prev) => {
        const next = new Map(prev)
        const breaks = [...(next.get(state.currentIdx) ?? [])]
        const i = breaks.findIndex((pb) => pb.pau_idx === pauIdx)
        if (i >= 0) {
          const pb = breaks[i]
          // Dragging implies intent to use this break
          breaks[i] = { ...pb, ms: newMs, use_break: true }
          // Update region end if playing this pau's noise zone
          if (pb.noise_zone_ms != null) {
            const newEnd = Math.max(newMs, pb.noise_zone_ms)
            waveformRef.current?.updateRegionEnd(newEnd)
          }
        }
        next.set(state.currentIdx, breaks)
        return next
      })
      setDirty(true)
    },
    [state, pushUndo],
  )

  const onMarkerClick = useCallback(
    (ms: number, pauIdx?: number) => {
      setLastClickedMs(ms)
      if (pauIdx != null) setLastClickedPauIdx(pauIdx)
    },
    [],
  )

  // Shift+click on pau marker creates noise zone (b) at click position
  const onNoiseZoneSplit = useCallback(
    (pauIdx: number, noiseZoneMs: number) => {
      if (state.kind !== 'labeling') return
      pushUndo()
      setLastClickedMs(noiseZoneMs)
      setLastClickedPauIdx(pauIdx)
      setPauBreaksMap((prev) => {
        const next = new Map(prev)
        const breaks = [...(next.get(state.currentIdx) ?? [])]
        const i = breaks.findIndex((pb) => pb.pau_idx === pauIdx)
        if (i >= 0) {
          breaks[i] = { ...breaks[i], noise_zone_ms: noiseZoneMs }
        }
        next.set(state.currentIdx, breaks)
        return next
      })
      setDirty(true)
    },
    [state, pushUndo],
  )

  // Drag noise zone (b) marker
  const onNoiseZoneDrag = useCallback(
    (pauIdx: number, noiseZoneMs: number) => {
      if (state.kind !== 'labeling') return
      pushUndo()
      setLastClickedMs(noiseZoneMs)
      setLastClickedPauIdx(pauIdx)
      setPauBreaksMap((prev) => {
        const next = new Map(prev)
        const breaks = [...(next.get(state.currentIdx) ?? [])]
        const i = breaks.findIndex((pb) => pb.pau_idx === pauIdx)
        if (i >= 0) {
          const pb = breaks[i]
          breaks[i] = { ...pb, noise_zone_ms: noiseZoneMs }
          // Update region end if playing this pau's noise zone
          if (pb.ms != null) {
            const newEnd = Math.max(pb.ms, noiseZoneMs)
            waveformRef.current?.updateRegionEnd(newEnd)
          }
        }
        next.set(state.currentIdx, breaks)
        return next
      })
      setDirty(true)
    },
    [state, pushUndo],
  )

  // Drag noise zone window — moves both (a) and (b) together
  const onNoiseZoneWindowDrag = useCallback(
    (pauIdx: number, newMs: number, newNoiseZoneMs: number) => {
      if (state.kind !== 'labeling') return
      pushUndo()
      setLastClickedMs(newMs)
      setLastClickedPauIdx(pauIdx)
      setPauBreaksMap((prev) => {
        const next = new Map(prev)
        const breaks = [...(next.get(state.currentIdx) ?? [])]
        const i = breaks.findIndex((pb) => pb.pau_idx === pauIdx)
        if (i >= 0) {
          breaks[i] = { ...breaks[i], ms: newMs, noise_zone_ms: newNoiseZoneMs }
          // Update region end if playing this pau's noise zone
          const newEnd = Math.max(newMs, newNoiseZoneMs)
          waveformRef.current?.updateRegionEnd(newEnd)
        }
        next.set(state.currentIdx, breaks)
        return next
      })
      setDirty(true)
    },
    [state, pushUndo],
  )

  // Esc key: removes noise zone OR prompts early exit
  useEffect(() => {
    if (state.kind !== 'labeling') return
    const handler = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // Don't handle if modal is open (let modal handler deal with it)
      if (confirmModal) return

      // If we have a noise zone selected, remove it
      if (lastClickedPauIdx != null) {
        const breaks = pauBreaksMap.get(state.currentIdx) ?? []
        const pb = breaks.find((b) => b.pau_idx === lastClickedPauIdx)
        if (pb?.noise_zone_ms != null) {
          // Push to undo stack before removing noise zone
          pushUndo()
          // Remove noise zone from the last clicked pau
          setPauBreaksMap((prev) => {
            const next = new Map(prev)
            const newBreaks = [...(next.get(state.currentIdx) ?? [])]
            const i = newBreaks.findIndex((b) => b.pau_idx === lastClickedPauIdx)
            if (i >= 0) {
              newBreaks[i] = { ...newBreaks[i], noise_zone_ms: null }
              next.set(state.currentIdx, newBreaks)
            }
            return next
          })
          setDirty(true)
          setLastClickedPauIdx(null)
          return
        }
      }

      // Otherwise, prompt for early exit
      setConfirmModal({
        message: `Exit session? (${savedIndices.size}/${state.batch.items.length} labeled)`,
        onConfirm: () => {
          setConfirmModal(null)
          exitToSessionList()
        },
      })
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [state, lastClickedPauIdx, confirmModal, pauBreaksMap, savedIndices, exitToSessionList, pushUndo])

  const onTrimDrag = useCallback(
    (edge: 'start' | 'end', newMs: number) => {
      if (state.kind !== 'labeling') return
      pushUndo()
      setLastClickedMs(newMs)
      setTrimMap((prev) => {
        const next = new Map(prev)
        const trim = { ...(next.get(state.currentIdx) ?? { startMs: 0, endMs: 0 }) }
        if (edge === 'start') {
          trim.startMs = newMs
        } else {
          trim.endMs = newMs
        }
        next.set(state.currentIdx, trim)
        return next
      })
      setDirty(true)
    },
    [state, pushUndo],
  )

  // Replay: seek to drag position (if dragging), last-clicked, or start
  const handleReplay = useCallback(() => {
    if (state.kind !== 'labeling') return
    const item = state.batch.items[state.currentIdx]
    const dragMs = waveformRef.current?.getDragMs()
    const seekMs = dragMs ?? lastClickedMs
    if (seekMs != null && item.duration_sec > 0) {
      waveformRef.current?.seekTo(seekMs / (item.duration_sec * 1000))
    } else {
      waveformRef.current?.seekTo(0)
    }
    waveformRef.current?.play()
  }, [state, lastClickedMs])

  // Replay noise zone window (F key): plays audio between (a) and (b) of last clicked pau
  // Falls back to 100ms snippet if no noise zone exists
  const handleReplaySnippet = useCallback(() => {
    if (state.kind !== 'labeling') return
    const item = state.batch.items[state.currentIdx]
    const durationMs = item.duration_sec * 1000
    if (durationMs <= 0) return

    // Find noise zone for last clicked pau
    const pauBreaks = pauBreaksMap.get(state.currentIdx) ?? []
    const pb = lastClickedPauIdx != null
      ? pauBreaks.find((p) => p.pau_idx === lastClickedPauIdx)
      : null

    if (pb?.ms != null && pb?.noise_zone_ms != null) {
      // Play the noise zone window (a to b) using region playback
      const startMs = Math.min(pb.ms, pb.noise_zone_ms)
      const endMs = Math.max(pb.ms, pb.noise_zone_ms)
      waveformRef.current?.playRegion(startMs, endMs)
    } else {
      // Fallback: 100ms snippet from last clicked position using region playback
      const dragMs = waveformRef.current?.getDragMs()
      const seekMs = dragMs ?? lastClickedMs
      if (seekMs != null) {
        waveformRef.current?.playRegion(seekMs, seekMs + 100)
      }
    }
  }, [state, lastClickedMs, lastClickedPauIdx, pauBreaksMap])

  // Direct play/pause handlers — called synchronously from click events
  const handlePlay = useCallback(() => waveformRef.current?.play(), [])
  const handlePause = useCallback(() => waveformRef.current?.pause(), [])
  const handlePlayPause = useCallback(() => waveformRef.current?.playPause(), [])

  // Render
  if (state.kind === 'loading') {
    return <div style={styles.container}><div style={styles.center}>Loading...</div></div>
  }

  if (state.kind === 'error') {
    return (
      <div style={styles.container}>
        <div style={styles.center}>
          <div style={{ color: '#ff6b6b' }}>Error: {state.message}</div>
        </div>
      </div>
    )
  }

  if (state.kind === 'pick_dataset') {
    return (
      <div style={styles.container}>
        <h1 style={styles.title}>KOE Segmentation Labeler</h1>
        <div style={styles.cardGrid}>
          {state.datasets.map((ds) => (
            <button key={ds.name} style={styles.card} onClick={() => selectDataset(ds.name)}>
              <div style={styles.cardTitle}>{ds.name}</div>
              <div style={styles.cardSub}>{ds.manifest_count} manifest(s)</div>
            </button>
          ))}
        </div>
      </div>
    )
  }

  if (state.kind === 'pick_session') {
    const ds = state.dataset
    const runs = ds.heuristic_runs ?? []
    const activeHash = selectedHeuristic ?? (runs.length > 0 ? runs[0].params_hash : null)
    const activeRun = runs.find((r) => r.params_hash === activeHash)
    const totalHeuristicHits = activeRun?.with_breaks ?? ds.strata.reduce((sum, s) => sum + s.heuristic_hits, 0)
    return (
      <div style={styles.container}>
        <h1 style={styles.title}>KOE Segmentation Labeler</h1>
        <h2 style={styles.subtitle}>{ds.name} - {ds.total_utterances.toLocaleString()} utterances</h2>

        {runs.length > 0 && (
          <>
            <h3 style={styles.sectionTitle}>Heuristic Run</h3>
            <div style={styles.filterRow}>
              <select
                value={activeHash ?? ''}
                onChange={(e) => setSelectedHeuristic(e.target.value || null)}
                style={styles.select}
              >
                {runs.map((r) => (
                  <option key={r.params_hash} value={r.params_hash}>
                    {r.name} — {r.with_breaks}/{r.processed} hits ({r.created_at})
                  </option>
                ))}
              </select>
            </div>
          </>
        )}

        <h3 style={styles.sectionTitle}>New Session</h3>
        <div style={styles.filterRow}>
          <label style={styles.filterLabel}>
            <input
              type="checkbox"
              checked={heuristicOnly}
              onChange={(e) => setHeuristicOnly(e.target.checked)}
              style={styles.checkbox}
            />
            Heuristic hits only
            <span style={styles.filterHint}>
              ({totalHeuristicHits} of {ds.total_utterances.toLocaleString()})
            </span>
          </label>
          <label style={styles.filterLabel}>
            Batch:
            <select
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              style={styles.select}
            >
              {[10, 25, 50, 100].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </label>
          <label style={styles.filterLabel}>
            Sample:
            <select
              value={samplingMode}
              onChange={(e) => setSamplingMode(e.target.value as 'random' | 'stratum')}
              style={styles.select}
            >
              <option value="random">Random</option>
              <option value="stratum">By stratum (pau)</option>
            </select>
          </label>
          {samplingMode === 'stratum' && (
            <label style={styles.filterLabel}>
              Stratum:
              <select
                value={selectedStratum}
                onChange={(e) => setSelectedStratum(Number(e.target.value))}
                style={styles.select}
              >
                {ds.strata.map((s) => {
                  const count = heuristicOnly ? s.heuristic_hits : s.count
                  return (
                    <option key={s.stratum} value={s.stratum} disabled={count === 0}>
                      {s.label} ({count.toLocaleString()})
                    </option>
                  )
                })}
              </select>
            </label>
          )}
        </div>
        <button
          style={styles.startBtn}
          onClick={() => startSession(
            ds.name,
            batchSize,
            samplingMode === 'stratum' ? selectedStratum : null,
            heuristicOnly,
            activeHash,
          )}
        >
          Start labeling
        </button>

        {ds.sessions.filter((s) => !s.published).length > 0 && (
          <>
            <div style={styles.sectionHeader}>
              <h3 style={styles.sectionTitle}>Resume Session</h3>
              <button
                style={styles.clearAllBtn}
                onClick={() => handleDeleteAll(ds.name)}
              >
                Clear all
              </button>
            </div>
            <div style={styles.cardGrid}>
              {ds.sessions.filter((s) => !s.published).map((s) => (
                <div key={s.session_id} style={styles.sessionCardWrap}>
                  <button
                    style={styles.card}
                    onClick={() => resumeSession(s.session_id, ds.name)}
                  >
                    <div style={styles.cardTitle}>{s.session_id.slice(0, 15)}</div>
                    <div style={styles.cardSub}>
                      {s.labeled + s.skipped}/{s.total} done ({s.pct}%)
                    </div>
                  </button>
                  {s.labeled > 0 && (
                    <button
                      style={styles.publishBtn}
                      onClick={(e) => {
                        e.stopPropagation()
                        handlePublish(s.session_id, ds.name)
                      }}
                      title="Publish labels to table"
                    >
                      &#9992;
                    </button>
                  )}
                  <button
                    style={styles.deleteBtn}
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDeleteSession(s.session_id, ds.name)
                    }}
                    title="Delete session"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </>
        )}

        {ds.sessions.filter((s) => s.published).length > 0 && (
          <>
            <h3 style={styles.sectionTitle}>Published</h3>
            <div style={styles.cardGrid}>
              {ds.sessions.filter((s) => s.published).map((s) => (
                <div key={s.session_id} style={styles.sessionCardWrap}>
                  <button
                    style={{ ...styles.card, ...styles.cardPublished }}
                    onClick={() => resumeSession(s.session_id, ds.name)}
                  >
                    <div style={styles.cardTitle}>
                      {s.session_id.slice(0, 15)}
                      <span style={styles.publishedBadge}>published</span>
                    </div>
                    <div style={styles.cardSub}>
                      {s.labeled + s.skipped}/{s.total} done ({s.pct}%)
                    </div>
                  </button>
                  <button
                    style={styles.deleteBtn}
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDeleteSession(s.session_id, ds.name)
                    }}
                    title="Delete session"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </>
        )}

        {confirmModal && (
          <div style={styles.modalOverlay}>
            <div style={styles.modal}>
              <div style={styles.modalText}>{confirmModal.message}</div>
              <div style={styles.modalActions}>
                <button style={styles.modalBtnY} onClick={confirmModal.onConfirm}>
                  Yes (Y)
                </button>
                <button style={styles.modalBtnN} onClick={() => setConfirmModal(null)}>
                  No (N)
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // Labeling view
  const { batch, currentIdx } = state
  const item = batch.items[currentIdx]
  const currentPauBreaks = pauBreaksMap.get(currentIdx) ?? []

  // Build waveform markers from pau breaks (all that have ms positions)
  const markers: WaveformMarker[] = currentPauBreaks
    .filter((pb) => pb.ms != null || pb.ms_proposed != null)
    .map((pb) => ({
      pauIdx: pb.pau_idx,
      ms: pb.ms,
      msProposed: pb.ms_proposed,
      useBreak: pb.use_break,
      noiseZoneMs: pb.noise_zone_ms ?? null,
    }))

  const durationMs = Math.round(item.duration_sec * 1000)
  const currentTrim = trimMap.get(currentIdx) ?? {
    startMs: Math.min(100, durationMs),
    endMs: Math.max(durationMs - 100, 0),
  }

  return (
    <div style={styles.container}>
      <BatchInfo
        dataset={batch.session_id.split('_').slice(0, -1).join('_')}
        sessionId={batch.session_id}
        total={batch.items.length}
        labeled={savedIndices.size}
      />

      <Waveform
        ref={waveformRef}
        audioUrl={item.audio_url}
        markers={markers}
        trim={currentTrim}
        duration={item.duration_sec}
        onMarkerDrag={onMarkerDrag}
        onMarkerToggle={onTogglePau}
        onMarkerClick={onMarkerClick}
        onTrimDrag={onTrimDrag}
        onPlayStateChange={setIsPlaying}
        onNoiseZoneSplit={onNoiseZoneSplit}
        onNoiseZoneDrag={onNoiseZoneDrag}
        onNoiseZoneWindowDrag={onNoiseZoneWindowDrag}
      />

      <PhonemePills
        phonemes={item.phonemes}
        pauBreaks={currentPauBreaks}
        onTogglePau={onTogglePau}
      />

      <TextDisplay text={item.text} speakerId={item.speaker_id} />

      <NavControls
        onPrev={() => navigate('prev')}
        onNext={() => navigate('next')}
        onPlay={handlePlay}
        onPause={handlePause}
        onPlayPause={handlePlayPause}
        onSave={saveCurrent}
        onSkip={skipCurrent}
        onUndoSkip={undoSkip}
        onReplay={handleReplay}
        onReplaySnippet={handleReplaySnippet}
        onUndo={handleUndo}
        onRedo={handleRedo}
        currentIdx={currentIdx}
        total={batch.items.length}
        isPlaying={isPlaying}
        isDirty={dirty}
        isSaved={savedIndices.has(currentIdx)}
        isSkipped={skippedIndices.has(currentIdx)}
      />

      {confirmModal && (
        <div style={styles.modalOverlay}>
          <div style={styles.modal}>
            <div style={styles.modalText}>{confirmModal.message}</div>
            <div style={styles.modalActions}>
              <button style={styles.modalBtnY} onClick={confirmModal.onConfirm}>
                Yes (Y)
              </button>
              <button style={styles.modalBtnN} onClick={() => setConfirmModal(null)}>
                No (N)
              </button>
            </div>
          </div>
        </div>
      )}

      {showHotkeyHelp && <HotkeyHelp onClose={() => setShowHotkeyHelp(false)} />}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: 900,
    margin: '0 auto',
    padding: '20px 16px',
  },
  center: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: '50vh',
    fontSize: 18,
  },
  title: {
    fontSize: 24,
    fontWeight: 600,
    marginBottom: 8,
    color: '#fff',
  },
  subtitle: {
    fontSize: 16,
    color: '#aaa',
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: 600,
    color: '#888',
    textTransform: 'uppercase' as const,
    letterSpacing: 1,
    marginBottom: 12,
    marginTop: 24,
  },
  cardGrid: {
    display: 'flex',
    gap: 12,
    flexWrap: 'wrap' as const,
  },
  card: {
    background: '#16213e',
    border: '1px solid #2a2a4a',
    borderRadius: 8,
    padding: '16px 20px',
    cursor: 'pointer',
    color: '#e0e0e0',
    fontSize: 14,
    textAlign: 'left' as const,
    minWidth: 150,
  },
  cardTitle: {
    fontWeight: 600,
    marginBottom: 4,
  },
  cardSub: {
    fontSize: 12,
    color: '#888',
  },
  filterRow: {
    display: 'flex',
    gap: 24,
    alignItems: 'center',
    flexWrap: 'wrap' as const,
    marginBottom: 8,
  },
  filterLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    fontSize: 14,
    color: '#ccc',
  },
  filterHint: {
    fontSize: 12,
    color: '#666',
  },
  checkbox: {
    accentColor: '#4a9eff',
  },
  select: {
    background: '#16213e',
    border: '1px solid #2a2a4a',
    borderRadius: 4,
    color: '#e0e0e0',
    padding: '4px 8px',
    fontSize: 14,
  },
  startBtn: {
    marginTop: 16,
    background: '#1a5cb0',
    border: 'none',
    borderRadius: 8,
    color: '#fff',
    fontSize: 15,
    fontWeight: 600,
    padding: '10px 32px',
    cursor: 'pointer',
  },
  sectionHeader: {
    display: 'flex',
    alignItems: 'baseline',
    gap: 12,
  },
  clearAllBtn: {
    background: 'none',
    border: '1px solid #553333',
    borderRadius: 4,
    color: '#cc6666',
    fontSize: 12,
    padding: '2px 8px',
    cursor: 'pointer',
  },
  sessionCardWrap: {
    position: 'relative' as const,
  },
  cardPublished: {
    borderColor: '#2e7d32',
    opacity: 0.7,
  },
  publishedBadge: {
    fontSize: 9,
    color: '#4caf50',
    background: '#1b3a1b',
    padding: '1px 5px',
    borderRadius: 3,
    marginLeft: 6,
    fontWeight: 400,
  },
  publishBtn: {
    position: 'absolute' as const,
    top: -6,
    right: 18,
    width: 20,
    height: 20,
    borderRadius: '50%',
    background: '#1b3a1b',
    border: '1px solid #2e7d32',
    color: '#4caf50',
    fontSize: 12,
    lineHeight: '18px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    opacity: 0.7,
    padding: 0,
  },
  deleteBtn: {
    position: 'absolute' as const,
    top: -6,
    right: -6,
    width: 20,
    height: 20,
    borderRadius: '50%',
    background: '#442222',
    border: '1px solid #663333',
    color: '#cc6666',
    fontSize: 14,
    lineHeight: '18px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    opacity: 0.5,
    padding: 0,
  },
  modalOverlay: {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 100,
  },
  modal: {
    background: '#1a1a2e',
    border: '1px solid #2a2a4a',
    borderRadius: 12,
    padding: '24px 32px',
    minWidth: 320,
    textAlign: 'center' as const,
  },
  modalText: {
    color: '#e0e0e0',
    fontSize: 16,
    marginBottom: 20,
  },
  modalActions: {
    display: 'flex',
    gap: 12,
    justifyContent: 'center',
  },
  modalBtnY: {
    background: '#cc3333',
    border: 'none',
    borderRadius: 6,
    color: '#fff',
    fontSize: 14,
    padding: '8px 20px',
    cursor: 'pointer',
    fontWeight: 600,
  },
  modalBtnN: {
    background: '#333',
    border: '1px solid #555',
    borderRadius: 6,
    color: '#ccc',
    fontSize: 14,
    padding: '8px 20px',
    cursor: 'pointer',
  },
}
