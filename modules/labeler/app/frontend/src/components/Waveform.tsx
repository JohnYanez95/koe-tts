import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'

export interface WaveformHandle {
  play: () => void
  pause: () => void
  playPause: () => void
  seekTo: (pct: number) => void
  getDragMs: () => number | null
  playRegion: (startMs: number, endMs: number) => void
  updateRegionEnd: (endMs: number) => void  // Update end position during region playback
}

export interface WaveformMarker {
  pauIdx: number
  ms: number | null
  msProposed: number | null
  useBreak: boolean
  noiseZoneMs: number | null  // (b) marker position — noise zone boundary
}

export interface TrimState {
  startMs: number
  endMs: number
}

interface Props {
  audioUrl: string
  markers: WaveformMarker[]
  trim: TrimState
  duration: number
  onMarkerDrag: (pauIdx: number, newMs: number) => void
  onMarkerToggle: (pauIdx: number) => void
  onMarkerClick: (ms: number, pauIdx?: number) => void
  onTrimDrag: (edge: 'start' | 'end', newMs: number) => void
  onPlayStateChange: (playing: boolean) => void
  onNoiseZoneSplit?: (pauIdx: number, noiseZoneMs: number) => void
  onNoiseZoneDrag?: (pauIdx: number, noiseZoneMs: number) => void
  onNoiseZoneWindowDrag?: (pauIdx: number, newMs: number, newNoiseZoneMs: number) => void
}

const SNAP_FINE = 5
const SNAP_COARSE = 10

// Special pauIdx values for trim markers (negative to avoid collision)
const TRIM_START_ID = -1
const TRIM_END_ID = -2

// Noise zone (b) markers use offset IDs: base pauIdx + NOISE_ZONE_OFFSET
const NOISE_ZONE_OFFSET = 10000
const isNoiseZoneId = (id: number) => id >= NOISE_ZONE_OFFSET
const toNoiseZoneId = (pauIdx: number) => pauIdx + NOISE_ZONE_OFFSET
const fromNoiseZoneId = (id: number) => id - NOISE_ZONE_OFFSET

function snapMs(ms: number, coarse: boolean): number {
  const snap = coarse ? SNAP_COARSE : SNAP_FINE
  return Math.round(ms / snap) * snap
}

// Special ID for noise zone window drag (encodes pauIdx)
const NOISE_ZONE_WINDOW_OFFSET = 20000
const isNoiseZoneWindowId = (id: number) => id >= NOISE_ZONE_WINDOW_OFFSET
const toNoiseZoneWindowId = (pauIdx: number) => pauIdx + NOISE_ZONE_WINDOW_OFFSET
const fromNoiseZoneWindowId = (id: number) => id - NOISE_ZONE_WINDOW_OFFSET

export const Waveform = forwardRef<WaveformHandle, Props>(function Waveform(
  { audioUrl, markers, trim, duration, onMarkerDrag, onMarkerToggle, onMarkerClick, onTrimDrag, onPlayStateChange, onNoiseZoneSplit, onNoiseZoneDrag, onNoiseZoneWindowDrag },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const regionEndMsRef = useRef<number | null>(null)  // For region playback: stop when reaching this position
  const [ready, setReady] = useState(false)
  const [dragging, setDragging] = useState<{
    pauIdx: number
    ms: number
    startX: number
    moved: boolean
    windowStartMs?: number  // for window drag: original (a) position
    windowEndMs?: number    // for window drag: original (b) position
  } | null>(null)
  const durationMs = duration * 1000

  useImperativeHandle(ref, () => ({
    play: () => {
      regionEndMsRef.current = null  // Clear region end on normal play
      if (wsRef.current && ready) wsRef.current.play()
    },
    pause: () => {
      regionEndMsRef.current = null  // Clear region end
      if (wsRef.current) wsRef.current.pause()
    },
    playPause: () => {
      if (wsRef.current && ready) wsRef.current.playPause()
    },
    seekTo: (pct: number) => {
      if (wsRef.current) wsRef.current.seekTo(Math.max(0, Math.min(1, pct)))
    },
    getDragMs: () => {
      return dragging?.ms ?? null
    },
    playRegion: (startMs: number, endMs: number) => {
      if (!wsRef.current || !ready || durationMs <= 0) return
      // Set the end position for the timeupdate handler to check
      regionEndMsRef.current = endMs
      // Seek to start and play
      wsRef.current.seekTo(startMs / durationMs)
      wsRef.current.play()
    },
    updateRegionEnd: (endMs: number) => {
      // Update the region end position (used when user drags boundary during playback)
      // If playback is past the new end, pause immediately
      if (regionEndMsRef.current != null) {
        regionEndMsRef.current = endMs
        if (wsRef.current) {
          const currentMs = wsRef.current.getCurrentTime() * 1000
          if (currentMs >= endMs) {
            regionEndMsRef.current = null
            wsRef.current.pause()
          }
        }
      }
    },
  }), [ready, dragging, durationMs])

  // Initialize WaveSurfer
  useEffect(() => {
    if (!containerRef.current) return

    setReady(false)

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#4a9eff',
      progressColor: '#1a5cb0',
      cursorColor: '#fff',
      cursorWidth: 1,
      height: 128,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
    })

    ws.on('ready', () => setReady(true))
    ws.on('finish', () => {
      regionEndMsRef.current = null
      onPlayStateChange(false)
    })
    ws.on('pause', () => onPlayStateChange(false))
    ws.on('play', () => onPlayStateChange(true))

    // Region playback: stop when reaching the end position
    ws.on('timeupdate', (currentTime: number) => {
      const endMs = regionEndMsRef.current
      if (endMs != null) {
        const currentMs = currentTime * 1000
        if (currentMs >= endMs) {
          regionEndMsRef.current = null  // Clear to prevent multiple pauses
          ws.pause()
        }
      }
    })

    ws.load(audioUrl)
    wsRef.current = ws

    return () => {
      ws.destroy()
      wsRef.current = null
      setReady(false)
    }
  }, [audioUrl, onPlayStateChange])

  // Right-click: seek to position
  const handleContextMenu = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      e.preventDefault()
      if (!overlayRef.current || !wsRef.current || duration <= 0) return

      const rect = overlayRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const pct = x / rect.width
      wsRef.current.seekTo(Math.max(0, Math.min(1, pct)))
    },
    [duration],
  )

  // Drag handlers — uses pointer capture for reliable tracking
  // Click (no significant movement) = toggle use_break (pau markers only)
  // Drag (moved > 2px) = reposition ms
  // Shift+click on pau marker = create noise zone (b) at click position
  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>, pauIdx: number, isShiftSplit = false, windowMs?: number, windowNoiseZoneMs?: number) => {
      e.preventDefault()
      e.stopPropagation()
      const target = e.currentTarget
      target.setPointerCapture(e.pointerId)

      if (!overlayRef.current || durationMs <= 0) return
      const rect = overlayRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const ms = snapMs(Math.max(0, Math.min(durationMs, (x / rect.width) * durationMs)), e.shiftKey)

      // Shift+click on a pau marker creates a noise zone at click position
      if (isShiftSplit && onNoiseZoneSplit && pauIdx >= 0 && !isNoiseZoneId(pauIdx) && !isNoiseZoneWindowId(pauIdx)) {
        onNoiseZoneSplit(pauIdx, ms)
        return
      }

      // Window drag stores original positions
      if (isNoiseZoneWindowId(pauIdx) && windowMs != null && windowNoiseZoneMs != null) {
        setDragging({ pauIdx, ms, startX: e.clientX, moved: false, windowStartMs: windowMs, windowEndMs: windowNoiseZoneMs })
      } else {
        setDragging({ pauIdx, ms, startX: e.clientX, moved: false })
      }
    },
    [durationMs, onNoiseZoneSplit],
  )

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging || !overlayRef.current || durationMs <= 0) return
      const moved = dragging.moved || Math.abs(e.clientX - dragging.startX) > 2
      const rect = overlayRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const ms = snapMs(Math.max(0, Math.min(durationMs, (x / rect.width) * durationMs)), e.shiftKey)
      setDragging({ ...dragging, ms, moved })
    },
    [dragging, durationMs],
  )

  // Middle-click on a pau marker toggles use_break
  const handleAuxClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>, pauIdx: number) => {
      if (e.button === 1) {
        e.preventDefault()
        onMarkerToggle(pauIdx)
      }
    },
    [onMarkerToggle],
  )

  const handlePointerUp = useCallback(
    (_e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging) return
      if (dragging.pauIdx === TRIM_START_ID || dragging.pauIdx === TRIM_END_ID) {
        // Trim marker — always drag (no toggle)
        if (dragging.moved) {
          const edge = dragging.pauIdx === TRIM_START_ID ? 'start' : 'end'
          onTrimDrag(edge, dragging.ms)
        } else {
          // Clicked trim marker without moving — report position for replay (no pauIdx)
          onMarkerClick(dragging.ms)
        }
      } else if (isNoiseZoneWindowId(dragging.pauIdx)) {
        // Noise zone window drag completed — move both (a) and (b) together
        const realPauIdx = fromNoiseZoneWindowId(dragging.pauIdx)
        if (dragging.moved && onNoiseZoneWindowDrag && dragging.windowStartMs != null && dragging.windowEndMs != null) {
          // Calculate delta: current position - original window center
          const windowCenter = (dragging.windowStartMs + dragging.windowEndMs) / 2
          const delta = dragging.ms - windowCenter
          const newMs = dragging.windowStartMs + delta
          const newNoiseZoneMs = dragging.windowEndMs + delta
          onNoiseZoneWindowDrag(realPauIdx, Math.max(0, newMs), Math.max(0, newNoiseZoneMs))
        } else {
          onMarkerClick(dragging.ms, realPauIdx)
        }
      } else if (isNoiseZoneId(dragging.pauIdx)) {
        // Noise zone (b) marker drag completed
        const realPauIdx = fromNoiseZoneId(dragging.pauIdx)
        if (dragging.moved && onNoiseZoneDrag) {
          onNoiseZoneDrag(realPauIdx, dragging.ms)
        } else {
          // Clicked (b) marker — report position with pauIdx for F key replay
          onMarkerClick(dragging.ms, realPauIdx)
        }
      } else if (dragging.moved) {
        // Pau marker drag completed — update ms position
        onMarkerDrag(dragging.pauIdx, dragging.ms)
      } else {
        // Clicked (a) marker without moving — report position with pauIdx for F key replay
        onMarkerClick(dragging.ms, dragging.pauIdx)
      }
      setDragging(null)
    },
    [dragging, onMarkerDrag, onMarkerClick, onTrimDrag, onNoiseZoneDrag, onNoiseZoneWindowDrag],
  )

  // Compute display ms for each marker (use drag position if currently dragging that marker)
  const getDisplayMs = (m: WaveformMarker): number | null => {
    if (dragging && dragging.pauIdx === m.pauIdx) return dragging.ms
    // Window drag: compute new (a) position
    if (dragging && isNoiseZoneWindowId(dragging.pauIdx) && fromNoiseZoneWindowId(dragging.pauIdx) === m.pauIdx) {
      if (dragging.windowStartMs != null && dragging.windowEndMs != null && m.ms != null) {
        const windowCenter = (dragging.windowStartMs + dragging.windowEndMs) / 2
        const delta = dragging.ms - windowCenter
        return Math.max(0, dragging.windowStartMs + delta)
      }
    }
    return m.ms
  }

  // Compute display ms for noise zone (b) marker
  const getNoiseZoneDisplayMs = (m: WaveformMarker): number | null => {
    if (dragging && dragging.pauIdx === toNoiseZoneId(m.pauIdx)) return dragging.ms
    // Window drag: compute new (b) position
    if (dragging && isNoiseZoneWindowId(dragging.pauIdx) && fromNoiseZoneWindowId(dragging.pauIdx) === m.pauIdx) {
      if (dragging.windowStartMs != null && dragging.windowEndMs != null && m.noiseZoneMs != null) {
        const windowCenter = (dragging.windowStartMs + dragging.windowEndMs) / 2
        const delta = dragging.ms - windowCenter
        return Math.max(0, dragging.windowEndMs + delta)
      }
    }
    return m.noiseZoneMs
  }

  const getTrimDisplayMs = (id: number, defaultMs: number): number => {
    if (dragging && dragging.pauIdx === id) return dragging.ms
    return defaultMs
  }

  // Trim positions
  const trimStartMs = getTrimDisplayMs(TRIM_START_ID, trim.startMs)
  const trimEndMs = getTrimDisplayMs(TRIM_END_ID, trim.endMs)
  const trimStartPct = durationMs > 0 ? (trimStartMs / durationMs) * 100 : 0
  const trimEndPct = durationMs > 0 ? (trimEndMs / durationMs) * 100 : 100
  const isDraggingTrimStart = dragging?.pauIdx === TRIM_START_ID
  const isDraggingTrimEnd = dragging?.pauIdx === TRIM_END_ID

  return (
    <div style={styles.wrapper}>
      <div style={styles.waveContainer}>
        <div ref={containerRef} style={styles.wave} />
        <div
          ref={overlayRef}
          style={styles.overlay}
          onContextMenu={handleContextMenu}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
        >
          {/* Trim shaded regions (dimmed areas outside S–E) */}
          {trimStartPct > 0 && (
            <div style={{ ...styles.trimShade, left: 0, width: `${trimStartPct}%` }} />
          )}
          {trimEndPct < 100 && (
            <div style={{ ...styles.trimShade, left: `${trimEndPct}%`, right: 0, width: 'auto' }} />
          )}

          {/* S trim marker */}
          <div
            style={{
              ...styles.trimLine,
              ...(isDraggingTrimStart ? styles.trimLineDragging : {}),
              left: `${trimStartPct}%`,
            }}
          />
          <div
            style={{
              ...styles.trimHandle,
              left: `${trimStartPct}%`,
            }}
            onPointerDown={(e) => handlePointerDown(e, TRIM_START_ID)}
            title={`Start trim: ${trimStartMs}ms`}
          >
            S
          </div>
          {isDraggingTrimStart && (
            <div style={{ ...styles.trimLabel, left: `${trimStartPct}%` }}>
              {trimStartMs}ms
            </div>
          )}

          {/* E trim marker */}
          <div
            style={{
              ...styles.trimLine,
              ...(isDraggingTrimEnd ? styles.trimLineDragging : {}),
              left: `${trimEndPct}%`,
            }}
          />
          <div
            style={{
              ...styles.trimHandle,
              left: `${trimEndPct}%`,
            }}
            onPointerDown={(e) => handlePointerDown(e, TRIM_END_ID)}
            title={`End trim: ${trimEndMs}ms`}
          >
            E
          </div>
          {isDraggingTrimEnd && (
            <div style={{ ...styles.trimLabel, left: `${trimEndPct}%` }}>
              {trimEndMs}ms
            </div>
          )}

          {/* Pau markers */}
          {markers.map((m) => {
            const displayMs = getDisplayMs(m)
            const noiseZoneMs = getNoiseZoneDisplayMs(m)
            const proposedPct = m.msProposed != null && durationMs > 0
              ? (m.msProposed / durationMs) * 100
              : null
            const currentPct = displayMs != null && durationMs > 0
              ? (displayMs / durationMs) * 100
              : null
            const noiseZonePct = noiseZoneMs != null && durationMs > 0
              ? (noiseZoneMs / durationMs) * 100
              : null
            const isDraggingThis = dragging?.pauIdx === m.pauIdx
            const isDraggingNoiseZone = dragging?.pauIdx === toNoiseZoneId(m.pauIdx)
            const deltaMs = displayMs != null && m.msProposed != null
              ? displayMs - m.msProposed
              : null
            // Noise zone region bounds (a to b)
            const hasNoiseZone = currentPct != null && noiseZonePct != null
            const noiseZoneLeft = hasNoiseZone ? Math.min(currentPct, noiseZonePct) : 0
            const noiseZoneWidth = hasNoiseZone ? Math.abs(noiseZonePct - currentPct) : 0

            const isDraggingWindow = dragging && isNoiseZoneWindowId(dragging.pauIdx) && fromNoiseZoneWindowId(dragging.pauIdx) === m.pauIdx

            return (
              <div key={m.pauIdx}>
                {/* Noise zone shaded region (between a and b) — clickable to drag whole window */}
                {hasNoiseZone && (
                  <div
                    style={{
                      ...styles.noiseZoneShade,
                      ...(isDraggingWindow ? styles.noiseZoneDragging : {}),
                      left: `${noiseZoneLeft}%`,
                      width: `${noiseZoneWidth}%`,
                      cursor: 'grab',
                      pointerEvents: 'auto',
                    }}
                    onPointerDown={(e) => handlePointerDown(e, toNoiseZoneWindowId(m.pauIdx), false, displayMs!, noiseZoneMs!)}
                    title={`Noise zone: ${Math.min(displayMs!, noiseZoneMs!)}–${Math.max(displayMs!, noiseZoneMs!)}ms [drag to move window]`}
                  />
                )}
                {/* Ghost line at ms_proposed (always visible, dashed) */}
                {proposedPct != null && (
                  <div
                    style={{
                      ...styles.ghostLine,
                      left: `${proposedPct}%`,
                    }}
                    title={`proposed: ${m.msProposed}ms`}
                  />
                )}
                {/* Current position line (a) - solid, colored by use_break */}
                {currentPct != null && (
                  <>
                    <div
                      style={{
                        ...styles.markerLine,
                        ...(m.useBreak ? styles.markerActive : styles.markerInactive),
                        ...(isDraggingThis ? styles.markerDragging : {}),
                        left: `${currentPct}%`,
                      }}
                    />
                    {/* Draggable handle (shift+click to create noise zone) */}
                    <div
                      style={{
                        ...styles.handle,
                        ...(m.useBreak ? styles.handleActive : styles.handleInactive),
                        left: `${currentPct}%`,
                      }}
                      onPointerDown={(e) => handlePointerDown(e, m.pauIdx, e.shiftKey && !hasNoiseZone)}
                      onAuxClick={(e) => handleAuxClick(e, m.pauIdx)}
                      title={`pau(${m.pauIdx}): ${displayMs}ms${deltaMs != null ? ` (Δ${deltaMs >= 0 ? '+' : ''}${deltaMs}ms)` : ''}${hasNoiseZone ? '' : ' [shift+click for noise zone]'} [middle-click to toggle]`}
                    >
                      {hasNoiseZone ? `${m.pauIdx}a` : m.pauIdx}
                    </div>
                    {/* Delta label while dragging */}
                    {isDraggingThis && deltaMs != null && (
                      <div
                        style={{
                          ...styles.deltaLabel,
                          left: `${currentPct}%`,
                        }}
                      >
                        {deltaMs >= 0 ? '+' : ''}{deltaMs}ms
                      </div>
                    )}
                  </>
                )}
                {/* Noise zone (b) marker - dashed line */}
                {noiseZonePct != null && (
                  <>
                    <div
                      style={{
                        ...styles.noiseZoneLine,
                        ...(isDraggingNoiseZone ? styles.markerDragging : {}),
                        left: `${noiseZonePct}%`,
                      }}
                    />
                    {/* Draggable handle for (b) — at bottom to distinguish from (a) */}
                    <div
                      style={{
                        ...styles.handle,
                        ...styles.handleNoiseZone,
                        left: `${noiseZonePct}%`,
                        top: 'auto',
                        bottom: 2,
                      }}
                      onPointerDown={(e) => handlePointerDown(e, toNoiseZoneId(m.pauIdx))}
                      title={`pau(${m.pauIdx}b): ${noiseZoneMs}ms [noise zone boundary, Esc to remove]`}
                    >
                      {m.pauIdx}b
                    </div>
                    {/* Ms label while dragging noise zone */}
                    {isDraggingNoiseZone && (
                      <div
                        style={{
                          ...styles.deltaLabel,
                          left: `${noiseZonePct}%`,
                        }}
                      >
                        {noiseZoneMs}ms
                      </div>
                    )}
                  </>
                )}
              </div>
            )
          })}
        </div>
      </div>
      <div style={styles.timeBar}>
        <span>0s</span>
        <span>{ready ? `${duration.toFixed(1)}s` : 'loading...'}</span>
      </div>
    </div>
  )
})

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    marginBottom: 16,
  },
  waveContainer: {
    position: 'relative',
    background: '#0f0f23',
    borderRadius: 8,
    overflow: 'hidden',
    border: '1px solid #2a2a4a',
  },
  wave: {
    width: '100%',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    cursor: 'default',
    zIndex: 10,
  },
  trimShade: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.45)',
    pointerEvents: 'none',
    zIndex: 11,
  },
  trimLine: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    background: '#4fc3f7',
    transform: 'translateX(-1px)',
    pointerEvents: 'none',
    zIndex: 12,
    transition: 'left 0.05s linear',
  },
  trimLineDragging: {
    background: '#ffb84d',
    boxShadow: '0 0 8px rgba(255, 184, 77, 0.8)',
    transition: 'none',
  },
  trimHandle: {
    position: 'absolute',
    bottom: 2,
    width: 18,
    height: 18,
    borderRadius: 3,
    transform: 'translateX(-9px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 9,
    fontWeight: 700,
    color: '#fff',
    background: '#0277bd',
    border: '2px solid #4fc3f7',
    cursor: 'grab',
    zIndex: 22,
    userSelect: 'none',
    touchAction: 'none',
  },
  trimLabel: {
    position: 'absolute',
    top: 4,
    transform: 'translateX(-50%)',
    fontSize: 10,
    fontWeight: 600,
    color: '#4fc3f7',
    background: 'rgba(0, 0, 0, 0.8)',
    padding: '1px 5px',
    borderRadius: 3,
    whiteSpace: 'nowrap',
    pointerEvents: 'none',
    zIndex: 25,
  },
  ghostLine: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 1,
    background: 'rgba(255, 255, 255, 0.15)',
    borderLeft: '1px dashed rgba(255, 255, 255, 0.25)',
    transform: 'translateX(-1px)',
    pointerEvents: 'none',
  },
  markerLine: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    transform: 'translateX(-1px)',
    pointerEvents: 'none',
    transition: 'left 0.05s linear',
  },
  markerActive: {
    background: '#4caf50',
    boxShadow: '0 0 6px rgba(76, 175, 80, 0.6)',
  },
  markerInactive: {
    background: '#666',
    boxShadow: 'none',
  },
  markerDragging: {
    background: '#ffb84d',
    boxShadow: '0 0 8px rgba(255, 184, 77, 0.8)',
    transition: 'none',
  },
  handle: {
    position: 'absolute',
    top: 2,
    width: 18,
    height: 18,
    borderRadius: '50%',
    transform: 'translateX(-9px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 9,
    fontWeight: 700,
    color: '#fff',
    cursor: 'grab',
    zIndex: 20,
    userSelect: 'none',
    touchAction: 'none',
  },
  handleActive: {
    background: '#388e3c',
    border: '2px solid #4caf50',
  },
  handleInactive: {
    background: '#555',
    border: '2px solid #777',
  },
  handleNoiseZone: {
    background: '#5c4033',
    border: '2px dashed #ff9800',
    borderRadius: 4,
    width: 22,
    transform: 'translateX(-11px)',
  },
  noiseZoneShade: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    background: 'rgba(255, 152, 0, 0.15)',
    borderTop: '2px dashed rgba(255, 152, 0, 0.4)',
    borderBottom: '2px dashed rgba(255, 152, 0, 0.4)',
    zIndex: 9,
  },
  noiseZoneDragging: {
    background: 'rgba(255, 184, 77, 0.3)',
    borderTop: '2px dashed rgba(255, 184, 77, 0.8)',
    borderBottom: '2px dashed rgba(255, 184, 77, 0.8)',
  },
  noiseZoneLine: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    background: 'transparent',
    borderLeft: '2px dashed #ff9800',
    transform: 'translateX(-1px)',
    pointerEvents: 'none',
    transition: 'left 0.05s linear',
    zIndex: 14,
  },
  deltaLabel: {
    position: 'absolute',
    bottom: 4,
    transform: 'translateX(-50%)',
    fontSize: 10,
    fontWeight: 600,
    color: '#ffb84d',
    background: 'rgba(0, 0, 0, 0.8)',
    padding: '1px 5px',
    borderRadius: 3,
    whiteSpace: 'nowrap',
    pointerEvents: 'none',
    zIndex: 25,
  },
  timeBar: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 11,
    color: '#666',
    padding: '4px 2px 0',
  },
}
