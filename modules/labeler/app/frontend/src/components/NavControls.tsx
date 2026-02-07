import { useEffect } from 'react'

interface Props {
  onPrev: () => void
  onNext: () => void
  onPlay: () => void
  onPause: () => void
  onPlayPause: () => void
  onSave: () => void
  onSkip: () => void
  onUndoSkip: () => void
  onReplay: () => void
  onReplaySnippet: () => void
  onUndo: () => void
  onRedo: () => void
  currentIdx: number
  total: number
  isPlaying: boolean
  isDirty: boolean
  isSaved: boolean
  isSkipped: boolean
}

export function NavControls({
  onPrev,
  onNext,
  onPlay,
  onPause,
  onPlayPause,
  onSave,
  onSkip,
  onUndoSkip,
  onReplay,
  onReplaySnippet,
  onUndo,
  onRedo,
  currentIdx,
  total,
  isPlaying,
  isDirty,
  isSaved,
  isSkipped,
}: Props) {
  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return

      switch (e.key) {
        case ' ':
          e.preventDefault()
          isPlaying ? onPause() : onPlay()
          break
        case 'ArrowLeft':
        case 'a':
          e.preventDefault()
          onPrev()
          break
        case 'ArrowRight':
        case 'd':
          e.preventDefault()
          onNext()
          break
        case 's':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            onSave()
          }
          break
        case 'r':
          e.preventDefault()
          onReplay()
          break
        case 'f':
          e.preventDefault()
          onReplaySnippet()
          break
        case 'z':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            if (e.shiftKey) {
              onRedo()
            } else {
              onUndo()
            }
          }
          break
        case 'y':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            onRedo()
          }
          break
        // Esc is handled in LabelView for early exit
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onPrev, onNext, onPlay, onPause, onSave, onSkip, onUndoSkip, onReplay, onReplaySnippet, onUndo, onRedo, isPlaying])

  return (
    <div style={styles.container}>
      <div style={styles.controls}>
        <button style={styles.btn} onClick={onPrev} disabled={currentIdx === 0} title="Previous (Left arrow)">
          Prev
        </button>

        <button style={styles.btnPrimary} onClick={onPlayPause} title="Play/Pause (Space)">
          {isPlaying ? 'Pause' : 'Play'}
        </button>

        <button style={styles.btn} onClick={onNext} title="Next (Right arrow / D)">
          Next
        </button>

        <button
          style={isSkipped ? styles.btnUndoSkip : styles.btnSkip}
          onClick={isSkipped ? onUndoSkip : onSkip}
          title={isSkipped ? "Mark as labeled instead of skipped" : "Skip — too hard to label"}
        >
          {isSkipped ? 'Undo Skip' : 'Skip'}
        </button>
      </div>

      <div style={styles.info}>
        <span style={styles.progress}>
          {currentIdx + 1} / {total}
        </span>
        {isDirty && <span style={styles.dirty}>unsaved</span>}
        {!isDirty && isSkipped && <span style={styles.skipped}>skipped</span>}
        {!isDirty && isSaved && !isSkipped && <span style={styles.saved}>saved</span>}
      </div>

      <div style={styles.shortcuts}>
        Space: play/pause &middot; A/D: navigate &middot; R: replay &middot; Ctrl+Z: undo &middot; Ctrl+S: save &middot; ?: help
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 8,
    padding: '16px 0',
  },
  controls: {
    display: 'flex',
    gap: 12,
    alignItems: 'center',
  },
  btn: {
    padding: '8px 20px',
    borderRadius: 6,
    border: '1px solid #2a2a4a',
    background: '#16213e',
    color: '#e0e0e0',
    fontSize: 14,
    cursor: 'pointer',
  },
  btnPrimary: {
    padding: '8px 28px',
    borderRadius: 6,
    border: '1px solid #4a9eff',
    background: '#1a3a6a',
    color: '#fff',
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
  },
  btnSkip: {
    padding: '8px 16px',
    borderRadius: 6,
    border: '1px solid #554400',
    background: '#2e2200',
    color: '#aa8833',
    fontSize: 13,
    cursor: 'pointer',
  },
  btnUndoSkip: {
    padding: '8px 16px',
    borderRadius: 6,
    border: '1px solid #2e7d32',
    background: '#1b3a1b',
    color: '#81c784',
    fontSize: 13,
    cursor: 'pointer',
  },
  info: {
    display: 'flex',
    gap: 12,
    alignItems: 'center',
  },
  progress: {
    fontSize: 14,
    color: '#aaa',
  },
  dirty: {
    fontSize: 11,
    color: '#ffb84d',
    padding: '2px 8px',
    background: '#3d2e00',
    borderRadius: 4,
  },
  saved: {
    fontSize: 11,
    color: '#81c784',
    padding: '2px 8px',
    background: '#1a3d1a',
    borderRadius: 4,
  },
  skipped: {
    fontSize: 11,
    color: '#90a4ae',
    padding: '2px 8px',
    background: '#263238',
    borderRadius: 4,
  },
  shortcuts: {
    fontSize: 11,
    color: '#555',
  },
}
