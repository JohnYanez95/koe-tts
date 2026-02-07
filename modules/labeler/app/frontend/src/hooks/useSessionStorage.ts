import { useCallback, useEffect, useRef } from 'react'
import type { PauBreak } from '../api/client'
import type { TrimState } from '../components/Waveform'

export interface SessionState {
  currentIdx: number
  pauBreaksMap: [number, PauBreak[]][]  // Serialized Map
  trimMap: [number, TrimState][]        // Serialized Map
  savedIndices: number[]                // Serialized Set
  skippedIndices: number[]              // Serialized Set
  lastClickedPauIdx: number | null
}

const STORAGE_PREFIX = 'koe-labeler:'

function getKey(sessionId: string): string {
  return `${STORAGE_PREFIX}${sessionId}`
}

export function useSessionStorage(sessionId: string | null) {
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Save state to localStorage with debounce
  const saveState = useCallback(
    (state: {
      currentIdx: number
      pauBreaksMap: Map<number, PauBreak[]>
      trimMap: Map<number, TrimState>
      savedIndices: Set<number>
      skippedIndices: Set<number>
      lastClickedPauIdx: number | null
    }) => {
      if (!sessionId) return

      // Debounce saves to avoid excessive writes
      if (debounceRef.current) {
        clearTimeout(debounceRef.current)
      }

      debounceRef.current = setTimeout(() => {
        const serialized: SessionState = {
          currentIdx: state.currentIdx,
          pauBreaksMap: Array.from(state.pauBreaksMap.entries()),
          trimMap: Array.from(state.trimMap.entries()),
          savedIndices: Array.from(state.savedIndices),
          skippedIndices: Array.from(state.skippedIndices),
          lastClickedPauIdx: state.lastClickedPauIdx,
        }
        try {
          localStorage.setItem(getKey(sessionId), JSON.stringify(serialized))
        } catch {
          // localStorage might be full or disabled
          console.warn('Failed to save session state to localStorage')
        }
      }, 500)
    },
    [sessionId],
  )

  // Load state from localStorage
  const loadState = useCallback((): {
    currentIdx: number
    pauBreaksMap: Map<number, PauBreak[]>
    trimMap: Map<number, TrimState>
    savedIndices: Set<number>
    lastClickedPauIdx: number | null
  } | null => {
    if (!sessionId) return null

    try {
      const raw = localStorage.getItem(getKey(sessionId))
      if (!raw) return null

      const parsed: SessionState = JSON.parse(raw)
      return {
        currentIdx: parsed.currentIdx,
        pauBreaksMap: new Map(parsed.pauBreaksMap),
        trimMap: new Map(parsed.trimMap),
        savedIndices: new Set(parsed.savedIndices),
        lastClickedPauIdx: parsed.lastClickedPauIdx,
      }
    } catch {
      // Corrupted data, remove it
      localStorage.removeItem(getKey(sessionId))
      return null
    }
  }, [sessionId])

  // Clear state from localStorage
  const clearState = useCallback(() => {
    if (!sessionId) return
    localStorage.removeItem(getKey(sessionId))
  }, [sessionId])

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current)
      }
    }
  }, [])

  return { saveState, loadState, clearState }
}
