import { useCallback, useRef } from 'react'

const MAX_HISTORY = 20

export interface UndoStack<T> {
  push: (state: T) => void
  undo: () => T | null
  redo: () => T | null
  canUndo: () => boolean
  canRedo: () => boolean
  clear: () => void
}

export function useUndoStack<T>(): UndoStack<T> {
  const historyRef = useRef<T[]>([])
  const indexRef = useRef(-1)

  const push = useCallback((state: T) => {
    // Remove any redo history (states after current index)
    historyRef.current = historyRef.current.slice(0, indexRef.current + 1)

    // Add new state
    historyRef.current.push(structuredClone(state))
    indexRef.current = historyRef.current.length - 1

    // Limit history size
    if (historyRef.current.length > MAX_HISTORY) {
      historyRef.current.shift()
      indexRef.current--
    }
  }, [])

  const undo = useCallback((): T | null => {
    if (indexRef.current <= 0) return null
    indexRef.current--
    return structuredClone(historyRef.current[indexRef.current])
  }, [])

  const redo = useCallback((): T | null => {
    if (indexRef.current >= historyRef.current.length - 1) return null
    indexRef.current++
    return structuredClone(historyRef.current[indexRef.current])
  }, [])

  const canUndo = useCallback(() => indexRef.current > 0, [])
  const canRedo = useCallback(() => indexRef.current < historyRef.current.length - 1, [])

  const clear = useCallback(() => {
    historyRef.current = []
    indexRef.current = -1
  }, [])

  return { push, undo, redo, canUndo, canRedo, clear }
}
