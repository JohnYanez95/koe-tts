import type { PauBreak } from '../api/client'

interface Props {
  phonemes: string
  pauBreaks: PauBreak[]
  onTogglePau: (pauIdx: number) => void
}

export function PhonemePills({ phonemes, pauBreaks, onTogglePau }: Props) {
  if (!phonemes) return null

  const tokens = phonemes.split(/\s+/).filter(Boolean)

  // Build a map from token_position -> PauBreak for quick lookup
  const pauByPosition = new Map<number, PauBreak>()
  for (const pb of pauBreaks) {
    pauByPosition.set(pb.token_position, pb)
  }

  return (
    <div style={styles.container}>
      {tokens.map((token, i) => {
        const pb = pauByPosition.get(i)
        const isSil = token === 'sil'

        if (pb) {
          const deltaMs = pb.ms != null && pb.ms_proposed != null
            ? pb.ms - pb.ms_proposed
            : null
          const hasDelta = deltaMs != null && deltaMs !== 0

          return (
            <span
              key={`pau-${pb.pau_idx}`}
              style={{
                ...styles.pill,
                ...(pb.use_break ? styles.pauActive : styles.pauInactive),
                cursor: 'pointer',
              }}
              onClick={() => onTogglePau(pb.pau_idx)}
              title={`pau(${pb.pau_idx})${pb.ms != null ? ` @ ${pb.ms}ms` : ''}${hasDelta ? ` (Δ${deltaMs >= 0 ? '+' : ''}${deltaMs}ms)` : ''} — click to toggle`}
            >
              pau({pb.pau_idx})
              {hasDelta && (
                <span style={styles.delta}>
                  {deltaMs >= 0 ? '+' : ''}{deltaMs}
                </span>
              )}
            </span>
          )
        }

        return (
          <span
            key={`${token}-${i}`}
            style={{
              ...styles.pill,
              ...(isSil ? styles.sil : {}),
            }}
          >
            {token}
          </span>
        )
      })}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 4,
    padding: '8px 0',
    marginBottom: 8,
  },
  pill: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    padding: '2px 8px',
    borderRadius: 12,
    fontSize: 12,
    fontFamily: 'monospace',
    background: '#1e2a4a',
    color: '#b0c0e0',
    border: '1px solid #2a3a5a',
  },
  pauActive: {
    background: '#1a3d1a',
    color: '#6fdc6f',
    border: '1px solid #2d6b2d',
    fontWeight: 700,
  },
  pauInactive: {
    background: '#2e2e2e',
    color: '#888',
    border: '1px solid #444',
    fontWeight: 700,
  },
  sil: {
    background: '#2e2e2e',
    color: '#888',
    border: '1px solid #444',
  },
  delta: {
    fontSize: 10,
    color: '#ffb84d',
    fontWeight: 400,
  },
}
