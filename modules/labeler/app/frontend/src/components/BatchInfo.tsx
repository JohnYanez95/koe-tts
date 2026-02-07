interface Props {
  dataset: string
  sessionId: string
  total: number
  labeled: number
}

export function BatchInfo({ dataset, sessionId, total, labeled }: Props) {
  const pct = total > 0 ? Math.round((100 * labeled) / total) : 0

  return (
    <div style={styles.container}>
      <div style={styles.left}>
        <span style={styles.dataset}>{dataset}</span>
        <span style={styles.session}>{sessionId.slice(0, 15)}</span>
      </div>
      <div style={styles.right}>
        <div style={styles.barOuter}>
          <div style={{ ...styles.barInner, width: `${pct}%` }} />
        </div>
        <span style={styles.stats}>
          {labeled}/{total} ({pct}%)
        </span>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 0',
    marginBottom: 12,
    borderBottom: '1px solid #2a2a4a',
  },
  left: {
    display: 'flex',
    gap: 12,
    alignItems: 'center',
  },
  dataset: {
    fontSize: 16,
    fontWeight: 600,
    color: '#fff',
  },
  session: {
    fontSize: 12,
    color: '#666',
    fontFamily: 'monospace',
  },
  right: {
    display: 'flex',
    gap: 12,
    alignItems: 'center',
  },
  barOuter: {
    width: 100,
    height: 6,
    background: '#1e1e3e',
    borderRadius: 3,
    overflow: 'hidden',
  },
  barInner: {
    height: '100%',
    background: '#4a9eff',
    borderRadius: 3,
    transition: 'width 0.3s ease',
  },
  stats: {
    fontSize: 12,
    color: '#888',
    minWidth: 80,
    textAlign: 'right' as const,
  },
}
