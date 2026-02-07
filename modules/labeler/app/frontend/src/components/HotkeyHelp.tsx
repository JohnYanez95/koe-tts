interface HotkeyHelpProps {
  onClose: () => void
}

const hotkeys = [
  { section: 'Navigation' },
  { key: 'A / Left', desc: 'Previous item' },
  { key: 'D / Right', desc: 'Next item' },
  { key: 'Esc', desc: 'Exit session (or remove noise zone)' },

  { section: 'Playback' },
  { key: 'Space', desc: 'Play / Pause' },
  { key: 'R', desc: 'Replay from last clicked marker' },
  { key: 'F', desc: 'Play noise zone snippet' },

  { section: 'Editing' },
  { key: 'Click marker', desc: 'Select marker' },
  { key: 'Drag marker', desc: 'Move break position' },
  { key: 'Shift+Click', desc: 'Create noise zone at position' },
  { key: 'Click pill', desc: 'Toggle pau break' },
  { key: 'Ctrl+Z', desc: 'Undo' },
  { key: 'Ctrl+Shift+Z / Ctrl+Y', desc: 'Redo' },

  { section: 'Saving' },
  { key: 'Ctrl+S', desc: 'Save current item' },
  { key: 'Skip button', desc: 'Mark as skipped and advance' },
  { key: 'Undo Skip button', desc: 'Convert skipped to labeled' },
]

export function HotkeyHelp({ onClose }: HotkeyHelpProps) {
  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h2 style={styles.title}>Keyboard Shortcuts</h2>
          <button style={styles.closeBtn} onClick={onClose}>
            &times;
          </button>
        </div>
        <div style={styles.content}>
          {hotkeys.map((item, i) =>
            'section' in item ? (
              <div key={i} style={styles.section}>
                {item.section}
              </div>
            ) : (
              <div key={i} style={styles.row}>
                <span style={styles.key}>{item.key}</span>
                <span style={styles.desc}>{item.desc}</span>
              </div>
            ),
          )}
        </div>
        <div style={styles.footer}>Press Esc or ? to close</div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 200,
  },
  modal: {
    background: '#1a1a2e',
    border: '1px solid #2a2a4a',
    borderRadius: 12,
    padding: '20px 28px',
    minWidth: 380,
    maxWidth: 480,
    maxHeight: '80vh',
    overflow: 'auto',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    margin: 0,
    fontSize: 18,
    fontWeight: 600,
    color: '#e0e0e0',
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    color: '#888',
    fontSize: 24,
    cursor: 'pointer',
    padding: 0,
    lineHeight: 1,
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  section: {
    fontSize: 11,
    fontWeight: 600,
    color: '#666',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginTop: 12,
    marginBottom: 4,
  },
  row: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '4px 0',
  },
  key: {
    fontFamily: 'monospace',
    fontSize: 13,
    color: '#4a9eff',
    background: '#16213e',
    padding: '3px 8px',
    borderRadius: 4,
    border: '1px solid #2a2a4a',
  },
  desc: {
    fontSize: 13,
    color: '#aaa',
    textAlign: 'right',
    flex: 1,
    marginLeft: 16,
  },
  footer: {
    marginTop: 20,
    paddingTop: 12,
    borderTop: '1px solid #2a2a4a',
    fontSize: 11,
    color: '#555',
    textAlign: 'center',
  },
}
