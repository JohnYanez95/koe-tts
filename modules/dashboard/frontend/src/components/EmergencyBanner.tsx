/**
 * Emergency banner for NaN detection, absolute grad limit, or emergency stop.
 * Shows prominently at the top of the run view when triggered.
 */

interface EmergencyBannerProps {
  active: boolean;
  title: string;
  detail?: string;
  step?: number;
  checkpointPath?: string;
}

export function EmergencyBanner({
  active,
  title,
  detail,
  step,
  checkpointPath,
}: EmergencyBannerProps) {
  if (!active) return null;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.icon}>!</span>
        <span style={styles.title}>{title}</span>
      </div>
      {detail && <div style={styles.detail}>{detail}</div>}
      <div style={styles.meta}>
        {step !== undefined && <span>Step: {step.toLocaleString()}</span>}
        {checkpointPath && (
          <span style={styles.checkpoint}>
            Checkpoint: <code>{checkpointPath}</code>
          </span>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '12px 16px',
    borderRadius: 8,
    marginBottom: 16,
    border: '2px solid #dc2626',
    backgroundColor: '#1c1917',
    color: '#fecaca',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  icon: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 20,
    height: 20,
    borderRadius: '50%',
    backgroundColor: '#dc2626',
    color: '#fff',
    fontSize: 12,
    fontWeight: 700,
  },
  title: {
    fontSize: 14,
    fontWeight: 700,
    color: '#fca5a5',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  detail: {
    fontSize: 13,
    color: '#d6d3d1',
    marginTop: 4,
    marginLeft: 28,
  },
  meta: {
    fontSize: 12,
    color: '#78716c',
    marginTop: 8,
    marginLeft: 28,
    display: 'flex',
    gap: 16,
  },
  checkpoint: {
    fontFamily: 'monospace',
  },
};
