import { useToast, type ToastType } from '../contexts/ToastContext'

const colors: Record<ToastType, { bg: string; border: string; text: string }> = {
  success: { bg: '#1a3d1a', border: '#2e7d32', text: '#81c784' },
  info: { bg: '#1a3d5c', border: '#1976d2', text: '#64b5f6' },
  warning: { bg: '#4d3d1a', border: '#f9a825', text: '#fdd835' },
}

export function ToastContainer() {
  const { toasts } = useToast()

  if (toasts.length === 0) return null

  return (
    <div style={styles.container}>
      {toasts.map((toast) => {
        const c = colors[toast.type]
        return (
          <div
            key={toast.id}
            style={{
              ...styles.toast,
              background: c.bg,
              borderColor: c.border,
              color: c.text,
            }}
          >
            {toast.message}
          </div>
        )
      })}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: 'fixed',
    bottom: 20,
    right: 20,
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    zIndex: 1000,
    pointerEvents: 'none',
  },
  toast: {
    padding: '12px 20px',
    borderRadius: 8,
    border: '1px solid',
    fontSize: 14,
    fontWeight: 500,
    animation: 'toastFadeIn 0.2s ease-out',
    pointerEvents: 'auto',
  },
}
