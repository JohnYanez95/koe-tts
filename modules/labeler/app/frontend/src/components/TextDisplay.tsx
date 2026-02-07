interface Props {
  text: string
  speakerId: string | null
}

export function TextDisplay({ text, speakerId }: Props) {
  return (
    <div style={styles.container}>
      <div style={styles.text}>{text}</div>
      {speakerId && <div style={styles.speaker}>Speaker: {speakerId}</div>}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '8px 0',
    marginBottom: 12,
  },
  text: {
    fontSize: 18,
    lineHeight: 1.6,
    color: '#fff',
    fontFamily: "'Noto Sans JP', sans-serif",
  },
  speaker: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
}
