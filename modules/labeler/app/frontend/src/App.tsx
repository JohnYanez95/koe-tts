import { ToastProvider } from './contexts/ToastContext'
import { ToastContainer } from './components/Toast'
import { LabelView } from './pages/LabelView'

export default function App() {
  return (
    <ToastProvider>
      <LabelView />
      <ToastContainer />
    </ToastProvider>
  )
}
