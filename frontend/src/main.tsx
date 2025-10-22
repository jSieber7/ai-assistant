import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ThemeProvider } from './context/ThemeContext.tsx'
import { Toaster } from 'sonner'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
      <App />
      <Toaster
        theme="dark"
        closeButton
        position="top-right"
        toastOptions={{
          style: {
            fontSize: '16px',
            padding: '16px',
          },
          error: {
            style: {
              background: '#ef4444',
              color: '#ffffff',
              fontSize: '18px',
              padding: '20px',
              position: 'relative',
            },
            closeButton: {
              backgroundColor: 'rgba(255, 255, 255, 0.2)',
              color: '#ffffff',
              position: 'absolute',
              top: '8px',
              right: '8px',
              width: '24px',
              height: '24px',
              borderRadius: '4px',
            },
          },
        }}
      />
    </ThemeProvider>
  </StrictMode>,
)