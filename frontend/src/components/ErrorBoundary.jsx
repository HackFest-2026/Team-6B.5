import React from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('[ClinicalAI ErrorBoundary Caught]', error, errorInfo)
    this.setState({ errorInfo })
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null })
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#f8fafc',
          padding: 24,
          fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
        }}>
          <div style={{
            maxWidth: 540,
            width: '100%',
            backgroundColor: '#ffffff',
            borderRadius: 14,
            border: '1px solid #e2e8f0',
            padding: 32,
            boxShadow: '0 10px 15px -3px rgba(15, 23, 42, 0.08)',
            textAlign: 'center'
          }}>
            <div style={{
              width: 52,
              height: 52,
              borderRadius: '50%',
              backgroundColor: '#fef2f2',
              color: '#dc2626',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px'
            }}>
              <AlertCircle size={28} />
            </div>

            <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 8 }}>
              ClinicalAI Interface Recovery
            </h2>
            <p style={{ fontSize: 14, color: '#64748b', marginBottom: 20, lineHeight: 1.5 }}>
              The application encountered an unexpected runtime error during view rendering. 
              The backend API and database remain intact.
            </p>

            <div style={{
              backgroundColor: '#f1f5f9',
              border: '1px solid #e2e8f0',
              borderRadius: 8,
              padding: 12,
              textAlign: 'left',
              fontSize: 12,
              color: '#334155',
              fontFamily: 'monospace',
              maxHeight: 140,
              overflowY: 'auto',
              marginBottom: 24
            }}>
              {this.state.error?.toString() || 'Unknown runtime error'}
            </div>

            <button
              onClick={this.handleReset}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '10px 20px',
                borderRadius: 8,
                backgroundColor: '#2563eb',
                color: '#ffffff',
                border: 'none',
                fontSize: 14,
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              <RefreshCw size={16} />
              <span>Reload ClinicalAI</span>
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
