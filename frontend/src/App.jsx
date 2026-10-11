import React, { useState, useEffect } from 'react'
import {
  LayoutDashboard,
  Users,
  Calculator,
  ClipboardList,
  TrendingUp,
  ShieldCheck,
  Database,
  FileText,
  AlertTriangle,
} from 'lucide-react'
import Overview from './pages/Overview.jsx'
import Worklist from './pages/Worklist.jsx'
import RiskCalculator from './pages/RiskCalculator.jsx'
import CarePlanTracker from './pages/CarePlanTracker.jsx'
import Governance from './pages/Governance.jsx'
import DataImport from './pages/DataImport.jsx'
import Provenance from './pages/Provenance.jsx'
import { ErrorBoundary } from './components/ErrorBoundary.jsx'
import { api } from './utils/api.js'

const NAV_ITEMS = [
  { id: 'overview', label: 'Command Center', icon: LayoutDashboard },
  { id: 'worklist', label: 'Discharge Worklist', icon: Users },
  { id: 'calculator', label: 'Risk Simulator', icon: Calculator },
  { id: 'careplans', label: 'Care Plan Tracker', icon: ClipboardList },
  { id: 'models', label: 'Model Evaluation', icon: TrendingUp },
  { id: 'fairness', label: 'Fairness & Audit', icon: ShieldCheck },
  { id: 'datasets', label: 'Data Ingestion', icon: Database },
  { id: 'provenance', label: 'Model Card & Provenance', icon: FileText },
]

export default function App() {
  const [page, setPage] = useState('overview')
  const [health, setHealth] = useState(null)
  const [selectedEncId, setSelectedEncId] = useState(null)

  useEffect(() => {
    api.health()
      .then(setHealth)
      .catch(() => setHealth({ status: 'down', model_loaded: false, db_connected: false }))
  }, [])

  const navigateTo = (p, encId = null) => {
    setPage(p)
    if (encId) setSelectedEncId(encId)
  }

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar" role="navigation" aria-label="Main navigation">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">
            <div className="brand-logo">CA</div>
            <div className="brand-text">
              <span className="brand-name">ClinicalAI</span>
              <span className="brand-sub">Readmission Risk</span>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <span className="nav-section-label">Clinical Modules</span>
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              id={`nav-${id}`}
              className={`nav-item ${page === id ? 'active' : ''}`}
              onClick={() => navigateTo(id)}
              aria-current={page === id ? 'page' : undefined}
            >
              <Icon className="nav-item-icon" aria-hidden="true" />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="demo-badge">
            <AlertTriangle size={12} aria-hidden="true" />
            <span>Research Demo</span>
          </div>
          {health && (
            <div className="flex items-center gap-2 mt-2" style={{ fontSize: 11, color: '#94a3b8' }}>
              <span
                className={`status-dot ${
                  health.status === 'ok'
                    ? 'online'
                    : health.status === 'degraded'
                    ? 'degraded'
                    : 'offline'
                }`}
                title={`API: ${health.status}`}
              />
              <span>
                {health.model_loaded ? 'Model v1.0.0 Online' : 'No Model'}
              </span>
            </div>
          )}
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content" role="main">
        <ErrorBoundary key={page}>
          {page === 'overview' && (
            <Overview onNavigate={navigateTo} />
          )}
          {page === 'worklist' && (
            <Worklist
              initialEncId={selectedEncId}
              onNavigateToCalc={(encId) => navigateTo('calculator', encId)}
            />
          )}
          {page === 'calculator' && (
            <RiskCalculator initialEncId={selectedEncId} />
          )}
          {page === 'careplans' && (
            <CarePlanTracker
              onNavigateToEncounter={(encId) => navigateTo('worklist', encId)}
            />
          )}
          {page === 'models' && (
            <Governance initialTab={0} />
          )}
          {page === 'fairness' && (
            <Governance initialTab={1} />
          )}
          {page === 'datasets' && (
            <DataImport />
          )}
          {page === 'provenance' && (
            <Provenance />
          )}
        </ErrorBoundary>
      </main>
    </div>
  )
}
