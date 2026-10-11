import React, { useState, useEffect } from 'react'
import {
  FileText,
  ShieldCheck,
  History,
  Layers,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  RefreshCw,
  GitBranch,
  Calendar
} from 'lucide-react'
import { api } from '../utils/api.js'

export default function Provenance() {
  const [modelCard, setModelCard] = useState(null)
  const [auditLogs, setAuditLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [cardData, logsData] = await Promise.all([
        api.modelCard(),
        api.auditLogs(20),
      ])
      setModelCard(cardData)
      setAuditLogs(logsData ?? [])
    } catch (err) {
      console.error('Provenance load error:', err)
      setError(err.message || 'Failed to load model card or audit logs.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  return (
    <div className="page-body">
      {/* Header */}
      <div className="section-header" style={{ marginBottom: 24 }}>
        <div>
          <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <FileText className="text-primary" size={26} />
            <span>AI Model Card & System Provenance</span>
          </h1>
          <p className="page-subtitle">
            Formal AI system documentation, intended use boundaries, training lineage, and immutable compliance audit trail.
          </p>
        </div>
        <button onClick={fetchData} className="btn btn-secondary btn-sm">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          <span>Refresh</span>
        </button>
      </div>

      {loading ? (
        <div className="card">
          <div className="card-body">
            <div className="skeleton skeleton-title" style={{ width: '40%' }} />
            <div className="skeleton skeleton-text" />
            <div className="skeleton skeleton-text" />
            <div className="skeleton skeleton-card" />
          </div>
        </div>
      ) : error ? (
        <div className="alert alert-error">
          <AlertTriangle className="alert-icon" />
          <div>{error}</div>
        </div>
      ) : (
        <>
          {/* Model Card */}
          <div className="card" style={{ marginBottom: 28 }}>
            <div className="card-header">
              <div className="card-title">
                <ShieldCheck size={20} className="text-primary" />
                <span>Standardized AI Model Card</span>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <span className="risk-badge low">v{modelCard?.model_version || '1.0.0'}</span>
                <span className="risk-badge elevated">Research Demo</span>
              </div>
            </div>

            <div className="card-body">
              {/* Core metadata pills */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                  gap: 16,
                  paddingBottom: 24,
                  borderBottom: '1px solid var(--color-border)',
                  marginBottom: 24,
                }}
              >
                <div>
                  <div className="stat-label">Model Architecture</div>
                  <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--color-text-primary)' }}>
                    {modelCard?.model_type || 'XGBoost (Calibrated Isotonic)'}
                  </div>
                </div>

                <div>
                  <div className="stat-label">Operating Risk Threshold</div>
                  <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--color-accent)' }}>
                    P ≥ {modelCard?.threshold ?? 0.15} (Screening)
                  </div>
                </div>

                <div>
                  <div className="stat-label">Primary Metric</div>
                  <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--color-text-primary)' }}>
                    {modelCard?.primary_metric || 'Recall / Sensitivity'}
                  </div>
                </div>

                <div>
                  <div className="stat-label">Governance Committee</div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--color-text-secondary)' }}>
                    ClinicalAI ML Safety Board
                  </div>
                </div>
              </div>

              {/* Intended Use & Boundaries */}
              <div style={{ marginBottom: 24 }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, color: 'var(--color-text-primary)', marginBottom: 8 }}>
                  1. Intended Use & Scope
                </h3>
                <p style={{ fontSize: 14, color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
                  {modelCard?.intended_use}
                </p>
                <div
                  style={{
                    backgroundColor: 'var(--color-bg-elevated)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 8,
                    padding: '12px 16px',
                    marginTop: 12,
                    fontSize: 13,
                    color: 'var(--color-text-secondary)',
                  }}
                >
                  <strong>Target Inpatient Population:</strong> {modelCard?.target_population}
                </div>
              </div>

              {/* Training Data & Preprocessing Lineage */}
              <div style={{ marginBottom: 24 }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, color: 'var(--color-text-primary)', marginBottom: 8 }}>
                  2. Dataset Lineage & Leakage Protection
                </h3>
                <p style={{ fontSize: 14, color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
                  {modelCard?.training_cohort_description}
                </p>
                <p style={{ fontSize: 14, color: 'var(--color-text-secondary)', lineHeight: 1.6, marginTop: 8 }}>
                  <strong>Evaluation Holdout Design:</strong> {modelCard?.evaluation_split_design} GroupShuffleSplit ensures
                  different encounters belonging to the same patient never appear in both train and test partitions.
                </p>
              </div>

              {/* Known Limitations */}
              <div style={{ marginBottom: 24 }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, color: 'var(--color-text-primary)', marginBottom: 8 }}>
                  3. Known Limitations & Caveats
                </h3>
                <ul style={{ paddingLeft: 20, fontSize: 14, color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
                  {(modelCard?.known_limitations ?? []).map((lim, i) => (
                    <li key={i} style={{ marginBottom: 6 }}>
                      {lim}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Ethical & Safety Considerations */}
              <div>
                <h3 style={{ fontSize: 15, fontWeight: 700, color: 'var(--color-text-primary)', marginBottom: 8 }}>
                  4. Ethical Governance & Clinical Safeguards
                </h3>
                <ul style={{ paddingLeft: 20, fontSize: 14, color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
                  {(modelCard?.ethical_and_safety_considerations ?? []).map((item, i) => (
                    <li key={i} style={{ marginBottom: 6 }}>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* Compliance Audit Trail */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <History size={18} />
                <span>System Compliance & Administrative Audit Trail</span>
              </div>
              <span style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>
                Non-sensitive telemetry & user actions
              </span>
            </div>

            <div className="card-body">
              {auditLogs.length === 0 ? (
                <div style={{ textAlign: 'center', padding: 24, color: 'var(--color-text-muted)', fontSize: 13 }}>
                  No administrative events logged yet. Actions such as care plan creation, task updates, and dataset imports are persisted here.
                </div>
              ) : (
                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Event Type</th>
                        <th>Actor</th>
                        <th>Entity</th>
                        <th>Details</th>
                      </tr>
                    </thead>
                    <tbody>
                      {auditLogs.map((log) => (
                        <tr key={log.id}>
                          <td style={{ whiteSpace: 'nowrap' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <Calendar size={12} className="text-muted" />
                              <span>
                                {log.created_at
                                  ? new Date(log.created_at).toLocaleString()
                                  : 'Recently'}
                              </span>
                            </div>
                          </td>
                          <td>
                            <span
                              style={{
                                display: 'inline-block',
                                padding: '2px 8px',
                                borderRadius: 4,
                                fontSize: 11,
                                fontWeight: 600,
                                textTransform: 'uppercase',
                                backgroundColor: 'var(--color-bg-elevated)',
                                color: 'var(--color-accent)',
                              }}
                            >
                              {log.event_type}
                            </span>
                          </td>
                          <td>
                            <strong>{log.acting_user || 'System'}</strong>
                          </td>
                          <td>
                            <code style={{ fontSize: 11 }}>
                              {log.entity_type}:{log.entity_id?.slice(0, 8)}...
                            </code>
                          </td>
                          <td style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
                            {log.details || '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
