import React, { useState, useEffect } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, BarChart, Bar, XAxis,
  YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line, CartesianGrid, Legend
} from 'recharts'
import { Shield, AlertTriangle, Info, Users, TrendingUp, BarChart2 } from 'lucide-react'
import { api } from '../utils/api.js'

const TABS = ['Model Comparison', 'Fairness Audit', 'Data Quality', 'Feature Importance']

function MetricRow({ label, value, highlight }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--color-border-subtle)' }}>
      <span style={{ fontSize: 13, color: 'var(--color-text-secondary)' }}>{label}</span>
      <span style={{
        fontSize: 13, fontWeight: 700,
        color: highlight ? 'var(--color-accent-light)' : 'var(--color-text-primary)'
      }}>{value}</span>
    </div>
  )
}

function ModelCard({ name, metrics, selected }) {
  if (!metrics || !metrics.roc_auc) return null
  const pct = v => `${(v * 100).toFixed(1)}%`
  return (
    <div className="card" style={{
      border: selected ? '1px solid var(--color-accent)' : undefined,
      boxShadow: selected ? 'var(--shadow-glow-accent)' : undefined,
    }}>
      <div className="card-header">
        <div>
          <div className="card-title">{name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
          {selected && <div style={{ fontSize: 11, color: 'var(--color-accent-light)', marginTop: 2 }}>✓ Selected model for live scoring</div>}
        </div>
      </div>
      <div className="card-body">
        <MetricRow label="ROC-AUC" value={metrics.roc_auc?.toFixed(3)} highlight />
        <MetricRow label="PR-AUC" value={metrics.pr_auc?.toFixed(3)} />
        <MetricRow label="Recall" value={pct(metrics.recall)} highlight />
        <MetricRow label="Precision" value={pct(metrics.precision)} />
        <MetricRow label="Accuracy" value={pct(metrics.accuracy)} />
        <MetricRow label="Brier Score" value={metrics.brier_score?.toFixed(4)} />
        <MetricRow label="Threshold" value={metrics.threshold?.toFixed(2)} />
        {metrics.confusion_matrix && (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 }}>Confusion Matrix</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
              {[
                { label: 'True Neg', value: metrics.confusion_matrix.tn, color: 'var(--color-risk-low)' },
                { label: 'False Pos', value: metrics.confusion_matrix.fp, color: 'var(--color-risk-elevated)' },
                { label: 'False Neg', value: metrics.confusion_matrix.fn, color: 'var(--color-risk-high)' },
                { label: 'True Pos', value: metrics.confusion_matrix.tp, color: 'var(--color-risk-low)' },
              ].map(({ label, value, color }) => (
                <div key={label} style={{
                  padding: '8px 12px', background: 'var(--color-bg-elevated)',
                  borderRadius: 8, textAlign: 'center',
                }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color }}>{value?.toLocaleString()}</div>
                  <div style={{ fontSize: 10, color: 'var(--color-text-muted)', marginTop: 2 }}>{label}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function FairnessSection({ fairnessData }) {
  if (!fairnessData) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {Object.entries(fairnessData).map(([attr, data]) => {
        const subgroups = Object.entries(data.subgroups || {})
        if (subgroups.length === 0) return null

        const chartData = subgroups.map(([grp, m]) => ({
          name: grp.length > 15 ? grp.slice(0, 14) + '…' : grp,
          TPR: m.tpr != null ? +(m.tpr * 100).toFixed(1) : null,
          FPR: m.fpr != null ? +(m.fpr * 100).toFixed(1) : null,
          'Selection Rate': m.selection_rate != null ? +(m.selection_rate * 100).toFixed(1) : null,
          n: m.n,
        }))

        const attrLabel = { age_group: 'Age Group', gender: 'Gender', race: 'Race' }[attr] || attr

        return (
          <div key={attr} className="card">
            <div className="card-header">
              <div>
                <div className="card-title"><Users size={15} /> {attrLabel} Subgroups</div>
                {data.demographic_parity_difference != null && (
                  <div style={{ fontSize: 12, color: 'var(--color-text-muted)', marginTop: 2 }}>
                    Demographic parity gap: {(data.demographic_parity_difference * 100).toFixed(1)}pp
                    {data.tpr_gap != null && ` · TPR gap: ${(data.tpr_gap * 100).toFixed(1)}pp`}
                  </div>
                )}
              </div>
            </div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={chartData} barSize={18}>
                  <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} />
                  <YAxis unit="%" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} />
                  <Tooltip contentStyle={{ background: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', borderRadius: 8 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="TPR" fill="var(--color-accent)" radius={[3, 3, 0, 0]} name="TPR (Recall)" />
                  <Bar dataKey="FPR" fill="var(--color-risk-elevated)" radius={[3, 3, 0, 0]} name="FPR" />
                  <Bar dataKey="Selection Rate" fill="var(--color-risk-low)" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>

              {/* Subgroup table */}
              <div className="table-wrapper mt-4" style={{ borderRadius: 8 }}>
                <table style={{ fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th>Subgroup</th>
                      <th>N</th>
                      <th>Pos Rate</th>
                      <th>Selection Rate</th>
                      <th>TPR</th>
                      <th>FPR</th>
                      <th>PPV</th>
                      <th>Small N?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {subgroups.map(([grp, m]) => (
                      <tr key={grp}>
                        <td style={{ fontWeight: 600 }}>{grp}</td>
                        <td>{m.n?.toLocaleString()}</td>
                        <td>{m.positive_rate != null ? `${(m.positive_rate * 100).toFixed(1)}%` : '—'}</td>
                        <td>{m.selection_rate != null ? `${(m.selection_rate * 100).toFixed(1)}%` : '—'}</td>
                        <td style={{ color: 'var(--color-accent-light)' }}>{m.tpr != null ? `${(m.tpr * 100).toFixed(1)}%` : '—'}</td>
                        <td>{m.fpr != null ? `${(m.fpr * 100).toFixed(1)}%` : '—'}</td>
                        <td>{m.ppv != null ? `${(m.ppv * 100).toFixed(1)}%` : '—'}</td>
                        <td>{m.small_sample_caution ? '⚠️ Yes' : '✓'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function FeatureImportanceChart({ importances }) {
  if (!importances || importances.length === 0) return null
  const top = importances.slice(0, 15)
  const max = top[0]?.importance || 1
  const data = top.map(d => ({
    name: d.feature.length > 30 ? d.feature.slice(0, 29) + '…' : d.feature,
    value: +(d.importance / max * 100).toFixed(1),
    raw: d.importance,
  }))

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title"><TrendingUp size={15} /> Top Feature Importances</div>
        <div style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>XGBoost — normalised to 100</div>
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={data} layout="vertical" barSize={14}>
            <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }} unit="" />
            <YAxis type="category" dataKey="name" width={200} axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--color-text-secondary)' }} />
            <Tooltip
              formatter={(v, _, { payload }) => [`${v} (raw: ${payload.raw?.toFixed(4)})`, 'Importance']}
              contentStyle={{ background: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', borderRadius: 8 }}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {data.map((entry, i) => (
                <Cell
                  key={i}
                  fill={`hsl(${220 - i * 8}, 80%, ${60 - i * 1.5}%)`}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default function Governance({ initialTab = 0 }) {
  const [gov, setGov] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState(initialTab)

  useEffect(() => {
    setActiveTab(initialTab)
  }, [initialTab])

  useEffect(() => {
    api.governance()
      .then(d => { setGov(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  const modelComparison = gov?.model_comparison || {}
  const selectedModel = modelComparison.selected_model
  const modelNames = Object.keys(modelComparison).filter(k =>
    !['selected_model', 'tuned_threshold', 'risk_tiers', 'metric_rationale'].includes(k)
  )

  return (
    <>
      <header className="page-header">
        <div className="page-header-content">
          <div>
            <h1 className="page-title">Model Governance & Fairness</h1>
            <p className="page-subtitle">
              Model comparison, fairness audit, data quality, and AI transparency
            </p>
          </div>
        </div>
      </header>

      <div className="page-body">
        {loading && (
          <div style={{ padding: 48, textAlign: 'center', color: 'var(--color-text-muted)' }}>
            Loading governance report...
          </div>
        )}

        {error && (
          <div className="alert alert-error mb-6">
            <AlertTriangle className="alert-icon" />
            <div>
              {error}
              <br />
              <span style={{ fontSize: 12 }}>Run <code>python src/train.py</code> to generate the governance report.</span>
            </div>
          </div>
        )}

        {gov && (
          <>
            {/* Metric Rationale */}
            <div className="alert alert-info mb-6">
              <Info className="alert-icon" />
              <div style={{ fontSize: 13, lineHeight: 1.6 }}>
                <strong>Metric Rationale:</strong> {gov.metric_rationale}
              </div>
            </div>

            {/* Tabs */}
            <div className="tabs">
              {TABS.map((tab, i) => (
                <button
                  key={i}
                  id={`gov-tab-${i}`}
                  className={`tab ${activeTab === i ? 'active' : ''}`}
                  onClick={() => setActiveTab(i)}
                  aria-selected={activeTab === i}
                  role="tab"
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Tab: Model Comparison */}
            {activeTab === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                {/* Mitigation Notice */}
                {gov.mitigation_comparison?.status && (
                  <div className="alert alert-warning">
                    <Shield className="alert-icon" />
                    <div style={{ fontSize: 13 }}>
                      <strong>{gov.mitigation_comparison.method}:</strong>{' '}
                      {gov.mitigation_comparison.note}
                    </div>
                  </div>
                )}

                <div className="grid-3">
                  {modelNames.map(name => (
                    <ModelCard
                      key={name}
                      name={name}
                      metrics={modelComparison[name]}
                      selected={name === selectedModel}
                    />
                  ))}
                </div>

                {/* Comparison Bar Chart */}
                {modelNames.length > 0 && (
                  <div className="card">
                    <div className="card-header">
                      <div className="card-title"><BarChart2 size={15} /> Performance Comparison</div>
                    </div>
                    <div className="card-body">
                      <ResponsiveContainer width="100%" height={220}>
                        <BarChart
                          data={modelNames.map(name => ({
                            name: name.replace(/_/g, ' ').replace('logistic', 'LR').replace('random forest', 'RF').replace('xgboost calibrated', 'XGB'),
                            'ROC-AUC': +(modelComparison[name]?.roc_auc * 100 || 0).toFixed(1),
                            Recall: +(modelComparison[name]?.recall * 100 || 0).toFixed(1),
                            Precision: +(modelComparison[name]?.precision * 100 || 0).toFixed(1),
                            'PR-AUC': +(modelComparison[name]?.pr_auc * 100 || 0).toFixed(1),
                          }))}
                          barSize={22}
                        >
                          <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                          <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: 'var(--color-text-muted)' }} />
                          <YAxis unit="%" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} domain={[0, 100]} />
                          <Tooltip contentStyle={{ background: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', borderRadius: 8 }} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Bar dataKey="ROC-AUC" fill="var(--color-accent)" radius={[3, 3, 0, 0]} />
                          <Bar dataKey="Recall" fill="var(--color-risk-high)" radius={[3, 3, 0, 0]} />
                          <Bar dataKey="Precision" fill="var(--color-risk-elevated)" radius={[3, 3, 0, 0]} />
                          <Bar dataKey="PR-AUC" fill="var(--color-risk-low)" radius={[3, 3, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Tab: Fairness */}
            {activeTab === 1 && (
              <div>
                <div className="alert alert-info mb-6">
                  <Info className="alert-icon" />
                  <div style={{ fontSize: 13 }}>
                    Fairness metrics compare readmission risk scores across demographic subgroups.
                    ⚠️ marks subgroups with N &lt; 100 — interpret with caution.
                    No demographic-specific thresholds are used in live scoring.
                  </div>
                </div>
                <FairnessSection fairnessData={gov.fairness} />
                {gov.mitigation_comparison && (
                  <div className="card mt-6">
                    <div className="card-header">
                      <div className="card-title"><Shield size={15} /> Fairness Mitigation (Offline)</div>
                    </div>
                    <div className="card-body" style={{ fontSize: 13, lineHeight: 1.6, color: 'var(--color-text-secondary)' }}>
                      <strong>{gov.mitigation_comparison.method}</strong><br />
                      {gov.mitigation_comparison.note}<br /><br />
                      {gov.mitigation_comparison.mitigation_tradeoff && (
                        <div className="alert alert-warning" style={{ marginTop: 8 }}>
                          <Info className="alert-icon" size={14} />
                          {gov.mitigation_comparison.mitigation_tradeoff}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Tab: Data Quality */}
            {activeTab === 2 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                {gov.data_quality && (
                  <>
                    <div className="stat-grid">
                      {[
                        { label: 'Total Encounters (raw)', value: gov.data_quality.total_raw?.toLocaleString() ?? '—' },
                        { label: 'After Filters', value: gov.data_quality.after_filters?.toLocaleString() ?? '—' },
                        { label: 'Training Split', value: gov.data_quality.train_size?.toLocaleString() ?? '—' },
                        { label: 'Test Split', value: gov.data_quality.test_size?.toLocaleString() ?? '—' },
                        { label: 'After SMOTE', value: gov.data_quality.after_smote_train_size?.toLocaleString() ?? '—' },
                        { label: 'Positive Rate (test)', value: gov.data_quality.positive_rate_test != null ? `${(gov.data_quality.positive_rate_test * 100).toFixed(1)}%` : '—' },
                      ].map(({ label, value }) => (
                        <div key={label} className="stat-card">
                          <div className="stat-label">{label}</div>
                          <div className="stat-value" style={{ fontSize: 22 }}>{value}</div>
                        </div>
                      ))}
                    </div>

                    {[
                      { label: 'Missing Value Strategy', value: gov.data_quality.missing_value_strategy },
                      { label: 'Outlier Strategy', value: gov.data_quality.outlier_strategy },
                      { label: 'Split Strategy', value: gov.data_quality.split_strategy },
                      { label: 'Imbalance Strategy', value: gov.data_quality.imbalance_strategy },
                      { label: 'Excluded Records', value: gov.data_quality.excluded_records },
                      { label: 'Feature Selection', value: gov.data_quality.feature_selection },
                    ].filter(r => r.value).map(({ label, value }) => (
                      <div key={label} className="card">
                        <div className="card-body">
                          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
                            {label}
                          </div>
                          <div style={{ fontSize: 13, color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
                            {value}
                          </div>
                        </div>
                      </div>
                    ))}
                  </>
                )}
              </div>
            )}

            {/* Tab: Feature Importance */}
            {activeTab === 3 && (
              <div>
                <FeatureImportanceChart importances={gov.feature_importances} />
                <div className="alert alert-info mt-4">
                  <Info className="alert-icon" />
                  <div style={{ fontSize: 13 }}>
                    Feature importances reflect the XGBoost model's use of each feature across all training examples.
                    They represent association strength, not causal effects.
                    Individual predictions use SHAP values for instance-level attribution.
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </>
  )
}
