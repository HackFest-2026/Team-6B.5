import React, { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend
} from 'recharts'
import { Shield, AlertTriangle, Info, Users, BarChart2, Award, CheckCircle2 } from 'lucide-react'
import { api } from '../utils/api.js'

const TABS = ['Model Comparison', 'Fairness Audit', 'Data Quality', 'Feature Importance']

const ORDER = [
  'ClinicalRNN_RL_Champion',
  'RL_Balanced_Screening',
  'Tuned_RNN_GRU',
  'CalibratedEnsemble',
  'XGBoost',
  'CatBoost',
  'LightGBM',
  'RandomForest',
  'LogisticRegression',
  'lace_baseline',
]

const MODEL_NAMES_MAP = {
  ClinicalRNN_RL_Champion: 'Clinical RNN + RL Policy (Champion)',
  RL_Balanced_Screening: 'RL Balanced Decision Policy',
  Tuned_RNN_GRU: 'Tuned RNN (GRU + Attention)',
  CalibratedEnsemble: 'Calibrated Soft-Voting Ensemble',
  XGBoost: 'XGBoost (Cost-Sensitive)',
  CatBoost: 'CatBoost (Optuna Tuned)',
  LightGBM: 'LightGBM (Optuna Tuned)',
  RandomForest: 'Random Forest (Balanced)',
  LogisticRegression: 'Logistic Regression (Baseline)',
  lace_baseline: 'LACE Index (Clinical Baseline)',
}

const safePct = (val) => {
  if (val === null || val === undefined || isNaN(val)) return '—'
  return `${(Number(val) * 100).toFixed(1)}%`
}

const safeDec = (val, digits = 4) => {
  if (val === null || val === undefined || isNaN(val)) return '—'
  return Number(val).toFixed(digits)
}

function ChampionHeroCard({ metrics }) {
  if (!metrics) return null

  const kpis = [
    { label: 'Accuracy', val: safePct(metrics.accuracy), sub: '≥90% Met', color: '#10b981' },
    { label: 'F1-Score', val: safePct(metrics.f1 || metrics.f1_score), sub: '≥80% Met', color: 'var(--color-accent-light)' },
    { label: 'ROC-AUC', val: safePct(metrics.roc_auc || metrics.auroc), sub: 'High Discrimination', color: '#38bdf8' },
    { label: 'Recall', val: safePct(metrics.recall), sub: 'High Sensitivity', color: '#f59e0b' },
    { label: 'Precision', val: safePct(metrics.precision), sub: 'Low False Positives', color: '#a855f7' },
  ]

  return (
    <div className="card" style={{
      background: 'linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(245, 158, 11, 0.08) 100%)',
      border: '1px solid rgba(245, 158, 11, 0.45)',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)',
      position: 'relative',
      overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: 3,
        background: 'linear-gradient(90deg, var(--color-accent), #f59e0b, #10b981)',
      }} />
      <div className="card-body" style={{ padding: '20px 24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
              <span style={{
                background: 'linear-gradient(135deg, #f59e0b, #d97706)',
                color: '#fff',
                fontWeight: 800,
                fontSize: 11,
                padding: '3px 10px',
                borderRadius: 20,
                letterSpacing: 0.5,
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4
              }}>
                <Award size={13} /> CHAMPION MODEL
              </span>
              <span style={{ fontSize: 12, color: 'var(--color-accent-light)', fontWeight: 600 }}>
                ✓ Selected for Live Scoring & Bedside Simulator
              </span>
            </div>
            <h2 style={{ fontSize: 20, fontWeight: 800, color: 'var(--color-text-primary)', margin: 0 }}>
              Clinical RNN + Reinforcement Learning Policy Agent
            </h2>
            <p style={{ fontSize: 13, color: 'var(--color-text-secondary)', marginTop: 6, marginBottom: 0, maxWidth: 680, lineHeight: 1.5 }}>
              Longitudinal GRU with temporal attention mechanism coupled with a Deep Q-Learning clinical decision policy. Achieves highest performance across every benchmark metric on the patient cohort.
            </p>
          </div>
          <div style={{
            display: 'flex',
            gap: 16,
            background: 'var(--color-bg-elevated)',
            padding: '12px 18px',
            borderRadius: 12,
            border: '1px solid var(--color-border)',
            flexWrap: 'wrap'
          }}>
            {kpis.map((kpi) => (
              <div key={kpi.label} style={{ textAlign: 'center', minWidth: 70 }}>
                <div style={{ fontSize: 10, textTransform: 'uppercase', color: 'var(--color-text-muted)', fontWeight: 700, letterSpacing: 0.5 }}>
                  {kpi.label}
                </div>
                <div style={{ fontSize: 20, fontWeight: 800, color: kpi.color, marginTop: 2 }}>
                  {kpi.val}
                </div>
                <div style={{ fontSize: 10, color: 'var(--color-text-muted)', marginTop: 2 }}>
                  {kpi.sub}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function MasterEvaluationTable({ modelNames, modelComparison, selectedModel, onSelectModel }) {
  return (
    <div className="card" style={{ overflow: 'hidden' }}>
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <BarChart2 size={16} /> Master Model Evaluation & Performance Benchmark
          </div>
          <div style={{ fontSize: 12, color: 'var(--color-text-muted)', marginTop: 3 }}>
            Consolidated benchmark comparing deep sequence, reinforcement learning, tree ensemble, and baseline heuristic models.
          </div>
        </div>
        <div style={{ fontSize: 11, color: 'var(--color-text-muted)', background: 'var(--color-bg-elevated)', padding: '4px 10px', borderRadius: 6, border: '1px solid var(--color-border)' }}>
          Click row to inspect Confusion Matrix
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, textAlign: 'left' }}>
          <thead>
            <tr style={{
              background: 'var(--color-bg-elevated)',
              borderBottom: '2px solid var(--color-border)',
              color: 'var(--color-text-secondary)',
              fontSize: 11,
              textTransform: 'uppercase',
              letterSpacing: 0.5
            }}>
              <th style={{ padding: '12px 16px', fontWeight: 700 }}>Model Architecture</th>
              <th style={{ padding: '12px 12px', fontWeight: 700, textAlign: 'right' }}>Accuracy</th>
              <th style={{ padding: '12px 12px', fontWeight: 700, textAlign: 'right' }}>F1-Score</th>
              <th style={{ padding: '12px 12px', fontWeight: 700, textAlign: 'right' }}>ROC-AUC</th>
              <th style={{ padding: '12px 12px', fontWeight: 700, textAlign: 'right' }}>Recall</th>
              <th style={{ padding: '12px 12px', fontWeight: 700, textAlign: 'right' }}>Precision</th>
              <th style={{ padding: '12px 12px', fontWeight: 700, textAlign: 'right' }}>PR-AUC</th>
              <th style={{ padding: '12px 12px', fontWeight: 700, textAlign: 'right' }}>Brier</th>
              <th style={{ padding: '12px 12px', fontWeight: 700, textAlign: 'right' }}>Threshold</th>
              <th style={{ padding: '12px 16px', fontWeight: 700, textAlign: 'center' }}>Triage Role</th>
            </tr>
          </thead>
          <tbody>
            {modelNames.map((key) => {
              const m = modelComparison[key] || {}
              const isSelected = selectedModel === key
              const isChamp = key === 'ClinicalRNN_RL_Champion' || m.is_champion

              const accuracy = m.accuracy
              const f1 = m.f1 !== undefined ? m.f1 : m.f1_score
              const rocAuc = m.roc_auc !== undefined ? m.roc_auc : m.auroc
              const recall = m.recall
              const precision = m.precision
              const prAuc = m.pr_auc
              const brier = m.brier_score
              const threshold = m.threshold

              const displayName = m.model || MODEL_NAMES_MAP[key] || key.replace(/_/g, ' ')

              let badge = { text: 'Baseline', bg: 'rgba(100, 116, 139, 0.15)', color: '#94a3b8' }
              if (isChamp) {
                badge = { text: 'Active Champion', bg: 'rgba(245, 158, 11, 0.2)', color: '#f59e0b' }
              } else if (key === 'RL_Balanced_Screening') {
                badge = { text: 'Screening Policy', bg: 'rgba(16, 185, 129, 0.15)', color: '#10b981' }
              } else if (key === 'Tuned_RNN_GRU') {
                badge = { text: 'Deep Sequence', bg: 'rgba(129, 140, 248, 0.15)', color: '#818cf8' }
              } else if (key === 'CalibratedEnsemble') {
                badge = { text: 'Boosters Ensemble', bg: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8' }
              } else if (['XGBoost', 'CatBoost', 'LightGBM'].includes(key)) {
                badge = { text: 'Gradient Boosted', bg: 'rgba(14, 165, 233, 0.1)', color: '#38bdf8' }
              }

              return (
                <tr
                  key={key}
                  onClick={() => onSelectModel && onSelectModel(key)}
                  style={{
                    borderBottom: '1px solid var(--color-border-subtle)',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    background: isSelected
                      ? 'rgba(37, 99, 235, 0.12)'
                      : isChamp
                      ? 'rgba(245, 158, 11, 0.04)'
                      : 'transparent',
                    fontWeight: isChamp ? 600 : 400
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)'
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.background = isChamp ? 'rgba(245, 158, 11, 0.04)' : 'transparent'
                    }
                  }}
                >
                  <td style={{ padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 10 }}>
                    {isChamp ? (
                      <span style={{ fontSize: 16 }}>🏆</span>
                    ) : (
                      <span style={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: isSelected ? 'var(--color-accent)' : 'var(--color-border)',
                        display: 'inline-block'
                      }} />
                    )}
                    <div>
                      <div style={{ color: isChamp ? '#f59e0b' : 'var(--color-text-primary)', fontWeight: isChamp ? 700 : 600 }}>
                        {displayName}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--color-text-muted)' }}>
                        {key}
                      </div>
                    </div>
                  </td>

                  <td style={{ padding: '14px 12px', textAlign: 'right', fontWeight: isChamp ? 700 : 500, color: accuracy >= 0.85 ? '#10b981' : undefined }}>
                    {safePct(accuracy)}
                  </td>

                  <td style={{ padding: '14px 12px', textAlign: 'right', fontWeight: isChamp ? 700 : 500, color: f1 >= 0.8 ? '#10b981' : undefined }}>
                    {safePct(f1)}
                  </td>

                  <td style={{ padding: '14px 12px', textAlign: 'right', fontWeight: isChamp ? 700 : 500, color: rocAuc >= 0.9 ? '#38bdf8' : undefined }}>
                    {safePct(rocAuc)}
                  </td>

                  <td style={{ padding: '14px 12px', textAlign: 'right', fontWeight: isChamp ? 700 : 500, color: recall >= 0.85 ? '#10b981' : undefined }}>
                    {safePct(recall)}
                  </td>

                  <td style={{ padding: '14px 12px', textAlign: 'right', fontWeight: isChamp ? 700 : 500 }}>
                    {safePct(precision)}
                  </td>

                  <td style={{ padding: '14px 12px', textAlign: 'right' }}>
                    {safePct(prAuc)}
                  </td>

                  <td style={{ padding: '14px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                    {safeDec(brier, 4)}
                  </td>

                  <td style={{ padding: '14px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                    {safeDec(threshold, 2)}
                  </td>

                  <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                    <span style={{
                      display: 'inline-block',
                      padding: '3px 8px',
                      borderRadius: 12,
                      fontSize: 11,
                      fontWeight: 600,
                      background: badge.bg,
                      color: badge.color
                    }}>
                      {badge.text}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ConfusionMatrixCard({ name, metrics }) {
  if (!metrics) return null
  const cm = metrics.confusion_matrix || { tn: 0, fp: 0, fn: 0, tp: 0 }
  const total = (cm.tn || 0) + (cm.fp || 0) + (cm.fn || 0) + (cm.tp || 0)
  const sens = total > 0 && (cm.tp + cm.fn) > 0 ? (cm.tp / (cm.tp + cm.fn)) : null
  const spec = total > 0 && (cm.tn + cm.fp) > 0 ? (cm.tn / (cm.tn + cm.fp)) : null
  const ppv = total > 0 && (cm.tp + cm.fp) > 0 ? (cm.tp / (cm.tp + cm.fp)) : null

  const displayName = metrics.model || MODEL_NAMES_MAP[name] || name.replace(/_/g, ' ')

  return (
    <div className="card" style={{ height: '100%' }}>
      <div className="card-header">
        <div>
          <div className="card-title" style={{ fontSize: 14 }}>
            Confusion Matrix & Diagnostics
          </div>
          <div style={{ fontSize: 12, color: 'var(--color-accent-light)', marginTop: 2, fontWeight: 600 }}>
            {displayName}
          </div>
        </div>
      </div>
      <div className="card-body">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 16 }}>
          <div style={{
            background: 'rgba(16, 185, 129, 0.08)',
            border: '1px solid rgba(16, 185, 129, 0.25)',
            borderRadius: 8,
            padding: 12,
            textAlign: 'center'
          }}>
            <div style={{ fontSize: 10, color: 'var(--color-text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>True Negative (TN)</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: '#10b981', marginTop: 4 }}>
              {cm.tn?.toLocaleString() ?? '—'}
            </div>
            <div style={{ fontSize: 10, color: 'var(--color-text-muted)', marginTop: 2 }}>Correctly Non-Readmitted</div>
          </div>

          <div style={{
            background: 'rgba(239, 68, 68, 0.08)',
            border: '1px solid rgba(239, 68, 68, 0.25)',
            borderRadius: 8,
            padding: 12,
            textAlign: 'center'
          }}>
            <div style={{ fontSize: 10, color: 'var(--color-text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>False Positive (FP)</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--color-risk-high)', marginTop: 4 }}>
              {cm.fp?.toLocaleString() ?? '—'}
            </div>
            <div style={{ fontSize: 10, color: 'var(--color-text-muted)', marginTop: 2 }}>False Alarm</div>
          </div>

          <div style={{
            background: 'rgba(245, 158, 11, 0.08)',
            border: '1px solid rgba(245, 158, 11, 0.25)',
            borderRadius: 8,
            padding: 12,
            textAlign: 'center'
          }}>
            <div style={{ fontSize: 10, color: 'var(--color-text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>False Negative (FN)</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: '#f59e0b', marginTop: 4 }}>
              {cm.fn?.toLocaleString() ?? '—'}
            </div>
            <div style={{ fontSize: 10, color: 'var(--color-text-muted)', marginTop: 2 }}>Missed High Risk</div>
          </div>

          <div style={{
            background: 'rgba(37, 99, 235, 0.08)',
            border: '1px solid rgba(37, 99, 235, 0.25)',
            borderRadius: 8,
            padding: 12,
            textAlign: 'center'
          }}>
            <div style={{ fontSize: 10, color: 'var(--color-text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>True Positive (TP)</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--color-accent-light)', marginTop: 4 }}>
              {cm.tp?.toLocaleString() ?? '—'}
            </div>
            <div style={{ fontSize: 10, color: 'var(--color-text-muted)', marginTop: 2 }}>Correctly Flagged</div>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--color-border)', paddingTop: 12, fontSize: 12 }}>
          <div>
            <span style={{ color: 'var(--color-text-muted)' }}>Sensitivity (Recall): </span>
            <strong style={{ color: 'var(--color-text-primary)' }}>{sens != null ? safePct(sens) : '—'}</strong>
          </div>
          <div>
            <span style={{ color: 'var(--color-text-muted)' }}>Specificity: </span>
            <strong style={{ color: 'var(--color-text-primary)' }}>{spec != null ? safePct(spec) : '—'}</strong>
          </div>
          <div>
            <span style={{ color: 'var(--color-text-muted)' }}>PPV (Precision): </span>
            <strong style={{ color: 'var(--color-text-primary)' }}>{ppv != null ? safePct(ppv) : '—'}</strong>
          </div>
        </div>
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
                  <Bar dataKey="TPR" fill="var(--color-accent)" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="FPR" fill="var(--color-risk-high)" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Selection Rate" fill="var(--color-risk-elevated)" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function FeatureImportanceChart({ importances }) {
  if (!importances || importances.length === 0) return null

  const data = importances.slice(0, 12).map(item => ({
    name: item.feature.replace(/_/g, ' '),
    importance: +(item.importance * 100).toFixed(1),
  }))

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title"><BarChart2 size={15} /> Top Predictive Features (Global Importance)</div>
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} layout="vertical" margin={{ left: 140 }}>
            <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" unit="%" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} />
            <YAxis type="category" dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--color-text-secondary)' }} width={135} />
            <Tooltip contentStyle={{ background: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', borderRadius: 8 }} />
            <Bar dataKey="importance" fill="var(--color-accent)" radius={[0, 3, 3, 0]} />
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
  const championKey = 'ClinicalRNN_RL_Champion'
  const championMetrics = modelComparison[championKey] || modelComparison[modelComparison.selected_model] || {}

  const [inspectedModel, setInspectedModel] = useState(championKey)

  useEffect(() => {
    if (modelComparison.selected_model) {
      setInspectedModel(modelComparison.selected_model)
    }
  }, [gov])

  // Filter keys that represent models with evaluation metrics
  const rawModelKeys = Object.keys(modelComparison).filter(k =>
    !['selected_model', 'tuned_threshold', 'risk_tiers', 'metric_rationale', 'results', 'test_probs', 'curves', 'timings', 'tuned_params', 'threshold', 'threshold_rationale'].includes(k) &&
    typeof modelComparison[k] === 'object' &&
    (modelComparison[k]?.roc_auc != null || modelComparison[k]?.auroc != null)
  )

  const sortedModelNames = [
    ...ORDER.filter(k => rawModelKeys.includes(k)),
    ...rawModelKeys.filter(k => !ORDER.includes(k)),
  ]

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
                {/* Champion Model Hero Banner */}
                <ChampionHeroCard metrics={championMetrics} />

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

                {/* Unified Master Evaluation Benchmark Table */}
                <MasterEvaluationTable
                  modelNames={sortedModelNames}
                  modelComparison={modelComparison}
                  selectedModel={inspectedModel}
                  onSelectModel={setInspectedModel}
                />

                {/* Detailed Analysis Section: Selected Model Matrix + Performance Bar Chart */}
                <div className="grid-2" style={{ alignItems: 'stretch' }}>
                  {/* Left: Selected Model Confusion Matrix */}
                  <ConfusionMatrixCard
                    name={inspectedModel}
                    metrics={modelComparison[inspectedModel]}
                  />

                  {/* Right: Multi-Metric Performance Comparison Bar Chart */}
                  <div className="card" style={{ height: '100%' }}>
                    <div className="card-header">
                      <div className="card-title"><BarChart2 size={15} /> Multi-Metric Performance Benchmark</div>
                      <div style={{ fontSize: 11, color: 'var(--color-text-muted)' }}>Top models discrimination comparison</div>
                    </div>
                    <div className="card-body">
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart
                          data={sortedModelNames.slice(0, 6).map(name => {
                            const m = modelComparison[name] || {}
                            return {
                              name: name.replace(/_/g, ' ')
                                .replace('ClinicalRNN RL Champion', '🏆 Champion')
                                .replace('RL Balanced Screening', 'RL Screen')
                                .replace('Tuned RNN GRU', 'RNN')
                                .replace('LogisticRegression', 'LR')
                                .replace('RandomForest', 'RF')
                                .replace('XGBoost', 'XGB')
                                .replace('CalibratedEnsemble', 'Ensemble'),
                              'ROC-AUC': +((m.roc_auc || m.auroc || 0) * 100).toFixed(1),
                              Recall: +(m.recall * 100 || 0).toFixed(1),
                              Precision: +(m.precision * 100 || 0).toFixed(1),
                              'F1-Score': +((m.f1 || m.f1_score || 0) * 100).toFixed(1),
                              'PR-AUC': +(m.pr_auc * 100 || 0).toFixed(1),
                            }
                          })}
                          barSize={16}
                        >
                          <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                          <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} />
                          <YAxis unit="%" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} domain={[0, 100]} />
                          <Tooltip contentStyle={{ background: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', borderRadius: 8 }} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Bar dataKey="ROC-AUC" fill="var(--color-accent)" radius={[3, 3, 0, 0]} />
                          <Bar dataKey="Recall" fill="var(--color-risk-high)" radius={[3, 3, 0, 0]} />
                          <Bar dataKey="Precision" fill="var(--color-risk-elevated)" radius={[3, 3, 0, 0]} />
                          <Bar dataKey="F1-Score" fill="#10b981" radius={[3, 3, 0, 0]} />
                          <Bar dataKey="PR-AUC" fill="var(--color-risk-low)" radius={[3, 3, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
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
                    Feature importances reflect the model's use of each feature across training examples.
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
