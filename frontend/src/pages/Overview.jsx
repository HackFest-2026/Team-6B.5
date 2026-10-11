import React, { useState, useEffect } from 'react'
import {
  BarChart, Bar, PieChart, Pie, Cell, Tooltip, ResponsiveContainer, XAxis, YAxis
} from 'recharts'
import {
  Activity, Users, AlertTriangle, CheckCircle, TrendingUp, Info,
  ArrowRight, Clock, Zap
} from 'lucide-react'
import { api } from '../utils/api.js'
import { formatPct } from '../utils/helpers.js'

const RISK_COLORS = {
  low: 'var(--color-risk-low)',
  elevated: 'var(--color-risk-elevated)',
  high: 'var(--color-risk-high)',
}

function StatCard({ label, value, sub, icon: Icon, variant, loading }) {
  if (loading) return (
    <div className={`stat-card ${variant || ''}`}>
      <div className="skeleton skeleton-title" />
      <div className="skeleton skeleton-text" />
    </div>
  )
  return (
    <div className={`stat-card ${variant || ''}`}>
      {Icon && <Icon className="stat-icon" aria-hidden="true" />}
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-value-sub">{sub}</div>}
    </div>
  )
}

export default function Overview({ onNavigate }) {
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tierFilter, setTierFilter] = useState('')
  const [facilityFilter, setFacilityFilter] = useState('')

  const fetchSummary = (tier, fac) => {
    setLoading(true)
    api.summary({
      risk_tier: tier || undefined,
      facility: fac ? parseInt(fac) : undefined,
    })
      .then(data => { setSummary(data); setLoading(false) })
      .catch(err => { setError(err.message); setLoading(false) })
  }

  useEffect(() => {
    fetchSummary(tierFilter, facilityFilter)
  }, [tierFilter, facilityFilter])

  const riskPieData = summary ? [
    { name: 'Low', value: summary.risk_distribution?.low ?? 0, color: RISK_COLORS.low },
    { name: 'Elevated', value: summary.risk_distribution?.elevated ?? 0, color: RISK_COLORS.elevated },
    { name: 'High', value: summary.risk_distribution?.high ?? 0, color: RISK_COLORS.high },
  ] : []

  const planBarData = summary ? [
    { name: 'Pending', value: summary.care_plan_counts?.pending ?? 0, fill: 'var(--color-warning)' },
    { name: 'In Progress', value: summary.care_plan_counts?.in_progress ?? 0, fill: 'var(--color-accent)' },
    { name: 'Completed', value: summary.care_plan_counts?.completed ?? 0, fill: 'var(--color-success)' },
  ] : []

  return (
    <>
      <header className="page-header">
        <div className="page-header-content">
          <div>
            <h1 className="page-title">Risk Command Center</h1>
            <p className="page-subtitle">
              Real-time 30-day readmission risk cohort overview
            </p>
          </div>
          <div className="flex items-center gap-3">
            {summary && (
              <span style={{ fontSize: 12, color: 'var(--color-text-muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className="risk-badge" style={{ background: '#f59e0b', color: '#fff', fontSize: 10, padding: '2px 8px', fontWeight: 700 }}>
                  🏆 Champion Model
                </span>
                Model v{summary.model_version} · Threshold {(summary.threshold * 100).toFixed(0)}%
              </span>
            )}
            <button
              id="btn-go-worklist"
              className="btn btn-primary btn-sm"
              onClick={() => onNavigate('worklist')}
            >
              View Worklist <ArrowRight size={14} />
            </button>
          </div>
        </div>
      </header>

      <div className="page-body">
        {error && (
          <div className="alert alert-error mb-6">
            <AlertTriangle className="alert-icon" />
            <div>
              <strong>API Error:</strong> {error}
              <br />
              <span style={{ fontSize: 12 }}>
                Ensure the backend is running: <code>uvicorn backend.main:app --reload</code>
              </span>
            </div>
          </div>
        )}

        {/* Research Notice */}
        <div className="alert alert-warning mb-6">
          <Info className="alert-icon" aria-hidden="true" />
          <div>
            <strong>Research Demonstration Notice</strong>
            <p style={{ marginTop: 4, fontSize: 13, lineHeight: 1.5 }}>
              {summary?.cohort_note ||
                'This cohort is a de-identified research demonstration. Model outputs are NOT a validated clinical tool and must not be used for individual patient care decisions.'}
            </p>
          </div>
        </div>

        {/* Filter Bar */}
        <div className="card mb-6">
          <div className="card-body" style={{ padding: '14px 20px' }}>
            <div className="filter-bar" style={{ justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text-secondary)' }}>
                  Cohort Filter:
                </span>
                <div className="filter-chips">
                  <button
                    className={`filter-chip ${tierFilter === '' ? 'active' : ''}`}
                    onClick={() => setTierFilter('')}
                  >
                    All Tiers
                  </button>
                  <button
                    className={`filter-chip ${tierFilter === 'high' ? 'active' : ''}`}
                    onClick={() => setTierFilter('high')}
                  >
                    High Risk Only
                  </button>
                  <button
                    className={`filter-chip ${tierFilter === 'elevated' ? 'active' : ''}`}
                    onClick={() => setTierFilter('elevated')}
                  >
                    Elevated Only
                  </button>
                  <button
                    className={`filter-chip ${tierFilter === 'low' ? 'active' : ''}`}
                    onClick={() => setTierFilter('low')}
                  >
                    Low Risk Only
                  </button>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <label style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>Admission Source:</label>
                <select
                  className="form-select"
                  style={{ width: 'auto', padding: '6px 12px', fontSize: 13 }}
                  value={facilityFilter}
                  onChange={(e) => setFacilityFilter(e.target.value)}
                >
                  <option value="">All Facilities</option>
                  <option value="1">Physician Referral (1)</option>
                  <option value="7">Emergency Room (7)</option>
                  <option value="2">Clinic Referral (2)</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <section className="section">
          <div className="section-header">
            <h2 className="section-title"><Activity size={16} /> Cohort Metrics</h2>
          </div>
          <div className="stat-grid">
            <StatCard
              label="Total Encounters"
              value={loading ? '—' : summary?.total_encounters?.toLocaleString() ?? '—'}
              sub="In demo cohort"
              icon={Users}
              loading={loading && !error}
            />
            <StatCard
              label="Flagged for Follow-up"
              value={loading ? '—' : summary?.flagged_count?.toLocaleString() ?? '—'}
              sub={summary ? `${summary.flagged_pct}% of cohort` : ''}
              icon={AlertTriangle}
              variant="risk-elevated"
              loading={loading && !error}
            />
            <StatCard
              label="High Risk"
              value={loading ? '—' : summary?.risk_distribution?.high?.toLocaleString() ?? '—'}
              sub="Needs urgent review"
              icon={Zap}
              variant="risk-high"
              loading={loading && !error}
            />
            <StatCard
              label="Care Plans Completed"
              value={loading ? '—' : summary?.care_plan_counts?.completed?.toLocaleString() ?? '—'}
              sub={summary ?
                `${summary.care_plan_counts.pending} pending · ${summary.care_plan_counts.in_progress} in progress`
                : ''}
              icon={CheckCircle}
              variant="risk-low"
              loading={loading && !error}
            />
          </div>
        </section>

        {/* Charts */}
        {!loading && !error && summary && (
          <div className="grid-2 mb-8">
            {/* Risk Distribution Pie */}
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title"><TrendingUp size={15} /> Risk Distribution</div>
                  <div className="card-subtitle">Readmission risk tier breakdown</div>
                </div>
              </div>
              <div className="card-body">
                <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
                  <ResponsiveContainer width={160} height={160}>
                    <PieChart>
                      <Pie
                        data={riskPieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={45}
                        outerRadius={70}
                        paddingAngle={3}
                        dataKey="value"
                      >
                        {riskPieData.map((entry, i) => (
                          <Cell key={i} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip
                        formatter={(v) => [v.toLocaleString(), 'Encounters']}
                        contentStyle={{
                          background: 'var(--color-bg-elevated)',
                          border: '1px solid var(--color-border)',
                          borderRadius: 8,
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {riskPieData.map(d => (
                      <div key={d.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{
                          width: 10, height: 10, borderRadius: '50%',
                          background: d.color, flexShrink: 0,
                        }} />
                        <span style={{ fontSize: 13, flex: 1 }}>{d.name} Risk</span>
                        <span style={{ fontSize: 13, fontWeight: 600, color: d.color }}>
                          {d.value?.toLocaleString()}
                        </span>
                        <span style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>
                          ({summary.total_encounters > 0
                            ? ((d.value / summary.total_encounters) * 100).toFixed(1)
                            : 0}%)
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Care Plan Status */}
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title"><CheckCircle size={15} /> Care Plan Status</div>
                  <div className="card-subtitle">Discharge planning workflow</div>
                </div>
              </div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={planBarData} barSize={32}>
                    <XAxis
                      dataKey="name"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fontSize: 12, fill: 'var(--color-text-muted)' }}
                    />
                    <YAxis hide />
                    <Tooltip
                      formatter={(v) => [v.toLocaleString(), 'Plans']}
                      contentStyle={{
                        background: 'var(--color-bg-elevated)',
                        border: '1px solid var(--color-border)',
                        borderRadius: 8,
                      }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {planBarData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <section className="section">
          <div className="section-header">
            <h2 className="section-title">Quick Actions</h2>
          </div>
          <div className="grid-3">
            <ActionCard
              title="Review Worklist"
              description="Search and filter encounters by risk tier and care-plan status."
              action="Open Worklist"
              onClick={() => onNavigate('worklist')}
              icon={Users}
              color="var(--color-accent)"
            />
            <ActionCard
              title="Scenario Simulator"
              description="Adjust clinical variables and see how the calibrated model responds."
              action="Open Simulator"
              onClick={() => onNavigate('calculator')}
              icon={Activity}
              color="var(--color-risk-elevated)"
            />
            <ActionCard
              title="Governance & Fairness"
              description="Model comparison, fairness audit, and data-quality decisions."
              action="View Report"
              onClick={() => onNavigate('governance')}
              icon={CheckCircle}
              color="var(--color-risk-low)"
            />
          </div>
        </section>
      </div>
    </>
  )
}

function ActionCard({ title, description, action, onClick, icon: Icon, color }) {
  return (
    <div
      className="card"
      style={{ cursor: 'pointer', transition: 'all 0.2s' }}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && onClick()}
    >
      <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%' }}>
        <div style={{
          width: 40, height: 40, borderRadius: 10,
          background: `${color}20`, border: `1px solid ${color}40`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Icon size={20} style={{ color }} />
        </div>
        <div>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>{title}</div>
          <div style={{ fontSize: 13, color: 'var(--color-text-muted)', lineHeight: 1.5 }}>{description}</div>
        </div>
        <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', gap: 4, color, fontSize: 13, fontWeight: 600 }}>
          {action} <ArrowRight size={13} />
        </div>
      </div>
    </div>
  )
}
