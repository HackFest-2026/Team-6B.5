import React, { useState, useEffect } from 'react'
import { Calculator, AlertTriangle, Info, ArrowUp, ArrowDown, RefreshCw } from 'lucide-react'
import { api } from '../utils/api.js'
import { formatPct, tierLabel, tierColor } from '../utils/helpers.js'

const AGE_BANDS = [
  '[20-30)', '[30-40)', '[40-50)', '[50-60)',
  '[60-70)', '[70-80)', '[80-90)', '[90-100)'
]

const DEFAULTS = {
  time_in_hospital: 4,
  num_lab_procedures: 44,
  num_procedures: 1,
  num_medications: 15,
  number_inpatient: 0,
  number_emergency: 0,
  number_outpatient: 0,
  number_diagnoses: 5,
  age: '[60-70)',
  gender: 'Male',
  max_glu_serum: 'None',
  a1c_result: 'None',
  change: 'No',
  insulin: 'No',
  metformin: 'Steady',
}

function SliderField({ id, label, value, min, max, onChange, hint }) {
  return (
    <div className="form-group">
      <label htmlFor={id} className="form-label">
        {label}
        <span style={{ marginLeft: 8, fontWeight: 700, color: 'var(--color-accent-light)' }}>
          {value}
        </span>
      </label>
      <input
        id={id}
        type="range"
        className="form-range"
        min={min}
        max={max}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        aria-valuenow={value}
        aria-valuemin={min}
        aria-valuemax={max}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--color-text-muted)' }}>
        <span>{min}</span>
        {hint && <span style={{ color: 'var(--color-text-muted)' }}>{hint}</span>}
        <span>{max}</span>
      </div>
    </div>
  )
}

function SelectField({ id, label, value, options, onChange }) {
  return (
    <div className="form-group">
      <label htmlFor={id} className="form-label">{label}</label>
      <select
        id={id}
        className="form-select"
        value={value}
        onChange={e => onChange(e.target.value)}
      >
        {options.map(o => (
          <option key={o.value ?? o} value={o.value ?? o}>
            {o.label ?? o}
          </option>
        ))}
      </select>
    </div>
  )
}

export default function RiskCalculator({ initialEncId }) {
  const [scenario, setScenario] = useState(DEFAULTS)
  const [encId, setEncId] = useState(initialEncId || '')
  const [baseline, setBaseline] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const set = (field) => (val) => setScenario(s => ({ ...s, [field]: val }))

  const runScore = async () => {
    setLoading(true)
    setError(null)
    try {
      const body = {
        ...scenario,
        encounter_id: encId ? parseInt(encId) : undefined,
      }
      const res = await api.predict(body)
      setResult(res)
      setBaseline(res.baseline_prob != null ? res.baseline_prob : null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setScenario(DEFAULTS)
    setResult(null)
    setBaseline(null)
    setEncId(initialEncId || '')
    setError(null)
  }

  const delta = result?.absolute_change
  const tier = result?.tier

  return (
    <>
      <header className="page-header">
        <div className="page-header-content">
          <div>
            <h1 className="page-title">Bedside Risk Simulator</h1>
            <p className="page-subtitle">
              Adjust encounter variables and score with the saved calibrated model
            </p>
          </div>
          <button id="btn-reset-simulator" className="btn btn-ghost btn-sm" onClick={reset}>
            <RefreshCw size={14} /> Reset
          </button>
        </div>
      </header>

      <div className="page-body">
        {/* Notice */}
        <div className="alert alert-warning mb-6">
          <AlertTriangle className="alert-icon" />
          <div style={{ fontSize: 13 }}>
            <strong>Research demonstration only.</strong> This simulator uses the saved trained model on historical de-identified data.
            It does not retrain or refit on each request. Outputs are NOT a validated clinical tool.
          </div>
        </div>

        <div className="grid-2" style={{ alignItems: 'start' }}>
          {/* Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="card">
              <div className="card-header">
                <div className="card-title"><Calculator size={15} /> Encounter Parameters</div>
              </div>
              <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                {/* Encounter ID (optional) */}
                <div className="form-group">
                  <label htmlFor="sim-enc-id" className="form-label">
                    Load Baseline Encounter ID (optional)
                  </label>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <input
                      id="sim-enc-id"
                      className="form-input"
                      style={{ flex: 1 }}
                      placeholder="e.g. 12345"
                      value={encId}
                      onChange={e => setEncId(e.target.value.replace(/\D/g, ''))}
                      aria-label="Encounter ID for baseline comparison"
                    />
                  </div>
                  <p style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 4 }}>
                    If provided, baseline probability is loaded from the saved encounter.
                  </p>
                </div>

                <SelectField
                  id="sim-age"
                  label="Age Band"
                  value={scenario.age}
                  options={AGE_BANDS}
                  onChange={set('age')}
                />

                <SelectField
                  id="sim-gender"
                  label="Gender"
                  value={scenario.gender}
                  options={['Male', 'Female', 'Unknown']}
                  onChange={set('gender')}
                />

                <SliderField
                  id="sim-time"
                  label="Length of Stay (days)"
                  value={scenario.time_in_hospital}
                  min={1} max={14}
                  onChange={set('time_in_hospital')}
                  hint="Hospital days"
                />

                <SliderField
                  id="sim-meds"
                  label="Number of Medications"
                  value={scenario.num_medications}
                  min={1} max={50}
                  onChange={set('num_medications')}
                  hint="Current meds"
                />

                <SliderField
                  id="sim-labs"
                  label="Lab Procedures"
                  value={scenario.num_lab_procedures}
                  min={1} max={100}
                  onChange={set('num_lab_procedures')}
                />

                <SliderField
                  id="sim-diag"
                  label="Number of Diagnoses"
                  value={scenario.number_diagnoses}
                  min={1} max={9}
                  onChange={set('number_diagnoses')}
                />

                <SliderField
                  id="sim-inpatient"
                  label="Prior Inpatient Visits (past year)"
                  value={scenario.number_inpatient}
                  min={0} max={10}
                  onChange={set('number_inpatient')}
                  hint="High-risk indicator"
                />

                <SelectField
                  id="sim-insulin"
                  label="Insulin"
                  value={scenario.insulin}
                  options={['No', 'Steady', 'Up', 'Down']}
                  onChange={set('insulin')}
                />

                <SelectField
                  id="sim-a1c"
                  label="HbA1c Result"
                  value={scenario.a1c_result}
                  options={['None', '>7', '>8', 'Norm']}
                  onChange={set('a1c_result')}
                />

                <SelectField
                  id="sim-glu"
                  label="Max Serum Glucose"
                  value={scenario.max_glu_serum}
                  options={['None', '>200', '>300', 'Norm']}
                  onChange={set('max_glu_serum')}
                />

                <SelectField
                  id="sim-change"
                  label="Medication Change"
                  value={scenario.change}
                  options={[{ value: 'No', label: 'No change' }, { value: 'Ch', label: 'Changed' }]}
                  onChange={set('change')}
                />

                {error && (
                  <div className="alert alert-error">
                    <AlertTriangle className="alert-icon" size={16} />
                    {error}
                  </div>
                )}

                <button
                  id="btn-run-simulation"
                  className="btn btn-primary"
                  onClick={runScore}
                  disabled={loading}
                  style={{ width: '100%' }}
                >
                  {loading ? 'Scoring...' : 'Run Simulation'}
                </button>
              </div>
            </div>
          </div>

          {/* Results Panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {result ? (
              <>
                {/* Probability Comparison */}
                <div className="card">
                  <div className="card-header">
                    <div className="card-title">Simulation Result</div>
                  </div>
                  <div className="card-body">
                    <div style={{ display: 'flex', gap: 24, alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap' }}>
                      {result.baseline_prob != null && (
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: 11, color: 'var(--color-text-muted)', textTransform: 'uppercase', marginBottom: 8, fontWeight: 600 }}>
                            Baseline
                          </div>
                          <div className={`risk-prob-circle ${result.baseline_tier || 'low'}`}>
                            <div className="risk-prob-value" style={{ color: tierColor(result.baseline_tier) }}>
                              {formatPct(result.baseline_prob)}
                            </div>
                            <div className="risk-prob-label">probability</div>
                          </div>
                          <span className={`risk-badge ${result.baseline_tier || 'low'}`} style={{ marginTop: 8, display: 'inline-flex' }}>
                            {tierLabel(result.baseline_tier)}
                          </span>
                        </div>
                      )}

                      {delta != null && (
                        <div style={{ textAlign: 'center', color: delta > 0 ? 'var(--color-risk-high)' : 'var(--color-risk-low)' }}>
                          {delta > 0 ? <ArrowUp size={24} /> : <ArrowDown size={24} />}
                          <div style={{ fontSize: 18, fontWeight: 700 }}>
                            {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}pp
                          </div>
                          <div style={{ fontSize: 11 }}>change</div>
                        </div>
                      )}

                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 11, color: 'var(--color-text-muted)', textTransform: 'uppercase', marginBottom: 8, fontWeight: 600 }}>
                          Scenario
                        </div>
                        <div className={`risk-prob-circle ${tier}`}>
                          <div className="risk-prob-value" style={{ color: tierColor(tier) }}>
                            {formatPct(result.scenario_prob)}
                          </div>
                          <div className="risk-prob-label">probability</div>
                        </div>
                        <span className={`risk-badge ${tier}`} style={{ marginTop: 8, display: 'inline-flex' }}>
                          {tierLabel(tier)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Key Drivers */}
                {result.drivers?.length > 0 && (
                  <div className="card">
                    <div className="card-header">
                      <div className="card-title">Top Risk Factors</div>
                      <div style={{ fontSize: 11, color: 'var(--color-text-muted)' }}>
                        Associations, not causal
                      </div>
                    </div>
                    <div className="card-body">
                      <div className="factor-list">
                        {result.drivers.slice(0, 6).map((d, i) => {
                          const dir = d.direction || (d.shap_value > 0 ? 'increases' : 'decreases')
                          const maxVal = Math.max(...result.drivers.map(x => Math.abs(x.shap_value || x.importance || 0)))
                          const barPct = maxVal > 0
                            ? Math.abs(d.shap_value || d.importance || 0) / maxVal * 100
                            : 0
                          return (
                            <div key={i} className="factor-item">
                              <div className={`factor-direction ${dir === 'increases' ? 'up' : 'down'}`}>
                                {dir === 'increases' ? '↑' : '↓'}
                              </div>
                              <div className="factor-info">
                                <div className="factor-label">{d.label}</div>
                                <div className="factor-detail">
                                  {d.modifiable ? '✏️ Modifiable' : '📋 Historical'} · {d.association_note}
                                </div>
                              </div>
                              <div className="factor-bar-wrapper">
                                <div className="factor-bar">
                                  <div
                                    className={`factor-bar-fill ${dir === 'increases' ? 'up' : 'down'}`}
                                    style={{ width: `${barPct}%` }}
                                  />
                                </div>
                                <span className="factor-value">
                                  {(d.shap_value != null)
                                    ? (d.shap_value > 0 ? '+' : '') + d.shap_value.toFixed(3)
                                    : d.importance?.toFixed(3)}
                                </span>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  </div>
                )}

                {/* Recommended Actions */}
                {result.recommended_actions?.length > 0 && (
                  <div className="card">
                    <div className="card-header">
                      <div className="card-title">Discharge Planning Checklist</div>
                      <div style={{ fontSize: 11, color: 'var(--color-text-muted)' }}>
                        Rule-based — not a clinical prescription
                      </div>
                    </div>
                    <div className="card-body">
                      <div className="checklist">
                        {result.recommended_actions.map((action, i) => (
                          <div key={i} className="checklist-item">
                            <div className="checklist-content">
                              <div className="checklist-label">
                                {action.label}
                                <span className={`priority-badge priority-${action.priority}`}>
                                  {action.priority}
                                </span>
                              </div>
                              <div className="checklist-rationale">{action.rationale}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Research Notice */}
                <div className="alert alert-info">
                  <Info className="alert-icon" size={16} />
                  <div style={{ fontSize: 12 }}>{result.research_demo_notice}</div>
                </div>
              </>
            ) : (
              <div className="card">
                <div className="empty-state">
                  <Calculator className="empty-state-icon" />
                  <div className="empty-state-title">Configure & Run</div>
                  <div className="empty-state-text">
                    Adjust the parameters on the left and click <strong>Run Simulation</strong> to see the model's
                    probability estimate for this scenario.
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
