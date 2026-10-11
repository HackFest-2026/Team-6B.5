import React, { useState, useEffect } from 'react'
import { X, AlertTriangle, Info, CheckCircle, Save, ExternalLink } from 'lucide-react'
import { api } from '../utils/api.js'
import { formatPct, tierLabel, tierColor, ageDisplay, statusLabel, statusColor } from '../utils/helpers.js'

const ACTION_LABELS = {
  medication_reconciliation: 'Medication Reconciliation',
  followup_appointment: 'Schedule Follow-up Appointment',
  diabetes_education: 'Diabetes Self-Management Education',
  telehealth_followup: 'Telehealth Follow-up (48–72h)',
  care_transitions_referral: 'Care Transitions Referral',
  glycemic_review: 'Glycaemic Management Review',
  pharmacy_consult: 'Pharmacy Consultation',
  home_health: 'Home Health Coordination',
}

function FieldRow({ label, value }) {
  return (
    <div style={{ display: 'flex', gap: 8, padding: '6px 0', borderBottom: '1px solid var(--color-border-subtle)' }}>
      <span style={{ fontSize: 12, color: 'var(--color-text-muted)', width: 160, flexShrink: 0 }}>{label}</span>
      <span style={{ fontSize: 13, color: 'var(--color-text-secondary)' }}>{value ?? '—'}</span>
    </div>
  )
}

export default function EncounterDrawer({ encId, onClose, onNavigateToCalc }) {
  const [encounter, setEncounter] = useState(null)
  const [carePlan, setCarePlan] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState(null)
  const [activeTab, setActiveTab] = useState(0)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      api.encounter(encId),
      api.getCarePlan(encId).catch(() => null), // 404 is fine
    ])
      .then(([enc, plan]) => {
        setEncounter(enc)
        setCarePlan(plan)
        setLoading(false)
      })
      .catch(e => {
        setError(e.message)
        setLoading(false)
      })
  }, [encId])

  const initCarePlan = async () => {
    if (!encounter) return
    setSaving(true)
    try {
      const actionKeys = encounter.recommended_actions?.map(a => a.key) || []
      const plan = await api.upsertCarePlan(encId, {
        status: 'in_progress',
        actions: actionKeys,
      })
      setCarePlan(plan)
      setSaveMsg('Care plan created')
      setTimeout(() => setSaveMsg(null), 3000)
    } catch (e) {
      setSaveMsg(`Error: ${e.message}`)
    } finally {
      setSaving(false)
    }
  }

  const toggleAction = async (actionId, completed) => {
    try {
      const updated = await api.updateAction(encId, actionId, !completed)
      setCarePlan(updated)
    } catch (e) {
      console.error('Action update failed:', e)
    }
  }

  const tier = encounter?.risk_tier || 'low'

  return (
    <>
      <div className="drawer-overlay" onClick={onClose} aria-hidden="true" />
      <div
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label={`Encounter ${encId} details`}
      >
        <div className="drawer-header">
          <div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>
              Encounter #{encId}
            </div>
            {encounter && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                <span className={`risk-badge ${tier}`}>{tierLabel(tier)}</span>
                <span style={{ fontSize: 13, fontWeight: 700, color: tierColor(tier) }}>
                  {formatPct(encounter.readmission_prob)} readmission risk
                </span>
              </div>
            )}
          </div>
          <button
            id="btn-close-drawer"
            className="drawer-close"
            onClick={onClose}
            aria-label="Close encounter drawer"
          >
            <X size={16} />
          </button>
        </div>

        <div style={{ display: 'flex', borderBottom: '1px solid var(--color-border)' }}>
          {['Details', 'Risk Factors', 'Care Plan'].map((t, i) => (
            <button
              key={i}
              className={`tab ${activeTab === i ? 'active' : ''}`}
              onClick={() => setActiveTab(i)}
              aria-selected={activeTab === i}
              style={{ padding: '10px 16px', fontSize: 13 }}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="drawer-body">
          {loading && (
            <div>
              {[...Array(6)].map((_, i) => (
                <div key={i} className="skeleton skeleton-text mb-2" style={{ width: `${60 + (i % 4) * 10}%` }} />
              ))}
            </div>
          )}

          {error && (
            <div className="alert alert-error">
              <AlertTriangle className="alert-icon" />
              {error}
            </div>
          )}

          {encounter && (
            <>
              {/* Tab 0: Details */}
              {activeTab === 0 && (
                <div>
                  {/* Research notice */}
                  <div className="alert alert-warning mb-4" style={{ fontSize: 12 }}>
                    <Info className="alert-icon" size={14} />
                    {encounter.research_demo_notice}
                  </div>

                  {/* Data quality flags */}
                  {encounter.data_quality_flags?.length > 0 && (
                    <div className="alert alert-info mb-4" style={{ fontSize: 12 }}>
                      <AlertTriangle className="alert-icon" size={14} />
                      <div>
                        <strong>Data quality:</strong>
                        <ul style={{ marginTop: 4, paddingLeft: 16 }}>
                          {encounter.data_quality_flags.map((f, i) => (
                            <li key={i}>{f}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}

                  <div className="section-title mb-4" style={{ fontSize: 13 }}>Encounter Profile</div>
                  <FieldRow label="Age Band" value={ageDisplay(encounter.age)} />
                  <FieldRow label="Gender" value={encounter.gender} />
                  <FieldRow label="Race" value={encounter.race} />
                  <FieldRow label="Length of Stay" value={encounter.time_in_hospital ? `${encounter.time_in_hospital} days` : null} />
                  <FieldRow label="Lab Procedures" value={encounter.num_lab_procedures} />
                  <FieldRow label="Medications" value={encounter.num_medications} />
                  <FieldRow label="Number of Diagnoses" value={encounter.number_diagnoses} />
                  <FieldRow label="Prior Inpatient Visits" value={encounter.number_inpatient} />
                  <FieldRow label="Prior Emergency Visits" value={encounter.number_emergency} />
                  <FieldRow label="Prior Outpatient Visits" value={encounter.number_outpatient} />
                  <FieldRow label="Primary Diagnosis" value={encounter.diag_1} />
                  <FieldRow label="Secondary Diagnosis" value={encounter.diag_2} />
                  <FieldRow label="HbA1c Result" value={encounter.a1c_result} />
                  <FieldRow label="Max Serum Glucose" value={encounter.max_glu_serum} />
                  <FieldRow label="Model Version" value={encounter.model_version} />
                  <FieldRow label="Explanation Method" value={encounter.explanation_method} />

                  {onNavigateToCalc && (
                    <button
                      id="btn-open-in-simulator"
                      className="btn btn-secondary mt-4"
                      style={{ width: '100%' }}
                      onClick={() => { onClose(); onNavigateToCalc(encId) }}
                    >
                      <ExternalLink size={14} /> Open in Risk Simulator
                    </button>
                  )}
                </div>
              )}

              {/* Tab 1: Risk Factors */}
              {activeTab === 1 && (
                <div>
                  <div className="risk-probability-display">
                    <div className={`risk-prob-circle ${tier}`}>
                      <div className="risk-prob-value" style={{ color: tierColor(tier) }}>
                        {formatPct(encounter.readmission_prob)}
                      </div>
                      <div className="risk-prob-label">30-day risk</div>
                    </div>
                    <span className={`risk-badge ${tier}`}>{tierLabel(tier)}</span>
                  </div>

                  {encounter.drivers?.length > 0 ? (
                    <>
                      <div style={{ fontSize: 12, color: 'var(--color-text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
                        Top factors associated with this encounter's risk estimate.
                        All values represent model associations — not causal claims.
                      </div>
                      <div className="factor-list">
                        {encounter.drivers.map((d, i) => {
                          const dir = d.direction || (d.shap_value > 0 ? 'increases' : 'decreases')
                          const maxVal = Math.max(...encounter.drivers.map(x => Math.abs(x.shap_value || x.importance || 0)))
                          const barPct = maxVal > 0 ? Math.abs(d.shap_value || d.importance || 0) / maxVal * 100 : 0
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
                    </>
                  ) : (
                    <div className="alert alert-info mt-4">
                      <Info className="alert-icon" size={14} />
                      Factor attribution not available. Ensure model artifacts are loaded.
                    </div>
                  )}

                  {/* Recommended Actions */}
                  {encounter.recommended_actions?.length > 0 && (
                    <div style={{ marginTop: 20 }}>
                      <div className="section-title mb-3" style={{ fontSize: 13 }}>Recommended Follow-up Actions</div>
                      <div style={{ fontSize: 12, color: 'var(--color-text-muted)', marginBottom: 10 }}>
                        Rule-based suggestions — not clinical prescriptions.
                      </div>
                      {encounter.recommended_actions.map((a, i) => (
                        <div key={i} className="checklist-item mb-2">
                          <div className="checklist-content">
                            <div className="checklist-label">
                              {a.label}
                              <span className={`priority-badge priority-${a.priority}`}>{a.priority}</span>
                            </div>
                            <div className="checklist-rationale">{a.rationale}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Tab 2: Care Plan */}
              {activeTab === 2 && (
                <div>
                  {saveMsg && (
                    <div className={`alert ${saveMsg.startsWith('Error') ? 'alert-error' : 'alert-success'} mb-4`}>
                      <CheckCircle className="alert-icon" size={14} />
                      {saveMsg}
                    </div>
                  )}

                  {!carePlan ? (
                    <div style={{ textAlign: 'center', padding: '24px 0' }}>
                      <div style={{ fontSize: 14, color: 'var(--color-text-muted)', marginBottom: 16 }}>
                        No care plan exists for this encounter yet.
                      </div>
                      <button
                        id="btn-init-care-plan"
                        className="btn btn-primary"
                        onClick={initCarePlan}
                        disabled={saving}
                      >
                        <Save size={14} />
                        {saving ? 'Creating...' : 'Create Care Plan'}
                      </button>
                    </div>
                  ) : (
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                        <div>
                          <div style={{ fontSize: 13, fontWeight: 600 }}>
                            Care Plan Status:{' '}
                            <span style={{ color: statusColor(carePlan.status) }}>
                              {statusLabel(carePlan.status)}
                            </span>
                          </div>
                          {carePlan.updated_at && (
                            <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 3 }}>
                              Last updated: {new Date(carePlan.updated_at).toLocaleString()}
                            </div>
                          )}
                        </div>
                        <div style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>
                          {carePlan.actions?.filter(a => a.completed).length}/{carePlan.actions?.length} complete
                        </div>
                      </div>

                      {carePlan.actions?.length === 0 ? (
                        <div style={{ fontSize: 13, color: 'var(--color-text-muted)', marginBottom: 16 }}>
                          No actions in this care plan.
                        </div>
                      ) : (
                        <div className="checklist mb-4">
                          {carePlan.actions?.map(action => (
                            <div
                              key={action.id}
                              className={`checklist-item ${action.completed ? 'completed' : ''}`}
                              style={{ cursor: 'pointer' }}
                              onClick={() => toggleAction(action.id, action.completed)}
                              role="checkbox"
                              aria-checked={action.completed}
                              tabIndex={0}
                              onKeyDown={e => e.key === 'Enter' && toggleAction(action.id, action.completed)}
                            >
                              <div className={`checklist-checkbox ${action.completed ? 'checked' : ''}`}>
                                {action.completed && <CheckCircle size={12} color="white" />}
                              </div>
                              <div className="checklist-content">
                                <div className="checklist-label">{action.label}</div>
                                {action.completed_at && (
                                  <div className="checklist-rationale">
                                    Completed: {new Date(action.completed_at).toLocaleString()}
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                      <div className="alert alert-info" style={{ fontSize: 12 }}>
                        <Info className="alert-icon" size={14} />
                        Actions are saved to PostgreSQL. Click any item to toggle its completion status.
                        Plan status updates automatically.
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  )
}
