import React, { useState, useEffect, useCallback } from 'react'
import { Search, Filter, ChevronLeft, ChevronRight, X, ArrowRight, AlertCircle } from 'lucide-react'
import { api } from '../utils/api.js'
import { formatPct, tierLabel, ageDisplay, statusLabel, statusColor, tierColor } from '../utils/helpers.js'
import EncounterDrawer from '../components/EncounterDrawer.jsx'

const TIERS = ['', 'low', 'elevated', 'high']
const STATUSES = ['', 'pending', 'in_progress', 'completed', 'unplanned']

export default function Worklist({ initialEncId, onNavigateToCalc }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [tier, setTier] = useState('')
  const [status, setStatus] = useState('')
  const [selectedEncId, setSelectedEncId] = useState(initialEncId || null)

  const fetchWorklist = useCallback(() => {
    setLoading(true)
    setError(null)
    api.worklist({ page, page_size: 25, tier, status, search: search.trim() || undefined })
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [page, tier, status, search])

  useEffect(() => {
    const debounce = setTimeout(fetchWorklist, 300)
    return () => clearTimeout(debounce)
  }, [fetchWorklist])

  // Reset to page 1 when filters change
  useEffect(() => { setPage(1) }, [tier, status, search])

  return (
    <>
      <header className="page-header">
        <div className="page-header-content">
          <div>
            <h1 className="page-title">Encounter Worklist</h1>
            <p className="page-subtitle">
              {data ? `${data.total.toLocaleString()} encounters` : 'Loading...'}{' '}
              · Sorted by readmission risk (highest first)
            </p>
          </div>
        </div>
      </header>

      <div className="page-body">
        {/* Filters */}
        <div className="card mb-6">
          <div className="card-body" style={{ padding: '16px 20px' }}>
            <div className="filter-bar">
              {/* Search */}
              <div className="input-icon-wrapper" style={{ flex: '1 1 200px', maxWidth: 300 }}>
                <Search className="input-icon" size={16} aria-hidden="true" />
                <input
                  id="worklist-search"
                  className="form-input"
                  placeholder="Search encounter ID..."
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  aria-label="Search encounter ID"
                />
              </div>

              {/* Risk Tier Filter */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Filter size={14} style={{ color: 'var(--color-text-muted)' }} aria-hidden="true" />
                <span style={{ fontSize: 12, color: 'var(--color-text-muted)', fontWeight: 600 }}>Risk:</span>
                <div className="filter-chips">
                  {TIERS.map(t => (
                    <button
                      key={t}
                      id={`filter-tier-${t || 'all'}`}
                      className={`filter-chip ${tier === t ? 'active' : ''}`}
                      onClick={() => setTier(t)}
                      aria-pressed={tier === t}
                    >
                      {t ? tierLabel(t) : 'All'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Status Filter */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 12, color: 'var(--color-text-muted)', fontWeight: 600 }}>Plan:</span>
                <select
                  id="filter-status"
                  className="form-select"
                  style={{ width: 'auto' }}
                  value={status}
                  onChange={e => setStatus(e.target.value)}
                  aria-label="Filter by care plan status"
                >
                  <option value="">All Statuses</option>
                  <option value="unplanned">Unplanned</option>
                  <option value="pending">Pending</option>
                  <option value="in_progress">In Progress</option>
                  <option value="completed">Completed</option>
                </select>
              </div>

              {/* Clear */}
              {(tier || status || search) && (
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => { setTier(''); setStatus(''); setSearch('') }}
                  aria-label="Clear all filters"
                >
                  <X size={14} /> Clear
                </button>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div className="alert alert-error mb-4">
            <AlertCircle className="alert-icon" />
            <div>{error}</div>
          </div>
        )}

        {/* Table */}
        <div className="card">
          {loading ? (
            <div style={{ padding: 32 }}>
              {[...Array(8)].map((_, i) => (
                <div key={i} className="skeleton skeleton-text mb-2" style={{ width: `${70 + (i % 3) * 10}%` }} />
              ))}
            </div>
          ) : data?.items?.length === 0 ? (
            <div className="empty-state">
              <Search className="empty-state-icon" />
              <div className="empty-state-title">No encounters found</div>
              <div className="empty-state-text">
                Try adjusting your filters or search term. Make sure the database has been seeded by running
                <code> python src/train.py</code>.
              </div>
            </div>
          ) : (
            <>
              <div className="table-wrapper" style={{ border: 'none' }}>
                <table aria-label="Encounter worklist">
                  <thead>
                    <tr>
                      <th>Encounter ID</th>
                      <th>Age</th>
                      <th>Gender</th>
                      <th>LOS (days)</th>
                      <th>Medications</th>
                      <th>Readmission Risk</th>
                      <th>Risk Tier</th>
                      <th>Care Plan</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {data?.items?.map(enc => (
                      <tr
                        key={enc.encounter_id}
                        onClick={() => setSelectedEncId(enc.encounter_id)}
                        aria-label={`Open encounter ${enc.encounter_id}`}
                      >
                        <td style={{ fontFamily: 'monospace', color: 'var(--color-accent-light)' }}>
                          #{enc.encounter_id}
                        </td>
                        <td>{ageDisplay(enc.age)}</td>
                        <td>{enc.gender}</td>
                        <td>{enc.time_in_hospital ?? '—'}</td>
                        <td>{enc.num_medications ?? '—'}</td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div className="progress-bar" style={{ width: 60 }}>
                              <div
                                className={`progress-fill ${enc.risk_tier}`}
                                style={{ width: `${(enc.readmission_prob * 100).toFixed(0)}%` }}
                              />
                            </div>
                            <span style={{ fontWeight: 600, color: tierColor(enc.risk_tier) }}>
                              {formatPct(enc.readmission_prob)}
                            </span>
                          </div>
                        </td>
                        <td>
                          <span className={`risk-badge ${enc.risk_tier}`}>
                            {tierLabel(enc.risk_tier)}
                          </span>
                        </td>
                        <td>
                          {enc.care_plan_status ? (
                            <span style={{
                              fontSize: 12, fontWeight: 600,
                              color: statusColor(enc.care_plan_status),
                            }}>
                              {statusLabel(enc.care_plan_status)}
                            </span>
                          ) : (
                            <span style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>Unplanned</span>
                          )}
                        </td>
                        <td>
                          <ArrowRight size={16} style={{ color: 'var(--color-text-muted)' }} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {data && data.total_pages > 1 && (
                <div className="pagination">
                  <span className="pagination-info">
                    Page {data.page} of {data.total_pages} · {data.total.toLocaleString()} total
                  </span>
                  <div className="pagination-controls">
                    <button
                      id="btn-prev-page"
                      className="btn btn-secondary btn-sm"
                      onClick={() => setPage(p => Math.max(1, p - 1))}
                      disabled={page === 1}
                      aria-label="Previous page"
                    >
                      <ChevronLeft size={14} />
                    </button>
                    <button
                      id="btn-next-page"
                      className="btn btn-secondary btn-sm"
                      onClick={() => setPage(p => Math.min(data.total_pages, p + 1))}
                      disabled={page === data.total_pages}
                      aria-label="Next page"
                    >
                      <ChevronRight size={14} />
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Encounter Drawer */}
      {selectedEncId && (
        <EncounterDrawer
          encId={selectedEncId}
          onClose={() => setSelectedEncId(null)}
          onNavigateToCalc={onNavigateToCalc}
        />
      )}
    </>
  )
}
