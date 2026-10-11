import React, { useState, useEffect } from 'react'
import {
  ClipboardList,
  CheckCircle2,
  Clock,
  AlertCircle,
  Search,
  Filter,
  UserCheck,
  Calendar,
  ExternalLink,
  PlusCircle,
  RefreshCw
} from 'lucide-react'
import { api } from '../utils/api.js'

export default function CarePlanTracker({ onNavigateToEncounter }) {
  const [plans, setPlans] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [statusFilter, setStatusFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [updatingTaskId, setUpdatingTaskId] = useState(null)

  const fetchPlans = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.listCarePlans({
        status: statusFilter === 'all' ? undefined : statusFilter,
        limit: 50,
      })
      const items = Array.isArray(data) ? data : (data?.items ?? [])
      setPlans(items)
    } catch (err) {
      console.error('Failed to load care plans:', err)
      setError(err.message || 'Failed to connect to care-plans service.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPlans()
  }, [statusFilter])

  const handleToggleTask = async (encId, actionId, currentStatus) => {
    setUpdatingTaskId(actionId)
    try {
      await api.updateAction(encId, actionId, !currentStatus)
      // Update local state optimistically
      setPlans((prev) =>
        prev.map((plan) => {
          if (plan.encounter_id !== encId) return plan
          const updatedActions = plan.actions.map((act) =>
            act.id === actionId
              ? {
                  ...act,
                  completed: !currentStatus,
                  completed_at: !currentStatus ? new Date().toISOString() : null,
                }
              : act
          )
          const allDone = updatedActions.length > 0 && updatedActions.every((a) => a.completed)
          return {
            ...plan,
            status: allDone ? 'completed' : 'in_progress',
            actions: updatedActions,
          }
        })
      )
    } catch (err) {
      alert(`Could not update task: ${err.message}`)
    } finally {
      setUpdatingTaskId(null)
    }
  }

  const filteredPlans = plans.filter((plan) => {
    if (!search) return true
    const s = search.toLowerCase()
    return (
      plan.encounter_id?.toString().includes(s) ||
      plan.acting_user?.toLowerCase().includes(s) ||
      plan.actions?.some((a) => a.label?.toLowerCase().includes(s) || a.assigned_to?.toLowerCase().includes(s))
    )
  })

  // Summary counts
  const totalTasks = plans.reduce((acc, p) => acc + (p.actions?.length ?? 0), 0)
  const completedTasks = plans.reduce(
    (acc, p) => acc + (p.actions?.filter((a) => a.completed)?.length ?? 0),
    0
  )
  const inProgressCount = plans.filter((p) => p.status === 'in_progress').length
  const completedPlansCount = plans.filter((p) => p.status === 'completed').length

  return (
    <div className="page-body">
      {/* Header */}
      <div className="section-header" style={{ marginBottom: 24 }}>
        <div>
          <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <ClipboardList className="text-primary" size={26} />
            <span>Discharge Follow-up & Care Plan Tracker</span>
          </h1>
          <p className="page-subtitle">
            Transitional care management workflows, multidisciplinary assignments, and post-discharge accountability.
          </p>
        </div>
        <button
          onClick={fetchPlans}
          className="btn btn-secondary btn-sm"
          title="Refresh care plans"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Safety notice banner */}
      <div className="alert alert-info" style={{ marginBottom: 24 }}>
        <AlertCircle className="alert-icon" />
        <div>
          <strong>Responsible Use Protocol:</strong> Care plans and task assignments coordinate administrative
          and transitional follow-up for high-risk encounters. They reflect multidisciplinary workflow examples,
          not automated clinical orders or diagnostic determinations.
        </div>
      </div>

      {/* Metric Cards */}
      <div className="stat-grid" style={{ marginBottom: 24 }}>
        <div className="stat-card">
          <div className="stat-label">Total Active Plans</div>
          <div className="stat-value">{plans.length}</div>
          <div className="stat-value-sub">Encounter plans tracked</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">In-Progress Plans</div>
          <div className="stat-value" style={{ color: 'var(--color-warning)' }}>
            {inProgressCount}
          </div>
          <div className="stat-value-sub">Actively being coordinated</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Completed Plans</div>
          <div className="stat-value" style={{ color: 'var(--color-success)' }}>
            {completedPlansCount}
          </div>
          <div className="stat-value-sub">All checklist tasks finalized</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Task Fulfillment</div>
          <div className="stat-value">
            {totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0}%
          </div>
          <div className="stat-value-sub">{completedTasks} of {totalTasks} tasks closed</div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-body" style={{ padding: '16px 20px' }}>
          <div className="filter-bar" style={{ justifyContent: 'space-between' }}>
            <div className="filter-chips">
              <button
                className={`filter-chip ${statusFilter === 'all' ? 'active' : ''}`}
                onClick={() => setStatusFilter('all')}
              >
                All ({plans.length})
              </button>
              <button
                className={`filter-chip ${statusFilter === 'in_progress' ? 'active' : ''}`}
                onClick={() => setStatusFilter('in_progress')}
              >
                In Progress
              </button>
              <button
                className={`filter-chip ${statusFilter === 'completed' ? 'active' : ''}`}
                onClick={() => setStatusFilter('completed')}
              >
                Completed
              </button>
              <button
                className={`filter-chip ${statusFilter === 'pending' ? 'active' : ''}`}
                onClick={() => setStatusFilter('pending')}
              >
                Pending Review
              </button>
            </div>

            <div className="input-icon-wrapper" style={{ minWidth: 260 }}>
              <Search className="input-icon" />
              <input
                type="text"
                placeholder="Search by encounter or assignee..."
                className="form-input"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Main List */}
      {loading ? (
        <div className="card">
          <div className="card-body">
            <div className="skeleton skeleton-title" style={{ width: '30%' }} />
            <div className="skeleton skeleton-text" />
            <div className="skeleton skeleton-text" />
            <div className="skeleton skeleton-text" />
          </div>
        </div>
      ) : error ? (
        <div className="alert alert-error">
          <AlertCircle className="alert-icon" />
          <div>{error}</div>
        </div>
      ) : filteredPlans.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <ClipboardList className="empty-state-icon" />
            <div className="empty-state-title">No Care Plans Found</div>
            <div className="empty-state-text">
              No care plans match the current filter criteria. You can create a new care plan by opening
              any encounter in the Discharge Worklist.
            </div>
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {filteredPlans.map((plan) => {
            const actions = plan.actions ?? []
            const planCompleted = plan.status === 'completed'
            const doneCount = actions.filter((a) => a.completed).length

            return (
              <div key={plan.id} className="card">
                <div
                  className="card-header"
                  style={{
                    backgroundColor: 'var(--color-bg-elevated)',
                    borderBottom: '1px solid var(--color-border)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <span
                      style={{
                        fontSize: 14,
                        fontWeight: 700,
                        color: 'var(--color-text-primary)',
                      }}
                    >
                      Encounter #{plan.encounter_id}
                    </span>
                    <span
                      className={`risk-badge ${
                        plan.status === 'completed'
                          ? 'low'
                          : plan.status === 'in_progress'
                          ? 'elevated'
                          : 'high'
                      }`}
                    >
                      {plan.status.replace('_', ' ')}
                    </span>
                    <span style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>
                      Assigned by: <strong>{plan.acting_user || 'Care Team'}</strong>
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                    <span style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>
                      Tasks: <strong>{doneCount}/{actions.length}</strong>
                    </span>
                    {onNavigateToEncounter && (
                      <button
                        onClick={() => onNavigateToEncounter(plan.encounter_id)}
                        className="btn btn-ghost btn-sm"
                        style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
                      >
                        <span>View Encounter</span>
                        <ExternalLink size={12} />
                      </button>
                    )}
                  </div>
                </div>

                <div className="card-body">
                  {actions.length === 0 ? (
                    <div style={{ fontSize: 13, color: 'var(--color-text-muted)' }}>
                      No tasks assigned yet.
                    </div>
                  ) : (
                    <div className="checklist">
                      {actions.map((act) => (
                        <div
                          key={act.id}
                          className={`checklist-item ${act.completed ? 'completed' : ''}`}
                        >
                          <button
                            type="button"
                            className={`checklist-checkbox ${act.completed ? 'checked' : ''}`}
                            onClick={() =>
                              handleToggleTask(plan.encounter_id, act.id, act.completed)
                            }
                            disabled={updatingTaskId === act.id}
                            title={act.completed ? 'Mark as pending' : 'Mark as complete'}
                          >
                            {act.completed && <CheckCircle2 size={14} color="#ffffff" />}
                          </button>

                          <div className="checklist-content">
                            <div className="checklist-label">
                              <span>{act.label}</span>
                              {act.completed && (
                                <span
                                  style={{
                                    fontSize: 10,
                                    color: 'var(--color-success)',
                                    fontWeight: 500,
                                  }}
                                >
                                  (Completed{' '}
                                  {act.completed_at
                                    ? new Date(act.completed_at).toLocaleDateString()
                                    : 'recently'}
                                  )
                                </span>
                              )}
                            </div>

                            <div
                              style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 16,
                                marginTop: 6,
                                fontSize: 11,
                                color: 'var(--color-text-muted)',
                              }}
                            >
                              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                                <UserCheck size={12} />
                                <span>Assignee: {act.assigned_to || 'Care Coordinator'}</span>
                              </span>
                              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                                <Calendar size={12} />
                                <span>
                                  Target: Within 48-72h post-discharge
                                </span>
                              </span>
                              {act.notes && (
                                <span>Notes: {act.notes}</span>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
