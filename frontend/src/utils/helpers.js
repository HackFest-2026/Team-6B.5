/** helpers.js — shared formatting utilities */

export function formatPct(prob) {
  if (prob == null) return '—'
  return `${(prob * 100).toFixed(1)}%`
}

export function tierLabel(tier) {
  return { low: 'Low Risk', elevated: 'Elevated Risk', high: 'High Risk' }[tier] || tier
}

export function tierColor(tier) {
  return {
    low: 'var(--color-risk-low)',
    elevated: 'var(--color-risk-elevated)',
    high: 'var(--color-risk-high)',
  }[tier] || 'var(--color-text-muted)'
}

export function statusLabel(status) {
  return {
    pending: 'Pending',
    in_progress: 'In Progress',
    completed: 'Completed',
  }[status] || 'Unplanned'
}

export function statusColor(status) {
  return {
    pending: 'var(--color-warning)',
    in_progress: 'var(--color-accent)',
    completed: 'var(--color-success)',
  }[status] || 'var(--color-text-muted)'
}

export function ageDisplay(ageBand) {
  const map = {
    '[0-10)': '<10', '[10-20)': '10–19', '[20-30)': '20–29',
    '[30-40)': '30–39', '[40-50)': '40–49', '[50-60)': '50–59',
    '[60-70)': '60–69', '[70-80)': '70–79', '[80-90)': '80–89',
    '[90-100)': '90+',
  }
  return map[ageBand] || ageBand
}
