/**
 * api.js — Centralised API client for ClinicalAI
 * All calls go through the FastAPI backend. Never call the DB directly.
 */

const BASE_URL = '/api'

async function apiFetch(path, options = {}) {
  const url = `${BASE_URL}${path}`
  const isFormData = options.body instanceof FormData

  const headers = {
    ...(!isFormData && { 'Content-Type': 'application/json' }),
    ...options.headers,
  }

  const response = await fetch(url, {
    ...options,
    headers,
  })

  if (!response.ok) {
    let errorMsg = `API error ${response.status}`
    try {
      const err = await response.json()
      errorMsg = err.detail || errorMsg
    } catch (_) {}
    throw new Error(errorMsg)
  }

  return response.json()
}

export const api = {
  health: () => apiFetch('/health'),
  summary: (params = {}) => {
    const qs = new URLSearchParams()
    if (params.risk_tier) qs.set('risk_tier', params.risk_tier)
    if (params.facility) qs.set('facility', params.facility)
    return apiFetch(`/summary?${qs}`)
  },

  worklist: (params = {}) => {
    const qs = new URLSearchParams()
    if (params.page) qs.set('page', params.page)
    if (params.page_size) qs.set('page_size', params.page_size)
    if (params.tier) qs.set('tier', params.tier)
    if (params.status) qs.set('status', params.status)
    if (params.search) qs.set('search', params.search)
    return apiFetch(`/worklist?${qs}`)
  },

  encounter: (encId) => apiFetch(`/encounters/${encId}`),

  predict: (body) =>
    apiFetch('/predict', { method: 'POST', body: JSON.stringify(body) }),

  score: (body) =>
    apiFetch('/predictions/score', { method: 'POST', body: JSON.stringify(body) }),

  // Care plans
  listCarePlans: (params = {}) => {
    const qs = new URLSearchParams()
    if (params.status) qs.set('status', params.status)
    if (params.page) qs.set('page', params.page)
    if (params.page_size) qs.set('page_size', params.page_size)
    return apiFetch(`/care-plans?${qs}`)
  },

  getCarePlan: (encId) => apiFetch(`/care-plans/${encId}`),

  upsertCarePlan: (encId, body) =>
    apiFetch(`/care-plans/${encId}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    }),

  createCarePlan: (body) =>
    apiFetch('/care-plans', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  updateAction: (encId, actionId, completed, assignedTo = null, notes = null) =>
    apiFetch(`/care-plans/${encId}/actions/${actionId}`, {
      method: 'PATCH',
      body: JSON.stringify({
        completed,
        ...(assignedTo && { assigned_to: assignedTo }),
        ...(notes && { notes }),
      }),
    }),

  // Model & Governance
  governance: () => apiFetch('/governance'),
  models: () => apiFetch('/governance/models'),
  fairness: () => apiFetch('/governance/fairness'),
  modelCard: () => apiFetch('/model-card'),

  // Datasets & Audit
  validateDataset: (file) => {
    const formData = new FormData()
    formData.append('file', file)
    return apiFetch('/datasets/validate', {
      method: 'POST',
      body: formData,
    })
  },

  importDataset: (body) =>
    apiFetch('/datasets/import', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  auditLogs: (limit = 25) => apiFetch(`/audit-logs?limit=${limit}`),
}
