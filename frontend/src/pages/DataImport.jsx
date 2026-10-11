import React, { useState } from 'react'
import {
  UploadCloud,
  FileCheck2,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Database,
  ArrowRight,
  RefreshCw,
  FileText
} from 'lucide-react'
import { api } from '../utils/api.js'

export default function DataImport() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [validating, setValidating] = useState(false)
  const [validationResult, setValidationResult] = useState(null)
  const [validationError, setValidationError] = useState(null)

  const [importing, setImporting] = useState(false)
  const [importSuccess, setImportSuccess] = useState(null)
  const [importError, setImportError] = useState(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    if (!file.name.endsWith('.csv')) {
      setValidationError('Only CSV (.csv) files are supported for clinical dataset ingestion.')
      return
    }
    setSelectedFile(file)
    setValidationResult(null)
    setValidationError(null)
    setImportSuccess(null)
    setImportError(null)
  }

  const handleValidate = async () => {
    if (!selectedFile) return
    setValidating(true)
    setValidationError(null)
    setImportSuccess(null)
    try {
      const res = await api.validateDataset(selectedFile)
      setValidationResult(res)
    } catch (err) {
      console.error('Validation failed:', err)
      setValidationError(err.message || 'File validation pipeline failed.')
    } finally {
      setValidating(false)
    }
  }

  const handleConfirmImport = async () => {
    if (!validationResult || !validationResult.preview_rows) return
    setImporting(true)
    setImportError(null)
    try {
      const res = await api.importDataset({
        filename: validationResult.filename,
        records: validationResult.preview_rows,
        acting_user: 'Clinical Data Specialist',
      })
      setImportSuccess(res)
    } catch (err) {
      console.error('Import failed:', err)
      setImportError(err.message || 'Database import failed.')
    } finally {
      setImporting(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setValidationResult(null)
    setValidationError(null)
    setImportSuccess(null)
    setImportError(null)
  }

  return (
    <div className="page-body">
      {/* Header */}
      <div className="section-header" style={{ marginBottom: 24 }}>
        <div>
          <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Database className="text-primary" size={26} />
            <span>Dataset Ingestion & Validation Pipeline</span>
          </h1>
          <p className="page-subtitle">
            Upload inpatient diabetic CSV encounter cohorts with automated clinical schema validation and audit governance.
          </p>
        </div>
      </div>

      {/* Governance & Privacy Notice */}
      <div className="alert alert-info" style={{ marginBottom: 24 }}>
        <FileCheck2 className="alert-icon" />
        <div>
          <strong>De-Identification Mandate:</strong> Only upload de-identified or synthetic demonstration cohorts. 
          The ingestion pipeline checks required fields, identifies statistical outliers, audits missingness, 
          and requires explicit administrative confirmation before writing records to PostgreSQL.
        </div>
      </div>

      {/* Step 1: Upload Card */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <div className="card-title">
            <UploadCloud size={18} />
            <span>Step 1: Select CSV Encounter File</span>
          </div>
        </div>
        <div className="card-body">
          <div
            style={{
              border: '2px dashed var(--color-border)',
              borderRadius: 'var(--radius-lg)',
              padding: '36px 20px',
              textAlign: 'center',
              backgroundColor: 'var(--color-bg-elevated)',
              cursor: 'pointer',
            }}
            onClick={() => document.getElementById('csv-file-input').click()}
          >
            <input
              id="csv-file-input"
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
            <FileText size={44} style={{ color: 'var(--color-accent)', margin: '0 auto 12px' }} />
            <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: 4 }}>
              {selectedFile ? selectedFile.name : 'Click to select CSV cohort file'}
            </div>
            <div style={{ fontSize: 13, color: 'var(--color-text-muted)' }}>
              {selectedFile
                ? `${(selectedFile.size / 1024).toFixed(1)} KB — Ready to inspect`
                : 'Supports UCI Diabetes format (.csv) with encounter fields'}
            </div>
          </div>

          {selectedFile && (
            <div style={{ display: 'flex', gap: 12, marginTop: 16, justifyContent: 'flex-end' }}>
              <button
                className="btn btn-secondary btn-sm"
                onClick={handleReset}
                disabled={validating || importing}
              >
                Clear
              </button>
              <button
                className="btn btn-primary btn-sm"
                onClick={handleValidate}
                disabled={validating || !!validationResult}
              >
                {validating ? (
                  <>
                    <RefreshCw size={14} className="animate-spin" />
                    <span>Validating Schema...</span>
                  </>
                ) : (
                  <>
                    <span>Run Quality Audit</span>
                    <ArrowRight size={14} />
                  </>
                )}
              </button>
            </div>
          )}

          {validationError && (
            <div className="alert alert-error" style={{ marginTop: 16 }}>
              <XCircle className="alert-icon" />
              <div>{validationError}</div>
            </div>
          )}
        </div>
      </div>

      {/* Step 2: Quality Audit Report */}
      {validationResult && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-header">
            <div className="card-title">
              <FileCheck2 size={18} />
              <span>Step 2: Data Quality & Preprocessing Report</span>
            </div>
            <span
              className={`risk-badge ${validationResult.is_valid ? 'low' : 'high'}`}
            >
              {validationResult.is_valid ? 'Validation Passed' : 'Validation Failed'}
            </span>
          </div>

          <div className="card-body">
            {/* Stat row */}
            <div className="grid-3" style={{ marginBottom: 20 }}>
              <div
                style={{
                  padding: 16,
                  borderRadius: 10,
                  backgroundColor: 'var(--color-bg-elevated)',
                  border: '1px solid var(--color-border)',
                }}
              >
                <div className="stat-label">Total Records Found</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--color-text-primary)' }}>
                  {validationResult.total_rows}
                </div>
              </div>

              <div
                style={{
                  padding: 16,
                  borderRadius: 10,
                  backgroundColor: 'var(--color-bg-elevated)',
                  border: '1px solid var(--color-border)',
                }}
              >
                <div className="stat-label">Valid Clinical Rows</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--color-success)' }}>
                  {validationResult.valid_rows}
                </div>
              </div>

              <div
                style={{
                  padding: 16,
                  borderRadius: 10,
                  backgroundColor: 'var(--color-bg-elevated)',
                  border: '1px solid var(--color-border)',
                }}
              >
                <div className="stat-label">Columns Detected</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--color-accent)' }}>
                  {validationResult.columns_detected?.length ?? 0}
                </div>
              </div>
            </div>

            {/* Validation Details */}
            {validationResult.issues?.length > 0 ? (
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: 'var(--color-text-secondary)' }}>
                  Audit Findings & Discrepancies:
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {validationResult.issues.map((issue, idx) => (
                    <div
                      key={idx}
                      style={{
                        padding: '10px 14px',
                        borderRadius: 8,
                        backgroundColor: '#fffbeb',
                        border: '1px solid #fde68a',
                        color: '#92400e',
                        fontSize: 13,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                      }}
                    >
                      <AlertTriangle size={16} />
                      <span>{issue}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="alert alert-success" style={{ marginBottom: 20 }}>
                <CheckCircle2 className="alert-icon" />
                <div>All required clinical schema columns are present with 0 critical formatting errors.</div>
              </div>
            )}

            {/* Imputation Strategy Details */}
            <div
              style={{
                backgroundColor: 'var(--color-bg-elevated)',
                padding: 16,
                borderRadius: 8,
                border: '1px solid var(--color-border)',
                marginBottom: 20,
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: 4 }}>
                Automated Preprocessing & Imputation Policy:
              </div>
              <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
                Categoricals with missing data ('?') are mapped to 'Unknown' categories to preserve clinical signal.
                Continuous features (length of stay, laboratory procedures) are winsorized using train-only IQR bounds.
                Admission source IDs corresponding to hospice or expired discharges are flagged and excluded.
              </div>
            </div>

            {/* Preview Table */}
            {validationResult.preview_rows?.length > 0 && (
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: 'var(--color-text-secondary)' }}>
                  Sample Row Preview (First 5 Encounters):
                </div>
                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        {Object.keys(validationResult.preview_rows[0]).slice(0, 8).map((k) => (
                          <th key={k}>{k}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {validationResult.preview_rows.map((row, i) => (
                        <tr key={i}>
                          {Object.keys(validationResult.preview_rows[0]).slice(0, 8).map((k) => (
                            <td key={k}>{String(row[k])}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Step 3: Explicit Confirmation Action */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                paddingTop: 16,
                borderTop: '1px solid var(--color-border)',
              }}
            >
              <div style={{ fontSize: 13, color: 'var(--color-text-muted)' }}>
                {validationResult.is_valid
                  ? 'File is clinically valid. Confirm below to import into database.'
                  : 'Resolve missing mandatory columns before proceeding.'}
              </div>

              <button
                className="btn btn-primary"
                onClick={handleConfirmImport}
                disabled={!validationResult.is_valid || importing || !!importSuccess}
              >
                {importing ? (
                  <>
                    <RefreshCw size={16} className="animate-spin" />
                    <span>Importing & Live Scoring...</span>
                  </>
                ) : (
                  <>
                    <Database size={16} />
                    <span>Confirm & Ingest Encounters</span>
                  </>
                )}
              </button>
            </div>

            {importError && (
              <div className="alert alert-error" style={{ marginTop: 16 }}>
                <XCircle className="alert-icon" />
                <div>{importError}</div>
              </div>
            )}

            {importSuccess && (
              <div className="alert alert-success" style={{ marginTop: 16 }}>
                <CheckCircle2 className="alert-icon" />
                <div>
                  <strong>Import Completed!</strong> {importSuccess.message}
                  <div style={{ fontSize: 12, marginTop: 4 }}>
                    Batch reference ID: <code>{importSuccess.import_id}</code>. Records are now live in the Discharge Worklist and Command Center.
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
