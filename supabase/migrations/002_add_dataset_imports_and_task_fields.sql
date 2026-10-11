-- =============================================================================
-- 002_add_dataset_imports_and_task_fields.sql
-- Extend schema for configurable task tracking and dataset validation imports
-- =============================================================================

-- Add task tracking fields to care_plan_actions
ALTER TABLE care_plan_actions ADD COLUMN IF NOT EXISTS assigned_to TEXT DEFAULT 'Care Coordinator';
ALTER TABLE care_plan_actions ADD COLUMN IF NOT EXISTS due_date TEXT;
ALTER TABLE care_plan_actions ADD COLUMN IF NOT EXISTS notes TEXT;

-- Add details to audit_events
ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS details TEXT;

-- Create dataset_imports table
CREATE TABLE IF NOT EXISTS dataset_imports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    row_count       INTEGER DEFAULT 0,
    valid_count     INTEGER DEFAULT 0,
    invalid_count   INTEGER DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'validated'
                    CHECK (status IN ('validated', 'imported', 'rejected', 'failed')),
    summary_report  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dataset_imports_status ON dataset_imports(status);

-- Enable RLS for dataset_imports
ALTER TABLE dataset_imports ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS allow_all_dataset_imports ON dataset_imports;
CREATE POLICY allow_all_dataset_imports ON dataset_imports FOR ALL USING (TRUE) WITH CHECK (TRUE);
