-- =============================================================================
-- 001_initial_schema.sql
-- ClinicalAI: Hospital Readmission Risk — Initial Schema
-- =============================================================================
-- Run against your PostgreSQL / Supabase instance.
-- All tables are in the public schema. RLS is enabled for care_plans writes.

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── demo_encounters ───────────────────────────────────────────────────────────
-- De-identified encounter records (seed from training data).
-- No patient identifiers; encounter_id is a synthetic key from the dataset.
CREATE TABLE IF NOT EXISTS demo_encounters (
    id                  SERIAL PRIMARY KEY,
    encounter_id        INTEGER UNIQUE NOT NULL,
    age                 TEXT NOT NULL DEFAULT '[60-70)',
    gender              TEXT NOT NULL DEFAULT 'Unknown',
    race                TEXT NOT NULL DEFAULT 'Unknown',
    admission_type_id   INTEGER,
    discharge_disposition_id INTEGER,
    admission_source_id INTEGER,
    time_in_hospital    INTEGER,
    num_lab_procedures  INTEGER,
    num_procedures      INTEGER,
    num_medications     INTEGER,
    number_outpatient   INTEGER DEFAULT 0,
    number_emergency    INTEGER DEFAULT 0,
    number_inpatient    INTEGER DEFAULT 0,
    number_diagnoses    INTEGER,
    diag_1              TEXT,
    diag_2              TEXT,
    diag_3              TEXT,
    max_glu_serum       TEXT DEFAULT 'None',
    a1c_result          TEXT DEFAULT 'None',
    change              TEXT DEFAULT 'No',
    diabetes_med        TEXT DEFAULT 'Yes',
    -- Pre-computed model outputs
    readmission_prob    NUMERIC(5, 4),
    risk_tier           TEXT CHECK (risk_tier IN ('low', 'elevated', 'high')),
    model_version       TEXT DEFAULT '1.0.0',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_demo_encounters_tier ON demo_encounters(risk_tier);
CREATE INDEX IF NOT EXISTS idx_demo_encounters_prob ON demo_encounters(readmission_prob DESC);

-- ── care_plans ─────────────────────────────────────────────────────────────────
-- One care plan per encounter.
CREATE TABLE IF NOT EXISTS care_plans (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id    INTEGER NOT NULL REFERENCES demo_encounters(encounter_id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'in_progress', 'completed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    acting_user     TEXT,
    CONSTRAINT uq_care_plan_per_encounter UNIQUE (encounter_id)
);

CREATE INDEX IF NOT EXISTS idx_care_plans_encounter ON care_plans(encounter_id);
CREATE INDEX IF NOT EXISTS idx_care_plans_status ON care_plans(status);

-- ── care_plan_actions ──────────────────────────────────────────────────────────
-- Individual checklist items belonging to a care plan.
CREATE TABLE IF NOT EXISTS care_plan_actions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    care_plan_id    UUID NOT NULL REFERENCES care_plans(id) ON DELETE CASCADE,
    action_key      TEXT NOT NULL CHECK (action_key IN (
                        'medication_reconciliation',
                        'followup_appointment',
                        'diabetes_education',
                        'telehealth_followup',
                        'care_transitions_referral',
                        'glycemic_review',
                        'pharmacy_consult',
                        'home_health'
                    )),
    label           TEXT NOT NULL,
    completed       BOOLEAN NOT NULL DEFAULT FALSE,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_action_per_plan UNIQUE (care_plan_id, action_key)
);

CREATE INDEX IF NOT EXISTS idx_care_plan_actions_plan ON care_plan_actions(care_plan_id);

-- ── prediction_runs ─────────────────────────────────────────────────────────
-- Audit log of scoring calls (scenario simulations).
CREATE TABLE IF NOT EXISTS prediction_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_type        TEXT NOT NULL DEFAULT 'scenario'
                    CHECK (run_type IN ('baseline', 'scenario')),
    encounter_id    INTEGER REFERENCES demo_encounters(encounter_id),
    model_version   TEXT NOT NULL DEFAULT '1.0.0',
    input_hash      TEXT,                    -- SHA256 of validated scenario snapshot
    probability     NUMERIC(5, 4) NOT NULL,
    tier            TEXT NOT NULL CHECK (tier IN ('low', 'elevated', 'high')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prediction_runs_encounter ON prediction_runs(encounter_id);

-- ── audit_events ───────────────────────────────────────────────────────────
-- Lightweight application audit trail.
CREATE TABLE IF NOT EXISTS audit_events (
    id          SERIAL PRIMARY KEY,
    event_type  TEXT NOT NULL,
    entity_type TEXT,
    entity_id   TEXT,
    acting_user TEXT,
    details     JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── demo_governance ──────────────────────────────────────────────────────────
-- Pre-computed governance / model report metadata (JSON blob).
CREATE TABLE IF NOT EXISTS demo_governance (
    id          SERIAL PRIMARY KEY,
    report_key  TEXT UNIQUE NOT NULL,
    report_data JSONB NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Trigger: auto-update updated_at on care_plans ───────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_care_plans_updated_at ON care_plans;
CREATE TRIGGER trg_care_plans_updated_at
    BEFORE UPDATE ON care_plans
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ── RLS: care_plans (allow public demo reads; restrict writes) ──────────────
-- In a production Supabase setup, enable RLS and authenticated roles.
-- For the demo, allow all writes from the API (service role bypasses RLS).
ALTER TABLE care_plans ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS allow_read_care_plans ON care_plans;
CREATE POLICY allow_read_care_plans ON care_plans FOR SELECT USING (TRUE);
DROP POLICY IF EXISTS allow_write_care_plans ON care_plans;
CREATE POLICY allow_write_care_plans ON care_plans FOR ALL USING (TRUE) WITH CHECK (TRUE);

ALTER TABLE care_plan_actions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS allow_read_actions ON care_plan_actions;
CREATE POLICY allow_read_actions ON care_plan_actions FOR SELECT USING (TRUE);
DROP POLICY IF EXISTS allow_write_actions ON care_plan_actions;
CREATE POLICY allow_write_actions ON care_plan_actions FOR ALL USING (TRUE) WITH CHECK (TRUE);
