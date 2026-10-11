import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    _icd_to_category,
    cap_outliers,
    engineer_features,
    prepare_features,
    build_preprocessor,
    transform_single,
    apply_initial_filters,
    build_target,
    TERMINAL_DISCHARGE_IDS,
    ALL_MODEL_FEATURES,
    NUMERICAL_FEATURES,
)


def test_icd_category_mapping():
    assert _icd_to_category("250.01") == "Endocrine/Metabolic"
    assert _icd_to_category("410.01") == "Circulatory"
    assert _icd_to_category("428.0") == "Circulatory"
    assert _icd_to_category("496") == "Respiratory"
    assert _icd_to_category("585.9") == "Genitourinary"
    assert _icd_to_category("V45") == "External/Supplemental"
    assert _icd_to_category("E878") == "External/Supplemental"
    assert _icd_to_category("?") == "Unknown"
    assert _icd_to_category(None) == "Unknown"


def test_terminal_discharge_exclusion():
    df = pd.DataFrame({
        "encounter_id": [1, 2, 3, 4],
        "patient_nbr": [101, 102, 103, 104],
        "discharge_disposition_id": [1, 11, 2, 13],  # 11 and 13 are hospice/death
    })
    filtered = apply_initial_filters(df)
    assert len(filtered) == 2
    assert set(filtered["discharge_disposition_id"]) == {1, 2}


def test_target_construction():
    df = pd.DataFrame({"readmitted": ["<30", ">30", "NO", "<30"]})
    y = build_target(df)
    assert list(y) == [1, 0, 0, 1]


def test_outlier_capping():
    df = pd.DataFrame({
        "age_mid": [55, 65, 75],
        "time_in_hospital": [1, 2, 50],  # 50 is an extreme outlier
        "num_lab_procedures": [10, 20, 30],
        "num_procedures": [0, 1, 2],
        "num_medications": [5, 10, 15],
        "number_diagnoses": [3, 5, 8],
        "num_active_meds": [1, 2, 3],
        "prior_visits": [0, 1, 2],
    })
    capped, caps = cap_outliers(df)
    assert "time_in_hospital" in caps
    upper = caps["time_in_hospital"][1]
    assert capped["time_in_hospital"].max() <= upper


def test_transform_single_encounter():
    train_df = pd.DataFrame({
        "encounter_id": [1, 2, 3],
        "patient_nbr": [10, 11, 12],
        "age": ["[50-60)", "[60-70)", "[70-80)"],
        "gender": ["Male", "Female", "Male"],
        "race": ["Caucasian", "AfricanAmerican", "Other"],
        "admission_type_id": [1, 2, 1],
        "discharge_disposition_id": [1, 1, 3],
        "admission_source_id": [7, 7, 1],
        "time_in_hospital": [3, 4, 2],
        "num_lab_procedures": [35, 45, 50],
        "num_procedures": [0, 1, 2],
        "num_medications": [10, 15, 20],
        "number_outpatient": [0, 1, 0],
        "number_emergency": [0, 0, 1],
        "number_inpatient": [0, 1, 2],
        "number_diagnoses": [4, 6, 8],
        "diag_1": ["250.01", "410", "428"],
        "diag_2": ["401.9", "250", "?"],
        "diag_3": ["?", "272", "401"],
        "max_glu_serum": ["None", ">200", "Norm"],
        "A1Cresult": ["None", ">7", "Norm"],
        "change": ["No", "Ch", "No"],
        "diabetesMed": ["Yes", "Yes", "No"],
    })
    X_train, caps = prepare_features(train_df)
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # Test transforming a single dict
    single_enc = {
        "age": "[60-70)",
        "gender": "Female",
        "race": "Caucasian",
        "time_in_hospital": 4,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 12,
        "number_diagnoses": 6,
        "diag_1": "250.01",
        "diag_2": "401.9",
        "diag_3": "?",
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "change": "No",
        "diabetesMed": "Yes",
    }
    arr = transform_single(single_enc, preprocessor, caps)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 1
    assert arr.shape[1] > 0
