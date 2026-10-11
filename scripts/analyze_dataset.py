"""
analyze_dataset.py
Detailed clinical and statistical analysis of the UCI Diabetes 130-US Hospitals dataset (ID 296).
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/diabetic_data.csv")

def run_analysis():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    total_encounters = len(df)
    unique_patients = df["patient_nbr"].nunique()
    
    # Target distribution
    target_counts = df["readmitted"].value_counts().to_dict()
    target_pct = (df["readmitted"].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # Terminal discharges (hospice / death)
    terminal_ids = {11, 13, 14, 19, 20, 21}
    terminal_count = int(df["discharge_disposition_id"].isin(terminal_ids).sum())
    
    # Missing values
    missing_report = {}
    for col in df.columns:
        n_q = int((df[col] == "?").sum())
        n_na = int(df[col].isna().sum())
        total_missing = n_q + n_na
        if total_missing > 0:
            missing_report[col] = {
                "missing_count": total_missing,
                "missing_pct": round(total_missing / total_encounters * 100, 2)
            }
            
    # Demographics
    age_dist = df["age"].value_counts().sort_index().to_dict()
    race_dist = df["race"].replace("?", "Missing/Unknown").value_counts().to_dict()
    gender_dist = df["gender"].value_counts().to_dict()
    
    # Clinical continuous features
    num_stats = {}
    for col in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", 
                "number_diagnoses", "number_inpatient", "number_emergency", "number_outpatient"]:
        num_stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "median": round(float(df[col].median()), 2),
            "std": round(float(df[col].std()), 2),
            "min": int(df[col].min()),
            "max": int(df[col].max()),
            "p25": round(float(df[col].quantile(0.25)), 2),
            "p75": round(float(df[col].quantile(0.75)), 2),
        }
        
    # Glycemic indicators
    a1c_dist = df["A1Cresult"].value_counts().to_dict()
    glu_dist = df["max_glu_serum"].value_counts().to_dict()
    change_dist = df["change"].value_counts().to_dict()
    diab_med_dist = df["diabetesMed"].value_counts().to_dict()
    insulin_dist = df["insulin"].value_counts().to_dict()
    
    # Readmission rate by prior inpatient
    df["has_prior_inpatient"] = df["number_inpatient"] > 0
    df["target_30"] = (df["readmitted"] == "<30").astype(int)
    prior_inpatient_rates = df.groupby("has_prior_inpatient")["target_30"].agg(["count", "mean"]).to_dict("index")
    
    # Readmission rate by A1Cresult
    a1c_rates = df.groupby("A1Cresult")["target_30"].agg(["count", "mean"]).to_dict("index")
    
    # Readmission rate by change
    change_rates = df.groupby("change")["target_30"].agg(["count", "mean"]).to_dict("index")
    
    # Readmission rate by age
    age_rates = df.groupby("age")["target_30"].agg(["count", "mean"]).to_dict("index")
    
    # Readmission rate by race
    race_rates = df.groupby(df["race"].replace("?", "Missing/Unknown"))["target_30"].agg(["count", "mean"]).to_dict("index")
    
    output = {
        "total_encounters": total_encounters,
        "unique_patients": unique_patients,
        "encounters_per_patient": round(total_encounters / unique_patients, 2),
        "target_counts": target_counts,
        "target_pct": target_pct,
        "terminal_discharges": terminal_count,
        "missing_report": missing_report,
        "age_dist": age_dist,
        "race_dist": race_dist,
        "gender_dist": gender_dist,
        "num_stats": num_stats,
        "a1c_dist": a1c_dist,
        "glu_dist": glu_dist,
        "change_dist": change_dist,
        "diab_med_dist": diab_med_dist,
        "insulin_dist": insulin_dist,
        "prior_inpatient_rates": prior_inpatient_rates,
        "a1c_rates": a1c_rates,
        "change_rates": change_rates,
        "age_rates": age_rates,
        "race_rates": race_rates,
    }
    
    with open("data/dataset_analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Dataset analysis complete -> data/dataset_analysis.json")

if __name__ == "__main__":
    run_analysis()
