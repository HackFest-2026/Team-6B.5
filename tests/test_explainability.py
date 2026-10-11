import pytest
from src.explainability import (
    _readable_label,
    _is_modifiable,
    build_recommended_actions,
)


def test_readable_label():
    assert _readable_label("num_medications") == "Number of medications"
    assert _readable_label("prior_visits") == "Total prior visits (outpatient + ED + inpatient)"
    assert _readable_label("time_in_hospital") == "Length of stay (days)"


def test_is_modifiable():
    assert _is_modifiable("num_medications") is True
    assert _is_modifiable("any_med_change") is True
    assert _is_modifiable("a1c_tested") is True
    assert _is_modifiable("time_in_hospital") is False
    assert _is_modifiable("age_mid") is False
    assert _is_modifiable("gender") is False


def test_recommended_actions_high_risk():
    drivers = [
        {"feature": "num_medications", "modifiable": True},
        {"feature": "a1c_tested", "modifiable": True},
    ]
    actions = build_recommended_actions("high", drivers, {"discharge_disposition_id": 1})
    action_keys = [a["key"] for a in actions]

    # Must contain essential high-risk interventions
    assert "medication_reconciliation" in action_keys
    assert "telehealth_followup" in action_keys
    assert "followup_appointment" in action_keys
    assert "care_transitions_referral" in action_keys


def test_recommended_actions_low_risk():
    actions = build_recommended_actions("low", [])
    action_keys = [a["key"] for a in actions]

    assert "medication_reconciliation" in action_keys
    # Should not have intense telehealth or care transitions referrals for low risk
    assert "telehealth_followup" not in action_keys
    assert "care_transitions_referral" not in action_keys
