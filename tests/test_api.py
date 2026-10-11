import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.app.database import init_db
from backend.app import inference


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
async def setup_app():
    await init_db()
    inference.load_artifacts()


@pytest.mark.anyio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["db_connected"] is True
        assert data["model_loaded"] is True


@pytest.mark.anyio
async def test_summary_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_encounters"] >= 300
        assert data["risk_distribution"]["low"] >= 0
        assert data["care_plan_counts"]["completed"] >= 0
        assert data["threshold"] == 0.15


@pytest.mark.anyio
async def test_worklist_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Default pagination
        response = await ac.get("/api/worklist?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 10
        assert data["total"] >= 300
        assert data["page"] == 1

        # Filter by tier
        tier_resp = await ac.get("/api/worklist?tier=elevated&page_size=5")
        assert tier_resp.status_code == 200
        tier_data = tier_resp.json()
        for it in tier_data["items"]:
            assert it["risk_tier"] == "elevated"


@pytest.mark.anyio
async def test_governance_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/governance")
        assert response.status_code == 200
        data = response.json()
        assert "model_comparison" in data
        assert "fairness" in data
        assert "feature_importances" in data
        assert len(data["feature_importances"]) > 0


@pytest.mark.anyio
async def test_predict_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        scenario = {
            "age": "[60-70)",
            "gender": "Female",
            "race": "Caucasian",
            "time_in_hospital": 6,
            "num_lab_procedures": 48,
            "num_procedures": 2,
            "num_medications": 20,
            "number_diagnoses": 7,
            "diag_1": "250.01",
            "max_glu_serum": "None",
            "a1c_result": ">7",
            "change": "Ch",
            "diabetes_med": "Yes",
        }
        response = await ac.post("/api/predict", json=scenario)
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["scenario_prob"] <= 1.0
        assert data["tier"] in ("low", "elevated", "high")
        assert len(data["drivers"]) > 0
        assert len(data["recommended_actions"]) > 0


@pytest.mark.anyio
async def test_encounter_detail_and_care_plans():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Get worklist item
        wl_resp = await ac.get("/api/worklist?page=1&page_size=1")
        assert wl_resp.status_code == 200
        enc_id = wl_resp.json()["items"][0]["encounter_id"]

        # Get encounter detail
        enc_resp = await ac.get(f"/api/encounters/{enc_id}")
        assert enc_resp.status_code == 200
        enc_detail = enc_resp.json()
        assert enc_detail["encounter_id"] == enc_id
        assert "drivers" in enc_detail
        assert "recommended_actions" in enc_detail

        # Upsert care plan
        upsert_resp = await ac.put(
            f"/api/care-plans/{enc_id}",
            json={
                "status": "in_progress",
                "acting_user": "Dr. House",
                "actions": ["medication_reconciliation", "followup_appointment"],
            },
        )
        assert upsert_resp.status_code == 200
        plan = upsert_resp.json()
        assert plan["status"] == "in_progress"
        assert len(plan["actions"]) >= 2

        # Toggle action
        action_id = plan["actions"][0]["id"]
        patch_resp = await ac.patch(
            f"/api/care-plans/{enc_id}/actions/{action_id}",
            json={"completed": True},
        )
        assert patch_resp.status_code == 200
        updated_plan = patch_resp.json()
        toggled_action = next(a for a in updated_plan["actions"] if a["id"] == action_id)
        assert toggled_action["completed"] is True


@pytest.mark.anyio
async def test_dataset_validate_and_import():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Test CSV validation
        csv_content = (
            "encounter_id,age,gender,race,time_in_hospital,num_medications\n"
            "9990001,[60-70),Female,Caucasian,4,15\n"
            "9990002,[70-80),Male,AfricanAmerican,7,22\n"
        )
        files = {"file": ("test_encounters.csv", csv_content.encode("utf-8"), "text/csv")}
        val_resp = await ac.post("/api/datasets/validate", files=files)
        assert val_resp.status_code == 200
        val_data = val_resp.json()
        assert val_data["is_valid"] is True
        assert val_data["total_rows"] == 2
        assert len(val_data["preview_rows"]) == 2

        # 2. Test confirmed import
        import_payload = {
            "filename": "test_encounters.csv",
            "records": val_data["preview_rows"],
            "acting_user": "Dr. Smith",
        }
        import_resp = await ac.post("/api/datasets/import", json=import_payload)
        assert import_resp.status_code == 200
        import_data = import_resp.json()
        assert import_data["imported_count"] == 2
        assert import_data["status"] == "completed"

        # 3. Test audit logs
        audit_resp = await ac.get("/api/audit-logs")
        assert audit_resp.status_code == 200
        logs = audit_resp.json()
        assert len(logs) > 0
        assert any(l["event_type"] == "dataset_import" for l in logs)


@pytest.mark.anyio
async def test_provenance_and_model_card():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/model-card")
        assert resp.status_code == 200
        card = resp.json()
        assert card["model_name"] == "ClinicalAI Readmission Predictor"
        assert card["model_version"] == "1.0.0"
        assert "XGBoost" in card["model_type"]
        assert len(card["known_limitations"]) > 0
        assert card["threshold"] == 0.15

