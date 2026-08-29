from fastapi.testclient import TestClient
from tachikoma_connection_app import app


def test_identity_does_not_claim_memory_runtime():
    payload = TestClient(app).get("/healthz").json()
    assert payload["service"] == "bee-memory"
    assert payload["dependencies_ready"] is False
    assert payload["memory_runtime_started"] is False
    assert payload["business_entrypoint_present"] is True


def test_memory_reads_and_writes_are_blocked():
    client = TestClient(app)
    assert client.get("/readyz").status_code == 403
    assert client.post("/memory", json={}).status_code == 403


def test_contract_disables_execution_and_upgrades():
    payload = TestClient(app).get("/tachikoma/v1/contract").json()
    assert payload["production_execution_enabled"] is False
    assert payload["upgrade_execution_enabled"] is False
