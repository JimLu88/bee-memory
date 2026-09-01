from __future__ import annotations

import json

import run_sleep_cycle as runner


def _latest_receipt(path):
    files = sorted(path.glob("sleep-cycle-*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_check_is_read_only_and_writes_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_wait_for_nas", lambda: (True, 1))

    assert runner.main(["--check", "--receipt-dir", str(tmp_path)]) == 0

    receipt = _latest_receipt(tmp_path)
    assert receipt["mode"] == "health_check"
    assert receipt["status"] == "healthy"
    assert receipt["do_forget"] is False


def test_nas_failure_never_starts_local_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_wait_for_nas", lambda: (False, 4))

    assert runner.main(["--receipt-dir", str(tmp_path)]) == 1

    receipt = _latest_receipt(tmp_path)
    assert receipt["status"] == "nas_unavailable"
    assert receipt["health_attempts"] == 4


def test_remote_call_is_not_retried(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "_wait_for_nas", lambda: (True, 2))

    def fail_once(*args, **kwargs):
        calls.append((args, kwargs))
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr(runner.urllib.request, "urlopen", fail_once)

    assert runner.main(["--receipt-dir", str(tmp_path)]) == 1
    assert len(calls) == 1
    assert _latest_receipt(tmp_path)["status"] == "request_failed"


def test_forget_mode_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "DO_FORGET", "1")

    assert runner.main(["--receipt-dir", str(tmp_path)]) == 1

    receipt = _latest_receipt(tmp_path)
    assert receipt["status"] == "invalid_configuration"
    assert receipt["error_type"] == "do_forget_not_authorized"
