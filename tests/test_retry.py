import pytest

import src.data.retry as retry_module
from src.data.retry import DataProviderRetryError, call_with_backoff


def test_call_with_backoff_retries_until_success(monkeypatch):
    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("temporary")
        return "ok"

    assert call_with_backoff(flaky, label="fixture", attempts=3) == "ok"
    assert calls["count"] == 2


def test_call_with_backoff_raises_after_exhaustion(monkeypatch):
    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)

    with pytest.raises(DataProviderRetryError):
        call_with_backoff(lambda: (_ for _ in ()).throw(RuntimeError("nope")), label="fixture", attempts=2)
