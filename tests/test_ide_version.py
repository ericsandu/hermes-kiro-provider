from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "provider"))

import client
import pytest


@pytest.fixture(autouse=True)
def fresh_lookup(monkeypatch):
    """Empty cache, no env override and a recorded fake metadata endpoint."""
    monkeypatch.setattr(client, "_ide_version_cache", (0.0, ""))
    monkeypatch.delenv("KIRO_IDE_VERSION", raising=False)
    calls: list[dict] = []
    reply: dict = {"currentRelease": "9.8.7"}

    def fake_request_json(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return reply

    monkeypatch.setattr(client, "request_json", fake_request_json)

    def set_reply(value: dict) -> None:
        reply.clear()
        reply.update(value)

    return SimpleNamespace(calls=calls, set_reply=set_reply)


def test_env_override_wins(monkeypatch, fresh_lookup):
    """A valid KIRO_IDE_VERSION is used without any network request."""
    monkeypatch.setenv("KIRO_IDE_VERSION", " 2.3.4 ")
    assert client._ide_version() == "2.3.4"
    assert fresh_lookup.calls == []


def test_invalid_env_is_ignored(monkeypatch):
    """A malformed KIRO_IDE_VERSION falls through to the release metadata."""
    monkeypatch.setenv("KIRO_IDE_VERSION", "latest")
    assert client._ide_version() == "9.8.7"


def test_metadata_version_is_used(fresh_lookup):
    """The currentRelease from the metadata is used, fetched with a short timeout."""
    assert client._ide_version() == "9.8.7"
    assert fresh_lookup.calls[0]["url"] == client._IDE_VERSION_URL
    assert fresh_lookup.calls[0]["timeout"] <= 5


def test_metadata_failure_falls_back_to_pinned_default(monkeypatch):
    """A failing fetch returns the pinned default and does not raise."""
    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(client, "request_json", boom)
    assert client._ide_version() == client._IDE_VERSION_DEFAULT


def test_result_is_cached(fresh_lookup):
    """A second lookup within the day makes no request."""
    client._ide_version()
    client._ide_version()
    assert len(fresh_lookup.calls) == 1


def test_failure_is_cached_briefly(monkeypatch):
    """A failed fetch is not retried for 10 minutes, then is retried."""
    now = [1_000_000.0]
    attempts = []

    def boom(*args, **kwargs):
        attempts.append(1)
        raise OSError("offline")

    monkeypatch.setattr(client.time, "time", lambda: now[0])
    monkeypatch.setattr(client, "request_json", boom)
    client._ide_version()
    now[0] += 599
    client._ide_version()
    assert len(attempts) == 1
    now[0] += 2
    client._ide_version()
    assert len(attempts) == 2


def test_headers_user_agent_carries_version():
    """Both user-agent headers contain the looked-up version."""
    headers = client._headers(SimpleNamespace(access_token="token"))
    expected = "aws-sdk-python/1.0 KiroIDE-9.8.7-hermes"
    assert headers["User-Agent"] == expected
    assert headers["x-amz-user-agent"] == expected


def test_version_pattern_rejects_malicious_values(monkeypatch, fresh_lookup):
    """Neither the env nor the metadata can inject anything but ASCII digits and dots."""
    for bad in ["1.0.0; rm", "1.0.0\r\nX: y", "１.０.０", "1.0"]:
        monkeypatch.setattr(client, "_ide_version_cache", (0.0, ""))
        monkeypatch.setenv("KIRO_IDE_VERSION", bad)
        fresh_lookup.set_reply({"currentRelease": bad})
        assert client._ide_version() == client._IDE_VERSION_DEFAULT, bad
