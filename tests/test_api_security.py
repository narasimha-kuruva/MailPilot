"""Who may call the API: an API key when one is configured, otherwise this machine only."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mailpilot.config import Settings
from mailpilot.main import create_app
from tests.conftest import local_client

KEY = "k" * 32
REMOTE = ("203.0.113.7", 41000)  # a documentation address: not this machine

PROTECTED = ["/api/v1/metrics", "/api/v1/agent/c1/audit"]


@pytest.mark.parametrize("path", PROTECTED)
def test_without_a_key_this_machine_is_let_in(path: str) -> None:
    for host in ("127.0.0.1", "::1"):
        assert local_client(create_app(), client=(host, 50000)).get(path).status_code == 200


@pytest.mark.parametrize("path", PROTECTED)
def test_without_a_key_other_machines_are_refused(path: str) -> None:
    response = local_client(create_app(), client=REMOTE).get(path)

    assert response.status_code == 403
    assert "MAILPILOT_API_KEY" in response.json()["detail"]


@pytest.mark.parametrize("path", PROTECTED)
def test_with_a_key_every_caller_must_present_it(path: str) -> None:
    for client in (REMOTE, ("127.0.0.1", 50000)):  # this machine too: the key is the rule now
        api = local_client(create_app(), api_key=KEY, client=client)

        assert api.get(path).status_code == 401
        assert api.get(path, headers={"X-API-Key": "w" * 32}).status_code == 401
        assert api.get(path, headers={"X-API-Key": KEY}).status_code == 200
        assert api.get(path, headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_a_refused_request_is_refused_before_its_body_is_validated() -> None:
    api = local_client(create_app(), api_key=KEY, client=REMOTE)

    response = api.post("/api/v1/agent/run", json={"instruction": ""})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("path", ["/api/v1/health", "/docs", "/openapi.json"])
def test_health_and_docs_stay_open(path: str) -> None:
    assert local_client(create_app(), api_key=KEY, client=REMOTE).get(path).status_code == 200


def test_the_docs_offer_an_authorize_button() -> None:
    schemes = local_client(create_app()).get("/openapi.json").json()["components"]["securitySchemes"]

    assert {scheme["type"] for scheme in schemes.values()} == {"apiKey", "http"}


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_key_means_no_key(value: str) -> None:
    assert Settings(_env_file=None, mailpilot_api_key=value).mailpilot_api_key is None


def test_a_short_key_is_refused_at_startup() -> None:
    with pytest.raises(ValidationError, match="at least 16 characters"):
        Settings(_env_file=None, mailpilot_api_key="hunter2")


def test_the_key_never_shows_in_the_settings_repr() -> None:
    assert KEY not in repr(Settings(_env_file=None, mailpilot_api_key=KEY))
