"""The web UI: served with a strict CSP, self-contained, and never inserting text as HTML."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from mailpilot.main import UI_DIR

UI_FILES = {"index.html": "text/html", "app.js": "javascript", "app.css": "text/css"}


def test_the_root_redirects_to_the_ui(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


@pytest.mark.parametrize(("path", "content_type"), [("/ui/", "text/html"), *[(f"/ui/{f}", t) for f, t in UI_FILES.items()]])
def test_ui_files_are_served_with_a_strict_policy(client: TestClient, path: str, content_type: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    policy = response.headers["content-security-policy"]
    assert "script-src 'self'" in policy and "frame-ancestors 'none'" in policy
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert response.headers["x-content-type-options"] == "nosniff"


def test_the_api_is_not_given_the_ui_policy(client: TestClient) -> None:
    assert "content-security-policy" not in client.get("/api/v1/health").headers


def test_the_ui_needs_no_access_check_of_its_own(client: TestClient) -> None:
    """Static files reveal nothing; every request they make goes through the API's checks."""
    from mailpilot.main import create_app
    from tests.conftest import local_client

    remote = local_client(create_app(), api_key="k" * 32, client=("203.0.113.7", 41000))

    assert remote.get("/ui/").status_code == 200
    assert remote.get("/api/v1/metrics").status_code == 401


def test_the_script_never_inserts_markup() -> None:
    """Text shown in the UI can come from emails written by strangers."""
    script = (UI_DIR / "app.js").read_text(encoding="utf-8")

    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in script, sink


def test_the_ui_loads_nothing_from_other_sites_and_has_no_inline_script() -> None:
    page = (UI_DIR / "index.html").read_text(encoding="utf-8")

    # No inline code: the CSP would block it.
    assert re.findall(r"<script[^>]*>([^<]+)</script>", page) == []
    assert re.findall(r"<[^>]+\son[a-z]+\s*=", page, re.IGNORECASE) == []  # no onclick=... handlers
    for name in UI_FILES:
        assert not re.search(r"https?://", (UI_DIR / name).read_text(encoding="utf-8")), name


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_the_script_parses() -> None:
    result = subprocess.run(["node", "--check", str(UI_DIR / "app.js")], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
