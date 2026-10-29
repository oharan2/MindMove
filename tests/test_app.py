"""The dashboard states that it replays prerecorded EEG."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _visible_text(app: AppTest) -> str:
    chunks = []
    for name in (
        "markdown",
        "info",
        "warning",
        "error",
        "caption",
        "title",
        "header",
        "subheader",
        "text",
    ):
        for element in getattr(app, name, []):
            value = getattr(element, "value", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks).lower()


def test_app_says_this_is_prerecorded_replay():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=120)
    app.run(timeout=120)
    assert not app.exception, app.exception
    text = _visible_text(app)
    assert "prerecorded" in text
    assert "not a live" in text
