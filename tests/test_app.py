"""The Streamlit app is a deliverable too, so it gets run end to end here.

``AppTest`` executes ``app/streamlit_app.py`` exactly as the server would, without a browser: if a
widget signature, a cached function or a plot call breaks, this fails instead of the user finding
out with ``make run``.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT

# AppTest resolves relative paths against the calling file, so give it an absolute one.
APP_PATH = str(PROJECT_ROOT / "app" / "streamlit_app.py")


def test_app_runs_without_exceptions() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=180).run()
    assert not app.exception
    assert len(app.tabs) == 4
    assert app.title[0].value == "Hashing 101"


def test_bloom_playground_reports_no_false_negatives() -> None:
    """The app must tell the same story the tests do: zero, always."""
    app = AppTest.from_file(APP_PATH, default_timeout=180).run()
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["False negatives"] == "0"
    assert "Bits per item" in labels
