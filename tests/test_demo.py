"""Smoke test of the Streamlit demo: every mode runs without an exception."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "demo" / "app.py"


@pytest.mark.parametrize("mode", ["Automatic (blind)", "Semi-blind (manual PSF)", "Simulate blur on a sharp image"])
def test_demo_modes_run(mode):
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.radio[0].set_value(mode).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Directional contrast" in labels
    if mode.startswith("Simulate"):
        assert "PSNR vs. sharp original" in labels
