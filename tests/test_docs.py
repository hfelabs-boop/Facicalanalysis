"""dashboard/docs.html documents the metric maths; keep it in step with the code."""

import re
from pathlib import Path

from cogsense.config import CogSenseConfig

DOCS = (Path(__file__).resolve().parent.parent / "dashboard" / "docs.html").read_text(encoding="utf-8")


def test_internal_links_resolve():
    ids = set(re.findall(r'id="([^"]+)"', DOCS))
    assert set(re.findall(r'href="#([^"]+)"', DOCS)) <= ids


def test_documented_constants_match_config():
    c, s = CogSenseConfig(), CogSenseConfig().scales
    text = re.sub(r"\s+", " ", DOCS)
    expected = [
        f"clip( 0.6 · innerDrop / {s.brow_height:.2f} + 0.4 · gapDrop / {s.brow_gap:.2f} )",
        f"clip( outerRise / {s.outer_brow_height:.2f} )",
        f"0.5 · earDrop / {s.eye_aperture_ratio:.2f} + 0.5 · lidRise / {s.lower_lid_raise:.2f}",
        f"0.6 · Δlip_corner_depth / {s.lip_corner_depth:.2f} + 0.4 · Δmouth_width / {s.mouth_width:.2f}",
        f"{c.mes_au4_weight:.1f} · AU4 + {1 - c.mes_au4_weight:.1f} · S",
        f"AU4 &gt; {c.cfi_au4_threshold:.2f}",
        f"AU14 ≥ {c.cfi_au14_threshold:.2f}",
        f"exceeds <b>{c.cfi_marker_threshold:.2f}</b>",
        f"time constant of <b>{c.smoothing_tau_s} s</b>",
        f"<b>{c.default_blink_rate_per_min:.0f} blinks/min</b>",
        f"<b>≥ {c.microsleep_ms:.0f} ms</b>",
    ]
    for frag in expected:
        assert frag.replace("  ", " ") in text, frag


def test_fatigue_and_pupil_docs_match_config():
    c = CogSenseConfig()
    text = re.sub(r"\s+", " ", DOCS)
    for frag in [
        f"<code>perclos_window_s</code> {c.perclos_window_s:.0f}",
        f"<code>perclos_closed_openness</code> {c.perclos_closed_openness:.2f}",
        f"<code>perclos_min_coverage_s</code> {c.perclos_min_coverage_s:.0f}",
        f"<code>fatigue_max_dt_s</code> {c.fatigue_max_dt_s}",
        f"<code>pupil_smooth_s</code> {c.pupil_smooth_s}",
        f"<code>pupil_provisional_s</code> {c.pupil_provisional_s:.0f}",
        f"<code>pupil_luma_tolerance</code> {c.pupil_luma_tolerance:.2f}",
        f"ratio ≤ {c.perclos_closed_openness:.2f}",
        f"|luminance shift| ≤ {c.pupil_luma_tolerance * 100:.0f} %",
    ]:
        assert frag in text, frag
    assert c.perclos_alert is None  # the docs promise there is no default alert threshold
