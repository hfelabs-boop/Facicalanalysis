import pytest

from cogsense.report import analyze, export_csv, export_pdf, load_session
from cogsense.simulate import Phase, ScenarioSimulator

PRD_SCHEMA = {
    "timestamp_utc_ms": None, "frame_id": None, "tracking_status": None, "confidence": None,
    "head_pose": {"yaw_deg", "pitch_deg", "roll_deg"},
    "action_units": {"au04_brow_lowerer", "au07_lid_tightener", "au01_inner_brow_raiser", "au02_outer_brow_raiser",
                     "au14_dimpler", "au45_blink_state"},
    "cognitive_metrics": {"mental_effort_score", "cognitive_friction_index", "automation_surprise_flag",
                          "speech_interference_detected"},
    "fusion_context": {"active_aoi", "rula_grand_score", "rula_neck_score", "correlated_insight"},
}

SHORT = [
    Phase("CALIBRATION", 16, 0.0, {"COMMS_LOG": 1}, (2, 1, 1)),
    Phase("SWARM_ENGAGEMENT", 30, 0.55, {"TACTICAL_RADAR_WIDGET_PRIMARY": 5, "TRACK_TABLE": 1}, (4, 2, 2), friction_rate=10),
    Phase("DEGRADED_DISPLAY", 20, 0.3, {"TRACK_TABLE": 4}, (5, 3, 3), squint=0.6),
]


@pytest.fixture(scope="module")
def session():
    return list(ScenarioSimulator(SHORT, fps=30, seed=3).run())


def test_payload_matches_prd_schema(session):
    p = next(r for r in session if r["action_units"])
    for key, sub in PRD_SCHEMA.items():
        assert key in p, key
        if sub:
            assert sub <= set(p[key]), (key, sub - set(p[key]))
    assert isinstance(p["timestamp_utc_ms"], int)
    assert p["tracking_status"] in ("LOCKED", "DEGRADED")


def test_scenario_calibrates_and_flags(session):
    assert session[-1]["calibration"]["status"] == "CALIBRATED"
    kinds = {e["type"] for r in session for e in r["events"]}
    assert {"CFI_EVENT", "COMPOUND_POSTURE_RISK"} <= kinds


def test_report(session, tmp_path):
    rep = analyze(session)
    phases = {p["phase"]: p for p in rep.phases}
    assert phases["SWARM_ENGAGEMENT"]["mes_mean"] > phases["CALIBRATION"]["mes_mean"] + 20
    assert rep.top_aois[0]["aoi"] == "TACTICAL_RADAR_WIDGET_PRIMARY"
    assert len(rep.top_aois) <= 5
    assert rep.crosstab["3"]["HIGH (>=0.40)"] > 5
    assert any("neck flexion" in f for f in rep.findings)

    paths = export_csv(rep, tmp_path / "csv")
    assert {p.name for p in paths} >= {"phase_summary.csv", "top_friction_aois.csv", "posture_visual_crosstab.csv"}
    pdf = export_pdf(rep, tmp_path / "r.pdf")
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_load_session_round_trip(session, tmp_path):
    import json

    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in session[:50]) + "\n")
    assert load_session(path) == session[:50]
