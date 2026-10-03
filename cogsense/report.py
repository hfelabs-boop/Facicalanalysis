"""Automated HFE evaluation report: CSV dossier and PDF export (PRD §7)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

AU7_BINS = (("LOW (<0.20)", 0.0, 0.2), ("MODERATE (0.20-0.40)", 0.2, 0.4), ("HIGH (>=0.40)", 0.4, 1.01))
NECK_ROWS = ("1", "2", "3", "4+", "N/A")


def load_session(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _fmt_t(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass
class SessionReport:
    duration_s: float
    frames: int
    tracked_fraction: float
    phases: list[dict]
    top_aois: list[dict]
    crosstab: dict[str, dict[str, float]]  # neck row → au7 bin → seconds
    events: list[dict]
    timeseries: list[dict] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


def analyze(records: list[dict], marker_threshold: float = 0.70, top_n: int = 5) -> SessionReport:
    if not records:
        raise ValueError("session contains no telemetry")
    records = sorted(records, key=lambda r: r["timestamp_utc_ms"])
    t0 = records[0]["timestamp_utc_ms"]
    duration = (records[-1]["timestamp_utc_ms"] - t0) / 1000.0

    phases: dict[str, dict] = {}
    phase_order: list[str] = []
    aoi = defaultdict(lambda: {"cfi_seconds": 0.0, "dwell_s": 0.0, "cfi_events": 0, "markers": 0, "peak_cfi": 0.0})
    cross = {r: {b[0]: 0.0 for b in AU7_BINS} for r in NECK_ROWS}
    events: list[dict] = []
    series: list[dict] = []
    tracked = 0
    prev_ms = None
    prev_above: dict[str, bool] = {}

    for i, r in enumerate(records):
        ms = r["timestamp_utc_ms"]
        nxt = records[i + 1]["timestamp_utc_ms"] if i + 1 < len(records) else ms
        dt = max(min((nxt - ms) / 1000.0, 0.5), 0.0)  # cap gaps so dropouts don't inflate dwell
        rel = (ms - t0) / 1000.0
        fc = r.get("fusion_context") or {}
        phase = fc.get("mission_phase") or "UNSPECIFIED"
        if phase not in phases:
            phases[phase] = {"phase": phase, "start_s": rel, "end_s": rel, "frames": 0, "tracked": 0,
                             "mes_sum": 0.0, "mes_peak": 0.0, "cfi_sum": 0.0, "cfi_events": 0, "surprises": 0,
                             "compound_risks": 0, "perclos_sum": 0.0, "perclos_n": 0, "perclos_peak": 0.0,
                             "pupil_sum": 0.0, "pupil_n": 0, "pupil_peak": 0.0,
                             "lip_sum": 0.0, "lip_peak": 0.0}
            phase_order.append(phase)
        ph = phases[phase]
        ph["frames"] += 1
        ph["end_s"] = rel
        cm = r.get("cognitive_metrics")
        au = r.get("action_units")
        name = fc.get("active_aoi")
        if cm and au:
            tracked += 1
            ph["tracked"] += 1
            mes, cfi = cm["mental_effort_score"], cm["cognitive_friction_index"]
            ph["mes_sum"] += mes
            ph["mes_peak"] = max(ph["mes_peak"], mes)
            ph["cfi_sum"] += cfi
            lip = au.get("au24_lip_presser")
            if lip is not None:
                ph["lip_sum"] += lip
                ph["lip_peak"] = max(ph["lip_peak"], lip)
            fat = r.get("fatigue") or {}
            if fat.get("perclos") is not None:
                ph["perclos_sum"] += fat["perclos"]
                ph["perclos_n"] += 1
                ph["perclos_peak"] = max(ph["perclos_peak"], fat["perclos"])
            pu = fc.get("pupil") or {}
            if pu.get("change_mm") is not None and pu.get("reliable"):
                ph["pupil_sum"] += pu["change_mm"]
                ph["pupil_n"] += 1
                ph["pupil_peak"] = max(ph["pupil_peak"], pu["change_mm"])
            if name:
                a = aoi[name]
                a["cfi_seconds"] += cfi * dt
                a["dwell_s"] += dt
                a["peak_cfi"] = max(a["peak_cfi"], cfi)
                above = cfi > marker_threshold
                if above and not prev_above.get(name):
                    a["markers"] += 1
                prev_above[name] = above
            neck = fc.get("rula_neck_score")
            row = "N/A" if neck is None else ("4+" if neck >= 4 else str(max(int(neck), 1)))
            au7 = au["au07_lid_tightener"]
            for label, lo, hi in AU7_BINS:
                if lo <= au7 < hi:
                    cross[row][label] += dt
                    break
            if prev_ms is None or ms - prev_ms >= 500:
                series.append({"t_s": round(rel, 2), "mes": mes, "cfi": cfi, "au04": au["au04_brow_lowerer"],
                               "au07": au7, "rula_grand": fc.get("rula_grand_score"), "phase": phase, "aoi": name})
                prev_ms = ms
        for ev in r.get("events", []):
            e = dict(ev, t_s=round(rel, 3), phase=phase)
            events.append(e)
            kind = ev.get("type")
            if kind == "CFI_EVENT":
                ph["cfi_events"] += 1
                if ev.get("aoi"):
                    aoi[ev["aoi"]]["cfi_events"] += 1
            elif kind == "AUTOMATION_SURPRISE":
                ph["surprises"] += 1
            elif kind == "COMPOUND_POSTURE_RISK":
                ph["compound_risks"] += 1

    phase_rows = []
    for name in phase_order:
        p = phases[name]
        n = max(p["tracked"], 1)
        phase_rows.append({
            "phase": name, "start": _fmt_t(p["start_s"]), "duration_s": round(p["end_s"] - p["start_s"], 1),
            "mes_mean": round(p["mes_sum"] / n, 1), "mes_peak": round(p["mes_peak"], 1),
            "cfi_mean": round(p["cfi_sum"] / n, 3), "cfi_events": p["cfi_events"], "automation_surprises": p["surprises"],
            "compound_posture_risks": p["compound_risks"],
            "tracking_availability_pct": round(100.0 * p["tracked"] / max(p["frames"], 1), 1),
            "lip_press_mean": round(p["lip_sum"] / n, 3) if p["tracked"] else None,
            "lip_press_peak": round(p["lip_peak"], 3) if p["tracked"] else None,
            "perclos_mean": round(p["perclos_sum"] / p["perclos_n"], 3) if p["perclos_n"] else None,
            "perclos_peak": round(p["perclos_peak"], 3) if p["perclos_n"] else None,
            "pupil_change_mm_mean": round(p["pupil_sum"] / p["pupil_n"], 3) if p["pupil_n"] else None,
            "pupil_change_mm_peak": round(p["pupil_peak"], 3) if p["pupil_n"] else None,
        })

    ranked = sorted(aoi.items(), key=lambda kv: kv[1]["cfi_seconds"], reverse=True)[:top_n]
    top = [{"rank": i + 1, "aoi": k, "cumulative_cfi_s": round(v["cfi_seconds"], 2), "dwell_s": round(v["dwell_s"], 1),
            "mean_cfi_while_fixated": round(v["cfi_seconds"] / v["dwell_s"], 3) if v["dwell_s"] else 0.0,
            "peak_cfi": round(v["peak_cfi"], 3), "cfi_events": v["cfi_events"], "friction_markers": v["markers"]}
           for i, (k, v) in enumerate(ranked)]

    rep = SessionReport(duration, len(records), tracked / len(records), phase_rows, top,
                        {r: {k: round(v, 1) for k, v in cols.items()} for r, cols in cross.items()}, events, series)
    rep.findings = _findings(rep)
    return rep


def _findings(rep: SessionReport) -> list[str]:
    out = []
    for e in [e for e in rep.events if e["type"] == "CFI_EVENT"][:10]:
        out.append(f"[{_fmt_t(e['t_s'])}] Friction spike (CFI {e['cfi']:.2f}, trigger {e['trigger']}) "
                   f"mapped to AOI: {e.get('aoi') or 'none'}")
    if rep.top_aois:
        a = rep.top_aois[0]
        out.append(f"Highest cumulative friction: {a['aoi']} ({a['cumulative_cfi_s']} CFI-s over {a['dwell_s']} s dwell, "
                   f"{a['cfi_events']} friction events)")
    hi_neck = sum(rep.crosstab[r]["HIGH (>=0.40)"] for r in ("3", "4+"))
    if hi_neck > 5:
        out.append(f"Sustained neck flexion (RULA neck >= 3) co-occurred with AU7 >= 0.40 for {hi_neck:.0f} s: "
                   "workstation geometry or low-contrast display text is likely driving visual compensation")
    lo_neck_hi7 = sum(rep.crosstab[r]["HIGH (>=0.40)"] for r in ("1", "2"))
    if lo_neck_hi7 > 5:
        out.append(f"AU7 >= 0.40 with acceptable neck posture for {lo_neck_hi7:.0f} s: suspect display legibility/glare "
                   "rather than posture")
    n_s = sum(1 for e in rep.events if e["type"] == "AUTOMATION_SURPRISE")
    if n_s:
        out.append(f"{n_s} automation-surprise event(s) detected (brow raise followed by AU4 surge within 800 ms)")
    n_m = sum(1 for e in rep.events if e["type"] == "MICROSLEEP")
    if n_m:
        out.append(f"{n_m} microsleep / prolonged eye-closure event(s) (AU43)")
    return out


# ---------------------------------------------------------------------- exports
def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    fieldnames = fieldnames or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def export_csv(rep: SessionReport, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    p = out / "phase_summary.csv"; _write_csv(p, rep.phases); paths.append(p)
    p = out / "top_friction_aois.csv"
    _write_csv(p, rep.top_aois, ["rank", "aoi", "cumulative_cfi_s", "dwell_s", "mean_cfi_while_fixated", "peak_cfi",
                                 "cfi_events", "friction_markers"]); paths.append(p)
    p = out / "posture_visual_crosstab.csv"
    rows = [{"rula_neck_score": r, **{k: v for k, v in rep.crosstab[r].items()}} for r in NECK_ROWS]
    _write_csv(p, rows, ["rula_neck_score"] + [b[0] for b in AU7_BINS]); paths.append(p)
    p = out / "events.csv"
    _write_csv(p, rep.events, ["t_s", "phase", "type", "cfi", "au04", "au07", "trigger", "aoi", "rula_neck", "rula_trunk",
                               "closure_ms"]); paths.append(p)
    p = out / "timeseries_2hz.csv"
    _write_csv(p, rep.timeseries, ["t_s", "phase", "mes", "cfi", "au04", "au07", "rula_grand", "aoi"]); paths.append(p)
    return paths


def export_pdf(rep: SessionReport, path: str | Path, title: str = "CogSense HFE Evaluation Report") -> Path:
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title=title)
    tstyle = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2a37")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa5b1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f3f6")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])

    def table(rows: list[list]) -> Table:
        t = Table(rows, repeatRows=1)
        t.setStyle(tstyle)
        return t

    story = [Paragraph(title, styles["Title"]),
             Paragraph(f"Session duration {_fmt_t(rep.duration_s)} · {rep.frames} frames · "
                       f"face tracking availability {rep.tracked_fraction * 100:.1f}%", styles["Normal"]),
             Spacer(1, 6)]

    if rep.timeseries:
        d = Drawing(500, 170)
        lp = LinePlot()
        lp.x, lp.y, lp.width, lp.height = 40, 25, 440, 125
        mes = [(s["t_s"], s["mes"]) for s in rep.timeseries]
        rula = [(s["t_s"], (s["rula_grand"] or 0) * 100 / 7) for s in rep.timeseries]
        lp.data = [mes, rula]
        lp.lines[0].strokeColor = colors.HexColor("#2563eb")
        lp.lines[1].strokeColor = colors.HexColor("#d97706")
        lp.yValueAxis.valueMin, lp.yValueAxis.valueMax = 0, 100
        lp.xValueAxis.labelTextFormat = lambda v: _fmt_t(v)
        d.add(lp)
        d.add(String(40, 158, "Mental Effort Score (blue, 0-100) and RULA grand score (orange, scaled 1-7)", fontSize=8))
        story += [d, Spacer(1, 4)]

    story.append(Paragraph("Correlated diagnostic findings", styles["Heading2"]))
    for f in rep.findings or ["No significant findings."]:
        story.append(Paragraph("• " + f, styles["Normal"]))

    story.append(Paragraph("Mental effort by mission phase", styles["Heading2"]))
    cols = ["phase", "start", "duration_s", "mes_mean", "mes_peak", "cfi_mean", "cfi_events", "automation_surprises",
            "compound_posture_risks", "tracking_availability_pct"]
    hdr = ["Phase", "Start", "Dur (s)", "MES mean", "MES peak", "CFI mean", "CFI evts", "Surprises", "Posture risks", "Tracked %"]
    story.append(table([hdr] + [[p[c] for c in cols] for p in rep.phases]))

    if any(p["perclos_mean"] is not None or p["pupil_change_mm_mean"] is not None or p["lip_press_mean"] is not None
           for p in rep.phases):
        story.append(Paragraph("Fatigue, pupil and lip press by mission phase", styles["Heading2"]))
        fmt = lambda v: "n/a" if v is None else v
        story.append(table([["Phase", "PERCLOS mean", "PERCLOS peak", "Pupil chg mean (mm)", "Pupil chg peak (mm)",
                             "Lip press mean", "Lip press peak"]] +
                           [[p["phase"], fmt(p["perclos_mean"]), fmt(p["perclos_peak"]),
                             fmt(p["pupil_change_mm_mean"]), fmt(p["pupil_change_mm_peak"]),
                             fmt(p["lip_press_mean"]), fmt(p["lip_press_peak"])] for p in rep.phases]))
        story.append(Paragraph("PERCLOS: share of time the eyes were more than 80 % closed over the previous 60 s. "
                               "Pupil change: from the neutral baseline, only while display luminance was stable. "
                               "Lip press (AU24): 0-1 intensity of lips pressed together; a detection output, not yet part of the "
                               "friction index. None of these feeds the Mental Effort Score, and no alert thresholds are implied.", styles["Normal"]))

    story.append(Paragraph(f"Top {len(rep.top_aois)} UI widgets by cumulative cognitive friction", styles["Heading2"]))
    story.append(table([["#", "AOI", "Cum. CFI (CFI·s)", "Dwell (s)", "Mean CFI", "Peak", "Events", "Markers >0.70"]] +
                       [[a["rank"], a["aoi"], a["cumulative_cfi_s"], a["dwell_s"], a["mean_cfi_while_fixated"],
                         a["peak_cfi"], a["cfi_events"], a["friction_markers"]] for a in rep.top_aois]))

    story.append(Paragraph("Posture-to-visual strain cross-tabulation (seconds)", styles["Heading2"]))
    story.append(table([["RULA neck score"] + [b[0] for b in AU7_BINS]] +
                       [[r] + [rep.crosstab[r][b[0]] for b in AU7_BINS] for r in NECK_ROWS]))

    story.append(Paragraph("Event log (first 40)", styles["Heading2"]))
    story.append(table([["Time", "Phase", "Event", "CFI", "AU4", "AU7", "AOI"]] +
                       [[_fmt_t(e["t_s"]), e.get("phase"), e["type"], e.get("cfi", ""), e.get("au04", ""),
                         e.get("au07", ""), e.get("aoi") or ""] for e in rep.events[:40]]))
    doc.build(story)
    return Path(path)
