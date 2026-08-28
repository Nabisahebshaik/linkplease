"""Small, printable clinical screening report generator."""

from __future__ import annotations

from html import escape


def build_html_report(
    label: str,
    confidence: float,
    quality_grade: str,
    quality_feedback: str,
    lesion_summary: dict[str, int],
) -> str:
    """Build an HTML report suitable for browser printing or PDF export."""
    triage = "Immediate ophthalmologist referral" if label in {"Severe", "Proliferate_DR"} else "Routine clinical review"
    rows = "".join(
        f"<tr><td>{escape(name.replace('_', ' '))}</td><td>{count}</td></tr>"
        for name, count in lesion_summary.items()
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DR Screening Report</title>
<style>body{{font-family:Arial;max-width:760px;margin:32px auto}}table{{border-collapse:collapse}}
td,th{{border:1px solid #bbb;padding:8px}}</style></head><body>
<h1>Diabetic Retinopathy Screening Report</h1>
<p><b>Screening grade:</b> {escape(label)} ({confidence:.1%})</p>
<p><b>Triage:</b> {triage}</p>
<p><b>Image quality:</b> {escape(quality_grade)}. {escape(quality_feedback)}</p>
<h2>Candidate findings</h2><table><tr><th>Finding</th><th>Pixels</th></tr>{rows}</table>
<p><small>Decision support only. A qualified clinician must confirm this result.</small></p>
</body></html>"""
