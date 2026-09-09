"""Wording gate: automated tripwire for README.md and docs/*.md.

Two checks:
1. Banned sentence/word families -- zero hits. The list combines the historical
   process-language family (case-sensitive, as in the original leak gate) with the
   overclaim phrasings corrected by the wording guideline (docs/WORDING.md).
2. Numeric claim tracing -- every number printed in README.md must either appear
   verbatim inside results/** or be covered by an explicit entry below that reads /
   recomputes the value from a named results/ field.

docs/WORDING.md is excluded from the scan by design: it is the guardhouse reference
that must be able to quote the banned families verbatim.
"""

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BANNED = [
    # historical process-language family (case-sensitive)
    r"Branch A", r"Branch B", r"Branch C", r"Branch D", r"PASS-A", r"FAIL-A",
    r"S0", r"S\+", r"S-", r"red team", r"红队", r"CLAIMS_REGISTRY", r"claim lock",
    r"branch_decision", r"rt_f12", r"taskbook", r"任务书", r"EXPERT_BRIEFING",
    r"AGENT_HANDOFF", r"incident", r"事故", r"gate D", r"CI04R", r"ci04r",
    # process vocabulary that must not re-enter docs
    r"裁决", r"宪法", r"红线", r"CI0[0-9]",
    # overclaim phrasings (wording guideline §2)
    r"better than[^.\n]{0,80}E-optimality",
    r"on every resolution level",
    r"pass rate",
]

SCAN_FILES = sorted(
    [REPO / "README.md"]
    + list((REPO / "docs").rglob("*.md"))
    + list((REPO / "paper").rglob("*.md"))
    + list((REPO / "paper").rglob("*.tex"))
)
SCAN_FILES = [f for f in SCAN_FILES if f.name != "WORDING.md"]


def test_banned_families_absent():
    hits = []
    for f in SCAN_FILES:
        text = f.read_text(encoding="utf-8")
        for pat in BANNED:
            m = re.search(pat, text)
            if m:
                hits.append(f"{f.name}: /{pat}/ -> {text[max(0, m.start()-30):m.end()+30]!r}")
    assert not hits, "banned wording found:\n" + "\n".join(hits)


def _approx(token, value, decimals):
    return abs(float(token) - float(value)) <= 0.5 * 10 ** (-decimals) + 1e-9


def test_readme_numeric_claims_traceable():
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    # structural identity claim (frozen template): P_emin == P_mode within 1e-14
    assert "1e-14" in readme
    with open(REPO / "results/openillumination/level_severity.csv",
              newline="", encoding="utf-8") as f:
        ls = list(csv.DictReader(f))
    max_diff = max(abs(float(r["P_emin"]) - float(r["P_mode"])) for r in ls)
    assert max_diff <= 1e-14

    with open(REPO / "results/openillumination/predictor_comparison.csv",
              newline="", encoding="utf-8") as f:
        pc = {r["Predictor"]: r for r in csv.DictReader(f)}
    vs = json.loads((REPO / "results/openillumination/validation_summary.json")
                    .read_text(encoding="utf-8"))

    traced = {
        "0.536": _approx("0.536", pc["mode-resolved"]["Stratified_median_rho"], 3),
        "0.418": _approx("0.418", pc["logdet"]["Stratified_median_rho"], 3),
        "0.400": _approx("0.400", pc["trace"]["Stratified_median_rho"], 3),
        "0.455": _approx("0.455", pc["trace"]["L1"], 3),
        "0.245": _approx("0.245", pc["mode-resolved"]["L1"], 3),
        "0.700": _approx("0.700", pc["logdet"]["L3"], 3),
        "0.673": _approx("0.673", pc["mode-resolved"]["L3"], 3),
        "0.90": _approx("0.90", vs["RA"], 2),
        "0.95": _approx("0.95", vs["RA_ci95"][1], 2),
        "0.096": _approx("0.096", abs(vs["stratified_median_mode_ci95"][0]), 3),
        "0.858": _approx("0.858", vs["stratified_median_mode_ci95"][1], 3),
    }
    bad = [k for k, ok in traced.items() if not ok]
    assert not bad, f"frozen claims no longer match results/: {bad}"

    # generic sweep: every other number printed in README must appear verbatim
    # somewhere under results/** (claim-tracing rule, guideline §3)
    readme2 = re.sub(r"\d+e-\d+", " ", readme)      # scientific notation handled above
    tokens = set(re.findall(r"\d+\.\d+|\d+", readme2))
    results_text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted((REPO / "results").rglob("*")) if p.is_file())
    untraced = [t for t in sorted(tokens) if t not in results_text and t not in traced]
    assert not untraced, f"README numbers not traceable to results/**: {untraced}"
