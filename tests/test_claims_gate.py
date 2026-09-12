"""Wording gate: automated tripwire for README.md and docs/*.md.

Two checks:
1. Banned sentence/word families -- zero hits. The list combines the historical
   process-language family (case-sensitive, as in the original leak gate) with the
   overclaim phrasings listed in this file's BANNED table (docs/claims.md is the registry).
2. Numeric claim tracing -- every number printed in README.md must either appear
   verbatim inside results/** or be covered by an explicit entry below that reads /
   recomputes the value from a named results/ field.

docs/claims.md is the claim-to-evidence registry (product-facing, part of the
scan).
"""

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BANNED = [
    # overclaim phrasings (registry: docs/claims.md)
    r"better than[^.\n]{0,80}E-optimality",
    r"on every resolution level",
    r"pass rate",
    # ordinary-eigenvalue-ratio claim for retention (generalized form only)
    r"\\rho_j\s*=\s*\\lambda_j",
    r"rho_j\s*=\s*lambda_j\(",
    # pseudoinverse-precision notation for singular covariance (factor path
    # must be stated positively instead)
    r"Σ_c⁺",
    r"Sigma_c\^\+",
    r"\\Sigma_c\^\+",
    # lambda-star overclaim (registered name: directional crossover precision)
    r"validated real-world crossover",
    # allocation-grade overclaims (public text: "actionable outcome vs random")
    r"strong grade",
    r"strong result",
]

SCAN_FILES = sorted(
    [REPO / "README.md"]
    + list((REPO / "docs").rglob("*.md"))
    + list((REPO / "examples").rglob("*.md"))
    + list((REPO / "tutorials").rglob("*.md"))
    # community-facing .py files carry user-visible text too (docstrings)
    + list((REPO / "examples").rglob("*.py"))
)
SCAN_FILES = [f for f in SCAN_FILES
              if f.name != "claims.md" and "__pycache__" not in f.parts]


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
    """Every number in the library-first README traces to results/**:
    benchmark claims via summary JSON fields, allocation claims via
    allocation_summary.json, structural/count claims via the artifacts."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    vs = json.loads((REPO / "results/openillumination/validation_summary.json")
                    .read_text(encoding="utf-8"))
    al = json.loads((REPO / "results/openillumination/allocation/"
                     "allocation_summary.json").read_text(encoding="utf-8"))

    def _m(regime, policy):
        return al["regimes"][str(regime)]["policy_deltas"][policy]

    # benchmark reproduction section (frozen evidence)
    with open(REPO / "results/openillumination/level_severity.csv",
              newline="", encoding="utf-8") as f:
        lvl = list(csv.DictReader(f))
    assert len(lvl) == 66                                          # 66/66 cells
    with open(REPO / "results/openillumination/allocation/per_run_errors.csv",
              newline="", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    assert len({r["object"] for r in runs}) == 11                  # 11 objects
    assert len(runs) == 29700                                      # 29,700 runs

    # allocation key results (allocation_summary.json)
    traced = {
        "-0.150": _m(10, "mode_aware")["median"],
        "-0.279": _m(100, "mode_aware")["median"],
        "-0.550": _m(10, "mode_aware")["ci95"][0],
        "-0.077": _m(10, "mode_aware")["ci95"][1],
        "-0.684": _m(100, "mode_aware")["ci95"][0],
        "-0.102": _m(100, "mode_aware")["ci95"][1],
        "0.90": vs["RA"],
        "0.95": vs["RA_ci95"][1],
    }
    for tok, val in traced.items():
        assert _approx(tok, val, 3), (tok, val)

    # post-hoc paired policy comparison (allocation_policy_pairwise.csv): the
    # README quotes the range of the six median paired differences
    with open(REPO / "results/openillumination/allocation/"
              "allocation_policy_pairwise.csv", newline="",
              encoding="utf-8") as f:
        pw = list(csv.DictReader(f))
    assert len(pw) == 6 and {r["analysis_status"] for r in pw} == \
        {"posthoc_paired_comparison"}
    meds = [float(r["median_delta"]) for r in pw]
    assert _approx("0.019", min(meds), 3) and _approx("0.027", max(meds), 3)
    assert all(float(r["ci_lo"]) > 0 for r in pw)      # all CIs exclude 0

    # random_active48 attribution control (allocation_random48_summary.json):
    # README quotes the control deltas +0.014 / +0.019 with CIs spanning 0
    r48 = json.loads((REPO / "results/openillumination/allocation/"
                      "allocation_random48_summary.json").read_text(
                          encoding="utf-8"))
    assert r48["analysis_status"] == "posthoc_attribution_control"
    for regime, tok_med, tok_lo, tok_hi in (
            ("10", "0.014", "-0.024", "0.048"),
            ("100", "0.019", "-0.012", "0.063")):
        v = r48["regimes"][regime]["mode_minus_random_active48"]
        assert _approx(tok_med, v["median"], 3), (tok_med, v["median"])
        assert _approx(tok_lo, v["ci95"][0], 3), (tok_lo, v["ci95"][0])
        assert _approx(tok_hi, v["ci95"][1], 3), (tok_hi, v["ci95"][1])
        assert v["ci95"][0] < 0 < v["ci95"][1]         # CI spans 0

    # generic sweep: every other number printed in README must appear verbatim
    # somewhere under results/** (claim-tracing rule, guideline section 4)
    readme2 = re.sub(r"\d+e-\d+", " ", readme)
    tokens = set(re.findall(r"\d+\.\d+|\d+", readme2))
    results_text = "\n".join(
        q.read_text(encoding="utf-8", errors="ignore")
        for q in sorted((REPO / "results").rglob("*")) if q.is_file())
    untraced = [tk for tk in sorted(tokens)
                if tk not in results_text and tk not in traced]
    assert not untraced, f"README numbers not traceable to results/**: {untraced}"
