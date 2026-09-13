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

# numbers printed in a scan file that are NOT data claims (tooling versions)
NON_CLAIM_TOKENS = {"3.10"}          # README "Requires Python >= 3.10"

SCAN_FILES = sorted(
    [REPO / "README.md", REPO / "README.zh-CN.md"]
    + list((REPO / "docs").rglob("*.md"))
    + list((REPO / "examples").rglob("*.md"))
    + list((REPO / "tutorials").rglob("*.md"))
    # community-facing .py files carry user-visible text too (docstrings)
    + list((REPO / "examples").rglob("*.py"))
)
SCAN_FILES = [f for f in SCAN_FILES if "__pycache__" not in f.parts]


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
    # use the paper-facing arm D as the R_A anchor, not the frozen legacy
    # record (validation_summary.json is a frozen legacy artifact; the README
    # CI [0.7, 0.95] is arm D's)
    fd = json.loads((REPO / "results/openillumination/correctness/"
                     "mf0_factorial_summary.json").read_text(encoding="utf-8"))
    D = fd["variants"]["D"]
    assert D["noise_fit_convention"] == "corrected", "anchor must be paper-facing arm D"
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
        "0.90": D["RA"],
        "0.7":  D["RA_ci95"][0],      # CI lower bound, previously untraced
        "0.95": D["RA_ci95"][1],
    }
    for tok, val in traced.items():
        assert _approx(tok, val, 3), (tok, val)

    # the R_A CI pair printed in the README is the paper-facing anchor: it must
    # numerically match the arm-D field, not merely recur verbatim somewhere
    # under results/** (a plain lexical token like "0.6" can coincidentally
    # occur in a results file and slip past the generic sweep below).
    ci_m = re.search(r"CI\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", readme)
    assert ci_m, "README must print the R_A CI pair (e.g. 'CI [0.7, 0.95]')"
    lo, hi = ci_m.group(1), ci_m.group(2)
    assert _approx(lo, D["RA_ci95"][0], 3), (lo, D["RA_ci95"][0])
    assert _approx(hi, D["RA_ci95"][1], 3), (hi, D["RA_ci95"][1])

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

    # numbers quoted from the certified / level / channel evidence layer, and
    # from the corrected amplitude arm -- each verified against its own field
    # (decimals match the rounding printed in the README)
    cg = json.loads((REPO / "results/certification/certified_gaps.json")
                    .read_text(encoding="utf-8"))
    lr = json.loads((REPO / "results/certification/lowrank_fullres.json")
                    .read_text(encoding="utf-8"))
    ampD = json.loads((REPO / "results/magnitude/"
                       "directional_amplitude_summary.json")
                      .read_text(encoding="utf-8"))["arm_D_corrected"]
    for tok, val, dec in (
            ("0.014", r48["regimes"]["10"]["mode_minus_random_active48"]["median"], 3),
            ("0.019", r48["regimes"]["100"]["mode_minus_random_active48"]["median"], 3),
            ("0.027", max(meds), 3),
            ("0.64", max(v["random_mean_minus_lower_rel"]["median"]
                         for v in cg["by_k"].values()) * 100, 2),
            ("15.98", ampD["ratio_stats"]["median"], 2),
            ("714.2", ampD["ratio_stats"]["p95"], 1),
            ("27.3", lr["by_k"]["48"]["dynamic_range_pct"]["min"], 1),
            ("89.4", lr["by_k"]["48"]["dynamic_range_pct"]["max"], 1)):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val

    # generic sweep: every other number printed in README must appear as a
    # *token* under results/** (claim-tracing rule, guideline section 4).
    # Two hardening points over the original sweep:
    #   (a) the unicode minus is normalised and the token regex keeps the sign,
    #       so the signed `traced` keys actually match README tokens (before,
    #       "-0.150" was keyed but the tokenizer produced "0.150", so those six
    #       entries were never checked);
    #   (b) presence is tested on token boundaries -- a bare substring hit is
    #       not traceability (the literal "15.98" occurs inside the unrelated
    #       value "15.986150483597436").
    readme2 = re.sub(r"\d+e-\d+", " ", readme.replace("\u2212", "-"))
    tokens = set(re.findall(r"-?\d+\.\d+|-?\d+", readme2))
    results_text = "\n".join(
        q.read_text(encoding="utf-8", errors="ignore")
        for q in sorted((REPO / "results").rglob("*")) if q.is_file())
    untraced = [tk for tk in sorted(tokens)
                if tk not in traced and tk not in NON_CLAIM_TOKENS
                and re.search(r"(?<![\w.])" + re.escape(tk) + r"(?![\w])",
                              results_text) is None]
    assert not untraced, f"README numbers not traceable to results/**: {untraced}"
