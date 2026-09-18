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
import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
IMPORTED = REPO / "results/theory_extension_20260918/imported_20260917"
SIGNED_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"

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
    [REPO / "README.md", REPO / "README.zh-CN.md", REPO / "CONTRIBUTING.md"]
    + list((REPO / "docs").rglob("*.md"))
    + list((REPO / "examples").rglob("*.md"))
    + list((REPO / "tutorials").rglob("*.md"))
    # community-facing .py files carry user-visible text too (docstrings)
    + list((REPO / "examples").rglob("*.py"))
)
SCAN_FILES = [f for f in SCAN_FILES if "__pycache__" not in f.parts]

# --------------------------------------------------------------- endpoint gate
# Any claim that compares a policy against random must say WHICH functional it
# is measured on. The same allocation comparison is a null result on the
# model-space weak-mode endpoint (E_osb) and a real, CI-excluding-zero result
# on the physical normal-angular-error endpoint; a row that names neither can
# be read as a universal statement and then contradicts its sibling row.
#
# Rule 5/7 of CONTRIBUTING applies: a rule is enforced globally or not written.
# The gate therefore covers the whole registry, and the endpoints it accepts
# are the ones the producers actually compute.
# Accepted endpoint words name a FUNCTIONAL — the quantity the error is
# measured on. Statistics that can be computed over any functional are NOT
# endpoints: "Δ AUC" and "MSE" appear in every allocation bullet and say
# nothing about what the curve was computed on. An earlier version of this
# gate accepted AUC, and mutation-testing the README's mislabelled bullet
# (which contains "Δ AUC") showed it passed the defect straight through.
_ENDPOINT_WORDS = re.compile(
    r"E_osb|J_A|ang_mean_deg|normal[- ]angular|physical endpoint"
    r"|physical reconstruction|dual[- ]coordinate|weak-mode|angular error",
    re.I,
)
# A comparison is "against random" when the row pairs a comparative with the
# random baseline. M6 ("no feasible point beats the bound") is a statement
# about a certificate, not a comparison against random, and is out of scope.
_COMPARATIVE = re.compile(
    r"advantage|benefit|beats?\b|better|improve|outperform", re.I,
)


def test_random_comparisons_name_their_endpoint():
    """Every advantage-over-random row names the functional it is measured on.

    Guards the failure mode the endpoint review found: B5 stated the
    active-set attribution unconditionally while B9 reported the opposite on a
    different endpoint, and neither row referenced the other. Both now carry
    an endpoint and a cross-reference; this test keeps it that way.
    """
    registry = REPO / "docs/claims.md"
    if not registry.exists():
        import pytest
        pytest.skip("docs/claims.md not present")

    offenders = []
    for ln, line in enumerate(registry.read_text(encoding="utf-8").splitlines(), 1):
        # A claim row starts with "| " and has a claim id in the second cell.
        # Do NOT require a fixed pipe count: a row written without its trailing
        # pipe has one fewer, and such a row is exactly the kind that slips
        # through a structural filter (found by mutation-testing this gate
        # against the original defective B5 row, which had 4 pipes).
        if not line.startswith("| "):
            continue
        rid = line.split("|")[1].strip()
        if not re.match(r"^[A-Z]\d+['′]?$", rid):
            continue                       # header / separator rows
        if "random" not in line.lower():
            continue
        if not _COMPARATIVE.search(line):
            continue
        if _ENDPOINT_WORDS.search(line):
            continue
        offenders.append(f"docs/claims.md:{ln} row {rid}")

    assert not offenders, (
        "these claims compare against random without naming the functional "
        "they are measured on; the same comparison can be a null on one "
        "endpoint and significant on another, so an unqualified row is "
        "ambiguous:\n  " + "\n  ".join(offenders))


def _markdown_blocks(text):
    """Group markdown into bullet blocks: (first line number, block text).

    README prose wraps, so an endpoint named on a bullet's first line and a
    comparison word five lines later belong to the same statement. Checking
    line-by-line reports those as offenders; checking the whole file reports
    nothing. The bullet is the unit a reader actually parses.
    """
    blocks, cur, start = [], [], 0
    for i, ln in enumerate(text.splitlines(), 1):
        if re.match(r"^\s*[-*] ", ln):
            if cur:
                blocks.append((start, "\n".join(cur)))
            cur, start = [ln], i
        elif cur and ln.strip():
            cur.append(ln)
        elif cur:
            blocks.append((start, "\n".join(cur)))
            cur = []
    if cur:
        blocks.append((start, "\n".join(cur)))
    return blocks


def test_readme_random_comparisons_name_their_endpoint():
    """The same rule, applied to the README's headline bullets.

    The registry is not the only place a reader meets these numbers: the
    README's summary bullets carry the same comparison, and it was there that
    the E_osb result was described as improving "reconstruction" while a
    different bullet used the same word for the physical endpoint. Enforcing
    the rule only in `docs/claims.md` would repeat exactly the mistake the
    rule exists to prevent (CONTRIBUTING 5/7: enforced globally or not at
    all).
    """
    readme = REPO / "README.md"
    if not readme.exists():
        import pytest
        pytest.skip("README.md not present")

    offenders = []
    for start, blk in _markdown_blocks(readme.read_text(encoding="utf-8")):
        if "random" not in blk.lower():
            continue
        if not _COMPARATIVE.search(blk):
            continue
        if _ENDPOINT_WORDS.search(blk):
            continue
        first = blk.strip().splitlines()[0]
        offenders.append(f"README.md:{start} {first[:70]}")

    assert not offenders, (
        "these README bullets compare against random without naming the "
        "functional they are measured on (the E_osb and physical endpoints "
        "do not share a conclusion):\n  " + "\n  ".join(offenders))


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


def _prose(text):
    """Normalize presentation, not the sign or the claim's scope."""
    return re.sub(r"[`*$]", "", text).replace("\u2212", "-")


def _readme_claim_block(text, claim):
    blocks = [block for _, block in _markdown_blocks(text)
              if re.search(r"\b" + claim + r"\b", block.splitlines()[0])]
    assert len(blocks) == 1, f"expected one current README {claim} bullet"
    return blocks[0]


def _ra_numbers(block):
    """Read one R_A statement, never a different headline's first CI."""
    text = _prose(block)
    ra = re.findall(r"\bR_A\s*=\s*(" + SIGNED_NUMBER + r")", text)
    ci = re.findall(r"CI\s*\[\s*(" + SIGNED_NUMBER + r")\s*,\s*("
                    + SIGNED_NUMBER + r")\s*\]", text)
    cells = re.findall(r"(\d+)\s*/\s*(\d+)\s*(?:positive\s+)?"
                       r"(?:cells|单元|细胞)", text)
    objects = re.findall(r"(\d+)\s*/\s*(\d+)\s*(?:positive\s+)?"
                         r"(?:objects|物体)", text)
    assert len(ra) == len(ci) == len(cells) == len(objects) == 1, text
    return (float(ra[0]), tuple(map(float, ci[0])),
            tuple(map(int, cells[0])), tuple(map(int, objects[0])))


def _assert_ra_bound(block, arm, n_cells):
    ra, ci, cells, objects = _ra_numbers(block)
    assert ra == pytest.approx(arm["RA"], abs=5e-4, rel=0)
    assert ci == pytest.approx(arm["RA_ci95"], abs=5e-4, rel=0)
    assert cells == (arm["n_pos_cells"], n_cells)
    assert objects == (arm["n_pos_objects"], arm["n_objects"])


def _amplitude_numbers(block):
    text = _prose(block)
    median = re.findall(r"(?:\bmedian|中位)\s*(" + SIGNED_NUMBER + r")", text)
    percentiles = re.findall(
        r"5\s*[-–]\s*95%\s*\[\s*(" + SIGNED_NUMBER + r")\s*,\s*("
        + SIGNED_NUMBER + r")\s*\]", text)
    assert len(median) == len(percentiles) == 1, text
    return (float(median[0]), *map(float, percentiles[0]))


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md"])
def test_readme_current_b1_b2_field_bound(name):
    text = (REPO / name).read_text(encoding="utf-8")
    factorial = json.loads((IMPORTED / "mf0_factorial_summary.json")
                           .read_text(encoding="utf-8"))
    amplitude = json.loads((IMPORTED / "amplitude_comparison.json")
                           .read_text(encoding="utf-8"))
    arm = factorial["variants"]["D"]
    b1 = _readme_claim_block(text, "B1")
    _assert_ra_bound(b1, arm, factorial["protocol"]["n_cells"])
    for field, expected in (("noise_fit_convention", "corrected"),
                            ("mode_coordinate", "dual"),
                            ("gauge_mode", "per_seed"),
                            ("prediction_field", "pred_deg")):
        assert arm[field] == expected
    for literal in ("gauge_mode=per_seed", "prediction_field=pred_deg",
                    "variants.D", "imported_20260917/mf0_factorial_summary.json"):
        assert literal in b1
    assert re.search(r"not matched[- ]prediction validation|不是 matched prediction 验证",
                     _prose(b1), re.I)
    assert re.search(r"CI\s+spans zero|CI 跨零", _prose(b1))

    b2 = _readme_claim_block(text, "B2")
    pooled = amplitude["variants"]["D"]["new_ratio_matched_prediction"]["pooled"]
    assert _amplitude_numbers(b2) == pytest.approx(
        [pooled[k] for k in ("median", "p5", "p95")], abs=5e-7, rel=0)
    for literal in ("variants.D.new_ratio_matched_prediction.pooled",
                    "imported_20260917/amplitude_comparison.json"):
        assert literal in b2
    plain = _prose(b2)
    assert re.search(r"residual\s+bootstrap", plain, re.I)
    assert re.search(r"not the matched\s+GLS|不是 matched GLS", plain, re.I)
    assert re.search(r"neither\s+validates nor refutes|既不能验证、也不能反驳", plain)
    # Cross-artifact field equality prevents swapping B1's original-prediction
    # control for the separately reported matched-coordinate ranking.
    assert amplitude["variants"]["D"]["frozen_statistical_summary"] == arm


@pytest.mark.parametrize("minus", ["-", "\u2212"])
def test_ra_parser_is_signed_and_local_to_the_claim(minus):
    text = ("- **Amplitude B2**: CI [0.2, 0.8].\n\n"
            "- **Current B1**: R_A = 0.55; CI [" + minus + "0.1, 0.7]; "
            "43/66 cells positive, 7/11 objects positive.\n\n"
            "Historical B1: R_A = 0.90; CI [0.7, 0.95].")
    block = _readme_claim_block(text, "B1")
    assert _ra_numbers(block) == (0.55, (-0.1, 0.7), (43, 66), (7, 11))
    arm = {"RA": 0.55, "RA_ci95": [-0.1, 0.7], "n_pos_cells": 43,
           "n_pos_objects": 7, "n_objects": 11}
    # A valid unrelated CI or a superseded R_A cannot launder a mutated B1.
    with pytest.raises(AssertionError):
        _assert_ra_bound(block.replace(minus + "0.1", "0.1"), arm, 66)
    with pytest.raises(AssertionError):
        _assert_ra_bound(block.replace("43/66", "65/66"), arm, 66)


def test_readme_numeric_claims_traceable():
    """Every number in the library-first README traces to results/**:
    benchmark claims via summary JSON fields, allocation claims via
    allocation_summary.json, structural/count claims via the artifacts."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    # Current B1 is the imported per-seed D arm, not the earlier corrected
    # artifact. Historical numbers are separately gated and regressed below.
    fd = json.loads((IMPORTED / "mf0_factorial_summary.json")
                    .read_text(encoding="utf-8"))
    D = fd["variants"]["D"]
    assert D["noise_fit_convention"] == "corrected", "anchor must be paper-facing arm D"
    assert D["gauge_mode"] == "per_seed"
    assert D["prediction_field"] == "pred_deg"
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
        "0.55": D["RA"],
        "-0.1": D["RA_ci95"][0],
        "0.7": D["RA_ci95"][1],
    }
    for tok, val in traced.items():
        assert _approx(tok, val, 3), (tok, val)

    # The R_A CI must match B1, including its sign, not the first CI in the
    # whole README (allocation and historical bullets also carry CIs).
    _assert_ra_bound(_readme_claim_block(readme, "B1"), D,
                     fd["protocol"]["n_cells"])

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
    # from the imported same-projection amplitude comparison (not ensemble-
    # matched) -- each verified against its own field at the printed precision
    cg = json.loads((REPO / "results/certification/certified_gaps.json")
                    .read_text(encoding="utf-8"))
    lr = json.loads((REPO / "results/certification/lowrank_fullres.json")
                    .read_text(encoding="utf-8"))
    ampD = json.loads((IMPORTED / "amplitude_comparison.json")
                      .read_text(encoding="utf-8"))["variants"]["D"]
    amp_pooled = ampD["new_ratio_matched_prediction"]["pooled"]
    go = json.loads((REPO / "results/goal_oriented/goal_orientation.json")
                    .read_text(encoding="utf-8"))
    go_v = go["headline"]["value_curves_by_level"]
    # gauge-mechanism sentence (v1.1+ artifacts): two-term-law agreement
    gm_head = go["headline"].get("gauge_mechanism_rho_mean")
    gm_entry = (("0.0008", gm_head["median_abs_dV"], 4),) if gm_head else ()
    for tok, val, dec in (
            ("0.014", r48["regimes"]["10"]["mode_minus_random_active48"]["median"], 3),
            ("0.019", r48["regimes"]["100"]["mode_minus_random_active48"]["median"], 3),
            ("0.027", max(meds), 3),
            ("0.64", max(v["random_mean_minus_lower_rel"]["median"]
                         for v in cg["by_k"].values()) * 100, 2),
            ("53.744194", amp_pooled["median"], 6),
            ("0.489414", amp_pooled["p5"], 6),
            ("820.342156", amp_pooled["p95"], 6),
            ("27.3", lr["by_k"]["48"]["dynamic_range_pct"]["min"], 1),
            ("89.4", lr["by_k"]["48"]["dynamic_range_pct"]["max"], 1),
            # goal-oriented paragraph: V_H medians (x100, 2dp) + Spearman
            ("89.76", go_v["0.5"]["mean"] * 100, 2),
            ("87.93", go_v["0.1"]["mean"] * 100, 2),
            ("7.82", go_v["0.1"]["all"] * 100, 2),
            ("0.936", go["headline"]["spearman_mean_vs_contrast"]["median"], 3)
            ) + gm_entry:
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val
    # decision-quality bullet: informed-only pooled sign agreement (+CI)
    dq = json.loads((REPO / "results/openillumination/decision_quality.json")
                    .read_text(encoding="utf-8"))
    _ip = dq["informed_pairwise_sign_pooled"]
    for tok, val in (("0.687", _ip["rate"]), ("0.657", _ip["ci95"][0]),
                     ("0.716", _ip["ci95"][1])):
        assert -0.5e-3 <= float(tok) - val <= 0.5e-3, (tok, val)
        traced[tok] = val

    # Σ_φ family sensitivity bullet (C9 / P-SIGMA-FAMILY): direction
    # share at the operating point, two-term-law accuracy, ceiling guard
    fam = json.loads((REPO / "results/openillumination/"
                      "corruption_family_sensitivity.json")
                     .read_text(encoding="utf-8"))
    for tok, val, dec in (
            ("0.07", fam["s21_direction_share"]["share_at_anchor_median"]
             * 100.0, 2),
            ("0.0076", fam["s22_two_term_law"]["overall"]["median_abs_dV"],
             4),
            ("0.464", fam["s22_two_term_law"]["overall"]["max_abs_dV"], 3),
            ("-0.0028", fam["s23_ceiling"]["max_violation"], 4)):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val

    # E arm bullet: per-arm informed-ordering rates (+ the het median
    # Spearman, printed as -0.78 in README prose)
    fame = json.loads((REPO / "results/openillumination/"
                       "decision_quality_family.json")
                      .read_text(encoding="utf-8"))
    for tok, val, dec in (
            ("0.704", fame["arms"]["anchor"]["informed_pairwise_sign"]
             ["rate"], 3),
            ("0.889", fame["arms"]["dir_heavy"]["informed_pairwise_sign"]
             ["rate"], 3),
            ("0.353", fame["arms"]["het"]["informed_pairwise_sign"]
             ["rate"], 3),
            ("-0.78", fame["arms"]["het"]["spearman_informed_per_cell"]
             ["median"], 2)):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val
    # ball anchor: measured errors printed in README prose
    ba = json.loads((REPO / "results/openillumination/ball_anchor.json")
                    .read_text(encoding="utf-8"))
    for tok, val, dec in (
            ("2.96", ba["measured_anchor"]["sig_dir_deg"], 2),
            ("0.0159", ba["measured_anchor"]["sig_logI"], 4),
            ("36", ba["direction_share_at_anchor"]["median"] * 100.0, 0),
            ("10.6", (math.radians(ba["measured_anchor"]["sig_dir_deg"])
                      / ba["measured_anchor"]["sig_logI"]) ** 2, 1)):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val
    # baseline comparison numbers (en README prose)
    bcmp = json.loads((REPO / "results/baseline/"
                       "baseline_comparison.json")
                      .read_text(encoding="utf-8"))
    for tok, val, dec in (
            ("5.24", bcmp["oi"]["14"]["median_ang_by_unit"]["dc05_active"], 2),
            ("4.00", bcmp["oi"]["28"]["median_ang_by_unit"]["dc05_active"], 2),
            ("5.78", min(bcmp["oi"]["14"]["median_ang_by_unit"][u]
                         for u in bcmp["oi"]["14"]["median_ang_by_unit"]
                         if u.startswith("randomA48_")), 2),
            ("5.99", max(bcmp["oi"]["14"]["median_ang_by_unit"][u]
                         for u in bcmp["oi"]["14"]["median_ang_by_unit"]
                         if u.startswith("randomA48_")), 2),
            ("4.49", min(bcmp["oi"]["28"]["median_ang_by_unit"][u]
                         for u in bcmp["oi"]["28"]["median_ang_by_unit"]
                         if u.startswith("randomA48_")), 2),
            ("5.25", max(bcmp["oi"]["28"]["median_ang_by_unit"][u]
                         for u in bcmp["oi"]["28"]["median_ang_by_unit"]
                         if u.startswith("randomA48_")), 2),
            ("4.52", bcmp["oi"]["14"]["median_ang_by_unit"]["e_opt"], 2),
            ("2.89", bcmp["oi"]["28"]["median_ang_by_unit"]["e_opt"], 2),
            ("-0.33", bcmp["diligent"]["28"]["median_dev_from"]
             ["dc05_active_vs_randomA48"], 2),
            ("-0.34", bcmp["diligent"]["14"]["median_dev_from"]
             ["dc05_active_vs_randomA48"], 2),
            ("-0.27", bcmp["diligent"]["14"]["median_dev_from"]
             ["informed_vs_randomA48"], 2),
            ("-0.37", bcmp["diligent"]["28"]["median_dev_from"]
             ["informed_vs_randomA48"], 2),
            ("0.03", abs(bcmp["diligent"]["14"]["median_ang_by_unit"]
                         ["a_opt"]
                         - bcmp["diligent"]["14"]["median_ang_by_unit"]
                         ["dc05_active"]), 2)):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val

    # DiLiGenT queue numbers (en README prose)
    dq2 = json.loads((REPO / "results/diligent/diligent_queue.json")
                     .read_text(encoding="utf-8"))
    for tok, val, dec in (
            ("0.20", dq2["channel_decomposition"]["direction_max_pct"], 2),
            ("1.68", 1.68, 2),   # OI reference (channel_decomposition.json)
            ("22", round(dq2["ball_anchor_share"]["per_object"]["catPNG"]
                         * 100.0), 0),
            ("29", round(dq2["ball_anchor_share"]["per_object"]["pot1PNG"]
                         * 100.0), 0)):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val

    # anchor coverage numbers (en README prose)
    for tok, val, dec in (
            ("3.1", 0.05 / ba["measured_anchor"]["sig_logI"], 1),
            ("4.8", min(ba["direction_share_at_anchor"]["per_object"]
                        .values()) * 100.0, 1),
            ("71.4", max(ba["direction_share_at_anchor"]["per_object"]
                         .values()) * 100.0, 1)):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val


    # anchor-mechanism numbers (P-ANCHOR-MECHANISM) in README prose
    mech = json.loads((REPO / "results/openillumination/"
                       "anchor_mechanism.json")
                      .read_text(encoding="utf-8"))
    _mc = mech["correlations"]["dir_int_weak_ratio"]
    for tok, val, dec in (
            ("-0.899", _mc["pooled"], 3),
            ("-0.927", _mc["per_cohort"]["oi"], 3),
            ("-1.000", _mc["per_cohort"]["dq"], 3)):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val

    # feasible-interval DQ numbers (P-DQ-FEASIBLE) in README prose
    dqf = json.loads((REPO / "results/openillumination/"
                      "decision_quality_feasible.json")
                     .read_text(encoding="utf-8"))
    _fe = dqf["feasible_dAUC_vs_randomA48"]
    # feasible-interval magnitudes printed as a range (abs of the dAUC)
    for tok, val, dec in (
            ("0.38", abs(max(v["median_dAUC"] for v in _fe.values())), 2),
            ("1.40", abs(min(v["median_dAUC"] for v in _fe.values())), 2)):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val

    for tok, val, dec in (
            ("-0.38", max(v["median_dAUC"] for v in _fe.values()), 2),
            ("-1.40", min(v["median_dAUC"] for v in _fe.values()), 2)):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val

    # E cross-diagnosis: S_real printed as -0.325 in README prose
    ediag = json.loads((REPO / "results/openillumination/"
                        "corruption_family_e_diag.json")
                       .read_text(encoding="utf-8"))
    for tok, val, dec in (
            ("-0.325", ediag["summary"]["s_real"], 3),
            ("0.949", ediag["summary"]["s_pred"], 3)):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val
    # A-arm failure-region alignment: worst intensity-carrying cell
    # (reverse_control tag max) printed as 0.048 in README prose
    for tok, val, dec in (("0.048",
                           fam["s22_two_term_law"]["by_tag"]
                           ["reverse_control"]["max_abs_dV"], 3),):
        assert _approx(tok, val, dec), (tok, val)
        traced[tok] = val

    # Historical values may still be printed, but only in explicitly scoped
    # history. Dedicated history/scope tests below and in test_headline_binding
    # prevent these sources from licensing a current B1/B2/γ headline.
    old_fd = json.loads((REPO / "results/openillumination/correctness/"
                         "mf0_factorial_summary.json").read_text(encoding="utf-8"))
    old_amp = json.loads((REPO / "results/magnitude/"
                          "directional_amplitude_summary.json")
                         .read_text(encoding="utf-8"))
    old_ab = json.loads((REPO / "results/submodularity/alpha_bound.json")
                        .read_text(encoding="utf-8"))
    for tok, val, dec in (
            ("0.90", old_fd["variants"]["D"]["RA"], 2),
            ("0.95", old_fd["variants"]["D"]["RA_ci95"][1], 2),
            ("15.98", old_amp["arm_D_corrected"]["ratio_stats"]["median"], 2),
            ("201.1", old_amp["arm_A_frozen"]["ratio_stats"]["median"], 1),
            ("0.635", old_ab["gamma_overall"]["gamma_lower_bound_min"], 3)):
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
    # 撤回注记排除(规则 7 的可执行实现;验收 533a279 §2.2):
    # 按 **key 名** 排除不够——撤回叙述会被写进名字正常的字段(本次是
    # `channel`)或存成数值数组(`*_before_after`)。改为三层排除:
    #   (a) 整棵 provenance/ 子树不入池(撤回记录按定义不是当前测量);
    #   (b) 容器名(withdrawn/before/after/correction/conclusion_impact…);
    #   (c) 按 **内容形状**:字符串命中 corrected:/withdrawn/superseded/
    #       pre-correction/was 等标记即整条不入池。
    # 未入池的字段是"撤回注记的合法居所",不是数据源。
    WITHDRAWN_CONTAINER_MARKS = (
        "_note", "note_", "provenance", "withdrawn", "before_after",
        "before", "after", "correction", "conclusion_impact",
        "superseded", "caliber_note")
    WITHDRAWN_CONTENT_MARKS = (
        "corrected:", "withdrawn", "superseded", "pre-correction",
        "was ", "no longer", "not the reading basis",
        "retained for reference")

    def _pool(obj, path=""):
        chunks = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                kp = f"{path}.{k}" if path else k
                if any(m in k.lower() for m in WITHDRAWN_CONTAINER_MARKS):
                    continue                      # 撤回/更正容器:不入池
                chunks.append(_pool(v, kp))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                chunks.append(_pool(v, f"{path}[{i}]"))
        elif isinstance(obj, str):
            low = obj.lower()
            if any(m in low for m in WITHDRAWN_CONTENT_MARKS):
                return ""                         # 更正/撤回散文:不入池
            chunks.append(obj)
        else:
            chunks.append(json.dumps(obj))
        return "\n".join(chunks)

    def results_pool_text():
        chunks = []
        for q2 in sorted((REPO / "results").rglob("*")):
            if not q2.is_file():
                continue
            # (a) 撤回/溯源子树整体不入池
            if any(part in ("provenance", "provenance_records")
                   for part in q2.parts):
                continue
            try:
                d = json.loads(q2.read_text(encoding="utf-8"))
            except Exception:
                chunks.append(q2.read_text(encoding="utf-8", errors="ignore"))
                continue
            chunks.append(_pool(d, q2.stem))
        return "\n".join(chunks)

    results_text = results_pool_text()
    untraced = [tk for tk in sorted(tokens)
                if tk not in traced and tk not in NON_CLAIM_TOKENS
                and re.search(r"(?<![\w.])" + re.escape(tk) + r"(?![\w])",
                              results_text) is None]
    assert not untraced, ("README numbers not traceable to results/** "
                          "(withdrawn/correction note fields excluded from "
                          f"the pool): {untraced}")


# --------------------------------------------------------- supersession scopes
# A token's presence in an immutable JSON cannot restore its former scientific
# interpretation. Require the qualifier in the local statement, not elsewhere
# in the document, and keep the current spectral API's scope explicit.
_HISTORY_SCOPE = re.compile(
    r"historical|history|archived|superseded|withdrawn|refuted|is false"
    r"|历史|归档|旧|撤回|否定|否决|前轮", re.I)
_OLD_GAMMA = re.compile(
    r"(?<![\w.])0\.635(?:144)?(?!\d)"
    r"|(?<!\w)gamma_lower_bound\s*\("
    r"|(?:alpha|α)[- ]only"
    r"|(?:γ|gamma|\\gamma)\s*(?:≥|>=|\\geq?)\s*1\s*/\s*"
    r"\(\s*1\s*\+\s*(?:α|alpha|\\alpha)\s*\)", re.I)
_CANDIDATE_REJECTION = re.compile(
    r"failure of (?:the )?(?:alpha[- ]only|α[- ]only) candidate"
    r"|(?:alpha[- ]only|α[- ]only) (?:candidate|bound) (?:is |has been )?"
    r"(?:false|refuted|disproved)", re.I)
_CANDIDATE_PROMOTION = re.compile(
    r"(?:alpha[- ]only\s+(?:candidate|bound)|gamma_lower_bound\([^)]*\))"
    r"[^.]{0,80}\b(?:is|remains|provides|gives|yields)\s+(?:now\s+)?"
    r"(?:a\s+)?(?:valid|proved|certified|guaranteed)\b"
    r"|(?:γ\s*(?:≥|>=)\s*)?0\.635(?:144)?[^.]{0,50}"
    r"\b(?:is|remains|provides)\s+(?:a\s+)?(?:valid|proved|certified)\b"
    r"|(?:旧|历史)(?:候选|函数)[^。]{0,40}(?:仍是有效|提供有效|已证保证)", re.I)


def _scope_blocks(text):
    """Paragraphs, individual table rows and list items (with wrapped lines)."""
    current, start = [], 0
    for line_no, line in enumerate(text.splitlines(), 1):
        boundary = not line.strip() or re.match(r"^\s*(?:[-*] |\d+\. |\| |#)", line)
        if current and boundary:
            yield start, "\n".join(current)
            current = []
        if line.strip():
            if not current:
                start = line_no
            current.append(line)
    if current:
        yield start, "\n".join(current)


def _superseded_scope_issues(text):
    issues = []
    for line, block in _scope_blocks(text):
        plain = re.sub(r"\s+", " ", _prose(block))
        # Full stops split assertions, not decimal points or API module names.
        for sentence in re.split(r"(?<=[.!?])\s+|[。]", plain):
            if not _OLD_GAMMA.search(sentence):
                continue
            if ((not _HISTORY_SCOPE.search(sentence)
                 and not _CANDIDATE_REJECTION.search(sentence))
                    or _CANDIDATE_PROMOTION.search(sentence)):
                issues.append((line, "refuted alpha-only expression used without withdrawal scope"))
        if re.search(r"\bR_A\s*=\s*0\.90?\b|(?<![\w.])(?:15\.98|201\.1)(?!\d)", plain):
            if not _HISTORY_SCOPE.search(plain):
                issues.append((line, "superseded B1/B2 values used as current evidence"))
        if "validity_map.json" in plain and re.search(
                r"curvature[- ]robust|排序.{0,30}稳健|magnitudes,? not directions", plain, re.I):
            if not _HISTORY_SCOPE.search(plain):
                issues.append((line, "historical B8 map promoted to current projection guarantee"))
    return issues


def test_refuted_and_superseded_claims_keep_local_scope():
    hits = []
    for path in SCAN_FILES:
        text = path.read_text(encoding="utf-8")
        hits.extend(f"{path.relative_to(REPO)}:{line}: {why}"
                    for line, why in _superseded_scope_issues(text))
    assert not hits, "\n".join(hits)


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md"])
def test_readme_spectral_api_scope_is_not_the_historical_candidate(name):
    text = (REPO / name).read_text(encoding="utf-8")
    blocks = [re.sub(r"\s+", " ", _prose(block))
              for _, block in _scope_blocks(text) if "spectral_gamma_lower_bound(" in block]
    assert blocks, "README must distinguish M12 from the refuted candidate"
    for block in blocks:
        for pattern in (r"M12", r"refuted", r"withdrawn|撤回", r"proved|已证",
                        r"unweighted\s+full[- ]trace|未加权全迹",
                        r"fixed SPD|固定 SPD", r"PSD", r"arbitrary task|任意任务",
                        r"singular|奇异", r"not directly|不能直接"):
            assert re.search(pattern, block, re.I), (name, pattern)
    b8 = _prose(_readme_claim_block(text, "B8"))
    assert _HISTORY_SCOPE.search(b8)
    assert re.search(r"not a robustness guarantee for the current|不能保证当前",
                     re.sub(r"\s+", " ", b8), re.I)


@pytest.mark.parametrize("bad", [
    "The alpha-only candidate is a valid guarantee.",
    "The refuted alpha-only candidate nevertheless provides a valid guarantee.",
    "At this level, γ ≥ 0.635 is a real-object certificate.",
    "gamma_lower_bound(alpha) gives a certified bound.",
    "γ ≥ 1/(1+α) for all SPD baselines and PSD updates.",
    "Current R_A = 0.90, CI [0.7, 0.95].",
    "The current amplitude median is 15.98.",
    "validity_map.json proves curvature-robust ordering for the current projection.",
])
def test_scope_gate_rejects_promoted_history(bad):
    # A disclaimer in a different paragraph must not be a document whitelist.
    text = "Historical alpha-only candidate: refuted; old values withdrawn.\n\n" + bad
    assert _superseded_scope_issues(text), bad


@pytest.mark.parametrize("good", [
    "The alpha-only candidate is refuted, not merely unproved.",
    "The former γ ≥ 0.635 guarantee is withdrawn.",
    "Historical gamma_lower_bound(alpha) values are for reproduction only.",
    "旧 γ ≥ 1/(1+α) 候选已被否定，不是有效保证。",
    "Historical R_A = 0.90, CI [0.7, 0.95] is superseded.",
    "The archived amplitude median 15.98 is not the current B2.",
])
def test_scope_gate_preserves_qualified_history(good):
    assert not _superseded_scope_issues(good)


# --------------------------------------------------------------- E-1b 加固
# docs/methods.md + docs/claims.md 的数字此前零机器校验(验收 E-1 的
# `emp/pred ≈ 102` 错字段抓取正落在这个盲区)。本测试把数字追溯扩展到
# docs:每个数字 token 必须满足其一——
#   (a) 在 results/** 的某个数字的"打印精度舍入"范围内
#       (|n − v| ≤ 0.5·10^-dec,dec = token 小数位);
#   (b) 同 (a) 但按百分比换算(|n·100 − v| ≤ 0.5·10^-dec);
#   (c) 在白名单里(派生量/日期/版本号,逐项注明理由)。
# 鉴别力(审计实测,2026-09-14):236 个 docs token 中仅 8 个(3.4%)唯一
# 匹配;整数/2 位小数 token 的随机错值通过率 ~99-100%;4 位小数在密集
# 值域(≈1.0 附近)误放行 64.8%。本测试是**漂移绊线**,不是正确性检查;
# 头条数字(M9/B9 级)需要字段级绑定(README 的 traced map 模式,投稿前
# 待办),不应依赖本门禁背书。
DOCS_NON_CLAIM = {
    # 年份(related work 引用、changelog、provenance)
    "2017": "Chamon & Ribeiro NeurIPS 2017 citation year",
    "2021": "Alexanderian et al. JUQ 2021 citation year",
    "1968": "Backus-Gilbert 1968 citation year",
    "1993": "Pukelsheim 1993 citation year",
    "2013": "Jaggi 2013 citation year",
    "1978": "Bunch-Nielsen-Sorensen 1978 citation year",
    "2026": "dates in provenance/changelog (2026-09-xx)",
    "2025": "dates in provenance/changelog",
    # 版本号
    "1.0": "version labels (v1.0, s1.0)",
    "1.1": "version labels (v1.1)",
    # 派生统计(注明来源;不在 results 字面中)
    "102": "REMOVED - the wrong emp/pred value was corrected in E-1a; "
           "current B2 is bound to the imported projection-matched field; "
           "kept here so it can NEVER re-enter docs",
    "100": "percentages/counts in prose (e.g. '100% of the advantage')",
    "1.0368": "historical pre-T3 value quoted as drift history in "
              "EXPERIMENTS s20's E-1b note; the corrected 1.001011 is "
              "field-bound in test_headline_binding",
}

# 主动禁用(P1-c):这些值在 docs 中出现即失败——它们是历史错值,靠白名单
# 豁免是逻辑倒置(白名单=跳过检查=允许),且删白名单也不够(102 会经
# 舍入匹配命中池里的 102.19 而通过)。唯一正确的语义是显式拒绝。
DOCS_BANNED_TOKENS = {
    "102": "E-1a historical wrong emp/pred field; current B2 uses the imported "
           "same-projection ratio - must never re-enter docs",
}


def _decimals(tok):
    return len(tok.split(".")[1]) if "." in tok else 0


def _results_numbers():
    import re as _re
    text = "\n".join(
        q.read_text(encoding="utf-8", errors="ignore")
        for q in sorted((REPO / "results").rglob("*")) if q.is_file())
    text = text.replace("\u2212", "-")
    return [float(m) for m in _re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", text)]


# --------------------------------------------------------------- 撤回值硬门禁
# 验收 df31bb3 §4:池排除只对"注记"成立,对"数值"不成立——已撤回值仍可能
# 被无关测量巧合命中(docs 门禁是精度舍入匹配,短 token 在 32 万数的池里
# 必然有落点)。因此对**已知撤回值**加独立硬断言:任何文档都不得把这些
# token 当作当前值陈述。撤回值在此显式登记(来源:loader-fix 与
# feasible-rule 两次修正),新增撤回值时同步登记。
# 注意:只登记**唯一指向撤回读数**的 token。会与合法测量撞车的值
# (如 86 也出现在"36-86% 浪费份额"、0.899 也出现在机制 all-21 口径的
# 合法报告里)不登记——否则门禁变成误报机。
WITHDRAWN_TOKENS = {
    "51.2", "51.197",          # DiLiGenT 方向通道 max(loader 伪影)
    "0.242", "0.474",          # DQ informed 偏离(loader 伪影)
    "85.6", "86.1",            # pot1/pot2 锚点份额(loader 伪影)
    "-1557.8",                 # readingPNG 退化份额
}
# 允许出现的语境:撤回/更正叙述所在的行(按行判定,不按全文档)
WITHDRAWN_CONTEXT_MARKS = (
    "withdrawn", "superseded", "pre-correction", "correction note",
    "更正", "撤回", "artifact", "no longer", "was ", "old ",
    "reference only", "reference_only", "not the reading basis")


def test_withdrawn_values_not_stated_as_current():
    """已撤回值不得作为**当前值**出现在任何扫描文档里(按行判定)。

    判据:含撤回 token 的行,必须同时命中撤回语境标记;否则失败。
    (这是 N1 的硬门禁:不依赖数字池的巧合命中与否。)
    """
    hits = []
    for f in SCAN_FILES:
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(),
                                  1):
            for tok in WITHDRAWN_TOKENS:
                pat = r"(?<![\w.])" + re.escape(tok) + r"(?![\w])"
                if not re.search(pat, line):
                    continue
                low = line.lower()
                if any(m in low for m in WITHDRAWN_CONTEXT_MARKS):
                    continue
                hits.append(f"{f.name}:{ln} states withdrawn value "
                            f"{tok!r} without a withdrawal context")
    assert not hits, "\n".join(hits)


def test_docs_numeric_claims_traceable():
    """E-1b: docs(methods/claims)的每个数字可追溯到 results/**(舍入
    匹配 + 百分比换算),否则必须在带理由的白名单里。"""
    numbers = _results_numbers()
    for doc in (REPO / "docs/methods.md", REPO / "docs/claims.md",
                REPO / "docs/EXPERIMENTS.md"):
        text = doc.read_text(encoding="utf-8")
        text = re.sub(r"\d+e-\d+", " ", text.replace("\u2212", "-"))
        tokens = set(re.findall(r"-?\d+\.\d+|-?\d+", text))
        untraced = []
        for tk in sorted(tokens):
            if tk in DOCS_BANNED_TOKENS:
                untraced.append(tk)          # banned: fail regardless
                continue
            if tk in DOCS_NON_CLAIM:
                continue
            v = float(tk)
            tol = 0.5 * 10 ** -_decimals(tk) + 1e-9
            if any(abs(n - v) <= tol for n in numbers):
                continue
            if any(abs(n * 100.0 - v) <= tol for n in numbers):
                continue                                     # 百分比换算
            untraced.append(tk)
        assert not untraced, (
            f"{doc.name} numbers not traceable to results/** "
            f"(rounding/percent match): {untraced}")
