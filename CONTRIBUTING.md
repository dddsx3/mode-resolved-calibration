# Contributing

Thanks for your interest in improving `mode-resolved-calibration`. Short
version: open an issue first for anything semantic; keep PRs small; make sure
`pytest` is green.

## Ground rules

1. **Committed evidence is immutable.** The benchmark artifacts under
   `results/` reproduce published numbers — never edit them. Every committed
   file is pinned by `checksums.sha256` (CI verifies), so any artifact change
   is visible in the diff and must be accompanied by a checksum regeneration
   plus a re-derived test.
2. **Single source of truth.** The math lives once, in `src/calibinfo`
   (docstrings carry the normative definitions; `docs/methods.md` is the
   prose statement). Docs, examples, and papers reference it — they never
   restate formulas independently.
3. **Numeric claims are gated.** `tests/test_claims_gate.py` scans the repo
   for overclaiming sentence patterns; `docs/claims.md` is the
   claim-to-evidence registry. New numeric claims must trace to a file under
   `results/` and be registered there.
4. **Experimental code is quarantined.** New analysis capabilities go into
   `experiments/` with a header
   `# status: experimental — NOT part of the published results`, and never
   into `src/calibinfo/` (the published library path).
5. **Any claim of an advantage over random must carry the active-set
   control.** Comparing against universe-random measures VISIBILITY, not
   selection quality: a policy that cannot see which lights illuminate
   the object is indistinguishable from universe-random for that reason
   alone (C8). Every such claim must also report the active48-restricted
   random baseline (permute the illuminating set only), and baselines
   whose candidate pool spans all lights must either be restricted to
   the active set or report their active-set overlap. This rule was
   learned twice (the E-arm dAUC headline, C12 v1's DC05 baseline) — do not
   learn it a third time. The E-arm budget-grid saturation (acceptance
   9515039) is the same lesson in its budget-axis form: k > |active|
   makes the two sides select identical sets, so cross-budget statistics
   must be restricted to the feasible interval or flagged degenerate.
   **A rule is enforced globally or not written at all** (acceptance
   665d01e section 7): every artifact that uses a cross-budget /
   cross-sample integrated statistic must carry the degeneracy check
   for EVERY budget it uses -- new artifacts in-file, already-committed
   artifacts via a zero-cost sidecar diagnostic with its own gate
   (`decision_quality_feasible.json` for the E arm, whose budgets
   57/85/114 > 48 die entirely; `active_set_ablation_feasible.json`
   for the C8 ablation, whose k=48 = |active| kills only the ordering
   sub-term). `tests/test_reproduce_coverage.py` keeps the reproduction
   entry point covering every registered experiment; the same
   no-silent-locality principle applies to it.

## Development setup

```bash
git clone https://github.com/dddsx3/mode-resolved-calibration.git
cd mode-resolved-calibration
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev,examples]
pytest
```

### Regenerating `checksums.sha256`

Run `scripts/make_checksums.sh` **only when the working tree is
byte-identical to what you are about to commit** — any unstaged
modification leaks into the hashes and silently desynchronises the ledger
(this bit us twice: unstaged docs were hashed while only code was staged,
so the committed `checksums.sha256` did not verify against its own commit).
Safe sequence: `git stash` (the files you are *not* committing) →
regenerate → `git add` → `git stash pop`; then verify with a clean
`git worktree add /tmp/wt HEAD && (cd /tmp/wt && sha256sum -c
checksums.sha256)` before pushing.

### Drift annotations never go into frozen artifacts

Provenance annotations for an already-committed artifact (config drift,
supersession notes, hash clarifications) belong in a SIDEcar - this file,
`docs/REPRODUCIBILITY.md`, or a fresh provenance JSON - never edited into
the artifact itself. Hand-editing a frozen artifact changes its sha256 and
silently invalidates every pin that references it (acceptance P2-7: the
hand-added manifest annotation shifted the blob so the reuse-record's
v1_artifact_sha256 only resolved to git history, not the working tree).
Regenerate-and-recommit via the producing script if the annotation must
live in the artifact.

## Pull requests

- one purpose per PR; reference the issue it closes;
- add or extend a known-answer test for any new numerical capability
  (`tests/test_allocation_known_answers.py` is the pattern);
- `pytest` green, and the wording gate green;
- update `CHANGELOG.md` under the *Unreleased* heading.
