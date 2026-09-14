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
