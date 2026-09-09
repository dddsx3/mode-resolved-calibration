# Contributing

Thanks for your interest in improving `mode-resolved-calibration`. Short
version: open an issue first for anything semantic; keep PRs small; make sure
`pytest` is green.

## Ground rules

1. **The frozen benchmark is immutable.** `results/`, `configs/`, and
   `tests/test_reproduction.py` reproduce published numbers — never edit them.
   CI fails if `git diff science-closed HEAD` touches those paths.
2. **Single source of truth.** The math lives once, in `src/calibinfo`
   (docstrings carry the normative definitions). Docs, examples, and papers
   reference it — they never restate formulas independently.
3. **Claims are gated.** `tests/test_wording_gate.py` scans the repo for
   overclaiming sentence patterns; `docs/WORDING.md` is the approved wording
   registry. New numeric claims must trace to a file under `results/`.
4. **Experimental code is quarantined.** New analysis capabilities go into
   `src/calibinfo/experimental/` with a header
   `# status: experimental — NOT part of the published results`, and never
   into the benchmark path.

## Development setup

```bash
git clone https://github.com/dddsx3/mode-resolved-calibration.git
cd mode-resolved-calibration
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev,examples]
pytest
```

## Pull requests

- one purpose per PR; reference the issue it closes;
- add or extend a known-answer test for any new numerical capability
  (`tests/test_allocation_known_answers.py` is the pattern);
- `pytest` green, and the wording gate green;
- update `CHANGELOG.md` under the *Unreleased* heading.
