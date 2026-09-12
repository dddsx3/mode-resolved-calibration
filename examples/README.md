# Examples

Self-contained scripts — each generates its own synthetic data and figures.
Run from the repository root after `pip install -e .[examples]`:

```bash
python examples/ex1_retention_gauge.py     # retention spectrum + gauge response
python examples/ex2_mode_vs_scalar.py      # mode-resolved vs scalar criteria
python examples/ex3_allocation_demo.py     # minimal calibration-allocation demo
python examples/ex4_calibration_tour.py   # two-question guided tour (fragile modes + certified budget value)
```

Each script prints the path of the PNG it writes. No downloads, no benchmark
data, no network access.

| Script | What it shows | Output |
|---|---|---|
| `ex1_retention_gauge.py` | per-mode retention spectrum across the calibration path; closed-form gauge spectral response with slope/saturation | `retention_gauge.png` |
| `ex2_mode_vs_scalar.py` | predicted damage on the fragile subspace: mode-resolved (weakest retained mode) vs trace- and log-det-based criteria under a growing nuisance prior | `mode_vs_scalar.png` |
| `ex3_allocation_demo.py` | a tiny version of the calibration-allocation experiment: which lights to recalibrate first, mode-aware vs E/A-opt greedy vs random | `allocation_demo.png` |
| `ex4_calibration_tour.py` | the two questions of the whole project on one figure: which albedo directions are fragile (retention modes), and what a recalibration budget buys (certified landscape with a convex lower bound) | `calibration_tour.png` |

These examples use the library API only (`src/calibinfo`). The frozen research
benchmark under `results/` is separate — see `docs/EXPERIMENTS.md` and
`docs/REPRODUCIBILITY.md`.
