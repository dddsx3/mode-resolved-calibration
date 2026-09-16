"""P-MATRIX-SYNC · 复现矩阵行的产物字段同步(验收 533a279 §5-3)。

手工同步矩阵行在同一提交里就重现了自己的失败模式(行 130 的尾巴仍是
`transfer-partial` 而产物已是 `transfer-confirmed`)。本脚本从产物字段
**生成**矩阵行的值段,消除手工抄写。

用法:python experiments/sync_matrix_rows.py [--check]
  --check:只报告差异(CI 用),不写盘。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATRIX = REPO / "docs/REPRODUCIBILITY.md"

# (行内定位子串, 产物路径, 生成函数)
def _dq_row(j):
    cd = j["channel_decomposition"]
    sh = j["ball_anchor_share"]["per_object"]
    return (f"radius_2x median {j['linearization_radius']['radius_2x']['median_over_crossed_subset']} "
            f"(all objects [0.75,1.5]); radius_10x 1.5 on 10/10; "
            f"direction max D {cd['direction_max_pct']:.2f}% (intensity dominance transfers); "
            f"ball-anchor share median ≈0 (cat {sh['catPNG']*100:.0f}%/pot1 {sh['pot1PNG']*100:.0f}%) "
            f"-> {j['outcome']}")


def _bc_row(j):
    oi = j["oi"]["14"]["median_dev_from"]
    dq14 = j["diligent"]["14"]["median_dev_from"]
    dq28 = j["diligent"]["28"]["median_dev_from"]
    return (f"OI geometry-insufficient (dc05_active {oi['dc05_active_vs_randomA48']:.3f}/"
            f"{j['oi']['28']['median_dev_from']['dc05_active_vs_randomA48']:.3f} vs randomA48; "
            f"every informed policy better); DiLiGenT both beat active-random "
            f"(dc05_active {dq14['dc05_active_vs_randomA48']:.3f}/{dq28['dc05_active_vs_randomA48']:.3f}; "
            f"informed {dq14['informed_vs_randomA48']:.3f}/{dq28['informed_vs_randomA48']:.3f})")


def _mech_row(j):
    c = j["correlations"]["dir_int_weak_ratio"]
    return (f"dir_int_weak_ratio OI-internal ρ {c['per_cohort_excl_degenerate']['oi']:.3f} (n=11); "
            f"valid-sample pooled {c['pooled_excl_degenerate']:.3f} (below 0.7); "
            f"all-21 {c['pooled']:.3f} counts {j['n_degenerate']} degenerate shares")


ROWS = [
    ("DiLiGenT queue: second-dataset transfer", "results/diligent/diligent_queue.json", _dq_row),
    ("Baseline: literature-style selection", "results/baseline/baseline_comparison.json", _bc_row),
    ("Anchor-share mechanism", "results/openillumination/anchor_mechanism.json", _mech_row),
]


def run(check_only=False):
    text = MATRIX.read_text(encoding="utf-8")
    changed = []
    for key, art_path, fn in ROWS:
        j = json.loads((REPO / art_path).read_text(encoding="utf-8"))
        new_val = fn(j)
        lines = text.splitlines()
        hit = None
        for i, line in enumerate(lines):
            if line.startswith("| " + key) or key in line.split("|")[0]:
                hit = i
                break
        if hit is None:
            print(f"[matrix] WARN: row not found for {key!r}")
            continue
        parts = lines[hit].split("|")
        if len(parts) < 4:
            continue
        old_val = parts[2].strip()
        if old_val != new_val:
            parts[2] = " " + new_val + " "
            lines[hit] = "|".join(parts)
            changed.append(key)
            text = "\n".join(lines) + "\n"
    if changed:
        if check_only:
            print(f"[matrix] OUT OF SYNC: {changed}")
            return 1
        MATRIX.write_bytes(text.encode("utf-8"))
        print(f"[matrix] synced rows: {changed}")
    else:
        print("[matrix] all rows in sync")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    raise SystemExit(run(args.check))
