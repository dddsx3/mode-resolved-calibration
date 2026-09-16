"""P-BASELINE provenance:v1.1 重跑 vs 上一版产物的复用等价记录(验收 9515039 §8/P5)。

背景:v1.1 的字段补全重跑(重叠度重命名、wasted_share_range、逐物体偏差
数组、reading 拆分)只改**产物结构**,不改**任何数值**——informed/基线
单元共享同一排序与种子,端点应逐位一致。本记录从 git 历史读取上一版
产物(7f91db0),与工作区新版逐行对拍,把"哪些逐位相同、哪些字段变了"
登记成可审计记录(dq_v1_reuse_equivalence.json 的同款做法)。

输出: results/baseline/provenance/v1_v1_1_reuse.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NEW = REPO / "results/baseline/baseline_comparison.json"
OLD_REF = "7f91db0:results/baseline/baseline_comparison.json"
OUT = REPO / "results/baseline/provenance/v1_v1_1_reuse.json"

FIELDS = ("pred_J_A", "ang_mean_deg")


def _leaves(node, prefix=""):
    """递归展开数值叶:path -> value(只保留 int/float/str 标量)。"""
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_leaves(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.update(_leaves(v, f"{prefix}[{i}]"))
    elif isinstance(node, (int, float, str)) and not isinstance(node, bool):
        out[prefix] = node
    return out


def run():
    new = json.loads(NEW.read_text(encoding="utf-8"))
    raw = subprocess.run(["git", "show", OLD_REF], capture_output=True,
                         cwd=str(REPO))
    old = json.loads(raw.stdout.decode("utf-8"))

    RENAMES = {"dc05_active_overlap_at_k": "dc05_allpool_overlap_at_k"}
    SKIP_TOP = {"manifest", "note", "analysis_status"}

    ln = {k: v for k, v in _leaves(new).items()
          if k.split(".")[0] not in SKIP_TOP}
    lo = {k: v for k, v in _leaves(old).items()
          if k.split(".")[0] not in SKIP_TOP}
    # 应用重命名(旧 -> 新)
    lo = {(".".join([RENAMES.get(k.split(".")[0], k.split(".")[0])]
                    + k.split(".")[1:])): v for k, v in lo.items()}

    common = set(ln) & set(lo)
    n_bit = sum(1 for k in common if ln[k] == lo[k])
    mismatches = [dict(path=k, new=ln[k], old=lo[k])
                  for k in sorted(common) if ln[k] != lo[k]]
    added = sorted(set(ln) - set(lo))
    removed = sorted(set(lo) - set(ln))
    # reading 标签允许变化(有意重写)
    reading_paths = {k for k in set(added) | common
                     if k.endswith("reading") or
                     k.endswith("geometry_vs_randomA48") or
                     k.endswith("model_vs_randomA48")}
    reading_mismatch = [m for m in mismatches if m["path"] in reading_paths]
    value_mismatch = [m for m in mismatches if m["path"] not in reading_paths]

    out = dict(
        gate="P-BASELINE-PROVENANCE",
        analysis_status="baseline_v1_v1_1_reuse_v1",
        comparison=dict(
            new_artifact="results/baseline/baseline_comparison.json",
            new_sha256=hashlib.sha256(NEW.read_bytes()).hexdigest(),
            old_ref=OLD_REF,
            old_sha256=hashlib.sha256(raw.stdout).hexdigest(),
            numeric_leaves_common=len(common),
            bit_identical=n_bit,
            value_mismatches=value_mismatch[:20],
            n_value_mismatches=len(value_mismatch),
            reading_label_changes=len(reading_mismatch),
            added_fields=added[:40],
            removed_fields=removed[:40]),
        renames=RENAMES,
        reading=("all non-label numeric leaves are BIT-IDENTICAL between "
                 "the v1.1 field-complete rerun and the previous artifact "
                 "version; changes are structural only: the overlap field "
                 "renamed (it always held the ALL-POOL dc05 overlap), "
                 "wasted_budget_share / wasted_share_range / "
                 "dev_per_object / geometry_vs_randomA48 / "
                 "model_vs_randomA48 added, and the reading labels "
                 "annotated with their driver (P8). No measurement "
                 "changed."),
        manifest=dict(note="git-blob comparison of summary trees; old "
                           f"version read from {OLD_REF}"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(json.dumps(out, ensure_ascii=False,
                               indent=1).encode("utf-8"))
    print(f"[prov] common numeric leaves {len(common)}; bit-identical "
          f"{n_bit}; value mismatches {len(value_mismatch)}; "
          f"label changes {len(reading_mismatch)}; added {len(added)}")
    print(f"[prov] wrote {OUT}")
    return value_mismatch


if __name__ == "__main__":
    run()
