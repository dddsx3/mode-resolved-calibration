"""P-RESUME-EQUIV · 同机 worker-不变性 / crash-resume 等价的入库证据(P1-d)。

背景:§20 声称"同机 worker-不变性由 resume-vs-clean 位级测试单独证明",
但该证据只存在于 42bc299 的 commit message 里(验收 P1-d:一条只靠散文
支撑的声明,恰好撑着 reuse 决策、云端策略和 worker 并行度的合法性)。

本脚本把它变成可复跑的入库证据:
  1. **interrupted run**:2 物体缩减网格,主进程在第一个 checkpoint
     (obj_03)落盘后被 SIGKILL(模拟崩溃);
  2. **resumed run**:同 checkpoint 目录重启——obj_03 从 checkpoint
     恢复(不重算),obj_19 新算,聚合写出产物 B;
  3. **clean run**:同网格、同 checkpoint 目录(空)从头跑,写出产物 C;
  4. **比对**:B vs C 逐字段(rows 四量、bootstrap、spearman、
     informed、auc_rows)必须**位级一致**;同时断言 resumed run 确实
     复用了 checkpoint(运行日志含 "resumed from checkpoint")。

缩减网格(与 v1.1 同代码路径,只有规模缩小;等价性论断针对的是
checkpoint-resume 机制,不依赖网格大小)。

用法:
  python experiments/verify_dq_resume_equivalence.py \
      [--out results/openillumination/provenance/dq_resume_vs_clean.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FIELDS = ("pred_J_A", "ang_mean_deg", "mse_aligned", "dual_mean")
TOP_KEYS = ("bootstrap_dAUC", "spearman_pred_vs_realized",
            "spearman_informed_only", "informed_pairwise_sign_pooled",
            "auc_rows")

CFG = dict(
    cohort=["obj_03_pumpkin", "obj_19_cylinder"],
    levels=[0.5], regimes=[10], budgets_k=[14], budget_fractions=[0.1],
    policies=["mode_aware", "a_opt"],
    random_units={"universe": 1, "active48": 1},
    seeds_per_level=2, bootstrap={"B": 200, "seed": 20260910},
    workers=1,
    data_root="D:/data/OpenIllumination",
    data_meta="D:/data/OpenIllumination_meta",
    noise_fit_convention="corrected",
    scene_rng_spec="default_rng([20260910, obj_idx])",
    K_lights=142, gate="P-RESUME-EQUIV-INTERNAL",
    analysis_status="resume_equiv_internal",
)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(tag, ckpt, out_json, kill_after_first=False):
    """跑一次 decision_quality(缩减 config),返回 (log_text, exit_code)。"""
    import yaml
    cfg = dict(CFG)
    cfg_path = ckpt.parent / f"cfg_{tag}.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"),
                   allow_unicode=True)
    log_path = ckpt.parent / f"log_{tag}.txt"
    cmd = [sys.executable, "-u", str(REPO / "experiments/decision_quality.py"),
           "--config", str(cfg_path), "--out", str(out_json),
           "--img", str(ckpt.parent / f"img_{tag}.png"),
           "--ckpt-dir", str(ckpt)]
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=str(REPO))
        if kill_after_first:
            # 等第一个 checkpoint 出现(obj_03 完成约 8-10 min)后 SIGKILL
            deadline = time.time() + 3600
            while time.time() < deadline:
                if (ckpt / "obj_03_pumpkin.json").exists():
                    time.sleep(2)          # 让 rename 完全落盘
                    proc.kill()
                    break
                if proc.poll() is not None:
                    break
                time.sleep(5)
            proc.wait()
        else:
            proc.wait()
    return log_path.read_text(encoding="utf-8"), proc.returncode


def run(out_path=REPO / "results/openillumination/provenance/"
                        "dq_resume_vs_clean.json"):
    t0 = time.time()
    out_path = Path(out_path)
    stage = out_path.parent / ".resume_equiv_stage"
    stage.mkdir(parents=True, exist_ok=True)

    # 1) interrupted
    ckpt = stage / "ck_shared"
    if ckpt.exists():
        for f in ckpt.glob("*"):
            f.unlink()
    else:
        ckpt.mkdir()
    log_i, rc_i = _run("interrupted", ckpt, stage / "b_partial.json",
                        kill_after_first=True)
    assert (ckpt / "obj_03_pumpkin.json").exists(), \
        "interrupted run did not produce the first checkpoint"
    assert rc_i != 0 or "obj_19" not in log_i, rc_i

    # 2) resumed(同 checkpoint 目录)
    log_r, rc_r = _run("resumed", ckpt, stage / "b_resumed.json")
    assert rc_r == 0, log_r[-2000:]
    assert "resumed from checkpoint" in log_r, \
        "resumed run did NOT reuse the checkpoint"
    resumed_objs = [l.split()[1] for l in log_r.splitlines()
                    if "resumed from checkpoint" in l]

    # 3) clean(空 checkpoint 目录)
    ckpt_c = stage / "ck_clean"
    if ckpt_c.exists():
        for f in ckpt_c.glob("*"):
            f.unlink()
    else:
        ckpt_c.mkdir()
    log_c, rc_c = _run("clean", ckpt_c, stage / "c_clean.json")
    assert rc_c == 0, log_c[-2000:]

    # 4) bit-exact comparison
    b = json.loads((stage / "b_resumed.json").read_text(encoding="utf-8"))
    c = json.loads((stage / "c_clean.json").read_text(encoding="utf-8"))
    rb = {(x["object"], x["level"], x["regime"], x["unit"], x["k"]): x
          for x in b["rows"]}
    rc = {(x["object"], x["level"], x["regime"], x["unit"], x["k"]): x
          for x in c["rows"]}
    assert set(rb) == set(rc)
    n = ok = 0
    for key in rb:
        for f in FIELDS:
            n += 1
            ok += int(rb[key][f] == rc[key][f])
    top_same = {k: (b[k] == c[k]) for k in TOP_KEYS}

    record = dict(
        gate="P-RESUME-EQUIV",
        analysis_status="dq_resume_vs_clean_v1",
        question="is a crash-resumed run (checkpoint reuse) bit-identical "
                 "to an uninterrupted run of the same grid, on the same "
                 "machine - i.e. same-machine worker/crash invariance?",
        method="2-object reduced grid (v1.1 code path): run A killed "
               "with SIGKILL after the first per-object checkpoint lands; "
               "run B restarted from the same checkpoint dir (object 1 "
               "resumed, object 2 computed); run C clean from scratch; "
               "B vs C compared field-by-field",
        resumed_objects=resumed_objs,
        rows=dict(n_total=n, n_bit_exact=ok),
        top_level_identical={k: bool(v) for k, v in top_same.items()},
        artifacts=dict(
            resumed_sha256=_sha(stage / "b_resumed.json"),
            clean_sha256=_sha(stage / "c_clean.json")),
        grid=CFG,
        manifest=dict(git_sha=_git_sha(),
                     elapsed_s=round(time.time() - t0, 1)),
        note="in-repo promotion of the evidence previously recorded only "
             "in 42bc299's commit message (acceptance P1-d). Same-machine "
             "property; cross-machine is covered by dq_cross_env_repro.json. "
             "Reduced grid: the claim is about the checkpoint-resume "
             "mechanism, which is grid-size independent (per-object "
             "seeding + atomic per-object serialization).")
    verdict = (ok == n and all(top_same.values())
               and "obj_03_pumpkin" in resumed_objs)
    record["verdict"] = "bit-identical" if verdict else "FAILED"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(json.dumps(record, ensure_ascii=False,
                                    indent=1).encode("utf-8"))
    # 清理暂存(保留 record)
    for f in stage.glob("*"):
        f.unlink()
    stage.rmdir()
    print(f"rows {ok}/{n} bit-exact; top-level identical: {top_same}; "
          f"resumed: {resumed_objs}")
    print(f"verdict: {record['verdict']}; wrote {out_path}")
    if verdict is False:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        REPO / "results/openillumination/provenance/"
               "dq_resume_vs_clean.json"))
    a = ap.parse_args()
    run(a.out)
