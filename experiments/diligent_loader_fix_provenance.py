"""P-DILIGENT-LOADER-FIX provenance:归一化修正的前后对照(缺陷报告判据 #3)。

记录(全部可复算):
  1. 决定性实测:同一 48 灯子集、同一 1200 像素、同一估计器,只改图像
     归一化 → 名义重建 vs GT 的中位/mean 角误差(未除光强 vs 除光强);
  2. 队列产物前后对照(半径、通道分解、球锚点份额),从 git blob 读取
     修正前版本与工作区新版对比;
  3. 结论影响标注:哪些主张加固、哪些被修正。

输出: results/diligent/provenance/loader_normalization_fix.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

DATA = Path("D:/data/DiLiGenT/pmsData")
OLD_REF = "29f9b9d:results/diligent/diligent_queue.json"
NEW = REPO / "results/diligent/diligent_queue.json"
OUT = REPO / "results/diligent/provenance/loader_normalization_fix.json"
GT_OBJECTS = ("ballPNG", "catPNG", "buddhaPNG", "pot1PNG")


def _nominal_vs_gt(objects):
    """名义重建(scen.n) vs 数据集 GT 法向:两条归一化路径的中位/mean。"""
    from experiments.openillumination_validation import NominalScene
    out = {}
    for name in objects:
        d = DATA / name
        N_gt = sio.loadmat(d / "Normal_gt.mat")["Normal_gt"]
        mask = np.array(Image.open(d / "mask.png").convert("L")) > 128
        ngt = N_gt[mask]
        dirs = np.loadtxt(d / "light_directions.txt")
        ints = np.loadtxt(d / "light_intensities.txt")[:, 0]
        K = dirs.shape[0]
        imgs = np.stack([np.array(Image.open(str(d / f"{i:03d}.png"))
                                  .convert("L")).astype(float)
                         for i in range(1, K + 1)])
        row = {}
        for tag, arr in (("unnormalized", imgs / 255.0),
                         ("intensity_normalized", imgs / ints[:, None, None])):
            obj = dict(images=arr[..., None].repeat(3, axis=-1), mask=mask,
                       light_directions=dirs, meta={})
            scen = NominalScene(obj, np.random.default_rng([20260915, 0]),
                                noise_fit_convention="corrected",
                                n_lights_total=96)
            gt = ngt[scen.pidx]
            ang = np.degrees(np.arccos(np.clip((scen.n * gt).sum(1), -1, 1)))
            row[tag] = dict(median_deg=round(float(np.median(ang)), 2),
                            mean_deg=round(float(ang.mean()), 2))
        out[name] = row
    return out


def run():
    old = json.loads(subprocess.run(
        ["git", "show", OLD_REF], capture_output=True,
        cwd=str(REPO)).stdout.decode("utf-8"))
    new = json.loads(NEW.read_text(encoding="utf-8"))

    def q(a, path):
        cur = a
        for p in path:
            cur = cur[p]
        return cur

    radius = dict(
        radius_2x_median=[q(old, ("linearization_radius", "radius_2x",
                                  "median_over_crossed_subset")),
                          q(new, ("linearization_radius", "radius_2x",
                                  "median_over_crossed_subset"))],
        radius_10x_median=[q(old, ("linearization_radius", "radius_10x",
                                   "median_over_crossed_subset")),
                           q(new, ("linearization_radius", "radius_10x",
                                   "median_over_crossed_subset"))],
        per_object_changed=[o for o in q(new, ("linearization_radius",
                                               "radius_2x",
                                               "per_object"))
                            if q(old, ("linearization_radius", "radius_2x",
                                       "per_object", o))
                            != q(new, ("linearization_radius", "radius_2x",
                                       "per_object", o))])
    channel = dict(
        direction_max_pct=[q(old, ("channel_decomposition",
                                   "direction_max_pct")),
                           q(new, ("channel_decomposition",
                                   "direction_max_pct"))],
        joint_D_at_0p5_per_object={
            o: [next(r["D"] for r in old["channel_decomposition"]["rows"]
                     if r["object"] == o and r["channel"] == "joint"
                     and r["level"] == 0.5),
                next(r["D"] for r in new["channel_decomposition"]["rows"]
                     if r["object"] == o and r["channel"] == "joint"
                     and r["level"] == 0.5)]
            for o in sorted({r["object"]
                             for r in new["channel_decomposition"]["rows"]})})
    anchor = dict(
        median=[q(old, ("ball_anchor_share", "median")),
                q(new, ("ball_anchor_share", "median"))],
        per_object={o: [q(old, ("ball_anchor_share", "per_object", o)),
                        q(new, ("ball_anchor_share", "per_object", o))]
                    for o in q(new, ("ball_anchor_share", "per_object"))},
        degenerate_objects=[q(old, ("ball_anchor_share",
                                    "degenerate_objects")),
                            q(new, ("ball_anchor_share",
                                    "degenerate_objects"))])

    out = dict(
        gate="P-DILIGENT-LOADER-FIX",
        analysis_status="loader_normalization_fix_v1",
        defect="src/calibinfo/datasets/diligent_oi_adapter.py dropped the "
               "per-light intensity normalization (v1 divided by 255 only), "
               "biasing the DiLiGenT cohort's nominal reconstruction vs GT "
               "by 15.6-26.3 deg while the drift endpoints were blind to it "
               "(two-sided same-source bias cancels in nominal-vs-corrupted "
               "differences). Fixed to gray / light_intensities[:, 0], the "
               "diligent.py:43 convention; cross-loader gate added "
               "(tests/test_diligent_loader_consistency.py, bit-identical "
               "masked intensities).",
        decisive_measurement=_nominal_vs_gt(GT_OBJECTS),
        queue_before_after=dict(
            source_old=OLD_REF,
            source_new_sha256=hashlib.sha256(NEW.read_bytes()).hexdigest(),
            radius=radius, channel=channel, anchor=anchor,
            outcome=[old["outcome"], new["outcome"]]),
        conclusion_impact=dict(
            radius="HARDENED: radius_2x/10x medians unchanged (1.0/1.5), "
                   "radius_10x now 1.5 on 10/10 objects; the cross-system "
                   "envelope reproduction survives the correction",
            channel="CORRECTED: the direction-channel maximum was 51.2% "
                    "under the biased loader and is 0.20% corrected -- the "
                    "DiLiGenT direction channel is NOT large; the "
                    "'dataset-geometry axis' reading of C11 was a loader "
                    "artifact. The ball-anchor share split also collapses "
                    "(pot1 85.6%->29.5%, pot2 86.1%->-0.1%, cat "
                    "-0.1%->+22.1%, median -0.67%->-0.005%)",
            baseline="DQ half of C12 rerun on the corrected loader (see "
                     "baseline_comparison.json; the OI half is "
                     "deterministic-unchanged and the OI hard anchor is "
                     "re-asserted at runtime)"),
        manifest=dict(note="git-blob + raw-data comparison; the decisive "
                           "measurement re-runs the nominal estimator on "
                           "both normalization conventions"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(json.dumps(out, ensure_ascii=False,
                               indent=1).encode("utf-8"))
    print("[loader-fix] decisive (median deg, unnorm -> norm):",
          {k: (v["unnormalized"]["median_deg"],
               v["intensity_normalized"]["median_deg"])
           for k, v in out["decisive_measurement"].items()})
    print("[loader-fix] direction_max_pct:", channel["direction_max_pct"])
    print("[loader-fix] radius medians:", radius["radius_2x_median"],
          radius["radius_10x_median"])
    print(f"[loader-fix] wrote {OUT}")


if __name__ == "__main__":
    run()
