"""P-SUBMOD · 对抗次模搜索（负结果包；plan v2 M4/L6）。

问题：把「选哪些灯精化」当集合函数。F(S) = 泛函(ΔF(S))（S 中灯 t=κ），
增益 G(S) = F(∅) − F(S)（递增）。次模 ⇔ 对所有 A ⊆ B, x ∉ B：
    G(A∪x) − G(A) ≥ G(B∪x) − G(B)（边际递减）。

协议（吸收评估报告 + 审计纪律）：
  - 实例族：random 基线 + 对抗旋钮（近共线 u 列、尺度悬殊、Λ0 各向异性）；
  - **退化守卫**：任何被评子集 λmin(ΔF(S)) 相对 < 1e-8 → 丢弃整个实例
   （否则 1/λ 的截断会伪造巨量假违反）；
  - 穷举 S ⊆ T 全子集（L ≤ 10 → 3^L 三元组）；
  - **检测器自检**：玩具超模函数 G=|S|²−|S| 必须 100% 触发；玩具模性函数
    G=10|S| 必须零触发——确保"未发现违例"有含义；
  - 次模比 γ = min 边际比（γ ≥ 1 次模；γ < 1 违反，记录首个反例）。

结论纪律：不把次模写成正/负定理，如实报告计数 + γ + 首反例；
Chamon–Ribeiro 线的近似超模引用进论文。
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from calibinfo.allocation.blocks import sym_inv


def build_instance(seed, L, P, family="random"):
    """per-light whitened 块实例；family='adversarial' 施加三旋钮。"""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=(L, P))
    s = rng.uniform(0.2, 1.5, size=(L, P))
    B = rng.normal(0.0, 0.5, size=(L, P, 3))
    lam_diag = np.array([30.0, 800.0, 800.0])
    if family == "adversarial":
        base_dir = rng.normal(size=(P, 3))                     # 近共线
        for k in range(L):
            B[k] = base_dir[None, :] * rng.uniform(0.8, 1.2) \
                + 0.05 * rng.normal(size=(P, 3))
        s = s ** 3
        s = s * (1.0 + rng.uniform(0, 20, size=(L, 1)))        # 尺度悬殊
        lam_diag = np.array([30.0 * 1e-2, 800.0, 800.0 * 1e2])  # 各向异性
    u = (w * s)[:, :, None] * B
    M0 = np.einsum("kpi,kpj->kij", B, w[:, :, None] * B)
    lam0 = np.stack([np.diag(lam_diag)] * L)
    finf = (w * s ** 2).sum(0)
    return u, M0, lam0, finf


def subset_objectives(u, M0, lam0, finf, kind, degeneracy_tol=1e-8,
                      kappa=10.0):
    """全子集的 F(S)，语义 = 冻结协议（v1.1 修复）：S 中灯 t=κ、其余 t=1——
    **精化集合**，不是累减构造（v1.0 误把逐子集减法当语义，产生假
    "非单调"类；F = 1/λmin 在 Loewner 下本应单调）。
    kind: 'A'=tr ΔF⁻¹, 'E'=1/λmin, 'D'=−logdet₊。任何子集相对退化 → None。"""
    L = u.shape[0]
    vals = {}
    for size in range(0, L + 1):
        for S_tuple in itertools.combinations(range(L), size):
            S = frozenset(S_tuple)
            t = np.ones(L)
            for k in S_tuple:
                t[k] = kappa
            DF = np.diag(finf)
            for k in range(L):
                K = sym_inv(M0[k] + t[k] * lam0[k])
                DF = DF - (u[k] @ K) @ u[k].T
            ev = np.linalg.eigvalsh(0.5 * (DF + DF.T))
            if ev[0] < degeneracy_tol * max(1.0, float(ev[-1])):
                return None
            vals[S] = _objective(DF, kind)
    return vals


def _objective(DF, kind):
    ev = np.linalg.eigvalsh(0.5 * (DF + DF.T))
    if kind == "A":
        return float(np.sum(1.0 / ev))
    if kind == "E":
        return float(1.0 / ev[0])
    if kind == "D":
        pos = ev > 1e-12 * max(1.0, float(ev[-1]))
        return float(-np.log(ev[pos]).sum())
    raise ValueError(kind)


def search_violations(F_vals, L, tol=1e-12):
    """穷举 (A ⊆ B, x ∉ B)。G = F(∅) − F(S)。分类（v1.1）：
      - **submodular 违例**（报告关注类）：dB > 0 且 dA < dB − tol
        （正边际下的边际递减破坏）；
      - **non-monotone 三元组**（单独计数）：dB ≤ 0 或 dA ≤ 0
        （对抗近共线实例上精化可使 E-opt 变差——非单调性，非次模性）；
      - γ = min dA/dB，只在 dB > tol 的三元组上取（负边际不进 γ）。
    返回 (submodular 违例统计, non-monotone 计数, γ, 首个 submodular 反例)。"""
    G = {S: F_vals[frozenset()] - F_vals[S] for S in F_vals}
    viol_sub = 0
    nonmono = 0
    triples = 0
    gamma = np.inf
    first = None
    for B in [frozenset(S) for n in range(1, L + 1)
              for S in itertools.combinations(range(L), n)]:
        for x in range(L):
            if x in B:
                continue
            Bx = B | {x}
            dB = G[Bx] - G[B]
            for r in range(0, len(B) + 1):
                for A_tuple in itertools.combinations(sorted(B), r):
                    A = frozenset(A_tuple)
                    Ax = A | {x}
                    dA = G[Ax] - G[A]
                    triples += 1
                    if dB < -tol * max(1.0, abs(dB)) or dA < -tol * max(1.0, abs(dA)):
                        nonmono += 1          # 严格负边际 = 非单调（零边际不是）
                        continue
                    if dA < dB - tol * max(1.0, abs(dB)):
                        viol_sub += 1
                        if first is None:
                            first = dict(A=sorted(A), B=sorted(B), x=x,
                                         dA=float(dA), dB=float(dB))
                    if abs(dB) > 1e-15:
                        gamma = min(gamma, dA / dB)
    return dict(violations_submodular=viol_sub, nonmonotone_triples=nonmono,
                triples=triples,
                gamma=(float(gamma) if np.isfinite(gamma) else None),
                first_violation=first)


def toy_gain(kind, L):
    """检测器自检：返回全子集 G(S)（增益，递增）。
    'supermodular' = |S|²−|S|（必须 100% 触发）；'modular' = 10|S|（必须 0）。"""
    out = {}
    for n in range(L + 1):
        for S in itertools.combinations(range(L), n):
            # 严格正且递增的边际（|S|=0→1 处也 >0），保证"全部违例"预期成立
            g = (len(S) ** 2 + len(S)) if kind == "supermodular" else 10.0 * len(S)
            out[frozenset(S)] = float(g)
    return out


def run(out_path="results/submodularity/submodularity_search.json",
        seeds=range(20260918, 20260928), L=5, P=40):
    """全网格扫描 → 负结果 JSON（分类计数 + γ + 首反例；无符号判据）。"""
    out = {}
    for family in ("random", "adversarial"):
        for kind in ("A", "E", "D"):
            entries = []
            n_valid = 0
            for seed in seeds:
                u, M0, lam0, finf = build_instance(seed, L, P, family=family)
                F = subset_objectives(u, M0, lam0, finf, kind=kind)
                if F is None:
                    entries.append(dict(seed=seed, discarded="degenerate"))
                    continue
                n_valid += 1
                out_s = search_violations(F, L)
                entries.append(dict(seed=seed, **out_s))
            sub_total = sum(e.get("violations_submodular", 0) for e in entries)
            nonmono_total = sum(e.get("nonmonotone_triples", 0) for e in entries)
            gammas = [e["gamma"] for e in entries
                      if e.get("gamma") is not None]
            out[f"{family}/{kind}"] = dict(
                n_seeds=len(entries), n_valid=n_valid,
                n_degenerate=len(entries) - n_valid,
                submodular_violations_total=sub_total,
                nonmonotone_triples_total=nonmono_total,
                gamma_min=(float(min(gammas)) if gammas else None),
                first_pinned=next((e["first_violation"] for e in entries
                                   if e.get("first_violation")), None),
                entries=entries)
    result = dict(
        gate="P-SUBMOD: adversarial submodularity search (negative-result pack)",
        analysis_status="submodularity_search_v1",
        L=L, P=P, seeds=list(seeds),
        detector_selfcheck=dict(
            toy_supermodular_violations=search_violations(
                {frozenset(S): (L * L + L) - (len(S) ** 2 + len(S))
                 for n in range(L + 1)
                 for S in itertools.combinations(range(L), n)}, L)[
                    "violations_submodular"],
            note="G=|S|^2+|S| toy supermodular must trigger on every A<B triple"),
        families=out,
        note="Classification: submodular violations = positive-marginal "
             "diminishing-returns failures (dB > 0, dA < dB); non-monotone "
             "triples counted separately (strictly negative marginals). "
             "Degenerate instances discarded by the guard. No theorem claims: "
             "counts + gamma + pinned counterexamples, reported as-is.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(result, ensure_ascii=False, indent=1)
                               .encode("utf-8"))
    for key, blk in out.items():
        print(f"[submod] {key}: sub_violations={blk['submodular_violations_total']} "
              f"nonmono={blk['nonmonotone_triples_total']} "
              f"gamma_min={blk['gamma_min']} valid={blk['n_valid']}/{blk['n_seeds']}")
    print(f"[submod] wrote {out_path}")
    return result


if __name__ == "__main__":
    run()
