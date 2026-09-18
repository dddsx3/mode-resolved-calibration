# Mathematical foundations

This page states the methods implemented in `calibinfo` precisely enough to
use the library on its own. Statements marked **(proved)** have analytic
arguments; the tests listed at the end check identities and representative
instances, not the universal quantifiers of a theorem. Statements marked
**(cited)** are classical results we instantiate. The continuation proofs and
their scope are indexed in `docs/theory/README.md`.

## 1. Model

A local linearization of a calibrated inverse problem around a nominal
calibration gives the whitened system

    y = A x + B δc + ε,   ε ~ N(0, σ² I),

with `x ∈ Rⁿ` the parameters of interest, `δc ∈ R^q` calibration/nuisance
perturbations, `A = D(ŝ)` the per-pixel design after whitening and `B` the
whitened nuisance Jacobian. All downstream objects live in this whitened
coordinate system (`calibinfo.information.whitening`).

## 2. Effective information ΔF(Λ)

Two equivalent views of "the information about `x` that survives calibration
uncertainty":

- **Profiling** (penalized precision Λ ⪰ 0, null directions = flat):

      ΔF(Λ) = Aᵀ [ I − B (BᵀB + Λ)⁻¹ Bᵀ ] A.

- **Marginalization** (proper Gaussian Σ_c, Σ_y = σ²I):

      ΔF = σ² Aᵀ (σ² I + B Σ_c Bᵀ)⁻¹ A.

For Σ_c ≻ 0 the two coincide via Woodbury with Λ = σ²Σ_c⁻¹
(`schur.delta_f` vs `schur.delta_f_marginal`, elementwise agreement < 1e-10).
**Semantics warning**: the zero directions of a profiling precision mean
"flat/unpenalized", while the zero directions of a Gaussian covariance mean
"almost surely zero" — the opposite. For singular Σ_c the correct routes are
the covariance factorization Σ_c = LLᵀ, C = BL (`schur.nuisance_factor`,
`schur.delta_f_marginal_factor`) or the direct marginal; substituting a
pseudoinverse precision is a known-answer counterexample (factor/marginal
give 0.5, the shortcut gives 0 on the canonical instance).

## 3. Retention spectrum

On the identifiable subspace 𝒳 = range(F∞), F∞ = AᵀA:

    R(Λ) = F∞^{-1/2} ΔF(Λ) F∞^{-1/2},   0 ⪯ R ⪯ I,

so R has real eigenvalues 0 ≤ ρ₁ ≤ … ≤ ρ_r ≤ 1 with **r = rank(F∞)** — not
the nuisance dimension. ρ_j is the generalized Rayleigh quantity
`v_jᵀΔFv_j / v_jᵀF∞v_j`; it is *not* a ratio of ordinary eigenvalues of the
two matrices (`retention.retention_spectrum` returns both the values and the
eigenvectors; `retention.retention_spectrum_gen_eig` is an independent
generalized-eigenvalue route).

Under the matched linear-Gaussian GLS estimator the strict variance statement
is coordinate-specific: for the normalized dual error coordinate
`z_j = u_jᵀ F∞^{1/2}(x̂ − x)`,

    Var(z_j) = σ² / ρ_j,   equivalently   F∞^{1/2} Cov(x̂) F∞^{1/2} / σ² = R⁻¹.

The naive coordinate `u_jᵀ(x̂ − x)` does *not* have variance σ²/ρ_j unless
F∞ is isotropic — the reason the empirical pipeline projects errors through
`F∞^{1/2}` (`tests/test_math_gates.py`).

## 4. Gauge response and directional crossover

If a parameter direction `a` and nuisance direction `c̄` satisfy the gauge
identity `Aa = Bc̄`, then the information along `a` has the closed form

    aᵀΔF(λ)a = Σᵢ αᵢ² sᵢ² λ / (sᵢ² + λ),   (sᵢ², V) = eig(BᵀB),  α = Vᵀc̄,

(`gauge.gauge_response`): monotone increasing and concave in the isotropic
precision λ, with low-precision slope Σαᵢ² and saturation ‖Aa‖². The
**directional crossover precision** λ⋆ solves `aᵀΔF(λ)a = μ` for a
pre-specified floor μ; it exists iff `0 < μ < ‖Aa‖²` and is a within-scene
diagnostic only.

**Scope note (forward form vs downstream variance).** The closed form
`aᵀΔF(λ)a = Σᵢ αᵢ²sᵢ²λ/(sᵢ²+λ)` is an **information** statement. The task
functional is the **inverse** form `aᵀΔF⁻¹a`; the two coincide only when
`a` is an eigen-direction of ΔF. `gauge.py` therefore diagnoses how much
information a direction carries, and does **not** directly give that
direction's downstream variance — on the real cohort the forward-form
ratio predicts `V ≈ 0` where the realized task value is `≈ 0.9` (see §10
for the mechanism that does predict it).

## 5. Allocation as a convex program

With per-light precision multipliers `t ∈ [1, κ]^L` (t_k multiplies Λ0k:
larger t = higher precision; the parameterization is pinned by
`tests/test_math_foundations.py`),

    ΔF(t) = diag(F∞) − Σ_k u_k (M0k + t_k Λ0k)⁻¹ u_kᵀ.

- **L1 (per-light monotonicity)** — `t_k ↦ u_k(M0k + t_kΛ0k)⁻¹u_kᵀ` is
  Loewner decreasing, hence ΔF is jointly Loewner **increasing** (entrywise
  monotonicity does *not* hold).
- **L2 (joint operator concavity)** — `X ↦ X⁻¹` is operator convex on PD, so
  each rank-3 term is operator concave in t_k and ΔF is jointly operator
  concave (midpoint residual ~1e-15, machine-checked). Equivalently
  ΔF(Λ) = Aᵀ(I + BΛ⁻¹Bᵀ)⁻¹A.
- **L3 (convex objectives)** — on the feasible region (ΔF ≻ 0):
  `J_A = tr ΔF⁻¹` (smooth), `J_D = −logdet`, and `J_E = −λmin` (nonsmooth at
  eigenvalue crossings — use as certificate only with care) are convex in t.

  ![midpoint convexity of J_E](img/convexity_midpoint.png)
  The certificate functional is **J_A**; mode-tail functionals on the R
  spectrum are report-only (convexity unproven).
- **L4 (certificates)** — the budget-constrained program
  `min J_A(t) s.t. Σ_k (t_k − 1) ≤ B, t ∈ [1,κ]^L` is convex; the
  Frank–Wolfe duality gap `g(t) = ∇J_A(t)ᵀ(t − s*(t)) ≥ 0` at any feasible t
  is a **global** optimality certificate: `J_A(t) − J* ≤ g(t)`. With
  `B = k(κ−1)` the feasible set contains every "exactly k refined lights"
  allocation, so each per-k bound is valid for that discrete family. The LMO
  is analytic (ascending-gradient budget prefix; spend only on coordinates
  with negative gradient). `allocation.convex.CertificateProblem`.

## 6. Exact low-rank structure

With per-light block-diagonal nuisance the retention operator factorizes as

    R = I − V Vᵀ,   V = F∞^{-1/2} [u_k K_k^{1/2}]_{k active}  (P × 3L),

so the multiplicity of the unit eigenvalue is exactly `P − rank(V)`.
Only the nonzero eigenvalues of `VᵀV` produce nontrivial values `1 − λ`;
zero eigenvalues of the skinny Gram matrix are not counted a second time.
There are at least `max(0, P − 3L)` unit modes, with equality `P − 3L`
only when `rank(V)=3L`. The same structure
gives the Woodbury route for `tr ΔF⁻¹` and its gradient
(`information.lowrank`), which is what makes full-resolution (all masked
pixels, all 142 lights) certification feasible on a workstation.

## 7. Certified dynamic range (the question the library answers)

For each real scene the certified dynamic range

    (J_A(t = 1) − J_A(t = κ·1)) / J_A(t = 1)

is computed with a valid lower bound on every feasible allocation (the FW
bound), together with policy candidates on the same functional (greedy
prefix, seeded random subsets, all/none). On the 11 real objects this range
is 27.3–89.4% per object at full resolution — instance-dependent, which is
why per-instance certification, not a universal constant, is the deliverable.
`allocation.rank_invariance` additionally certifies that the numerical rank
of ΔF is preserved along any allocation, which licenses standard
A/D-optimality naming for the classical greedy baselines.

### Universal ceiling on the calibration-budget value (H2)

**Structural assumptions (all objects come from one whitened model).** The
ceiling is a theorem about the *model* `y = A x + B δc + ε`, not a universal
matrix identity. Concretely, for every light-indexed block the objects share
a single `(w_k, s_k, B_k)`:

    u_k = w_k s_k B_k       (= A^T B_k in the whitened coordinate system),
    M0_k = B_k^T (w_k B_k)  (identity-precision block, PSD),
    F∞ = diag( Σ_k w_k s_k² )   (= A^T A, the calibration-limit Fisher).

Under the shared design the full Gram block is positive semidefinite;
there is no general matrix identity `u_k u_kᵀ = F∞ M0_k` (the factors even
have different dimensions in the per-light model). The scalar one-row
special case has `u² = F∞ m`, but that equality is not a matrix proof.

**The ceiling.** For fixed whitened `A_obs`, `B` and precision `Λ0 ⪰ 0`,
completion of squares gives the variational form

    xᵀΔF(t)x = min_c { ‖A_obs x − Bc‖² + t cᵀΛ0 c }.

For each c and `t ≥ 1`, the expression lies between its value at one and
t times that value. Taking the minimum on both sides proves

    ΔF(1) ⪯ ΔF(t) ⪯ t ΔF(1).

This argument uses the common observation model, not an invalid subtraction
of unrelated kernel bounds. It also holds for per-light `1 ≤ t_k ≤ κ`,
yielding `ΔF(1) ⪯ ΔF(t_vec) ⪯ κΔF(1)`. The kernel is fixed on that box;
on one fixed positive-definite quotient, inverse order therefore proves

    D = 1 − J_A(κ·1)/J_A(1) ≤ 1 − 1/κ.

The "90.0% at κ=10" headline is this ceiling being **met** — when the
identity precision block `M0` dominates `Λ0` (large level = small `Λ0`),
the bound is tight to ≤ 1e-4 (`results/magnitude/calibration_value_ceiling.json`,
5 objects at level 64): κ=10 gives D ∈ [0.89992, 0.89999] against the
ceiling 0.9. The ceiling is a property of the functional, not an empirical
saturation artifact. The bound holds on every instance/admissible level
(machine-checked, CI-safe: `tests/test_math_foundations.py`).

**The bound holds across the Σ_φ family.** The ceiling has *zero*
Σ_φ-dependence: it is a theorem about the functional under the structural
model, valid for **any** per-light prior precision `Λ0 ≻ 0` — not a
property of the joint parameterization (logI/dir variance ratio ~3283 at
level 0.5) used to measure it. The 45-point three-axis family sweep
(independent channel ratio σ_dir ∈ [0.1°, 25°] at fixed σ_logI, per-light
lognormal heterogeneity het ∈ [0, 1], channel coupling |ρ| ≤ 0.999; 11
objects, 495 rows) therefore runs the ceiling as a **runtime guard, not
as evidence**: the maximum violation over all family rows is −0.0028 —
every D stays below the bound by at least 0.0028
(`results/openillumination/corruption_family_sensitivity.json`, S2-3).

**Non-legitimate counterexample (why the structure is necessary).** In the
single-light scalar model `ΔF(t) = F∞ − u²/(m + t·λ)` the ceiling ``D ≤ 1−1/κ``
is *not* implied by `M0 ⪰ 0` alone. If `u` and `m` are decoupled — e.g.
`u² = 1, m = 0.01, F∞ = 1, λ = 1, κ = 2` — both `ΔF(1) ≈ 0.0099` and
`ΔF(κ) ≈ 0.50` are positive definite, yet

    D ≈ 0.9803  >  1 − 1/κ = 0.5.

This `(u, M0, F∞)` triple cannot arise from any whitened model: its shared
Gram matrix `[[F∞,u],[u,m]]` has negative determinant. It exists only as a
detached matrix example. It demonstrates that the ceiling rests on the
common observation model, not on PSD-ness of `M0` by itself. The exact
numbers are pinned by `tests/test_math_foundations.py::test_structural_link_required_for_ceiling`,
which also checks the legitimate one-row scalar family `u²=F∞m`.
The variational argument above supplies the general matrix proof.

**Extension to any quadratic functional (M10′).** The ceiling holds for
*every* task functional, not only the trace: by the same Loewner chain
(`ΔF(κ·1) ⪯ κΔF(1)` under the structural assumptions above),
`ΔF(κ·1)^{-1} ⪰ ΔF(1)^{-1}/κ`, so for any task operator `H`,

    tr(H ΔF(κ·1)^{-1} Hᵀ) ≥ tr(H ΔF(1)^{-1} Hᵀ)/κ   ⟹   V_H ≤ 1 − 1/κ.

This step uses only operator monotonicity of `X ↦ X^{-1}` (Löwner), which
holds unconditionally — unlike the α-bound collapse step in §9, no
framework reduction and no numerical support are needed; the proof chain
is closed. Empirically saturated by the gauge-aligned mean functionals
(§10: `V_rho_mean` → 0.9000 at the ceiling from below, 44/44 cells at or
below it; `results/goal_oriented/goal_orientation.json`).

## 8. What the library does not claim

- No claim that mode-resolved objectives beat scalar criteria — the
  submodularity structure required by such guarantees is violated for E-opt
  gains on real-shaped instances (documented negative result,
  `results/submodularity/submodularity_search.json`).
- No claim that the linearized theory predicts real-data error magnitudes.
  The historical matched synthetic check has ratio ≈ 1.0045. The historical
  corrected-interface median 15.98 (`directional_amplitude_summary.json`)
  used a legacy gauge branch and is superseded for current amplitude reading.
  The imported per-sample-projection D-arm median is 53.744 relative to the
  same-coordinate prediction, but its denominator is residual bootstrap,
  not the matched GLS baseline. It does not by itself refute the theorem.
  B1/B2 identify the exact fields; the new S1/S2/S3 study is separate evidence.
- No claim of overall reconstruction advantage for any allocation policy —
  the certified result is about the information landscape itself plus the
  mode-targeted intervention endpoint.

## 9. γ 数学修正：被否定的 α 候选式与有效全迹谱界

**当前结论。** 历史候选 `γ ≥ 1/(1+α)` 在一般 SPD 基线加 PSD 更新类上
**已被精确反例否定**，而不只是证明尚未完成；一般共享线性高斯 Jacobian
结构也不能挽救它。它是仓库自己的历史候选式，不是 Chamon–Ribeiro
(NeurIPS 2017) 的已证定理。新有效保证是下文的 leave-one-out 谱界。
解析证明与有限数值回归必须区分：测试核验恒等式和实例，不代替一般证明。

### 定义、局部恒等式与零更新

在同一个固定的正定参数空间（或预先固定的可辨识子空间）上，令候选集
`U` 有限，`A ≻ 0`，`W_i ⪰ 0`，并定义

    M(S) = A + Σ_{i∈S} W_i,
    F(S) = tr M(S)^{-1},   G(S) = F(∅) − F(S),
    d_x(S) = G(S∪{x}) − G(S),
    γ = inf_{S⊆T⊆U\{x}, W_x≠0} d_x(S)/d_x(T).

这里允许 `S=T`。对未加权全迹，`d(M,W)>0` 当且仅当 `W≠0`；零更新
产生恒为零的边际，不能把 `0/0` 放进商的下确界。若候选集为空或所有更新
均为零，**另外约定 γ=1**，而非声称空集上的商下确界自动等于一。

标定模型中 `A=ΔF(1)`、`W_k=u_k[K_k(1)−K_k(κ)]u_kᵀ` 给出原 L9 的
精确加性分解；原 L10 的边际恒等式与 L11 的局部夹逼仍然成立：

    d(M,W) = tr[M^{-1} − (M+W)^{-1}]
           = tr[M^{-1}W(M+W)^{-1}],
    ρ = λmax(M^{-1/2} W M^{-1/2}),
    tr(W M^{-2})/(1+ρ) ≤ d(M,W) ≤ tr(W M^{-2}).

证明可用 `X=M^{-1/2}WM^{-1/2} ⪰ 0` 和
`X/(1+ρ) ⪯ X(I+X)^{-1} ⪯ X`，作合同变换并取迹。
这里不要求 `M^{-1}W` 对称，更不能由 `M⪯N` 推出 `M^{-2}⪰N^{-2}`。

### 精确反例与任意固定 α 的不可能性

取二维有理矩阵

    A  = diag(1, 1/100),
    W1 = [[1, 0], [0, 0]],
    W2 = [[1/2, 1/20], [1/20, 1/200]].

两更新为秩一 PSD，白化更新的谱均为 `{1,0}`，故
`α=max_i λmax(A^{-1/2}W_iA^{-1/2})=1`。直接求逆得到

    d_1(∅) = 1/2,   d_1({2}) = 109/28,
    d_1(∅)/d_1({2}) = 14/109 < 1/2 = 1/(1+α).

另一个严格嵌套比值为 `707/802`，其余四个 `S=T` 的比值为一，所以
**真实 γ=14/109**。同时 `tr(W1 A^{-2})=1`，而
`tr[W1(A+W2)^{-2}]=109/16`。这直接否决了“边际递减已证”和旧全局界，
但不影响上述局部夹逼。

更一般地，对任意固定 `p>0` 和 `ε>0`，取

    A=diag(1,ε²),
    W1=p·diag(1,0),   W2=(p/2)·[[1,ε],[ε,ε²]].

此时 `α=p` 精确不变，并有

    d_1(∅) = p/(1+p),
    d_1({2}) = p[p²+ε²(p+2)²]/[2ε²(p+1)(p²+4p+2)],
    d_1(∅)/d_1({2}) = 2ε²(p²+4p+2)/[p²+ε²(p+2)²] → 0.

`p=1` 时比值为 `14ε²/(1+9ε²)`。因此每个固定 `α>0` 的实例族都有
`inf γ=0`，即使维数和候选数都固定为二；不存在仅依赖 α 的普适正下界。
各个 `ε>0` 的实例仍然正定、非零边际严格为正，不能把奇异端点 `ε=0`
冒充合法实例。`α=0` 则意味着所有更新为零。非正交白化也不能无损地设
`A=I`：目标变为 `tr(A^{-1} M̃^{-1})`，不能丢掉这个权重。

### 一般共享实现与前轮物理范围边界

对 `p=1` 的 ε 族写 `W_i=l_i l_iᵀ`，其中 `l1=(1,0)ᵀ`、
`l2=(1,ε)ᵀ/√2`。令每块观测 `y_i=A_i x+B_i c_i+η_i`，独立单位噪声、
独立标量 nuisance 的先验精度从一精化到 `κ=10`，取

    A_i=(√22/3)l_iᵀ,   B_i=√10,
    F_i(t)=22t/[9(10+t)]·W_i,
    F_i(1)=(2/9)W_i,   F_i(10)=(11/9)W_i,
    A0=(1/√54)·[[6,−ε],[0,√47 ε]].

无 nuisance 观测块满足 `A0ᵀA0=R=A−(2/9)(W1+W2) ≻ 0`，
`det R=47ε²/81`。按行堆叠 `A0,A1,A2` 为共同观测设计 `A_obs`，并令
`B` 的前两行为零、后两行为 `diag(√10,√10)`，则四个子集均精确满足

    A_obsᵀA_obs − A_obsᵀB(BᵀB+diag(t1,t2))^{-1}BᵀA_obs
      = A + Σ_{i∈S}W_i,   t_i=10 当且仅当 i∈S，否则 t_i=1.

故这不是独立拼接 `F∞、U、M0` 的伪反例。任意 `p>0` 的共享实现也可取
`κ=B_i²=1+4p`、`A_i=√(1+2p)l_iᵀ`（此处 `l_i l_iᵀ=W_i/p`），
每块基线为 `l_i l_iᵀ/2`、增量为 `W_i`；无 nuisance 残余为
`A−(l1l1ᵀ+l2l2ᵀ)/2 ≻ 0`。这里 κ 随 p 改变；不能宣称任意 α 都能在
任意预先固定 κ 下实现。固定 κ 的共享模型有 `ΣW_i⪯(κ−1)A`。

**前轮构造的模型边界：** 上述观测行是一般线性 Jacobian，并未满足实际
Lambertian 每灯对角设计及受限方向导数，不能将这一个构造直接改称物理反例。
本轮 M13 另给出满足着色、单位法向和方向切基约束的独立物理构造；其逐灯
正定耦合先验也是构造的一部分。该新结果不等于对共享同一对角先验等更窄
子类的完整判定，详见 `docs/theory/physical_gamma_and_tightness.md`。

### 正定全迹谱界及证明

**成对定理。** 若 `0≺M⪯N`、`0≠W⪰0`，则

    d(M,W)/d(N,W) ≥ λmin(N)/λmax(M).

取同一个因子 `W=LLᵀ`（可为秩亏更新），对 `P=M,N` 定义
`X_P=LᵀP^{-1}L`、`Y_P=LᵀP^{-2}L`、`f(X)=I−(I+X)^{-1}`。
Woodbury 恒等式给出

    d(P,W)=tr[(I+X_P)^{-1}Y_P].

对每个单独的 SPD 矩阵 P，谱分解给出

    P^{-1}/λmax(P) ⪯ P^{-2} ⪯ P^{-1}/λmin(P).

作 L 的合同变换，再对正定权重 `(I+X_P)^{-1}` 取迹，得到

    d(M,W) ≥ tr f(X_M)/λmax(M),
    d(N,W) ≤ tr f(X_N)/λmin(N).

这些迹不等式不要求 `X_P` 与 `Y_P` 可交换。由 `M⪯N` 的逆反序与合同
变换有 `X_M⪰X_N`；再次对 `I+X_P` 使用逆反序，得到
`f(X_M)⪰f(X_N)`。`W≠0` 保证 `tr f(X_N)>0` 及 `d(N,W)>0`，故相除
即得成对谱界。证明从未比较 `M^{-2}` 与 `N^{-2}`。

**全局定理。** 记 `U+={x:W_x≠0}`。若它非空，则

    γ ≥ λmin(A)/max_{x∈U+} λmax(M(U\{x}))
      ≥ λmin(A)/λmax(M(U)) > 0.

因为任意有效三元组满足 `M(T)⪰A`、`M(S)⪯M(U\{x})⪯M(U)`，代入
成对界再取下确界即可。排除零候选不影响 γ，并可能加强 leave-one-out 界。
记 `m=|U|`、`χ_A=λmax(A)/λmin(A)`，还可得保守推论

    γ ≥ 1/[χ_A(1+(m−1)α)],
    γ ≥ 1/(κχ_A)  （满足 M(U)⪯κA 的共享模型）.

基线条件数不能删掉。本二维有理反例的 leave-one-out 谱界为 `1/200`，
小于真实 `14/109`；它不是任何真实物体重新测得的谱界读数。

**实现。** `calibinfo.allocation.alpha_bound.spectral_gamma_lower_bound(A, updates)`
返回上述仅对非零更新取 max 的 `float`；支持矩阵列表、生成器及三维数组。
`A` 必须非空、有限、实对称 SPD，更新必须同形、有限、实对称 PSD。
空/全零更新返回 `1.0`；非法输入抛出 `ValueError`，不使用 ridge 或伪逆。
仅允许相对于各自矩阵尺度 `64*n*eps_float` 量级的对称/PSD 舍入误差，
随后对称化并将更新的微小负特征值截零；保证针对这些校验后的矩阵。
按共同标量缩放以降低溢出问题，逐个排除候选后直接求和以避免 `total-W_x`
的大数相消；不按阈值删掉微小正更新。输入不被原地修改。无法分辨 SPD 或
表示正谱比值时拒绝输入；结果不是区间算术证书。该通用稠密实现的特征值
开销为 `O(m n³)`、直接 leave-one-out 求和为 `O(m² n²)`，不声称达到
历史低秩 α 路径的现实规模性能。

### 可交换情形与任务范围

若 `A` 和**所有**更新可同时正交对角化（两两可交换），共同基下

    d_x(S)=Σ_j w_xj/[m_j(S)(m_j(S)+w_xj)],
    m_j(S)=b_j+Σ_{i∈S}w_ij,   b_j>0, w_ij≥0.

每项随 S 增大而不增，因此真实 `γ=1`（包含 `S=T` 和全零约定）。
保守谱界函数不检测此特例，返回值不一定等于一。

新谱界面向**未加权全迹**。任务 `J_H(M)=tr(QM^{-1})`、`Q=HᵀH≻0`
时可在 `M̃=Q^{-1/2}MQ^{-1/2}` 的加权坐标中应用同一定理；原矩阵上的
同一谱常数不能直接沿用。若 Q 秩亏，取

    M=I, N=I+(1/2)·[[1,1],[1,1]], W=diag(1,0), H=[0,1],
    d_H(M,W)=0, d_H(N,W)=1/28.

这给出 `γ_H=0`，尽管 `λmin(N)/λmax(M)=1`。不能排除零分子来掩盖反例，
也不能用伪逆机械恢复正界。此任务范围限制不影响独立成立的结构价值上限
`V_H≤1−1/κ`，亦不影响凸性与 Frank–Wolfe 证书。

### 历史搜索处置与测试说明（不覆盖原数值）

`results/submodularity/alpha_bound.json` 与 `experiments/alpha_bound.py`
的历史结果和计算路径保留；`gamma_lower_bound(alpha)` 的名称和数值行为
也为复现保留，但其 docstring 明示是**已被反例否决的历史候选表达式**。
文件中旧字段名、旧措辞以及下列有限搜索结果，不是当前有效 γ 保证。
归档检索摘要仍为 162,000 个三元组、12,493 个平方逆反序样本、最小比值
0.271；保留这些可追溯数字并不保留原先的普适保证解释。

- P-SUBMOD 历史 toy 族为二十个实例（十个 random、十个 adversarial），
  原候选式在这些实例上未违反；random α 为 0.059–0.119，adversarial α 为
  0.025–0.090，实测 γ 为 0.9999–1.0000。这不是一般类上的定理。
- 有效的局部 sandwich 在 1600 个随机 `(S,x)` 对上零违反，历史端点比值
  `min val/lb=1.000096`、`min ub/val=1.001011` 保留。
- 真实对象在 level=0.5 的历史 α 为 0.188–0.515；旧候选表达式约为
  0.66–0.84。历史最小值 `0.635144`（当时按 `0.635` 打印，
  `obj_10_pumpkin3`、level=0.1）及 level=0.5 的 `0.660259` 均仅为
  **历史候选数值**，撤回“所有对象至少 0.635-supermodular”的保证解释。
  未重新计算真实设计的有效谱界，不用本节反例谱值替代真实读数。
- 原 proof-limits 搜索的四百个实例、**162,000** 个穷举三元组仍记为旧
  候选式零违反，包括 **12,493** 个平方逆迹反序样本、最小实测比值
  `0.271`、最小历史候选 slack 倍数 `1.31`。这些有限记录不抵消精确反例。

`tests/test_alpha_bound.py` 和 `tests/test_gamma_bound_proof_limits.py`
继续作为历史实例、局部夹逼和归档字段的回归；其历史名称及说明中“theorem”
或“bound”不再被解释为普适证明。新增 `tests/test_gamma_general_bound.py`
覆盖有理反例、ε/固定 α 族、零更新、可交换真实 γ、小维穷举谱界、输入拒绝
和旧函数数值兼容。独立精确脚本保留在原证据包的
`verification_v5/verify_gamma_exact.py`；其原始输出以新增副本存于
`results/theory_extension_20260918/imported_20260917/gamma_exact_verification.json`。
该副本是导入的前轮证据，不冒充本轮新运行。本轮命令与测试记录另见
`docs/theory/README.md`，不修改历史 results 或幅值 run 的 code_snapshot。

## 10. Goal-oriented calibration value (J_H)

The A-opt functional prices calibration for *all* parameters equally. A
downstream task — a set of linear functionals `H ∈ R^{m×P}` of the parameter
estimate — has its own price:

    J_H(t) = tr(H ΔF(t)^{-1} H^T).

The low-rank route extends exactly (push-through/Woodbury on
`ΔF = D − V Vᵀ`):

    J_H = tr(H D^{-1} H^T) + tr(G^{-1} (H D^{-1} V)^T (H D^{-1} V)),
    G = I − V^T D^{-1} V,

with the analytic gradient
`∇_{t_k} J_H = −tr(K_k Λ0_k K_k (u_k^T Z)(Z^T u_k))`, `Z = ΔF^{-1} H^T`
(`information.lowrank.woodbury_quad_risk[_grad]`; `H = I` reproduces
`tr ΔF^{-1}` to < 1e-10 and the gradient matches FD to ≤ 1e-6 —
`tests/test_goal_oriented.py`).

**Finding** (11 held-out objects × 4 levels, κ = 10;
`results/goal_oriented/goal_orientation.json`; descriptive, no sign gate):

- **Value is task-dependent.** The same uniform refinement of all 48
  Fisher-active lights recovers a median dynamic range of 62.87% for the
  all-parameter A-opt functional at level 0.5 — numerically the P-CERT
  certified headline, an independent cross-check of the quad-risk route —
  but 89.76% for the mean-albedo task; at level 0.1 the gap is 7.82%
  vs 87.93%.
- **Rankings are task-dependent.** Per-light single-refinement gains
  ranked under the mean task vs the bright/dim-quartile contrast task
  disagree: median Spearman 0.936 (min 0.586), and the top-3 light sets
  are disjoint in 11 of 44 object×level cells.

**Mechanism (why mean-type tasks sit at the ceiling).** The
albedo-weighted mean functional `a = ρ/‖ρ‖` is *exactly* gauge-aligned
with the light-intensity nuisance component — per light `B_phi[:,0] = ŝ·ρ`,
so `A_k a = B_k c̄` with `c̄_k = e_1/‖ρ‖` (alignment residual at machine
precision on all 11 objects). For exactly gauge-aligned functionals the
inverse form obeys the two-term law

    J_a(t) ≈ 1/(c̄ᵀΛ(t)c̄) + 1/‖Aa‖²  ⟹  V_a = (1 − 1/κ)/(1 + q·r),

with `q = c̄ᵀΛ0c̄` (confusable-prior precision) and `r = 1/‖Aa‖²` — note this is
the *inverse*-form law; the forward closed form of §4 does not give it
(see the scope note there)
(residual-information precision): the functional's posterior variance is
dominated by the intensity-confusable calibration prior, which uniform
refinement shrinks by exactly 1/t. Verified on all 44 cells
(`gauge_mechanism_rho_mean`): `V_rho_mean` rises 0.878 → 0.9000 along the
level grid toward the ceiling. The two-term law is accurate to ≤ 2e-4 at
large `level`; the worst case over the cohort is 3.9e-2 at `level` 0.1,
where the prior term `1/q` is smallest (per-level `|ΔV|` medians:
1.6e-2 / 2.3e-3 / 3e-4 / 0.0 for `level` = 0.1 / 0.5 / 2.0 / 8.0). The registered uniform-mean task is *not*
exactly aligned (texture residual 0.31–34.6) but inherits the mechanism
through its ρ-component; the all-parameter A-opt mixes in
gauge-orthogonal directions whose variance is calibration-insensitive —
that contrast is the value gap above.

**Scope.** The implemented `A_k=diag(sqrt(w_k) s_k)` acts on **additive
albedo perturbations**, not log-albedo; `B_phi[:,0]=rho*s_hat` confirms this
by differentiation. For log-albedo, write `D_rho=diag(rho)` and transform
`A_log=A_add D_rho`, `F_log=D_rho F_add D_rho`, and the task consistently.
The unit-normal joint model has intrinsic scene dimension `3P` (one
log-albedo plus two normal-tangent coordinates per pixel); `4P` is only an
ambient representation with P constraints. The continuation M17 derives
its information, fixed gauge quotient, surviving ceiling and certificate,
and validates a local synthetic profile estimator; it does not claim new
real-data reconstruction accuracy. The old `joint_map.py` estimates albedo
and lights while holding normals fixed, so it is not that joint scene estimator.

**Decision-layer reading: aggregate vs per-light.** Two results at
different granularities combine into one boundary statement. At the
*budget-aggregate* level the A-opt criterion is predictive:
Spearman(predicted `J_A`, realized normal angular error) is positive in
88/88 object×level×regime cells (median +0.917), and **within the
informed family alone** (n=4: mode_aware/e/a/d-opt) the pooled pairwise
sign agreement is 678/987 = 0.687 [binomial 95% CI 0.657, 0.716], median
per-cell Spearman +0.800 (min −1.000: at least one cell fully
rank-reversed) — well above chance, though far weaker than the
all-units headline (which includes random units sitting at the bad
corner of both axes by construction). At the *single-light* level the
criterion is **not** predictive (C8: the stable 100-seed oracle ranking
anti-correlates with single-light `J_A` gains, −0.36/−0.82). **A-opt
ranks how much budget to spend, not which light to spend it on** — which
is exactly the C8 decomposition's mechanism. The v1.1 grid carries the
active-set-restricted random control, and the decomposition closes
quantitatively: against universe-random the informed policies win
~3.6–5.1° of mean normal error, while against active48-random they win
only 0.17–0.55° — real (all 16 bootstrap CIs exclude zero) but an order
of magnitude smaller. The several-degree headline is mostly the
active-set effect; within the active set the ordering advantage is
genuine but modest.

## Tests binding these statements

| File | Binds |
|---|---|
| `tests/test_information_modules.py` | dual-route equivalence, whitening, gauge closed form, mode tracking |
| `tests/test_math_foundations.py` | L1 direction pinning, per-light monotonicity, midpoint convexity, low-rank identity, gradient FD |
| `tests/test_convex_certificates.py` | LMO optimality, FW gap validity, certificate-vs-random-feasible-points |
| `tests/test_lowrank_identity.py` | M4 spectral identity, Woodbury trace/gradient |
| `tests/test_math_gates.py` | singular-covariance counterexample, retention-covariance theorem, polyfit order, rank invariance |
| `tests/test_known_answer_precheck.py` | λmin reading, trace dilution, parallel sums, gauge identity |
| `tests/test_alpha_bound.py` | 历史 toy/实物 α 数值回归与仍有效的局部 sandwich；不是旧候选式的一般证明 |
| `tests/test_gamma_bound_proof_limits.py` | 历史平方逆反序实例、有限对抗搜索及归档字段回归；零违反不恢复被否定的候选式 |
| `tests/test_gamma_general_bound.py` | §9：精确反例的数值复现、固定 α 族、有效谱界穷举、零更新/可交换情形、输入校验、旧函数兼容 |
| `tests/test_goal_oriented.py` | §10: H=I parity, dense parity, row-mixing invariance, gradient FD; artifact headline fields |
