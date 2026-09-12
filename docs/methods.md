# Mathematical foundations

This page states the methods implemented in `calibinfo` precisely enough to
use the library without consulting any manuscript. Statements marked
**(proved)** are established in the unit tests listed at the end; statements
marked **(cited)** are classical results we instantiate. Notation follows the
table in the README.

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

  ![midpoint convexity of J_E](convexity_midpoint.png)
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

so `spec(R) = {1}^{P−rank(V)} ⊕ (1 − spec(VᵀV))`: exactly P − 3L modes sit at
ρ = 1 and the rest follow from a 3L × 3L eigenproblem. The same structure
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

For the **uniform** multiplier `t = κ·1` the per-light kernel is
`U_k(t) = u_k(M0_k + t Λ0_k)^{-1} u_k^T`, Loewner-decreasing in `t`. Since

    M0_k + t Λ0_k ⪰ (1/t)(M0_k + Λ0_k)  ⟺  (t−1)M0_k + (t²−1)Λ0_k ⪰ 0,  t ≥ 1,

we have `U_k(t) ⪯ t U_k(1)`, hence `ΔF(t) ⪯ t ΔF(1)` (Loewner) and
`J_A(t) ≥ J_A(1)/t` (tr X⁻¹ Loewner-decreasing). Therefore the certified
dynamic range obeys the **universal ceiling**

    D = 1 − J_A(κ·1)/J_A(1)  ≤  1 − 1/κ.

The "90.0% at κ=10" headline is this ceiling being **met** — when the
identity precision block `M0` dominates `Λ0` (large level = small `Λ0`),
the bound is tight to ≤ 1e-4 (`results/magnitude/calibration_value_ceiling.json`,
5 objects at level 64): κ=10 gives D ∈ [0.89992, 0.89999] against the
ceiling 0.9. The ceiling is a property of the functional, not an empirical
saturation artifact. The bound holds on every instance/admissible level
(machine-checked, CI-safe: `tests/test_math_foundations.py`).

## 8. What the library does not claim

- No claim that mode-resolved objectives beat scalar criteria — the
  submodularity structure required by such guarantees is violated for E-opt
  gains on real-shaped instances (documented negative result,
  `results/submodularity/submodularity_search.json`).
- No claim that the linearized theory predicts real-data error magnitudes —
  the matched-GLS variance identity holds on matched synthetic ensembles
  (ratio ≈ 1.0045) and the real-data deviation (emp/pred ≈ 102) is reported
  as a validity envelope.
- No claim of overall reconstruction advantage for any allocation policy —
  the certified result is about the information landscape itself plus the
  mode-targeted intervention endpoint.

## 9. α-approximate submodularity bound for the A-opt selection function

This section proves a **prior, computable lower bound** on the submodularity
ratio of the A-optimal light-refinement selection function. It answers the
"how close is greedy allocation to optimal?" question with a bound that
needs no ground truth, no measured data, and no exhaustive search — only the
nominal design (`A`, `B`, `Λ0`). It instantiates the Chamon & Ribeiro
(NeurIPS 2017) approximate-supermodularity framework with the calibration
precision `t` as the design variable; the novelty here is the explicit form
of `α`, not the framework (which is **not** claimed as first).

**Setup.** For a refined set `S ⊆ {1..L}` (light `k ∈ S` carries precision
multiplier `t_k = κ`, others `t_k = 1`), define the A-opt remaining cost

    F(S) = tr M(S)^{-1},   M(S) = ΔF(t_S),   G(S) = F(∅) − F(S).

**L9 (exact additive decomposition).** With `U_k(t) = u_k(M0_k + tΛ0_k)^{-1} u_k^T`
(Loewner-decreasing in `t`) and `A := ΔF(1)`,

    M(S) = A + Σ_{k∈S} W_k,   W_k = U_k(1) − U_k(κ) ⪰ 0.

**L10 (exact marginal gain).** By the Woodbury identity
`M^{-1} − (M+W)^{-1} = M^{-1}W(M+W)^{-1}`,

    Δ_x G(S) = tr[ M(S)^{-1} W_x (M(S)+W_x)^{-1} ].

**L11 (two-sided eigenvalue sandwich).** Let `ρ(S,x) = λmax(M(S)^{-1}W_x)`.
Simultaneously diagonalize `M^{-1}W_x` (both are symmetric: the product of
two PD/PSD symmetric matrices is diagonalizable with real eigenvalues, since
`M^{-1/2}(M^{-1}W_x)M^{1/2} = M^{-1/2}W_xM^{-1/2} ⪰ 0`), and use the fact
that `(I+X)^{-1}` has eigenvalues `1/(1+μ_j) ∈ [1/(1+ρ), 1]`:

    (1/(1+ρ)) · tr[W_x M(S)^{-2}]  ≤  Δ_x G(S)  ≤  tr[W_x M(S)^{-2}].

**L12 (monotonicity + uniform bound).** `S ⊆ T ⇒ M(S) ⪯ M(T)` (Loewner),
so `M(S)^{-1} ⪰ M(T)^{-1}` and the marginal gain has the correct
diminishing-returns direction; moreover `ρ(S,x) ≤ ρ_max := max_x λmax(A^{-1}W_x)`
(monotonicity of `λmax` under conjugation and `M(S) ⪰ A`).

**L13 (the bound).**

    γ  ≥  1/(1+α),    α := max_x λmax( ΔF(1)^{-1} W_x ).

`α` is a function of the nominal design only (`A`, `B`, `Λ0`, `κ`), so the
bound is available **a priori**, before any calibration data exists.

**Limit behavior.** `Λ0_x → ∞` (calibration excellent) or `Λ0_x → 0`
(calibration very poor) drive `W_x → 0`, hence `α → 0` and `γ → 1` (exact
submodularity); `α` peaks at intermediate precision levels. This is the
quantitative form of the assessment report's qualitative "low-SNR →
supermodular" tendency.

**Numerical verification** (`results/submodularity/alpha_bound.json`,
`experiments/alpha_bound.py`):

- **(a) theorem on the P-SUBMOD toy family** — 20 instances (10 random /
  10 adversarial, L=5, P=40), exhaustive triples: `γ_measured ≥ 1/(1+α)`
  on all 20 (random α ∈ [0.059, 0.119], adversarial α ∈ [0.025, 0.090];
  measured γ ∈ [0.9999, 1.0000]).
- **(b) two-sided sandwich** — 1600 random `(S,x)` pairs with zero
  violations (`min val/lb = 1.0021`, `min ub/val = 1.0368`).
- **(c) real objects** — on the 11 held-out OpenIllumination objects the
  nominal-design `α @ level=0.5` ranges 0.188–0.515 (γ lower bound
  0.66–0.84; `obj_10_pumpkin3` is the worst at 0.635 @ level=0.1). The
  most conservative statement is
  "A-optimal light-refinement selection is ≥ 0.635-supermodular on every
  held-out object at the probed levels."

**Honest framing.** The bound is about a factor 1.6 looser than the *measured*
`γ_min = 0.99989` (P-SUBMOD negative-result pack). Its value is the a-priori
computability (no exhaustive search), the explicit form of `α`, and the two
`α → 0` limits — not numerical tightness.

## Tests binding these statements

| File | Binds |
|---|---|
| `tests/test_information_modules.py` | dual-route equivalence, whitening, gauge closed form, mode tracking |
| `tests/test_math_foundations.py` | L1 direction pinning, per-light monotonicity, midpoint convexity, low-rank identity, gradient FD |
| `tests/test_convex_certificates.py` | LMO optimality, FW gap validity, certificate-vs-random-feasible-points |
| `tests/test_lowrank_identity.py` | M4 spectral identity, Woodbury trace/gradient |
| `tests/test_math_gates.py` | singular-covariance counterexample, retention-covariance theorem, polyfit order, rank invariance |
| `tests/test_known_answer_precheck.py` | λmin reading, trace dilution, parallel sums, gauge identity |
| `tests/test_alpha_bound.py` | L9–L13: γ ≥ 1/(1+α) on toy family, two-sided sandwich zero violations, real-object α in range |
