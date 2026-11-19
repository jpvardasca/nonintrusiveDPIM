# CONVENTIONS — single source of truth for the math (copied from PLAN.md §4; keep both in sync)

## 4. Conventions contract (the single source of truth for math)

### 4.1 Full-order model and signs

- **Thesis.** `M Ü + C U̇ + K U + g(U) + h(U) = F(t)`, with internal force `F_int(U) = K U + g(U) + h(U)`.
  - `g` is homogeneous quadratic and `h` homogeneous cubic. `U ∈ R^N` are the **free DOFs** (clamped DOFs removed), ordered as in §4.7.
- **MORFE.** `K U + C U̇ + M Ü = F(U, …)`, so the nonlinear closures must add **−g** and **−h**. The minus sign is applied in **one place only**: the Julia closure (`res .-= engine.quad(...)`). The Python engine always returns the **internal-force sign (+)**.
- **Damping.** `C = α M + β K` (Rayleigh), built in Julia from config.

### 4.2 Polarised forms and the multiplicity table

Define the **symmetric polarised forms**:

- `Ĝ(a,b)`: bilinear and symmetric, with `Ĝ(u,u) = g(u)`;
- `Ĥ(a,b,c)`: trilinear and fully symmetric, with `Ĥ(u,u,u) = h(u)`.

Then `K_tg(U) = ∂F_int/∂U = K + 2 Ĝ(U,·) + 3 Ĥ(U,U,·)`.

- **The engine's public API returns `Ĝ` and `Ĥ`**. This is what MORFE expects with `fully_asymmetric=false`.
- **The thesis/legacy "NSCs" are monomial coefficients:**

| Thesis/legacy symbol | Equals |
|---|---|
| `G̃_{i,i}` | `Ĝ(V_i,V_i)` |
| `G̃_{i,j}`, i<j | `2 Ĝ(V_i,V_j)` |
| `H̃_{i,i,i}` | `Ĥ(V_i,V_i,V_i)` |
| `H̃_{i,i,j}`, `H̃_{i,j,j}` | `3 Ĥ(·)` |
| `H̃_{i,j,k}`, all distinct | `6 Ĥ(V_i,V_j,V_k)` |

These factors are only applied when **producing or consuming thesis-style coefficients**: basis mode (§4.6) and legacy regression. Never apply them inside closures.

### 4.3 EED primitives for real directions (exact for cubic `F_int`)

Let `d` be a real direction with `max|d| = 1` and let `c = eps·d_ref` (default `eps = 0.24`). Write `K⁺(d) = K_tg(+c d)` and `K⁻(d) = K_tg(−c d)`.

- `G_op[d] = (K⁺(d) − K⁻(d)) / (4c)`. Then `G_op[d] @ v = Ĝ(d, v)`.
- `H_op[d,d] = (K⁺(d) + K⁻(d) − 2K) / (6c²)`. Then `H_op[d,d] @ v = Ĥ(d, d, v)`.
- **Pair** (d1 ≠ d2, both canonical). Choose `s = +1` if `max|c d1 + c d2| ≥ c`, else `s = −1`; this is the legacy rule. Compute one new job `K_mix = K_tg(c d1 + s c d2)`. Then
  `H_op[d1,d2] = (K_mix − K_tg(c d1) − K_tg(s c d2) + K) / (6 s c²)`, so that `H_op[d1,d2] @ v = Ĥ(d1, d2, v)`.
  Here `K_tg(c d1) = K⁺(d1)` and `K_tg(s c d2) = K⁺(d2)` if s=+1, otherwise `K⁻(d2)`.
- **K must be the backend's own `K_tg(0)`.** It comes from the same UEL and the same assembly, which is a job with U = 0. Otherwise `−2K` does not cancel.
- The maximum prescribed nodal displacement is `c` per probe and at most `2c = 0.48·d_ref` per pair. This is the thesis overlap rule.

### 4.4 Complex arguments (Appendix B generalised)

- **Decomposition.** For a complex vector `a = a_r + i a_i`, each nonzero part is written `a_r = α_r d_r` and `a_i = α_i d_i`, with canonical directions (§4.5). So `a = Σ_x κ_x d_x` with `κ = (α_r, i·α_i)`.
- **Quadratic.** `Ĝ(a,b) = Σ_x κ_x · (G_op[d_x] @ b)`. Only **one** argument (the probe) needs Abaqus data, and `b` stays complex.
- **Cubic.** `Ĥ(a,b,c) = Σ_x Σ_y κ_x λ_y · (H_op[d_x, d_y] @ c)`, where `(a,b)` is the chosen pair and `b = Σ_y λ_y d_y`.
  - If `d_x ≡ d_y` (same canonical direction), use `H_op[d,d]`; no pair job is needed.
  - This reproduces every case in Appendix B, including `Ĥ(a,a,·) = Ĥ(a_r,a_r) − Ĥ(a_i,a_i) + 2i Ĥ(a_r,a_i)`.
- **Choosing probe and pair** (deterministic, independent of cache state):
  - Every argument gets a registration id at first sight (§4.5).
  - Quadratic: the probe is the argument with the smaller id.
  - Cubic: if any id repeats, the pair is that argument twice. Otherwise the pair is the two smallest ids. The remaining argument is contracted.
  - Because MORFE solves in graded-lex order, the lowest id is the lowest-order coefficient. Only coefficients of order ≤ p/2 are ever probed, matching the legacy choice `sigma_idx <= nu_idx`.
- **Zero handling.**
  - If `max|a| == 0`, return zeros without registering.
  - A part is treated as zero if `max|part| ≤ zero_tol · max|a|` (default `zero_tol = 1e-13`).

### 4.5 Canonical directions and the registry

- **Canonical form.** For a real part `p`: `m = max|p|`; `d = p/m`; `σ = sign(w·d)` where `w` is a fixed seeded random vector (`numpy.random.default_rng(12345)`, length N); then `d ← σ d` and `α = σ m`.
  - This gives `p = α d`, `max|d| = 1`, and a sign fixed independently of ties.
  - A conjugate vector (`Im` negated) and real/imag parts that are proportional to each other map to the **same** direction. This is how conjugate monomials reuse jobs (legacy `global_index_cc`) automatically.
- **Matching.** A new `d` matches a stored direction if `‖d − d_k‖_∞ ≤ match_tol` (default `1e-9`). Pre-filter candidates by `‖d‖₂` within `match_tol·√N`. **Never key on exact bytes of MORFE vectors**, because reruns may differ by round-off.
- **Two kinds of registry.** Registration ids (§4.4) live per *vector* (complex arguments, matched the same way on the complex vector). The direction store persists to disk, so a resumed run reuses jobs.
- **Job keys.** The prescribed displacement of each job is computed from the *stored* direction. Its job key is `sha256(model_fingerprint ‖ U.tobytes())`, which is exact and reproducible.

### 4.6 Basis mode: the StEP.tex §NSCs variant

For a real basis `V = [V_1..V_n]` (N × n; modes, static derivatives, anything), register `V_1..V_n` in order. Then:

| Coefficient | Computation |
|---|---|
| `G̃_ii` | `quad(V_i,V_i)` |
| `G̃_ij` | `2·quad(V_i,V_j)` |
| `H̃_iii` | `cubic(V_i,V_i,V_i)` |
| `H̃_iij` | `3·cubic(V_i,V_i,V_j)` |
| `H̃_ijj` | `3·cubic(V_i,V_j,V_j)` |
| `H̃_ijk` | `6·cubic(V_i,V_j,V_k)` |

All are N-vectors, with `i ≤ j ≤ k`.

In code (`nistep/basis_mode.py`) the indices are **0-based** column indices of `V`: the thesis `G̃_11` is `coeffs["G", 0, 0]`.

- **Job count must equal** `N_EED = 2n + C(n−1,2) = (n²+n+2)/2`: probes ± for each `V_i`, plus one pair job for each `i<j≤n−1`.
- Also report `N_ED = (n³+6n²+5n)/6` for comparison only. ED itself is not implemented.
- These coincide with StEP.tex: single-vector `±V_i c_i`; two-vector contraction of `K_tg(±V_i c_i)` with `V_j`; three-vector `K_tg(V_i c_i + V_j c_j) V_k`.
- **Galerkin export** (optional ROM route):
  - `M_r = VᵀMV`, `K_r = VᵀKV`, `C_r = VᵀCV`.
  - `g_r[:, i, j] = Vᵀ Ĝ(V_i, V_j)` (symmetric in i, j).
  - `h_r[:, i, j, k] = Vᵀ Ĥ(V_i, V_j, V_k)` (fully symmetric).
  - These are dense arrays for small n. The constraint-force caveat of the StEP.tex footnote applies: this route is an approximation.

### 4.7 DOF ordering (one `DofMap` object, used everywhere)

- Free DOFs are nodes in **ascending node label**, excluding clamped nodes (deduplicated), each with DOFs 1, 2, 3.
  - `free_index(node, dof) = 3·rank(node) + (dof−1)`, where `rank` is the 0-based position among free nodes.
- The same map is used for: the BC lines written to `.inp`; assembly of K_tg/M/K from element matrices (global index for all nodes, then elimination); the monitor DOF; and vectors exchanged with Julia. Julia never re-orders.

### 4.8 Complex-safety and shape rules

- Python: `np.asarray(x, dtype=np.complex128)`; real sparse `@` complex vector is fine; never `.conj()` on arguments.
- Julia: `transpose`, not `'`; `collect(view)` before passing to Python; the result is a `Vector{ComplexF64}` of length N.
- Every engine output is checked in debug mode: `shape == (N,)`, all values finite.

---

## Thesis errata (do not implement these lines literally)

- `StEP.tex:201`: for n=10, `N_EED = (n²+n+2)/2 = 56`, not 65 (65 = 2n + C(n,2)). The reduction versus ED (275) is about 80%, not 76%.
- `StEP.tex:59`: the printed projection `G̃_{i,j} = Σ G_{r,s} V_{i,r} V_{j,s}` omits the swapped term for i≠j. It is not used by the code.
- `Thesis_Appendix_B.tex:88`: the last term of `K♠_{σr,σi}` should use `Im W_σ`, not `Re W_σ`.
- `step_flowchart_1.png`: the "σ<ν" G formula should contract with `W_ν`, not `W_σ`.

