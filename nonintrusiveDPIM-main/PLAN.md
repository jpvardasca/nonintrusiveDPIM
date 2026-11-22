# PLAN — Non-intrusive closure FE interface: Python EED-StEP ⇄ MORFE.jl

> This is the working plan. Tick checkboxes as steps finish and record the results under each step.
> Plan approved by the user on 2026-11-09.

---

## Status on the Abaqus PC (updated 2026-11-22)

This section is for the Claude session on the Abaqus PC, so it can carry on from the session
on the development PC. Chats are not shared between machines; this note and `git` are the
hand-over. Read §0 (rules) before doing anything.

**The machine.**

- Repository: `C:\Users\joanavpereira\Documents\nonintrusiveDPIM\nonintrusiveDPIM-main`,
  branch `feature/morfe-closure`.
- Windows with PowerShell.
- `abaqus` is on `PATH` (`C:\SIMULIA\Commands\abaqus.bat`), with Intel Fortran Classic
  2021.4.0.
- The Abaqus version is not recorded yet: run `abaqus information=release`.

**Already done there (by the user, guided from the development PC).**

1. Python virtual environment `.venv` (Python 3.11) with `nistep` installed.
2. Julia 1.11.6 environment `julia/MORFEStEP` instantiated and precompiled.
3. Tests before the fix below:
   - `pytest`: 193 passed.
   - Julia `Pkg.test()`: 621/621. The near-resonance warnings are expected.
4. P8.2, the UEL libraries. `abaqus make` COMPLETED for both sources:
   - `C:\Users\joanavpereira\Documents\nistep_uel\stiffness\standardU.dll` (12 272 640 bytes);
   - `C:\Users\joanavpereira\Documents\nistep_uel\mass\standardU.dll` (12 270 080 bytes).
5. Local config edits. These are **not in git; keep them, do not commit them**:
   - `configs/*_abaqus.toml` have `user_lib_dir` and `mass_lib_dir` set to those two folders, and
     `command = "abaqus"`.
   - `configs/single_element_abaqus.toml` has `keep_job_dirs = "all"` for diagnosis. Set it back to
     `"failed"` once the smoke test passes.

**Found on the first smoke test (P8.6, single element), and fixed.**

- **The failure.** The mass job failed with `matrix type STIFFNESS, expected MASS`.
  - Its `abaqus_v6.env` pointed at the mass folder, as it should.
  - The cause is that Abaqus labels the element-matrix output from the deck's
    `*Element Matrix Output` request (`Stiffness=Yes`), not from the UEL library.
- **The fix.** It is in the commit after `7eb2fab`:
  - `JobDeck.render(..., output="mass")` writes `Mass=Yes` for mass jobs;
  - the backend also checks the matrix values, so a swapped library is still caught
    (`translation_ratio`: a tangent stiffness maps rigid translations to zero, a mass does not;
    threshold `TRANSLATION_TOL = 1e-2`);
  - the mock Abaqus now labels its output the same way.
  - Result: Python suite 196 passed on the development PC.
- **What the same run already confirms** (record this under P8.3 once the full run passes):
  - the single-element jobs passed all four runner checks, so `usub_lib_dir` picks up the
    precompiled library and the `.sta` success sentence matches;
  - the raw `.mtx` file of a real Abaqus parsed without error.
- **VERIFY:** whether `Mass=Yes` with the mass library gives the consistent mass (check
  `mass_vs_pysvk`).

**Next, in order (on the Abaqus PC).**

1. Get the fix: `git pull`. The local config edits stay.
2. Run the Python tests: `.venv\Scripts\python -m pytest python/tests -q`. Expect 196 passed. The
   summary line may be hidden: check that the exit code is 0.
3. Run the single-element smoke test:
   `.venv\Scripts\python scripts\abaqus_smoke_test.py configs\single_element_abaqus.toml --json work\smoke_single.json`.
   - Exit code 0 means every check passed.
   - Record under P8.6: each printed value, the seconds per job and the Abaqus version.
   - If a check fails, stop and report the values to the user. Use the table in
     `scripts/build_uel_library.md` §4. **Never edit the Fortran UEL without asking (rule 3).**
4. If step 3 passes:
   - set `keep_job_dirs = "failed"` again;
   - run the same command with `configs\mems_abaqus.toml` (adds check `d_K0_vs_processed`);
   - record it under P8.6.
5. Tick P8.2 and P8.6. Record under P8.3 the outcome of its three VERIFY items.
6. Then the host items, one at a time:
   - P10.1–P10.4;
   - P11.1, the forced order-convergence run (`configs/mems_forced_abaqus.toml`);
   - P11.3.
7. Open decisions to ask the user when they matter:
   - D3: licence tokens, which set `max_parallel`;
   - D4: HBM export layout;
   - D5: legacy or MORFE2.0 results.
8. Before long unattended runs, make sure the PC does not sleep:
   - `powercfg /change standby-timeout-ac 0`;
   - `powercfg /requests` to check.

**Rules that matter here.**

- Commit or push only when the user asks.
- Never add thesis material to git (rule 4).
- Never change `test/`, `module/` or the Fortran.
- Every commit message ends with the attribution line the session gives.

---

## 0. Rules for the implementing agent

1. **One step at a time.** Do the step, run its acceptance checks, and record the results under the step: tick the box and paste the test summary. Then move to the next step. Never start a phase whose predecessors have failing checks.
2. **Never modify the legacy code** in `test/` or `module/`, except where a step says so explicitly. It is the reference implementation and the regression oracle.
3. **Never modify the Fortran UEL without asking the user** (see §9). Report UEL bugs; do not silently fix them.
4. **Never commit or push unless the user asks.** **Git must never track thesis material:** `Tese_limpa/`, any LaTeX source (`*.tex`, `*.bib`, `*.cls`, `*.sty`), PDFs or LaTeX build artefacts. These are ignored in `.gitignore` (user requirement, 2026-11-09). Never force-add them, and check `git status` before any commit. Reading them for reference is fine.
5. **Read §4 (Conventions contract) before writing any math.** Every factor of 2, 3, 6 and every sign in this project is defined there and nowhere else. If code and §4 disagree, §4 wins. If §4 looks wrong, stop and ask the user.
6. **Complex safety.** Never conjugate MORFE vectors: no `.conj()`, `np.vdot`, Julia `'`, `dot(a,b)` or `norm`-based inner products on the arguments of the nonlinear terms. Use `transpose`, `@` and `np.dot`.
7. **Prefer small, tested modules over clever code.** Each new Python module gets a `pytest` file; each Julia file gets a `@testset`.
8. **When an external API detail is marked "VERIFY"**, read the installed package source before using it: `~/.julia/packages/MORFE/<hash>/src/...`, or `pip show -f`.
9. **Windows specifics.** Use `pathlib`. Avoid long paths (keep work directories short, e.g. `C:\nistep\work`). Abaqus is a `.bat`, so call it through `cmd /c` or `shell=True` on Windows. Set `PYTHONUTF8=1` for any script that prints unicode; the console default is cp1252.

---

## 1. Context

The repository (thesis by J. R. Pereira, IST 2025, supervised by A. Vizzaccaro, T. Martins and F. Trainotti) builds nonlinear ROMs from Abaqus models non-intrusively:

- An Abaqus **UEL** (Total-Lagrangian Saint-Venant–Kirchhoff hex8) provides tangent stiffness matrices `K_tg(U)` at prescribed displacements.
- The **EED-StEP** (Enhanced Enforced Displacement Stiffness Evaluation Procedure) turns differences of `K_tg` into the quadratic and cubic nonlinear forces.
- A home-grown **Python DPIM** solves the cohomological equations order by order, calling the EED-StEP on the complex manifold coefficients `W_σ` at every order. This is a "closure" evaluation, not a fixed-basis tensor.

The Python DPIM is slow and fragile: dense solves, broken imports, files mutated in place, and no failure detection. **MORFE.jl** (MORFEproject, v0.2.0, by the same research group) is a mature Julia DPIM whose full-order model accepts nonlinear terms as **closures** `f!(res, a, b[, c])`. Those closures are evaluated on arbitrary complex full-order vectors and never need derivatives. Its tutorial calls this the `closure_model`. That is exactly the shape of the EED-StEP evaluation.

**Goal.** MORFE.jl owns the DPIM. A new, clean Python package owns the FE side: Abaqus I/O, EED-StEP and caching. A thin Julia package connects them through PythonCall. Every EED-StEP variant from the code and from `Tese_limpa/StEP.tex` / Appendix B is implemented, tested against exact analytic models, and validated end-to-end.

**Decisions taken with the user (2026-11-09):**

- Julia drives; the Python EED engine is called in-process via **PythonCall.jl**.
- **Abaqus runs on a separate machine.** Development and testing on this PC use mock backends; Abaqus jobs are shipped to the Abaqus host.
- The missing legacy source `test/Compute_all_options_K_tg.py` has been restored by the user and is the legacy reference.

---

## 2. Big picture

### 2.1 Architecture

```
┌──────────────────────────── Julia process (driver) ────────────────────────────┐
│  scripts/run_rom.jl  ──►  MORFEStEP.jl (thin glue, julia/MORFEStEP/)            │
│     • reads config TOML                                                         │
│     • gets K, M (sparse) from Python, builds C = αM + βK                        │
│     • builds NthOrderModel((K,C,M), (quad, cubic[, force]) [, ExternalSystem])  │
│     • spectrum / SpectralData / parametrise (MORFE.jl)                          │
│     • prefetch passes (record → batch jobs → rerun), exports W, R               │
│                                                                                 │
│   quad  = MultilinearMap((res,a,b)  -> res .-= engine.quad(a,b))                │
│   cubic = MultilinearMap((res,a,b,c)-> res .-= engine.cubic(a,b,c))             │
│                 │  PythonCall (in-process, same thread, zero-copy arrays)       │
└─────────────────┼───────────────────────────────────────────────────────────────┘
                  ▼
┌──────────────────────────── Python package nistep (python/nistep/) ────────────┐
│  EEDEngine  (closure evaluator: Ĝ(a,b), Ĥ(a,b,c) for complex N-vectors)        │
│     ├─ DirectionRegistry (canonical real directions, tolerance matching)        │
│     ├─ operator algebra: G_op[d], H_op[d,d], H_op[d1,d2]  (§4.3)               │
│     ├─ basis-mode EED (StEP.tex §NSCs): G̃_ij, H̃_ijk on a real basis V         │
│     └─ KtgStore (memory LRU + content-addressed disk cache, shared pattern)     │
│  TangentBackend (protocol: K0(), M(), tangent_many([U...]) -> [K_tg])           │
│     ├─ AnalyticBackend   (known polynomial G,H — exact; tests)                  │
│     ├─ PySVKBackend      (numpy hex8 TL-SVK on the real mesh — Abaqus stand-in) │
│     ├─ ReplayBackend     (serves K_tg recorded from earlier Abaqus runs)        │
│     └─ AbaqusBackend     (inp writer + Runner + .mtx parser + assembler)        │
│            Runner: LocalRunner | SpoolRunner(shared folder) | SSHRunner(opt.)   │
└─────────────────┼───────────────────────────────────────────────────────────────┘
                  ▼  (files: job .inp in → element-matrix .mtx out)
┌──────────── Abaqus host (separate machine) ────────────┐
│ scripts/abaqus_worker.py (stdlib only) + precompiled   │
│ UEL library (test/uel_nlmech_pk1.for + module/*.for)   │
└────────────────────────────────────────────────────────┘
```

### 2.2 One closure call, end to end

1. During `parametrise`, MORFE calls `quad.f!(res, W[:,1,β1], W[:,1,β2])`. The arguments are complex N-vectors (free DOFs). MORFE assumes the map is symmetric and multiplies by the multinomial factor itself.
2. The Julia closure passes both vectors to `engine.quad(a, b)` through PythonCall.
3. The engine registers `a` and `b` and splits each into real and imaginary parts. Each nonzero part is a scalar times a canonical real direction `d`. It picks the argument with the lowest registration id as the *probe*.
4. For each probe direction `d` it needs `K_tg(+c·d)` and `K_tg(−c·d)`. On a cache hit they are loaded; on a miss, jobs run through the backend.
5. It returns `Ĝ(a,b) = Σ_parts coef · G_op[d] @ b`, the complex N-vector of the symmetric polarised quadratic internal force.
6. Julia does `res .-= result`. The minus sign moves the internal force to MORFE's right-hand side.

### 2.3 Where things run

| Stage | Machine | Backend |
|---|---|---|
| Unit and integration tests (P1–P7) | this PC | Analytic, PySVK |
| Abaqus smoke tests, UEL library build (P8) | Abaqus host | Abaqus (LocalRunner) |
| Production ROMs (P10–P11) | Julia+Python here *or* on host (decide at P8, §9) | Abaqus via SpoolRunner/SSHRunner *or* LocalRunner |
| Offline debugging of recorded runs | this PC | Replay |

---

## 3. What exists today (analysis findings)

### 3.1 Legacy pipeline (`test/`, flat scripts, run from `test/` as the working directory)

- **`main-DPIM.py`** drives the legacy pipeline:
  - It is hard-coded to the MEMS model, `max_order=1`, `d_ref` (`minimum_distance`) = 10, `C = aM + bK`.
  - Setup: `WriteInputFile.get_node_information()` (reads `*Nset, nset=Set-2` as clamped) → `EmbeddedStiffnessMatrix` (K, M from processed `.mtx`, clamped DOFs removed) → `SecondOrderDPIM(...)`.
  - The DPIM loop is in `Compute_higher_order_terms.HigherOrderSolver`. Per γ: `rhs = xi − sum_H − sum_G`, then a dense `np.linalg.solve`.
  - Outputs: `Wu_matrix.txt`, `output_for_FRF.txt`, `my_array.pkl`.
- **The EED-StEP side** (closure style, Appendix B of the thesis):
  - `Compute_all_options_K_tg.py`, class `TangentStiffnessMatrixHandler`, has three options:
    - option 1: `±c·Re(W_σ)`, `±c·Im(W_σ)` (4 jobs);
    - option 2: `c_r Re W_σ + s c_i Im W_σ` (1 job);
    - option 3: `c_σ part(W_σ) + s c_ν part(W_ν)` (up to 4 jobs).
  - The scale factor is `c = 0.24·d_ref / max|part|`. The mixed sign rule is `s = +1 if max|sum| ≥ 0.24·d_ref else −1` (lines 354–369, 493–510).
  - G formulas: `StEP_validation_with_complex_numbers_abaqus_G_terms.py:134-184`. H formulas: `StEP_validation_complex_numbers_abaqus_H_terms.py`. Both were verified consistent with Appendix B. The divisions by `2c` (rather than `4c`) carry the monomial multiplicities (§4.2).
- **Abaqus I/O:**
  - `Write_BC_SETS_Input_file.py` adds a `*Nset, nset=node<i>` per node and a BC line per free DOF.
  - `Duplicate_Input_File.InputFileModifier` rewrites the `** Boundary conditions` … `** Output requests` block and the `File Name=` of `*Element Matrix Output`.
  - `Run_jobs.run_abaqus_job` calls `abaqus interactive double analysis ask_delete=off job=… user=uel_nlmech_pk1.for output_precision=full` through `os.system`.
  - `Organize_the_matrix_right_order.MatrixProcessor` parses the raw element-matrix `.mtx` and assembles it node-major (`3·rank(node) + dof−1`).
  - `EmbeddedStiffnessMatrix_Class` removes the clamped DOFs.
- **UEL** (`test/uel_nlmech_pk1.for`, which includes `../module/*.for`):
  - Formulation: TL-SVK hex8, 8 Gauss points, `AMATRX = K_tg`, `RHS = −F_int`.
  - The internal force is **exactly cubic** in U, so the EED identities are exact up to round-off.
  - The mass variant `uel_nlmech_pk1_mass_matrix.for` returns the consistent mass.
- **Repository models:**

  | File | Elements | Nodes | Clamped nset | Free DOFs | Note |
  |---|---|---|---|---|---|
  | `mems_abaqus_UEL_modified.inp` | 240 | 525 | Set-2, 18 unique (20 listed, 2 duplicated) | 1521 | thesis ω₁ = 1.819006, d_ref = 20 |
  | `Cantilever_Beam_500_elements_UEL_modified.inp` | 500 | 918 | Set-2, 18 nodes | 2700 | thesis ω₁ = 8.425818, d_ref = 10 |
  | `Clamped_Clamped_beam_200_elements_UEL_modified.inp` | 200 | 606 | Set-2, 12 nodes | 1782 | thesis used a 280-element mesh, so do not compare numbers with the thesis |
  | `Meu_input_file.inp` | 1 | 8 | 4 nodes | 12 | single-element tests |

  Matching processed `*_mass_matrix.mtx` / `*_stiffness_matrix.mtx` files exist for MEMS, cantilever and clamped-clamped (all nodes; text `i j v`, 1-based).

### 3.2 Legacy bugs: **do not port these behaviours**

| # | Where | Bug | New behaviour |
|---|---|---|---|
| B1 | `Compute_all_options_K_tg.py:274` | `np.all(np.any(disp == 0))` skips the job if *any* entry is 0, so a stale or missing file is loaded | Skip only if the whole part is zero (relative tolerance) |
| B2 | `Duplicate_Input_File.py:115`, `Compute_all_options_K_tg.py:280` | The base `.inp` is overwritten in place and the job always runs under the base name | Never touch the base file; one unique job directory and name per job |
| B3 | `Run_jobs.py:13` | `os.system`, exit code ignored; failures are not detected and stale `.mtx` files are reused | `subprocess.run`; check the exit code, `.sta`/`.msg`, and the `.mtx` element count |
| B4 | `Organize_the_matrix_right_order.py:144-150` | Elements with a wrong value count are silently skipped | Raise an error |
| B5 | `Organize_the_matrix_right_order.py:256-294` | Drops `abs(v) < 1e-20`, writes the result in place, parses with `Decimal` (slow) | Keep explicit zeros (stable sparsity pattern), never rewrite outputs, vectorised parsing |
| B6 | `WriteInputfile_Class.py:71-72` | Free nodes held in a `set`, so order is unspecified | `sorted(...)` everywhere |
| B7 | `G_terms.py:83/96/113` | Returns a matrix on one path and a tuple on another | Typed return values |
| B8 | file names `base_<idx>.mtx` | Collide across configurations | Content-addressed keys (sha256 of U bytes + model fingerprint) |
| B9 | `uel_nlmech_pk1.for:240` | `fbarFlag` is a local logical that is never initialised; if it is garbage-true, F-bar is enabled and the polynomial structure breaks | Detect at P8 with the cubic-exactness test (§8); ask the user before fixing |
| B10 | `uel_nlmech_pk1_mass_matrix.for` | `rho` is declared integer, so densities are truncated | Harmless for 7850/7860/2320; report it; ask before fixing |

### 3.3 Thesis errata found (implement from §4, not from these lines)

- `StEP.tex:201`: for n=10, `N_EED = (n²+n+2)/2 = 56`, not 65 (65 = 2n + C(n,2)). The reduction versus ED (275) is about 80%, not 76%.
- `StEP.tex:59`: the printed projection `G̃_{i,j} = Σ G_{r,s} V_{i,r} V_{j,s}` omits the swapped term for i≠j. It is not used by the code.
- `Thesis_Appendix_B.tex:88`: the last term of `K♠_{σr,σi}` should use `Im W_σ`, not `Re W_σ`.
- `step_flowchart_1.png`: the "σ<ν" G formula should contract with `W_ν`, not `W_σ`.

### 3.4 MORFE.jl facts that drive the design (v0.2.0, registered in General, Julia ≥ 1.10)

- **Model form.** `B0 u + B1 u̇ + B2 ü = F(u, u̇, r)`. Nonlinear terms are on the **right-hand side**. Structural: `NthOrderModel((K, C, M), terms[, ext])`. All three matrices must have the **same type** (e.g. `SparseMatrixCSC{Float64,Int}`).
- **Term API.** `MultilinearMap(f!; multiindex=(2,0) | (3,0), multiplicity_external=0, fully_asymmetric=false)`.
  - `f!` must **add** into `res`. It must be multilinear and symmetric.
  - MORFE evaluates each unordered factor tuple once and multiplies by the multinomial coefficient.
  - The arguments are complex `ComplexF64` column views of `W`. They are **not** in any modal subspace.
  - No Jacobians are ever requested.
- **Call pattern.** One call per factorisation per solved monomial. The solve is single-threaded and no values are cached, so the cache is ours to build.
  - Example from the MORFE tutorial: 486 calls for one mode pair at order 9 with quadratic and cubic terms.
- **Forcing.** `ExternalSystem((im*Ω, -im*Ω))` plus a term `MultilinearMap((res, r) -> …, (0,0), 1)`. Index `r[j]`; never sum it, because MORFE passes unit vectors. The force columns are `(F̂/2)·M·φ`.
- **Spectrum.**
  - `spectrum(model)` is dense and only suitable for small models.
  - `spectrum(model, StructureModalDampingEigensolver(nev, α, β))` needs `using Arpack, LinearMaps`; Arpack is pinned to 0.5.3. Alternatively, build `SpectralData(; diagonal, right_modes, …)` from eigen data computed in Python.
  - Then `SpectralData(model, sp; master = master_by_sorting(2))`.
- **Solve.** `W, R = parametrise(model, sd, order_or_mset; resonance = ResonanceConfig(style=:complex_normal_form, tol=0.05), options = ParametrisationOptions(...))`.
  - Pass the conjugate permutation (`:detect`) to halve the work. This requires a real FOM, i.e. `f!` maps real to real.
- **Outputs and helpers.**
  - `W.poly.coefficients` is `N × ORD × L`; `R.poly.coefficients` is `NVAR × L`.
  - Exponents: `W.poly.multiindex_set.exponents`.
  - `save_rom` writes a CSV of R that Python can read; W must be exported by us.
  - Also available: `realify`, `observable_polynomial`, `normal_form_branch`.
- **Not applicable here.** The `FEMMultilinearMap` element-loop interface needs quadrature-point kernels in Julia; a black-box Abaqus UEL cannot provide them. **Use the closure route.**
- **Known issue.** KLU fails on Julia 1.12.0–1.12.6. **Stay on the installed Julia 1.11.6.**

### 3.5 Environment facts (this PC)

- Julia 1.11.6. The global environment already has PyCall, Arpack, NPZ, MAT, MatrixMarket, JLD2, HDF5 and JSON. Our project uses **PythonCall**, in its own project environment. Never load PyCall and PythonCall in the same session.
- Python 3.11.9 is the **Windows Store** build, with numpy 2.2.2 and scipy 1.15.1. Embedding the Store build is unreliable, so PythonCall uses its own CondaPkg-managed Python by default.
- Abaqus here is the **2024 Learning Edition**, and there is no Fortran compiler on PATH. Do not use it for production; the Abaqus host is a separate machine.
- The git repository root is `nonintrusiveDPIM/` (the parent folder); the working directory is `nonintrusiveDPIM-main/`. `README.md` (repo root) describes an aspirational layout (`abaqus/`, `step/`, `dpim/`, `hbm/`) that does not exist.

---

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

## 5. Target layout (new files only; legacy untouched)

```
nonintrusiveDPIM-main/
├─ PLAN.md                       (this plan)
├─ CONVENTIONS.md                (copy of §4, kept in sync)
├─ configs/                      (TOML run configs: mems.toml, cantilever.toml, cc200.toml, single_element.toml, analytic_*.toml)
├─ python/
│  ├─ pyproject.toml             (package "nistep", deps: numpy, scipy; dev: pytest)
│  ├─ nistep/
│  │  ├─ __init__.py             (build_engine(config_path) factory)
│  │  ├─ config.py               (dataclasses + tomllib loader + validation)
│  │  ├─ dofmap.py               (DofMap, §4.7)
│  │  ├─ inp_io.py               (parse base .inp; prepare per-node Nsets; render job .inp text)
│  │  ├─ mtx_io.py               (raw element-matrix .mtx parser; processed "i j v" loader)
│  │  ├─ assembly.py             (element blocks → global CSR with fixed pattern; elimination)
│  │  ├─ store.py                (KtgStore: shared pattern, LRU, disk cache, manifest)
│  │  ├─ registry.py             (canonical directions + vector ids, §4.5)
│  │  ├─ engine.py               (EEDEngine: quad, cubic, modes lazy/record, pending, stats)
│  │  ├─ basis_mode.py           (§4.6 + Galerkin export + job-count formulas)
│  │  ├─ backends/{base,analytic,pysvk,replay,abaqus}.py
│  │  ├─ runners/{base,local,spool,ssh}.py
│  │  └─ diagnostics.py          (cubic-exactness test, d_ref sweep, symmetry/noise checks, preflight det F)
│  └─ tests/                     (pytest; one file per module + integration)
├─ julia/MORFEStEP/
│  ├─ Project.toml / Manifest.toml (MORFE 0.2.0, PythonCall, CondaPkg, Arpack 0.5.3, LinearMaps, SparseArrays, LinearAlgebra, TOML)
│  ├─ CondaPkg.toml              (python 3.11, numpy, scipy)
│  ├─ src/MORFEStEP.jl           (bridge, closures, model, driver, io)
│  └─ test/runtests.jl
└─ scripts/
   ├─ run_rom.jl                 (julia --project=julia/MORFEStEP scripts/run_rom.jl configs/mems.toml)
   ├─ abaqus_worker.py           (runs ON the Abaqus host, stdlib only)
   ├─ build_uel_library.md/.bat  (one-time UEL compile instructions/script for the host)
   └─ export_for_hbm.py          (reads run outputs → arrays for the user's HBM tool)
```

---

## 6. Phases and steps

Each step lists **Do**, **Accept** (must pass before moving on) and **Pitfalls**.

### P0 — Setup and baseline (this PC)

- [x] **P0.1** Copy this plan to `nonintrusiveDPIM-main/PLAN.md`, and §4 to `CONVENTIONS.md`.
  - *Done 2026-11-09:* `PLAN.md` (751 lines) and `CONVENTIONS.md` (§4 plus the §3.3 errata, 115 lines) created in `nonintrusiveDPIM-main/`. Nothing committed.
- [x] **P0.2** Create a branch `feature/morfe-closure` (no commits unless the user asks). Add a `.gitignore` covering `__pycache__/`, `*.pyc`, `runs/`, `work/`, `cache/`, `.venv/`, and the Abaqus scratch files (`*.odb *.msg *.sta *.dat *.prt *.com *.sim *.log`) under `work/`. Do **not** ignore `julia/MORFEStEP/Manifest.toml`; it pins the MORFE version.
  - *Done 2026-11-09:*
    - Branch `feature/morfe-closure` created from `main` (8c934cc).
    - `nonintrusiveDPIM-main/.gitignore` added. It also ignores `.pytest_cache/`, `*.egg-info/` and `julia/MORFEStEP/.CondaPkg/`.
    - At the user's request it also ignores **all thesis material** (`Tese_limpa/`, `*.tex`, `*.bib`, `*.cls`, `*.sty`, `*.bst`, `*.pdf`) and the LaTeX build artefacts. `git check-ignore` confirms `Tese_limpa/*` is ignored; no such file was ever tracked.
    - Nothing committed.
- [x] **P0.3** Python dev environment: `python -m venv .venv`, then `pip install numpy scipy pytest`. Check `pytest --version`.
  - *Done 2026-11-09:* `.venv` created from Python 3.11.9 (Windows Store build). It has pytest 9.2.0, numpy 2.4.6 and scipy 1.17.1. Use `.venv/Scripts/python`.
- [x] **P0.4** Julia environment: `julia --project=julia/MORFEStEP -e 'using Pkg; Pkg.add(["MORFE","PythonCall","CondaPkg","Arpack","LinearMaps"]); Pkg.add(["SparseArrays","LinearAlgebra","TOML"])'`. Record the MORFE version and tree hash in PLAN.md.
  - *Done 2026-11-10:*
    - The package was generated with `Pkg.generate` (uuid `c76c546a-3d71-4209-9460-519d858b4494`). The email was removed from `authors`.
    - **MORFE v0.2.0**, git-tree-sha1 `cf88ee822e45802cb07b912de1abdf6c83d38712`, installed at `~/.julia/packages/MORFE/2Jh0a/`.
    - Other packages: PythonCall 0.9.37, CondaPkg 0.2.38, Arpack 0.5.3 (pinned; upgrades exist but MORFEArpackExt needs 0.5.3), LinearMaps 3.11.4, NPZ 0.4.3 (added for the exact-closure comparisons), SparseArrays, LinearAlgebra, TOML.
    - `[compat]` is pinned to these versions. Julia 1.11.6.
    - Extensions present: `MORFEArpackExt` (Arpack + LinearMaps) and `MORFEBifurcationKitExt`.
- [x] **P0.5** MORFE smoke test: run the README Duffing example (2 DOF, order 5, complex normal form).
  - **Accept:** it runs, and `R.poly.coefficients` has a nonzero cubic backbone term.
  - *Done 2026-11-10:* `julia --project=julia/MORFEStEP julia/MORFEStEP/test/smoke_morfe.jl` gives `R[1,(2,1)] = 2.8e-17 − 0.50086i` (nonzero). Nonzero monomials: (1,0), (0,1), (2,1), (1,2), (3,2), (2,3).
    - Observed API facts:
      - The README form `MultilinearMap(f!, (3,0); fully_asymmetric=false)` uses a **positional** multiindex. **VERIFY** at P4 whether the keyword form also exists.
      - `R.poly.coefficients` is a `Matrix{ComplexF64}` of size (2, 20).
      - `W.poly.coefficients` is an `Array{ComplexF64,3}` of size (N=2, ORD=2, 20).
      - `R.poly.multiindex_set.exponents` is a `Vector{SVector{2,Int}}`.
      - The first master eigenvalue is `−0.0005 − 0.99999i` (negative imaginary part first).
- [x] **P0.6** PythonCall smoke test: in `julia --project=julia/MORFEStEP`, `using PythonCall; np = pyimport("numpy"); pyconvert(Vector{ComplexF64}, np.array([1+2im]))`.
  - **Accept:** it works. CondaPkg may download micromamba on first run, which needs internet.
  - *Done 2026-11-10:* `julia/MORFEStEP/test/smoke_pythoncall.jl` passes.
    - `julia/MORFEStEP/CondaPkg.toml` declares `python >=3.11,<3.13`, `numpy >=2.0` and `scipy >=1.13`. CondaPkg (pixi backend) installed **Python 3.12.16, numpy 2.5.4, scipy 1.18.1** in `julia/MORFEStEP/.CondaPkg/` (git-ignored).
    - `pyconvert(Vector{ComplexF64}, …)`, the zero-copy `PyArray{ComplexF64,1}` and the complex round trip are all correct, with no conjugation.
    - **Note:** the embedded Python (3.12, numpy 2.5) differs from `.venv` (3.11, numpy 2.4). `nistep` must stay compatible with both: Python ≥ 3.11 stdlib only, plus numpy and scipy.
- [x] **P0.7** Legacy import smoke test (the restored file): from `test/`, `python -c "import Compute_higher_order_terms"`.
  - **Accept:** it imports, except possibly `h5py`, which is unused. If so, note it and do not install anything into legacy without asking.
  - *Done 2026-11-09:*
    - `Compute_higher_order_terms` and `Compute_all_options_K_tg` import cleanly with both the system Python and `.venv`. h5py is not needed.
    - `.venv` needed `pandas` (3.1.1 installed, needed for the P12 regression).
    - Imports were run with `python -B`, so no `.pyc` files were written into `test/`. Legacy files unchanged.
- **Pitfalls:** Do not mix PyCall into this project. If CondaPkg cannot download, fall back to `ENV["JULIA_CONDAPKG_BACKEND"]="Null"` and `ENV["JULIA_PYTHONCALL_EXE"]=<a python.org or venv python, not the Store alias>`, set *before* `using PythonCall`.

### P1 — Python package skeleton, config, analytic backend

- [x] **P1.1** `python/pyproject.toml` and `nistep/__init__.py`, exposing `build_engine(config_path: str) -> EEDEngine` and `__version__`.
- [x] **P1.2** `config.py`: frozen dataclasses `ModelCfg`, `DampingCfg(alpha, beta)`, `StepCfg(d_ref, eps=0.24, zero_tol=1e-13, match_tol=1e-9)`, `BackendCfg(kind)`, `AbaqusCfg(runner, command, max_parallel, spool_dir, user_lib_dir, timeout_s, retries)`, `RomCfg(masters, order, forcing_order, style, resonance_tol)`, `ForcingCfg`, `OutputCfg`, `CacheCfg(dir, lru_items)`.
  - Load with `tomllib`. Paths are relative to the config file.
  - Validate: `d_ref > 0`, `0 < eps < 0.25`, files exist.
- [x] **P1.3** `backends/base.py`: a `TangentBackend` Protocol with `n_dof`, `fingerprint() -> str`, `stiffness0() -> csr_matrix`, `mass() -> csr_matrix`, `tangent_many(U_list: list[np.ndarray]) -> list[csr_matrix]` (real U only; batched so runners can parallelise), and `exact_quad/exact_cubic` (optional, `None` if unavailable).
- [x] **P1.4** `backends/analytic.py` with `AnalyticBackend(K, M, Gsym, Hsym)`.
  - `Gsym` and `Hsym` are symmetric dense or sparse-COO tensors.
  - It returns the exact `K_tg(U) = K + 2Ĝ(U,·) + 3Ĥ(U,U,·)` and exposes `exact_quad` and `exact_cubic`.
  - Factories:
    - `two_dof_example()`: StEP.tex 2-DoF system, `k_g=10, k_c=5`, `G11=G22=20`, `H112=H122=10`, m=1. Implement by symmetrising the given polynomial forces.
    - `random_sparse_model(N, seed, density)`: SPD K and M, random symmetric G and H.
    - `spring_chain(N)`: masses with quadratic and cubic springs, analytic.
- [x] **P1.5** Tests:
  - (a) For random U, `tangent_many` equals the finite-difference Jacobian of `F_int` (h = 1e-6, rel 1e-6).
  - (b) The 2-DoF tangent matches the matrix in StEP.tex lines 309–318 symbolically at sample points.
  - (c) `Ĝ`, `Ĥ` are symmetric to 1e-14.
  - *P1 done 2026-11-10:* `.venv/Scripts/python -m pytest` from `python/` gives **39 passed** (test_config, test_sparse, test_analytic).
    - **Install:** `pip install -e python[dev]` in `.venv`. The embedded Julia Python gets `python/` on `sys.path` instead (P4.1).
    - **Additions beyond the spec:**
      - `nistep/_sparse.py::FixedPattern`, a fixed CSR pattern with explicit zeros kept; to be reused by `assembly.py` in P6.4.
      - `EngineCfg(mode, debug)` (`[engine]` section) and `AnalyticCfg(model, n, seed, density, nl_scale)` (`[analytic]` section).
      - `StepCfg.seed` and the `StepCfg.c` property.
      - Unknown config keys raise `ConfigError`.
    - **Analytic models:** `random_sparse_model` and `spring_chain` are built from "generalised springs" (potential forces), so G and H are fully symmetric and K_tg is symmetric. The chain has linear ground springs (`kg=1`) to avoid a near 1:3 resonance.
    - **Configs:** `configs/analytic_{two_dof,chain,random}.toml`.
    - **Pending:** `build_engine` needs `engine.py` and `store.py` (P2).
- **Pitfalls:** The 2-DoF example's forces are given as monomials: `G_{1,1}U_1²` on row 1, `H_{1,2,2}U_1U_2²` on row 1. Build `Ĥ` by symmetrising over the three slots. A wrong factor here poisons every later test.

### P2 — EED core engine (the heart)

- [x] **P2.1** `registry.py`:
  - `canonicalize(p) -> (alpha, d) | None` (§4.5).
  - `DirectionStore` with `match(d) -> id | None`, `add(d) -> id` and persist/load (npz).
  - `VectorRegistry.id_of(a_complex) -> int`, using tolerance matching on `max|a−a_k| ≤ match_tol·max|a_k|`.
- [x] **P2.2** `store.py` with `KtgStore`:
  - `get(U) / put(U, K)`, keyed by `sha256(fingerprint‖U.tobytes())`.
  - In-memory LRU, sized by `lru_items`.
  - Optional disk cache under `cache/<fingerprint>/jobs/<key>.npz`, storing the `data` array, the U vector (for verification) and metadata.
  - A shared CSR pattern (`indptr`, `indices`) stored once. `put` asserts the pattern is identical; on mismatch, store the full matrix and warn.
  - Atomic writes: write `*.tmp`, then `os.replace`.
- [x] **P2.3** `engine.py`, class `EEDEngine(backend, step_cfg, store)`:
  - `K` property (= `backend.stiffness0()`), `M` property.
  - `quad(a, b) -> np.ndarray[complex]` and `cubic(a, b, c) -> np.ndarray[complex]`, exactly §4.4.
  - Internal: `_probe_pair(d_id) -> (Kp, Km)`, `_G_op_apply(d_id, v)` (apply as `(Kp@v − Km@v)/(4c)`; never form the difference matrix unless cached), `_H_diag_apply`, `_H_pair_apply(d1, d2, v)`.
  - `required_jobs(call)`: pure bookkeeping that lists the job U vectors needed.
  - **Modes:**
    - `"lazy"` (default): misses run immediately via `backend.tangent_many`, up to 4 jobs at once.
    - `"record"`: if any job is missing, add it to `pending` and return zeros.
  - `pending_count()`, `execute_pending(max_parallel)`, `stats()` (calls, probes, pairs, jobs run, cache hits, time in backend).
  - Optional `debug=True`: check finiteness and shape, and compare against `backend.exact_*` when available (logging the relative error).
- [x] **P2.4** Tests with `AnalyticBackend(random_sparse_model(N=40))`, at `d_ref` chosen so `c = 0.24`:
  - (a) Real a, b: `quad` equals `exact_quad` to ≤ 1e-10 relative.
  - (b) Complex a, b, c with independent real and imaginary parts: `quad` and `cubic` match exact to ≤ 1e-9.
  - (c) Symmetry: `quad(a,b) ≈ quad(b,a)` and all six permutations of `cubic` agree to ≤ 1e-9.
  - (d) Conjugate reuse: after `quad(a,b)`, calling `quad(conj(a), b)` runs **0** new jobs and returns `Ĝ(conj a, b)` correctly.
  - (e) `a = (1+2i)·p` with p real gives 2 jobs, not 4.
  - (f) Zero vectors and zero parts trigger no jobs.
  - (g) Record mode: zeros are returned, `pending > 0`; after `execute_pending`, the same call returns the exact value with 0 new jobs.
  - (h) The 2-DoF example gives `quad(V1,V1) = [20,20]` and `cubic(V1,V1,V1) = [10,10]` to 1e-12.
  - (i) Linearity: `quad(2a+3a', b) = 2quad(a,b) + 3quad(a',b)`.
  - *P2 done 2026-11-10:* `.venv/Scripts/python -m pytest` from `python/` gives **68 passed** (adds test_registry, test_store, test_engine; all of (a)-(i) pass).
    - **Pair jobs are cheaper than "both ± probes":** `H_op[d1,d2]` uses `K_tg(c d1)`, `K_tg(s c d2)` and the mixed job only. So a fresh `cubic(a,b,c)` with three distinct complex vectors costs 4 mixed jobs + 2 probes of `a` + 2 to 4 probes of `b` (8 to 10 jobs), not 12. Do not assert fixed counts where the pair sign `s` matters.
    - **Lazy mode** runs all misses of one call as a single `tangent_many` batch (the backend or runner decides the parallelism), not "up to 4".
    - **API details:** `K()` and `M()` are methods (easier through PythonCall). `required_jobs(kind, *args)` with `kind = "quad" | "cubic"`. `stats()["jobs_total"] = jobs_probe + jobs_pair`; the `K_tg(0)` job is counted separately as `jobs_k0`.
    - **Additions beyond the spec:** `save_state()` (directions saved to `<cache>/<fp16>/directions_seed<seed>.npz` after every batch); `KtgStore.manifest_keys()`; conjugate vectors share a vector id; real and imaginary parts that are parallel merge into one direction.
    - **Store:** without a disk cache nothing is evicted from memory (an evicted matrix would mean re-running a job). With a disk cache the LRU holds `lru_items` matrices.
- **Pitfalls:**
  - Use `abs()` and `max` on real parts separately; do not use `np.abs` on complex data for canonicalisation.
  - Scipy: `csr_matrix @ complex_vector` returns complex, which is correct. Do **not** cast the result to real.
  - Do not let the probe choice depend on what is cached (determinism, §4.4).

### P3 — Basis-mode EED (StEP.tex) and Galerkin export

- [x] **P3.1** `basis_mode.py`:
  - `basis_coefficients(engine, V) -> dict` with keys `('G',i,j)` and `('H',i,j,k)`, values N-vectors in thesis monomial convention (§4.6).
  - `galerkin_tensors(engine, V) -> (Mr, Kr, Cr?, g_r, h_r)`.
  - `n_eed(n)`, `n_ed(n)`.
  - Save to `npz`.
- [x] **P3.2** Tests:
  - (a) For n = 1..8 on a random model, jobs run equals `(n²+n+2)/2` exactly (fresh store).
  - (b) The coefficients match the exact multiplicity-scaled values (1e-9).
  - (c) The 2-DoF example: `G̃_11 = [20,20]`, `H̃_111 = [10,10]`.
  - (d) `n_eed(10) == 56` and `n_ed(10) == 275`.
- [x] **P3.3** Document the thesis erratum (§3.3) in `CONVENTIONS.md`.
  - *P3 done 2026-11-10:* `pytest` gives **87 passed** (adds test_basis_mode, 19 tests).
    - Job counts for n = 1..8 equal `(n²+n+2)/2` exactly (`jobs_probe = 2n`, `jobs_pair = C(n−1,2)`, plus one `K_tg(0)` job counted as `jobs_k0`).
    - **Indices are 0-based in code** (columns of `V`): `coeffs["G", i, j]`, `coeffs["H", i, j, k]`, `i ≤ j ≤ k`. The thesis `G̃_11` is `coeffs["G", 0, 0]`.
    - **API:** `polarised_forms`, `basis_coefficients`, `save_basis_coefficients` / `load_basis_coefficients` (npz arrays `G_i_j`, `H_i_j_k`), `galerkin_tensors(engine, V, alpha, beta) -> GalerkinModel` (`Mr, Kr, Cr, g, h`, `internal_force(q)`, `save`/`load`), `n_eed`, `n_ed`, `quad_multiplicity`, `cubic_multiplicity`.
    - **Prefetch:** by default the jobs are collected in record mode and run as **one batch** (`max_parallel` at a time), then the values are computed. `EEDEngine.register(*vectors)` was added to fix the ids in basis order.
    - **Extra checks:** `Σ G̃_ij q_i q_j = g(Vq)` and `Σ H̃_ijk q_i q_j q_k = h(Vq)` (validates the multiplicity table); the Galerkin `internal_force(q)` equals `Vᵀ F_int(Vq)`; 2-DoF example with `V = I` reproduces `G11 = G22 = 20`, `H112 = H122 = 10`.
    - The erratum was already in `CONVENTIONS.md` ("Thesis errata"); `test_d_formulas` pins `n_eed(10) == 56`.

### P4 — Julia package MORFEStEP and the PythonCall bridge

- [x] **P4.1** `src/MORFEStEP.jl`. **Bridge:**
  - `load_engine(config_path)` inserts `python/` into `sys.path`, imports `nistep` and calls `build_engine`. Keep the `Py` object in a `Ref`, so it is not garbage-collected.
  - `to_julia_sparse(py_csr)`: `tocsc()`, then `SparseMatrixCSC(m, n, colptr .+ 1, rowval .+ 1, nzval)` with `Int` indices and `Float64` values.
- [x] **P4.2** Closures (see Appendix B):
  - `quadratic_term(engine)` and `cubic_term(engine)` return `MultilinearMap(...; multiindex=(2,0)/(3,0), multiplicity_external=0, fully_asymmetric=false)`.
  - Inside: `@assert Threads.threadid() == 1`; `collect` the views; call Python; wrap the result with `PyArray{ComplexF64,1}`; `res .-= out`; `return res`.
  - `force_term(F::Matrix{Float64})` follows the MORFEFerrite pattern (index `r[j]`).
- [x] **P4.3** Model:
  - `build_model(cfg, engine)`: K and M from the engine; `C = α M + β K` as the same sparse type; `NthOrderModel((K, C, M), (quad, cubic))`. With forcing: `NthOrderModel((K,C,M), (quad, cubic, force), ExternalSystem((im*Ω, -im*Ω)))`. **VERIFY** the external-system constructor signature in the installed `FullOrderModel.jl`.
- [x] **P4.4** Spectral data:
  - `build_spectral(cfg, model)` uses `StructureModalDampingEigensolver(nev, α, β)` (needs `using Arpack, LinearMaps`); dense `spectrum(model)` is used only if N ≤ 200.
  - Then `SpectralData(model, sp; master = …)` with `conjugate_permutation = :detect`. **VERIFY** whether `:detect` belongs to `SpectralData` or `parametrise` in the installed version.
- [x] **P4.5** Test (`test/runtests.jl`) with `analytic` configs. Build the same model twice:
  - (i) via the engine closures;
  - (ii) via **pure-Julia exact closures** from the same tensors, exported by Python to an `.npz` and read with NPZ.jl (or passed through PythonCall).
  - Check `evaluate_nonlinear_terms!` on random complex vectors: the two must agree to ≤ 1e-9.
  - *P4 done 2026-11-10:* `julia --project=julia/MORFEStEP -e "using Pkg; Pkg.test()"` gives **260 passed** (includes the P5.2 end-to-end tests). Faster while developing: `julia --project=julia/MORFEStEP julia/MORFEStEP/test/runtests.jl`.
    - **VERIFIED in the installed MORFE 0.2.0 source:**
      - `MultilinearMap(f!, (2, 0); fully_asymmetric = false)` (positional multiindex, silent) and `MultilinearMap(f!, (0, 0), 1)` for forcing. `f!` must accept `deg + 1` arguments.
      - `:detect` (or an explicit permutation) belongs to `SpectralData(model, sp; master, conjugate_permutation)`. `parametrise` then uses `conjugate_permutation = :from_spectral` by default.
      - `parametrise(model, spectral, order_or_mset; resonance = ResonanceConfig(style, tol), options = ParametrisationOptions(show_progress, verbose, ...))`. `tol` is an **absolute** detuning threshold; `tol_relative` and `eigenvalue_projection = :imaginary_part_only` also exist.
      - `NthOrderModel((K, C, M), terms, ExternalSystem)` is the forced form (used in P11).
    - **Config:** Julia reads the settings from the Python config object (`run_config(engine) -> RunConfig`), so `nistep/config.py` stays the single place for defaults and validation. New field `[rom] n_modes` (0 = automatic, `max(masters) + 8`).
    - **Spectrum (`modal_spectrum`):** our own builder that follows MORFE's `StructureModalDampingEigensolver` path (real mass-normalised modes; mode `p` in entries `2p−1`, `2p`; positive imaginary part first; analytic left blocks via `left_eigenmode_orders_from_slice`). Differences from the plan:
      - dense `eigen` for N ≤ 200 gives the *same* normalisation as the sparse Arpack path (the plan's dense `spectrum(model)` would have used a different gauge); tested equal to Arpack and to MORFE's own solver;
      - the sign of each mode is fixed (largest-magnitude entry positive);
      - the conjugate permutation is passed explicitly (pairs `2p−1 ↔ 2p`), which is safer than `:detect` when eigenvalues repeat;
      - the eigenproblem uses symmetrised copies of K and M; the model keeps the originals.
    - **Reference closures:** `exact_terms(engine)` (pure Julia, from the analytic tensors passed through PythonCall) and `backend_exact_terms(engine)` (the backend's `exact_quad`/`exact_cubic` through Python; for PySVK in P7.2).
    - **Other API:** `load_engine`, `to_julia_sparse`, `build_matrices`, `build_model(engine, cfg; terms = :engine | :exact | :backend_exact)`, `build_spectral`, `resonance_config`, `solve_rom`, `solve_rom_by_order`, `rom_rel_diff`, `backbone_rel_diff`, `export_rom`, `engine_stats`, `force_term`.
    - `Project.toml` has `[extras]`/`[targets]` for `Test` and `Random`.
- **Pitfalls:**
  - Returning anything but `res` is fine, since MORFE ignores it, but **never overwrite `res`**; always accumulate.
  - Py objects must not be touched from other threads. Leave `JULIA_NUM_THREADS` unset or 1 for runs.
  - Python exceptions surface as `PyException`. Let them propagate; the disk cache makes reruns cheap.

### P5 — End-to-end on analytic models (no FE yet)

- [x] **P5.1** `scripts/run_rom.jl`: read the config, `load_engine`, `build_model`, `build_spectral`, `parametrise` (lazy mode), then export:
  - `runs/<name>/W.npz` (real and imaginary parts of `W.poly.coefficients`, plus the exponents);
  - `R.npz` and `R_coefficients.csv` (via `save_rom`);
  - `engine_stats.json`;
  - `summary.txt`.
- [x] **P5.2** **Accept:**
  - On `spring_chain(N=20)` and `random_sparse_model(N=50)`, one master pair, CNF, order 5 and 7: the engine-closure ROM and the exact-closure ROM agree, with `max|ΔR|/max|R| ≤ 1e-7` and `max|ΔW|/max|W| ≤ 1e-7`.
  - The backbone from `normal_form_branch` agrees to 1e-6.
- [x] **P5.3** Record the job counts per order in `engine_stats.json`. They are the template for the cost estimates in P10.
  - *P5 done 2026-11-10:* `julia --project=julia/MORFEStEP scripts/run_rom.jl configs/<model>.toml --order N --reference` (options: `--out DIR`, `--no-per-order`, `--verbose`). Exit code 1 if the reference check fails.
    - **Results (one master pair, CNF, engine closures against exact closures):**

      | Model | Order | max\|ΔR\|/max\|R\| | max\|ΔW\|/max\|W\| | backbone | Jobs (+1 for K) | New jobs per order 2..p |
      |---|---|---|---|---|---|---|
      | `spring_chain` N=20 | 5 | 2.4e-16 | 1.3e-15 | 2.2e-16 | 5 | 2, 0, 2, 1 |
      | `spring_chain` N=20 | 7 | 2.4e-16 | 1.3e-15 | 2.2e-16 | 17 | 2, 0, 2, 1, 8, 4 |
      | `random_sparse_model` N=50 | 5 | 2.6e-16 | 4.4e-14 | 3.9e-16 | 11 | 2, 0, 6, 3 |
      | `random_sparse_model` N=50 | 7 | 3.7e-14 | 6.8e-14 | 2.3e-14 | 26 | 2, 0, 6, 3, 9, 6 |

    - The same checks run in `julia/MORFEStEP/test/end_to_end.jl` (tolerances 1e-7 and 1e-6).
    - **Job pattern:** order 2 needs only the ± probes of the mode (2 jobs); order 3 needs nothing new; from order 4 on, new directions are the real and imaginary parts of the order-2, order-3, … coefficients. The chain needs fewer jobs because many of its coefficients are real or have parallel real and imaginary parts.
    - `engine_stats.json` holds the engine counters plus the `jobs_per_order` table (from `solve_rom_by_order`, which solves at order 2, 3, …, p in turn; the engine cache makes the repeats free of jobs).
    - The first solve in a Julia session takes about 25 s of compilation; the solves themselves take well under a second on these models.

### P6 — FE plumbing: .inp, DofMap, .mtx, PySVK stand-in (this PC, no Abaqus)

- [x] **P6.1** `inp_io.py`. Parse:
  - `*Node` blocks (label, x, y, z);
  - `*Element, type=U3, elset=allUEL` (connectivity);
  - `*Nset` blocks, including `generate` syntax (**VERIFY** in the files);
  - the clamped set (configurable name; deduplicate);
  - `*UEL PROPERTY` (E, ν, ρ, nInt, fbar, matID, nPost);
  - the existing BC block and `*Element Matrix Output`.
  - `render_job_inp(base_text, dofmap, U, matrix_file_name) -> str`: replace the BC block with `*Boundary` / `<clamped set>, ENCASTRE` / `node<i>, d, d, <U:.17e>` lines in DofMap order, and set `File Name=`. Port the logic of `Duplicate_Input_File.py:40-94`, but **return text and never write over the base file**.
  - `prepare_base_inp(raw_inp) -> text`: port `Write_BC_SETS_Input_file.py:86-102`, so per-node Nsets exist.
- [x] **P6.2** `dofmap.py` (§4.7). Tests on the repository decks:

  | Deck | Nodes | Elements | Clamped | Free DOFs |
  |---|---|---|---|---|
  | MEMS | 525 | 240 | 18 (duplicates removed) | 1521 |
  | cantilever | 918 | 500 | 18 | 2700 |
  | clamped-clamped | 606 | 200 | 12 | 1782 |
  | single element | 8 | 1 | 4 | 12 |

- [x] **P6.3** `mtx_io.py`:
  - `load_processed_ijv(path, n_total)`: the repository `*_matrix.mtx` format, 1-based, both triangles.
  - `parse_element_matrices(text)`: raw Abaqus element-matrix output. Port `Organize_the_matrix_right_order._read_general:101-134`, vectorised with `np.array(str.split(), float)`. It returns `{elem_label: (node_labels, dofs, 24×24 ndarray)}` and **raises** on any malformed block.
  - Test the processed loader on the repository files: symmetric to 1e-12 relative, diagonal > 0.
  - Test the raw parser on a synthetic file built from the format the legacy parser expects, then on a real file in P8.
- [x] **P6.4** `assembly.py`: `Assembler(dofmap, connectivity)` precomputes the global COO indices of every element block. `assemble(element_mats) -> csr_matrix` uses a fixed pattern with explicit zeros kept, eliminates clamped DOFs, and reuses the pattern for every job.
- [x] **P6.5** `backends/pysvk.py`, a numpy TL-SVK hex8 implementation matching the UEL:
  - Trilinear shape functions; 2×2×2 Gauss points; `F = I + ∇u`, `E = ½(FᵀF − I)`, `S = λ tr(E) I + 2μE`, `P = F S`, `f_int = ∫ Gᵀ P dV`, `K_tg = ∫ Gᵀ (∂P/∂F) G dV` (G = shape-function gradient operator, as in `Finite_element.tex` eq:K_tanget); consistent mass `∫ ρ NᵀN dV`.
  - Also provide `exact_quad` and `exact_cubic` via the polarised `P2(a,b)` and `P3(a,b,c)` (copy the formulas from the MORFE `fe_nonlinearity` tutorial), so the engine can be checked against an independent FE truth.
  - Vectorise over elements (einsum). A loop is acceptable for meshes of 500 elements or fewer.
- [x] **P6.6** **Accept:**
  - (a) PySVK `K_tg(0)` and `M` versus the repository `*_stiffness_matrix.mtx` / `*_mass_matrix.mtx` (cantilever and MEMS, after elimination): ≤ 1e-8 relative. If they disagree, **stop and investigate** (properties, ordering, which UEL produced the files) before going on. This check validates the DofMap and assembly ordering without Abaqus.
  - (b) PySVK `K_tg(U)` versus the finite difference of `f_int` (rel 1e-6).
  - (c) The engine on PySVK matches `exact_quad`/`exact_cubic` (≤ 1e-8) on a real mesh with complex vectors.
  - *P6 done 2026-11-10:* `pytest` gives **133 passed** in about 5 s (adds test_dofmap, test_inp_io, test_mtx_io, test_pysvk; shared deck table in `tests/decks.py`).
    - **(a) PySVK against the repository matrices** (after elimination, max entry difference relative to the largest entry):

      | Deck | Free DOFs | K | M |
      |---|---|---|---|
      | MEMS | 1521 | 4.3e-14 | 6.2e-15 |
      | cantilever | 2700 | 1.0e-14 | 2.8e-14 |
      | clamped-clamped | 1782 | 1.3e-14 | 1.9e-14 |
      | single element | 12 | 7.9e-15 | 5.9e-15 |

      So the DofMap, the assembly order, the hex8 formulation and the material constants all agree with the Abaqus UEL that produced those files. Tests assert ≤ 1e-10.
    - **(b)** Tangent equals the Jacobian of `internal_force` (full finite differences on the single element; directional on MEMS and cantilever), ≤ 1e-6.
    - **(c)** Engine on PySVK against the exact polarised forms with complex vectors: ≤ 1e-8 on MEMS (`d_ref = 20`), cantilever (`d_ref = 10`) and the single element.
    - **Correction to §3.1 and P6.2:** the clamped-clamped deck uses `Set-2` (12 nodes), like the others, not `Set-1`.
    - **UEL facts confirmed from the Fortran:** `props = (E, ν, ρ)`, `jprops = (nInt, fbarFlag, matID, nPostVars)`; `μ = E/(2(1+ν))`, `λ = νE/((1−2ν)(1+ν))`; C3D8 node order; 2×2×2 Gauss points at ±1/√3; local DOFs node-major; consistent mass with the same Gauss rule.
    - **Raw `.mtx` parser:** follows the layout the legacy parser reads (`** ELEMENT NUMBER`, `** ELEMENT NODES`, node line, DOF line, `*MATRIX,TYPE=…`, data). It accepts full (unsymmetric) and lower-triangle (symmetric) storage and raises on anything else. It has only been tested on synthetic files written by `format_element_matrices`; **check it on a real Abaqus file in P8.6**.
    - **Additions beyond the spec:**
      - `JobDeck` (build once, `render(U, name)` per job), `read_prescribed` (reads the prescribed vector back from a job deck; the round trip is exact with 17 digits), `bc_style = "nset" | "label"`.
      - `format_element_matrices` writes PySVK element matrices in the raw Abaqus layout, so a mock runner can stand in for Abaqus in P8/P9 tests.
      - `Assembler.check_element_order`, `assemble_vector`, `gather`; `DofMap.restrict_matrix`, `expand`, `restrict`, `node_dof`.
      - `PySVKBackend.internal_force`, `element_tangents`, `element_masses`, `min_det_F` (for the P10.1 preflight), `volume`.
      - `nistep.build_backend` now accepts `kind = "pysvk"`.
    - PySVK speed: about 0.05 s per tangent matrix on MEMS and 0.1 s on the cantilever.
- **Pitfalls:**
  - The UEL's node ordering inside an element and its Gauss-point ordering do not matter for assembled matrices, but the hex8 node numbering convention (Abaqus C3D8) does. Use the Abaqus convention.
  - `rho` integer truncation in the mass UEL (B10) means the repository M may have been made with ⌊ρ⌋. All repository densities are integers, so this should not matter.

### P7 — Linear model and spectrum on real meshes (no Abaqus)

- [x] **P7.1** Linear-only check: the config backend is `pysvk` (or a `processed-matrix` mode loading the repository K and M). Compare the MORFE `StructureModalDampingEigensolver` with Python `scipy.sparse.linalg.eigsh(K, M=M, sigma=0)`.
  - **Accept:** for MEMS `ω₁ ≈ 1.819006` and for cantilever `ω₁ ≈ 8.425818` (thesis values; tolerance 1e-4 relative).
  - If they disagree, report the found values and whether the thesis quoted MORFE2.0 eigenvalues (`Thesis_Applications.tex:28-31`). Do not force a match.
- [x] **P7.2** Full closure ROM with the **PySVK backend** on MEMS (N=1521), O(3) and O(5), CNF, one master pair, `d_ref = 20`.
  - **Accept:** it runs end-to-end. The engine-closure ROM agrees with the PySVK exact-closure ROM (P4.5 pattern) to ≤ 1e-6. Record the job counts and time.
  - *P7.1 and P7.2 done 2026-11-10:* Julia tests give **340 passed** (adds `test/real_meshes.jl` and `test/physics.jl`); Python **133 passed**. Configs: `configs/{mems,cantilever,cc200,single_element}_pysvk.toml`.
    - **P7.1 first natural frequencies** (PySVK matrices; our `modal_spectrum`, scipy `eigsh` and MORFE's `StructureModalDampingEigensolver` agree to 1e-8):

      | Model | ω₁ computed | ω₁ thesis | Relative difference |
      |---|---|---|---|
      | MEMS | 1.819004172 | 1.819006 | 1.0e-6 |
      | cantilever | 8.425782097 | 8.425818 | 4.3e-6 |
      | clamped-clamped (200 el.) | 41.498246 | (other mesh) | n/a |

    - **P7.2 MEMS closure ROM with PySVK** (N = 1521, `d_ref = 20`, `alpha = 5e-7`, one master pair, CNF), engine closures against the PySVK exact closures:

      | Order | Jobs (+1 for K) | New jobs per order | max\|ΔR\|/max\|R\| | max\|ΔW\|/max\|W\| | Worst per degree: R | Worst per degree: W | Backbone (relative to the shift) |
      |---|---|---|---|---|---|---|---|
      | 3 | 2 | 2, 0 | 1e-17 | 2e-13 | 4.3e-6 (degree 3) | 2.1e-8 | 4.3e-6 |
      | 5 | 11 | 2, 0, 6, 3 | 1e-17 | 2e-13 | 2.1e-5 (degree 5) | 6.3e-8 | 8.1e-7 |

      The plan's criterion (≤ 1e-6 relative to the largest coefficient) holds with a wide margin. Backend time: 0.43 s for the 11 tangent matrices; MORFE solve under 1 s after about 25 s of compilation.
    - **Why the per-degree numbers are larger (double-precision floor, not a method error).** Measured on MEMS:
      - The EED forces themselves are accurate to about 1e-13 (`Ĝ(ϕ,ϕ)`: 5e-14; `Ĥ(ϕ,ϕ,ϕ)`: 2e-13).
      - Solving for `W` amplifies this to about 1e-8, because K is ill-conditioned (soft bending against stiff membrane directions).
      - The cubic normal-form coefficient of this model is a difference of two terms that cancel to 8 digits (`ϕᵀĤ(ϕ,ϕ,ϕ)` gives about 2e-4, the result is 4.4e-12: the membrane stiffening is almost exactly relieved by the in-plane motion in `W₂`). So 1e-13 becomes about 1e-5 on that coefficient.
      - Physically: at a mode amplitude of `d_ref = 20` the frequency goes from 1.819004 to 1.820181 (order 5). The EED and exact backbones differ by under 1e-5 of that shift.
      - Raising `d_ref` helps only mildly (the noise scales like `1/c` for Ĝ and `1/c²` for Ĥ); min `det F` over the probe states is 0.998 at `d_ref = 20` and 0.975 at `d_ref = 320`, so there is room. This belongs to the `d_ref` calibration of P10.1.
    - **New comparison helpers:** `rom_rel_diff_by_degree` (per polynomial degree; use it on real models, where nonlinear coefficients are many orders smaller than linear ones and a global ratio hides them), `backbone_amplitudes(engine, spectrum)` (ρ up to a mode amplitude of `d_ref`) and `backbone_rel_diff` (now relative to the frequency *shift*). `run_rom.jl --reference` prints the per-degree table; `--tol` (default 1e-4) sets its pass threshold.
    - **Physics check of the conventions** (`test/physics.jl`, config `analytic_duffing.toml`): for `m ü + k u + k2 u² + k3 u³ = 0` the ROM gives `Im R[1, z²z̄] = (9α₃ω₀² − 10α₂²)/(24ω₀³)·4/m` to 1e-10, for hardening and softening cases, with both the engine closures and the exact closures; the second-order manifold coefficients match harmonic balance. This confirms the minus sign in the closures, the factor on the quadratic closure and MORFE's multiplicity handling against an independent closed-form result.
    - **Timing caution:** two runs on this PC stalled for minutes with no CPU work in our code (one pytest case took 339 s instead of 0.03 s; one MORFE solve reported 917 s instead of 25 s). Reruns were normal. Treat single wall-clock readings on this PC with care; the engine's `time_backend_s` is reliable.
      - *Cause found 2026-11-11:* the PC enters Modern Standby while a run is unattended, and every process is frozen meanwhile. The System event log shows Kernel-Power events 506 (entering) and 507 (exiting) at the times of the stalls. Keep the PC awake during long runs.
- [ ] **P7.3** *(Recommended)* Intrusive reference: install MORFEFerrite (`Pkg.add(url="https://github.com/MORFEproject/MORFEFerrite.jl.git")`) in a **separate** environment `julia/reference/`. Convert the deck with `MeshIO.abaqus_to_gmsh` and build its StructuralSVK model (**VERIFY** API) with the same E, ν, ρ, BCs and masters.
  - **Accept:** the backbone matches the P7.2 results within plotting accuracy, with the frequency at equal amplitude within 1e-4 relative. A mismatch points to element or BC differences; document it rather than tune.

  - *Deferred (optional step), 2026-11-10:* not run yet. It needs an unregistered package from GitHub and an API that must be verified first. Its purpose is already covered in part: PySVK reproduces the repository's Abaqus K and M to 1e-14 (P6.6), its polarised forms are consistent with its own internal force and tangent (P6 tests), and the conventions are checked against a closed-form backbone (P7 physics test). Revisit at P10.3, where it would serve as the third reference.

### P8 — Abaqus backend, runners, and the Abaqus host

Before starting, **ask the user (§9 D1, D2)**:

- whether Julia and Python will run on the Abaqus host (LocalRunner) or here (SpoolRunner or SSH);
- whether the Fortran fixes (B9, B10) are approved.

- [x] **P8.1** Host prerequisites, as a checklist in `scripts/build_uel_library.md`:
  - a full Abaqus (not LE) with a matching Intel Fortran and Visual Studio;
  - `abaqus verify -user_std` passes;
  - the repository is copied with the `test/` and `module/` layout intact.
  - *Done 2026-11-10 (the document):* `scripts/build_uel_library.md` holds the checklist, the library build, the two ways to reach Abaqus (local runner, spool worker), the smoke test and a troubleshooting list. The checks themselves are run by the user on the host; items marked **VERIFY** there are still untested.
- [ ] **P8.2** One-time UEL build on the host. From `test/` (so `include '../module/*.for'` resolves), run `abaqus make library=uel_nlmech_pk1.for` into `uel_lib/stiffness/`, and the mass variant into `uel_lib/mass/`.
  - Jobs use an `abaqus_v6.env` in the job directory with `usub_lib_dir=r'<abs path>'` and **no** `user=` argument. **VERIFY** on the host that the produced library name (`standardU.dll` on Windows, `libstandardU.so` on Linux) is picked up.
  - Fallback: a script that concatenates `module/*.for` and the UEL into one flat `uel_flat.for` (resolving includes), used with `user=uel_flat.for`.
  - *Open (host only), 2026-11-10:* the build itself needs the Abaqus host. Ready on this side: the runner writes `abaqus_v6.env` with `usub_lib_dir = r'<dir>'` into every job directory (config `user_lib_dir`, `mass_lib_dir`); the fallback script `scripts/make_flat_uel.py` exists and is tested on both UEL sources (config `user_source`, `mass_source`; the originals are not changed).
  - *Built on the host 2026-11-22* (see "Status on the Abaqus PC" at the top): Intel Fortran Classic 2021.4.0; `abaqus make` COMPLETED for both sources; `standardU.dll` in `nistep_uel\stiffness` and `nistep_uel\mass`. The single-element jobs ran with it through `usub_lib_dir`. Tick this box once the P8.6 single-element run passes.
- [x] **P8.3** `runners/local.py` with `LocalRunner.run(jobs) -> results`.
  - One directory per job, `work/<jobname>/` (jobname `j` + 12 hex characters of the key), containing `<job>.inp` and `abaqus_v6.env`.
  - Command: `abaqus job=<job> input=<job>.inp double=both output_precision=full interactive ask_delete=OFF cpus=1`, via `subprocess.run(..., shell=True on Windows, timeout=timeout_s, capture_output=True)`.
  - Up to `max_parallel` concurrent jobs (ThreadPoolExecutor; each runs in its own process).
  - **Success requires all of:** exit code 0; `.sta` contains "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" (**VERIFY** the exact text); the `.mtx` exists; the element count equals the number of UEL elements.
  - On failure, keep the directory and return the tail of `.msg`/`.dat`/`.log`. Retry up to `retries` times, then raise.
  - *Done 2026-11-10 (tested against a mock Abaqus, not a real one):*
    - `runners/abaqus_exec.py` (standard library only, Python ≥ 3.8, shared with the host worker) builds the command, writes the job files, runs the process (`Popen`; the whole process tree is killed on timeout) and applies the four success checks. `runners/local.py` adds the thread pool, retries and the `keep_job_dirs` policy (`failed` (default), `all`, `none`). All jobs of a batch finish before the first error is raised.
    - `runners/mock_abaqus.py` stands in for the `abaqus` command (`python -m nistep.runners.mock_abaqus job=… input=…`): it reads the job deck, computes the element matrices with PySVK and writes them in the raw Abaqus layout plus `.sta`/`.msg`/`.dat`. Faults are injected with `NISTEP_MOCK_MODE` (`fail`, `no_mtx`, `truncated`, `no_sta`, `bad_sta`, `symmetric`, `sleep=`, `fail_once=`, `nonpoly`).
    - Tests (`test_abaqus_backend.py`): each fault is detected with the right reason, the job directory is kept and the log tail is in the error; retry after one failure; timeout; missing command; parallel batches keep their order.
    - **Still to VERIFY on a real Abaqus:** the success sentence in `.sta`, the raw `.mtx` layout (the parser follows the legacy parser and has only seen synthetic files), and that `usub_lib_dir` picks up the library.
    - *Bug found and fixed 2026-11-11 (text encoding):* the job files were written with the default encoding of the platform. Every repository deck has a UTF-8 "ν" in a comment, and the Windows default (cp1252) cannot write it, so the first job crashed with `UnicodeEncodeError` when the engine ran inside Julia. The Python that PythonCall embeds is **not** in UTF-8 mode, even with `PYTHONUTF8=1` set; the pytest runs had hidden the bug because they were. Now every text read and write names its encoding: decks are read as UTF-8 and written as UTF-8 bytes, and the host worker passes the deck through as bytes. `make_flat_uel.py` copies the Fortran sources byte for byte (latin-1). Two tests run with UTF-8 mode off (`-X utf8=0`): `test_job_files_do_not_depend_on_the_locale_encoding` and `test_spool_roundtrip_without_utf8_mode_on_the_host`. The whole Python suite also passes with UTF-8 mode off.
    - **Rule for new code:** never call `open`, `read_text` or `write_text` on a text file without `encoding=`.
- [x] **P8.4** `backends/abaqus.py` with `AbaqusBackend(cfg, runner)`:
  - `stiffness0()` is a job at U=0 with the stiffness library. `mass()` is a job at U=0 with the mass library; alternatively, if configured, load the repository processed M (store the provenance in the fingerprint).
  - `tangent_many(Us)` renders the inputs, runs them through the runner, parses, assembles, eliminates and returns the matrices.
  - `fingerprint()` is the sha256 of: base deck text, UEL and module sources, library file hashes, clamped set, properties and backend kind.
  - *Done 2026-11-10 (mock-tested):* `AbaqusBackend(model, runner, clamped_nset, mass_from, mass_mtx, provenance)` and `from_config`. Through the mock it returns matrices bit-identical to PySVK on the single element and on MEMS, with one fixed sparsity pattern. A mass job that returns stiffness (or the reverse) is rejected (`matrix type … expected …`), as is any mismatch of element labels or connectivity. The engine runs on it (`build_engine` with `[backend] kind = "abaqus"`), and a second engine on the same cache runs 0 jobs.
    - *Change from the text above:* the fingerprint hashes the model data (coordinates, connectivity, element labels, DOF map, clamped nodes, properties), the mass source, and the `provenance` entries (library folders and Fortran sources from the config: by content if the file exists on this PC, by name otherwise). The deck text itself is not hashed, because the boundary block changes per job.
    - New config fields in `[abaqus]`: `work_dir`, `user_lib_dir`, `mass_lib_dir`, `user_source`, `mass_source`, `mass_from` (`uel` | `processed`), `keep_job_dirs`, `poll_s`, `heartbeat_stale_s`; `command` may be a string or a list. Templates: `configs/{single_element,mems,cantilever}_abaqus.toml` (the host paths in them are placeholders).
- [x] **P8.5** `runners/spool.py` and `scripts/abaqus_worker.py`, for remote use through a shared folder (stdlib only, runnable with any Python ≥ 3.8 on the host; `abaqus python` as fallback):
  - **Client:** write `inbox/<job>.inp.tmp`, then rename to `.inp`, with `inbox/<job>.json` (library: stiffness/mass, timeout).
  - **Worker:** claim a job by renaming it into `running/`. Run it with the LocalRunner logic. Write `outbox/<job>.mtx`, then `outbox/<job>.json` (status, timings, log tail) **last**, as the completion marker. Keep a heartbeat file `worker.alive`.
  - **Client polling:** poll `outbox/*.json` with a timeout. If the heartbeat is stale, raise "worker down".
  - *(Optional)* `runners/ssh.py` using the OpenSSH `ssh`/`scp` CLIs, if the user prefers SSH (§9 D1).
  - *Done 2026-11-10 (mock-tested; SSH runner not written, waiting for D1):*
    - Folder protocol: `inbox/`, `running/`, `outbox/`, `worker.alive`, optional `stop` file. Every hand-over is a rename, and the `.json` marker is always written last.
    - `scripts/abaqus_worker.py` uses only the standard library. Options: `--spool --work --stiffness-lib --mass-lib --stiffness-source --mass-source --parallel --timeout --retries --poll --heartbeat --keep --once --idle-exit`, and `--command …` which must be the **last** option. Jobs left in `running/` by a killed worker are put back in the inbox at start.
    - `SpoolRunner` judges the heartbeat by whether the file's content changes (no clock comparison between machines), reuses finished results left in the outbox, and withdraws its unclaimed jobs on error.
    - Tests (`test_spool.py`, with a real worker subprocess driving the mock): round trip, failure with log tail, "worker down", worker killed mid-job and restarted (the job completes once), stop file.
- [ ] **P8.6** Smoke tests on the host (or through the spool):
  - (a) Single element: `K_tg(0)` from Abaqus versus PySVK `K_tg(0)`, ≤ 1e-8 relative.
  - (b) Single element: random U with `max|U| = 0.1` (edge 1): Abaqus versus PySVK `K_tg(U)`, ≤ 1e-8.
  - (c) **Cubic-exactness test** (detects B9 and any non-polynomial behaviour). For a direction d and scales `c, 2c, 3c`, `K_tg(t d)` must fit `A + B t + C t²` exactly: the third difference `K(3c) − 3K(2c) + 3K(c) − K(0)` must be ≈ 0 relative to `‖K(c) − K(0)‖`, at the round-off level (record the measured value).
  - (d) MEMS: Abaqus `K_tg(0)` versus the repository processed K (≤ 1e-8).
  - (e) Record the wall time per job, with and without the precompiled library.
  - *Open (host only), 2026-11-10:* the test is written and passes against the mock; it has not run on a real Abaqus.
    - Run `python scripts/abaqus_smoke_test.py configs/single_element_abaqus.toml` and the same for `mems_abaqus.toml` (after editing the host paths). Six jobs per deck; exit code 0 if all checks pass; `--json` saves the report.
    - `nistep/diagnostics.py`: `smoke_test`, `cubic_exactness`, `matrix_rel_diff`, `symmetry_error`, `shortest_edge`. The default test amplitude is 10 % of the shortest element edge.
    - Tests (`test_diagnostics.py`): all checks pass through the mock (single element and MEMS); a non-polynomial tangent (mock mode `nonpoly`, imitating B9) fails `c_cubic_exactness` and `b_KU_vs_pysvk` while the `U = 0` checks still pass; the script returns exit code 1 in that case.
    - **Record here after the host run:** each printed value, the seconds per job (library and flat source), the Abaqus version, and the outcome of the three VERIFY items of P8.3.
  - *First host run, 2026-11-22 (single element):*
    - **What failed.** The run stopped at the mass job with `matrix type STIFFNESS, expected MASS`. Abaqus takes the label of the element-matrix output from the deck's `*Element Matrix Output` request, not from the UEL library, and every job deck asked for `Stiffness=Yes`.
    - **The fix.**
      - Mass jobs now ask for `Mass=Yes` (`JobDeck.render(..., output="mass")`).
      - `AbaqusBackend` checks that the values match the expected kind, using the rigid-translation ratio `translation_ratio`. This still catches swapped library folders.
      - New tests: `test_render_mass_output_request`, `test_mass_job_requests_mass_output`, `test_translation_ratio_separates_stiffness_and_mass`.
      - Python suite: 196 passed.
    - **Next.** Rerun on the host after `git pull`.
- [x] **P8.7** `backends/replay.py`: serves `K_tg` from a disk cache copied from the host. A miss raises `ReplayMiss` with the U key. Used to debug production runs on this PC.
  - *Done 2026-11-10:* `ReplayBackend(cache_dir, fingerprint=None)`; config `[backend] kind = "replay"` with `[cache] dir` (and `replay_fingerprint` if the folder holds more than one model). The engine now writes `backend_info.npz` (M, size, recorded backend kind) next to the cache, so a replay needs no deck and no Abaqus. Tested: a replayed engine reproduces the recorded results bit for bit with 0 jobs; a state that was never recorded raises `ReplayMiss` with its 64-character key.
  - **P8 status:** Python suite 172 passed (180 on 2026-11-11, after P9 and the encoding fix). Open: P8.2 and P8.6 (both need the Abaqus host) and decisions D1–D3 (§9).
- **Pitfalls:**
  - Abaqus `.bat` quoting.
  - Licence-token limits cap `max_parallel`; ask the user how many tokens are available.
  - Network-share latency: never read a file before its `.json` marker exists.
  - The `.mtx` output precision is unknown. Measure it in P8.6 (b) and (c) and record it; it sets the noise floor for `d_ref`.
  - Windows `MAX_PATH`.
  - The `*Step, inc=1` single increment must stay as is: K_tg is taken at the prescribed state with no path dependence (Total Lagrangian).

### P9 — Batch prefetch, parallelism, resume

- [x] **P9.1** Julia `run_with_prefetch(model, sd, cfg, engine)`:
  ```
  for q in 2:p_max
      engine.set_mode("record"); parametrise(model, sd, mset(q); quiet options)
      n = engine.execute_pending(max_parallel)   # all jobs of order q in one parallel batch
  end
  engine.set_mode("lazy"); W, R = parametrise(model, sd, mset(p_max); options)
  ```
  - Pass q is correct because every call of order < q hits the cache, so the requests recorded at order q use exact vectors.
  - The final pass runs in lazy mode, so a tolerance miss can never corrupt results; it only costs a synchronous job.
  - *Done 2026-11-11:* `run_with_prefetch(model, spectral, cfg, engine; order, max_parallel, forcing_order, verbose) -> (W, R, batches)` in `julia/MORFEStEP/src/MORFEStEP.jl`.
    - For each order `q = 2 … order`: one record pass, then `engine.execute_pending(max_parallel)`. The engine is put back in lazy mode even if a pass throws.
    - *Change from the sketch above:* the final lazy solve is skipped when the last record pass had no incomplete call, because that pass is then already the full solve (this is the case for every rerun on a filled cache). When the final solve does run and a job runs inside it, a warning says how many jobs were missed by the batches. No test or run has produced a missed job so far.
    - `batches` has one entry per order: `new_jobs`, cumulative `jobs_total`, `calls_quad`, `calls_cubic`, `seconds_jobs` (time in the job batch) and `seconds_pass` (time in MORFE). An entry with `order = 0` describes the final solve.
    - Switches: `[engine] prefetch = "auto" | "on" | "off"` in the config (`auto` = on for the `abaqus` backend only), and `--prefetch` / `--no-prefetch` in `scripts/run_rom.jl`. The per-order table is saved in `engine_stats.json` (`jobs_per_order`).
    - Cost of the extra passes: MORFE cannot restart from a lower order, so the pass at order `q` solves orders `2 … q` again. This costs solver time only, never jobs. MEMS at order 5 (PySVK): 26.6 s with prefetch, 26.0 s without; about 22 s of both is Julia compilation.
- [x] **P9.2** Persist the direction store and the job cache to disk. Vector ids are not persisted, because they are re-derived deterministically from MORFE's call order. Killing and restarting must then resume with 0 repeated jobs.
  - **Accept:** with the Analytic backend plus a fake-slow runner (sleep), kill the process mid-run, restart, and see 0 duplicate jobs (check the manifest).
  - *Done 2026-11-11:*
    - What is on disk: one `.npz` per tangent matrix plus `manifest.jsonl` (one line per finished job, appended after the matrix file is complete), and the direction store `directions_seed<seed>.npz`, saved by `engine.save_state()` after every batch. All writes go to a temporary file first and are then renamed. `manifest_keys()` skips a torn last line.
    - Test aids: `nistep/backends/wrappers.py` has `ThrottledBackend(inner, delay_s, fail_after)` (sleeps before each job; raises `SimulatedFailure` after a set number of jobs). `[backend] delay_s` in a config wraps any backend with it.
    - Python (`tests/test_resume.py`, 6 tests): a driver process is killed (hard kill, no clean-up) once at least 6 jobs are in the manifest, then started again. No key appears twice, the number of keys is the clean total (or one less if the kill hit between the matrix file and its manifest line, which costs no job), and the result equals a clean run to ≤ 1e-12. A third run runs 0 jobs. A failure in the middle of a batch is covered in the same way.
    - Julia (`test/prefetch.jl`, "P9.2 resume after an interruption"): the backend fails half-way through `run_with_prefetch`; a new engine on the same cache folder runs exactly the missing jobs (`jobs_total == total − done_before`), the manifest has no duplicate key, and W and R equal a clean run to ≤ 1e-12. A third run runs 0 jobs and returns identical coefficients.
    - End to end, with real processes: `scripts/run_rom.jl` on the single-element deck through `AbaqusBackend` → `LocalRunner` → mock Abaqus (1.5 s per job, 2 at a time, order 5), killed when 5 matrices were on disk (K and 4 jobs). The second run ran the 3 missing jobs (order 4: 2, order 5: 1). The manifest then had 8 keys, none twice, the first 5 unchanged, and no job directory was left behind. R and W are identical to the PySVK run of the same model (7 jobs).
- [x] **P9.3** **Accept (Analytic and PySVK):** the prefetch result equals the lazy result (≤ 1e-12 relative), with the same total number of distinct jobs. Record the per-order batch sizes.
  - *Done 2026-11-11:*
    - Julia tests (`test/prefetch.jl`, "P9.3 prefetch equals lazy") on `analytic_chain` (order 5), `analytic_random` (order 6) and `single_element_pysvk` (order 5): the difference in R and in W is ≤ 1e-12, the new jobs per order equal those of `solve_rom_by_order`, nothing ran twice, and the final solve ran 0 jobs.
    - MEMS (PySVK, 1521 DOFs, order 5, `scripts/run_rom.jl --prefetch` against `--no-prefetch`): R and W are identical (difference 0). Both ran 11 jobs plus one for K.

      | Order | Jobs in the batch | Cumulative jobs |
      |---|---|---|
      | 2 | 2 | 2 |
      | 3 | 0 | 2 |
      | 4 | 6 | 8 |
      | 5 | 3 | 11 |

    - Single element through the mock Abaqus, order 5: 7 jobs, the same as PySVK.
- [x] **P9.4** Memory: with the shared pattern, each cached K_tg costs `nnz × 8` bytes. Log the RAM estimate. Default LRU of 64 matrices, with the rest on disk.
  - *Done 2026-11-11:* `KtgStore.stats()` reports `store_MB_per_matrix`, `store_ram_estimate_MB` (now) and `store_ram_limit_MB` (`lru_items × nnz × 8` bytes). `KtgStore.memory_report()` gives them as one line; the engine logs it once and `run_rom.jl` prints it. The default is `[cache] lru_items = 64`.
    - MEMS: 75 663 stored entries, 0.577 MB per matrix, so at most 37 MB in memory with a disk cache.
    - Without `[cache] dir` nothing is evicted (there would be no copy to reload), so every matrix stays in memory. Set `[cache] dir` for any long run.
    - **P9 status:** Python suite 180 passed (also with UTF-8 mode off); Julia suite 423 passed.
- **Pitfalls:**
  - **VERIFY** how to build `mset(q)` with a forcing-degree cap (anisotropic multi-index set; see `src/Multiindices.jl` and the `multiindex_sets` tutorial).
    - *Verified 2026-11-11 in MORFE 0.2.0:* `parametrise` takes an integer or a `MultiindexSet`. `expansion_set(model, spectral, q; forcing_order)` returns `q` for an autonomous model. With an external system it returns `filter(pred, all_multiindices_up_to(ROM + N_EXT, q; min_degree = 1))`, where `pred` keeps the exponents whose forcing degree is ≤ `min(q, forcing_order)`. The reduced coordinates come first in each exponent, the forcing variables last. `filter` keeps the graded-lex order and the result is downward closed (`is_downward_closed`). The test checks the size, the cap and the closure of the set; the set has not been used in a forced solve yet (P11.1).
  - The total-degree cap must be q, and the forcing cap `min(q, forcing_order)`.

### P10 — Autonomous ROMs on real models; d_ref calibration; validation

- [x] **P10.1** `diagnostics.d_ref_sweep(engine_factory, φ, d_refs)`: compute `Ĝ(φ,φ)` and `Ĥ(φ,φ,φ)` for the first mode at each `d_ref`, and report the relative change between successive values (look for a plateau) together with any Abaqus failures.
  - Also report the O(3) CNF backbone coefficient per `d_ref`, which needs only a few jobs each.
  - Optional preflight: compute the minimum `det F` over Gauss points for each prescribed U (PySVK geometry) and refuse jobs with `det F ≤ 0.2`.
  - *Done 2026-11-11 (PySVK and mock Abaqus; the sweep on a real Abaqus is still to run, see the last bullet):*
    - How to run: `julia --project=julia/MORFEStEP scripts/d_ref_sweep.jl configs/<model>.toml --d-refs 1,5,10,20,40,80 [--json FILE]`. It prints two tables: the forces of the first master mode, and the ROM at order 3.
    - *Change from the text above:* the signature is `diagnostics.d_ref_sweep(engine, phi, d_refs, reference=None, det_f_min=0.2, plateau_tol=1e-6)`. Instead of an engine factory it uses the new `EEDEngine.with_d_ref(d_ref)`: a new engine on the same backend, matrix store and direction store. Each `d_ref` costs two jobs (`K_tg(±c d)`); `K_tg(0)` is shared.
    - Columns of the force table: `min det F` of the two prescribed states, `φ·Ĝ`, `φ·Ĥ`, the change against the previous row, and the error against the exact forms of a reference backend. The reference is the backend itself for analytic and PySVK, and the PySVK twin built from the same deck for Abaqus (`diagnostics.reference_backend`). A `d_ref` with `min det F ≤ 0.2` is refused before any job runs. A failed job is recorded in its row and the sweep goes on.
    - ROM table (Julia, `rom_d_ref_sweep(engine, d_refs; order = 3, reference = :auto)`): the coefficient of `z² z̄` in the reduced dynamics (`cubic_coefficient(R)`; its imaginary part is the slope of the backbone, `ω ≈ Im λ + Im(coefficient) ρ²`), its change from row to row and its error against the ROM of the exact closures. With one master pair it needs no job beyond the two of the force table.
    - New mock mode `NISTEP_MOCK_MODE=digits=<n>`: the mock Abaqus writes the matrices with `n` significant digits. The repository's processed matrices carry 15 digits (e.g. `3.18589451941020000000E+10`), so `digits=15` imitates a real `.mtx` file. **VERIFY** the number of digits of a raw `.mtx` on the host (P8.6).
    - Tests: `python/tests/test_d_ref_sweep.py` (9) and `julia/MORFEStEP/test/d_ref_sweep.jl` (49). Suites: Python 189 passed, Julia 472 passed.
    - **Results.** Forces: on both meshes `φ·Ĥ` is on a plateau from `d_ref = 1` up (error ≤ 1e-10), and `min det F` stays above 0.97 up to `d_ref = 80`, because the first mode is a bending mode. Error of the order-3 backbone coefficient against the exact-closure ROM:

      | `d_ref` | MEMS, PySVK (exact doubles) | MEMS, 15-digit matrices | Cantilever, PySVK | Cantilever, 15-digit matrices |
      |---|---|---|---|---|
      | 0.01 | 9.8 (wrong) | not run | 4.9 (wrong sign) | 4.1 at `d_ref = 0.02` (wrong) |
      | 0.1 | 1.4e-1 | not run | 3.5e-2 | 3.2e-1 |
      | 1 | 1.3e-3 | 3.0e-2 | 6.4e-4 | 1.5e-3 |
      | 5 | 1.7e-5 | 1.5e-3 | 3.3e-5 | not run |
      | 10 | 1.0e-5 | 4.7e-4 | 4.3e-6 | 1.6e-5 |
      | 20 | 2.5e-6 | 1.6e-4 | 2.7e-6 | 1.2e-5 |
      | 40 | 6.9e-8 | 5.6e-5 | 2.7e-7 | 7.3e-6 |
      | 80 | 4.3e-7 | 2.6e-5 | 5.9e-7 | not run |
      | 160 | 2.3e-7 | 1.9e-5 | not run | not run |

      The 15-digit columns come from the mock Abaqus (`digits=15`) and are a prediction for the real one.

    - **Why the ROM is so much more sensitive than the forces.** The backbone coefficient is a small difference of large terms. MEMS: `Im(coefficient) = 4.370e-12`, while its direct cubic part `3 φ·Ĥ(φ,φ,φ) / (2ω)` is 1.95e-4, so about 4.5e7 times larger; the quadratic terms cancel the rest. Cantilever: 1.658e-5 against 16.85, a factor of 1.0e6. The error of `Ĥ` is multiplied by that factor. The error of `Ĥ` itself falls as `1/c²`.
    - **This explains the calibration reported in the thesis** (`Thesis_Applications.tex`): the cantilever gave a non-physical softening backbone at `d_ref = 0.02`, a relative frequency error of 0.024 at `d_ref = 1` and 0.0004 at `d_ref = 10`; on the MEMS model only `d_ref = 10` and 20 gave the hardening response. The cause is the rounding of the tangent matrices, amplified by `1/c²` and by the cancellation factor above. It is not nodal overlap.
    - **Consequence for the choice of `d_ref`:** take it as large as the elements allow. The thesis values (MEMS 20, cantilever 10) are usable: with 15-digit matrices the predicted error of the MEMS backbone coefficient is 1.6e-4 at `d_ref = 20` and 5.6e-5 at 40. A value below 1 gives a wrong backbone.
    - **Open (host):** run the same command with `configs/mems_abaqus.toml` and `configs/cantilever_abaqus.toml` and record both tables here. The `err` columns then measure the noise of the real `.mtx` files.
- [ ] **P10.2** MEMS, conservative backbone (C=0 or tiny α), CNF, O(3), O(5), O(7), `d_ref` from P10.1 (thesis: 20).
  - **Accept:** convergence with order. Agreement with the PySVK closure ROM (P7.2), which separates Abaqus noise from method error. Agreement with the thesis figures, or the legacy outputs if the user provides them.
  - *Open (host only), 2026-11-11. Done without the host:*
    - Tool: `julia --project=julia/MORFEStEP scripts/order_convergence.jl configs/<model>.toml --orders 3,5,7,9 [--amplitude A] [--reference-config configs/<model>_pysvk.toml] [--json FILE]`. One engine solves every order, so lower orders' matrices are reused. Per order it prints the new jobs, the time, the backbone at the end of the amplitude range, its change against the previous order, and the differences against the exact-closure ROM. The JSON file holds the backbone curves (`rho`, `omega`, `amplitude` of the monitor DOF).
    - New helpers in `MORFEStEP.jl`: `monitor_index(engine, cfg, ϕ)` and `physical_amplitude(W, dof, amplitudes)` (largest displacement of a DOF over one period of the orbit). Tests: `test/amplitude.jl` (15).
    - **Config fix:** the monitor DOF of MEMS was wrong. Node 45 moves along y in the first mode (the plate is 1 thick in y), so `monitor_dof = 2`, not 1. The cantilever configs now have `monitor_node = 768`, `monitor_dof = 1` (thesis node, bending along x). A test checks that both monitor DOFs carry the largest entry of the first mode.
    - **MEMS, `d_ref = 20`, `alpha = 5e-7`, backbone up to a mode amplitude of 20 at node 45** (frequency at the end of the range; ω₁ = 1.819004):

      | Order | New jobs | Jobs so far | ω at the end | Change against the previous order (of the shift) | PySVK engine against exact closures (of the shift) | Mock Abaqus with 15-digit matrices against exact closures |
      |---|---|---|---|---|---|---|
      | 3 | 2 | 2 | 1.819965 | n/a | 4.3e-6 | 1.3e-4 |
      | 5 | 9 | 11 | 1.820181 | 18 % | 8.1e-7 | 4.7e-5 |
      | 7 | 15 | 26 | 1.820258 | 6.1 % | 9.6e-7 | 2.9e-5 |
      | 9 | 30 | 56 | 1.820291 | 2.6 % | 9.0e-7 | 2.8e-5 |

      The backbone converges with the order (each step changes it 0.3 to 0.4 times as much as the one before). The engine and the exact closures give the same frequencies to 7 digits, also through the mock Abaqus with prefetch (4 jobs at a time).
    - **Still to do on the host:** the same command with `configs/mems_abaqus.toml --reference-config configs/mems_pysvk.toml`; record the table here. Then compare with the thesis figure or the legacy outputs (decision D5).
- [ ] **P10.3** Cantilever, O(3)–O(9), `d_ref = 10` (thesis).
  - **Accept:** order convergence. Agreement with PySVK and MORFEFerrite (P7.3) on the same mesh. Thesis claim: absolute error < 0.003 against MORFE2.0 at `d_ref = 10`.
  - *Open (host only), 2026-11-11. Done without the host:*
    - **Cantilever, `d_ref = 10`, no damping, backbone up to a mode amplitude of 6 at node 768** (58 % of the beam length; ω₁ = 8.425782):

      | Order | New jobs | Jobs so far | ω at the end | Change against the previous order (of the shift) | PySVK engine against exact closures (of the shift) | Mock Abaqus with 15-digit matrices against exact closures |
      |---|---|---|---|---|---|---|
      | 3 | 2 | 2 | 8.484328 | n/a | 6.1e-6 | 2.0e-5 |
      | 5 | 5 | 7 | 8.487689 | 5.4 % | 5.8e-6 | 1.9e-5 |
      | 7 | 7 | 14 | 8.488131 | 0.71 % | 5.5e-6 | 1.9e-5 |
      | 9 | 13 | 27 | 8.488198 | 0.11 % | 5.5e-6 | 1.9e-5 |

      The frequency shift at the end of the range is 0.0624 (hardening, as in the thesis). With 15-digit matrices the predicted error of the frequency is 1.9e-5 × 0.0624 ≈ 1.2e-6, far below the 0.003 the thesis reports against MORFE2.0. Up to an amplitude of 3 the order-9 backbone is converged to 1.8e-5 of the shift.
    - MORFEFerrite (P7.3) was not installed; the references here are the exact closures of PySVK.
    - **Still to do on the host:** `scripts/order_convergence.jl configs/cantilever_abaqus.toml --orders 3,5,7,9 --amplitude 6 --reference-config configs/cantilever_pysvk.toml`; record the table here.
- [ ] **P10.4** Cost report: jobs per order, wall time, cache hit rate. Compare with the thesis run times (e.g. cantilever O(9): 29 min).
  - *Open (host only: the seconds per Abaqus job), 2026-11-11. Done without the host:* the job counts do not depend on the backend, so they are final.

    | Model | Order | Tangent jobs (probe + pair) | New jobs at orders 2 … 9 | Directions | Closure calls in one solve (quadratic + cubic) |
    |---|---|---|---|---|---|
    | Cantilever (2700 DOFs, no damping) | 9 | 27 (16 + 11) | 2, 0, 4, 1, 5, 2, 9, 4 | 8 | 167 + 319 |
    | MEMS (1521 DOFs, `alpha = 5e-7`) | 9 | 56 (26 + 30) | 2, 0, 6, 3, 9, 6, 18, 12 | 13 | 167 + 319 |

    - Add one job for `K_tg(0)` and one for the mass matrix. MEMS needs more jobs because its damping makes the manifold coefficients complex with independent real and imaginary parts.
    - Reuse: the 486 closure calls of an order-9 solve are served by 27 (or 56) matrices; every other request is a cache hit. No job ran twice in any run.
    - Time on this PC: PySVK under 0.1 s per cantilever matrix; MORFE 3.4 s for order 9 after about 25 s of compilation; mock Abaqus about 19 s of job time for the 28 cantilever jobs at 4 in parallel.
    - **Projection for Abaqus:** cantilever order 9 needs 29 jobs. At the 19 s per job of the legacy set-up (which recompiled the UEL every time) that is about 9 minutes one at a time and under 3 minutes with 4 licence tokens, against the 29 minutes of the legacy pipeline and the 4 minutes of MORFE2.0 quoted in the thesis. **Record the measured seconds per job (P8.6 e) and the real total here.**

### P11 — Forcing, FRF and exports

- [x] **P11.1** Forced model:
  - `force_term` with columns `(F̂/2)·M·φ₁` (thesis: `F̂ M Ỹᵘ cos(ω̂t)`).
  - `ExternalSystem((im*Ω, -im*Ω))` with `Ω = Im(λ₁)`.
  - Anisotropic multi-index set O(|γ|, |γ̂|), e.g. MEMS O(5,3) with `α = 5e-7` and `F̂ = 2e-3` (thesis values).
  - **VERIFY** in the MORFE tests (`test_external_coupling.jl`) how forced columns and resonance sets are configured.
  - *Done 2026-11-11 (PySVK and mock Abaqus; the run on a real Abaqus is open, see the last bullet):*
    - **VERIFIED in MORFE 0.2.0:**
      - `ExternalSystem((im*Ω, -im*Ω))` builds a diagonal external system (no re-basing). `NthOrderModel((K, C, M), (quad, cubic, force), ext)` is the forced model. The force term is `MultilinearMap(f!, (0, 0), 1)`; during the solve MORFE calls it with unit vectors `r`.
      - `parametrise` reads the external eigenvalues from the model and passes them to the resonance set (`build_resonance_set`). The default `conjugate_permutation = :from_spectral` extends the master pairing with the external one (`full_conjugate_permutation`), so the explicit `[2, 1]` of `build_spectral` is enough.
      - The degree-1 forcing monomials are solved first (`ExternalDirections.jl`), from the force term alone; the engine is not called for them.
    - **API (`MORFEStEP.jl`):**
      - `build_model(engine, cfg; forcing = (F, Ω))` builds the forced model; without `forcing` the model is autonomous.
      - `forcing_matrix(M, ϕ, F̂)` gives both columns `(F̂/2) M ϕ`. `harmonic_forcing(M, spectrum, cfg)` uses the shape of `[forcing] mode` and `Ω = Im λ` of the first master.
      - `setup_rom(engine, cfg)` returns `(model, spectral, spectrum, forcing)` as the config asks (forced when `[forcing] enabled`).
      - `build_spectral(...; spectrum = sp)` reuses a computed spectrum.
      - `solve_rom(...; order = q)` on a forced model solves O(q, `[rom] forcing_order`) (`expansion_set`); a forcing order of 0 means no cap. `run_with_prefetch` does the same.
    - **Frequency response (for P11.3):**
      - `forced_response(R; amplitudes, level = 1)` finds the steady states of the reduced dynamics in polar form, `z = ρ e^{i(ω̂t + ψ)}`, `r₁ = level · e^{iω̂t}`. Every monomial of the first row has `(a − b) + (p − q) = 1`, so a steady state solves `Re G(ρ, ψ) = 0` and gives `ω̂ = Im G / ρ`. The roots in `ψ` come from sampling plus bisection. The curve is traced by amplitude, so folds need no continuation, and stability comes from the 2 × 2 Jacobian. This is the orbit a harmonic balance of the reduced dynamics converges to (the thesis used pyhbm on the same equations).
      - `orbit_amplitude(W, dof, ρ, ψ)` gives the largest displacement of a DOF over the period.
      - `forced_response_amplitudes(R)` runs from 0 to 1.5 times the linear peak.
      - `forced_response_rel_diff(R1, R2; amplitudes)` is the frequency difference between two curves, relative to the largest detuning `|ω̂ − Ω|`.
      - `level` scales the load without a new solve, because R is polynomial in `r` (tested against a ROM with twice the load: equal to 1e-12).
    - **Scripts:**
      - `run_rom.jl` builds the forced model for a forced config and writes `frf.csv` (ρ, ψ, ω̂, stable, sheet, displacement of the monitor DOF). With `--reference` it compares the frequency response curves instead of the backbones.
      - `order_convergence.jl` handles forced configs: the peak of the curve replaces the end of the backbone.
      - New configs: `configs/mems_forced_pysvk.toml` and `configs/mems_forced_abaqus.toml` (O(5,3), `F̂ = 2e-3`, `α = 5e-7`, `d_ref = 20`).
    - **Conventions checked:**
      - `R₁[r₁] = −i F̂ / (4 Im λ₁)` (the modal projection of the load, for a mass-normalised mode).
      - The external rows of R are `diag(iΩ, −iΩ)`.
      - Forced Duffing (`α = 0.02`, `F̂ = 0.01`): the stable points of the ROM curve, started on the ROM orbit and integrated with the full equation (RK4, 400 periods), keep the ROM amplitude to 1.5e-3 near resonance at O(7,3). Error at the largest amplitude: 2.7e-3 at O(3,3), 6.3e-4 at O(7,3). Far from resonance (|ω̂ − Ω| ≈ 0.12) the error is 3.8e-3 at every order: the ROM is built at Ω and used at ω̂.
      - Engine against exact closures, same load: Duffing and 2-DoF ≤ 1e-12; prefetch equals lazy for the forced 2-DoF model, with the same jobs.
    - **Robustness fix:** `nistep.config.load_config` now accepts a UTF-8 byte-order mark. PowerShell 5.1 and Notepad write one, and `tomllib` rejected it. A file that is not UTF-8 gives a `ConfigError`.
    - Tests: `julia/MORFEStEP/test/forcing.jl` (45) and `python/tests/test_config.py` (+1). Suites: Julia 532 passed, Python 190 passed.
    - **MEMS, O(q, 3), `F̂ = 2e-3`, `α = 5e-7`, `d_ref = 20`** (`order_convergence.jl configs/mems_forced_pysvk.toml --orders 3,5,7`). Ω = 1.81900417; the half-power bandwidth is only `α = 5e-7`, so the curve folds strongly:

      | Order | New jobs | Jobs so far | ω̂ at the peak | Peak displacement, node 45 y | Change against the previous order (of the largest detuning) | PySVK engine against exact closures | Mock Abaqus with 15-digit matrices against exact closures |
      |---|---|---|---|---|---|---|---|
      | O(3,3) | 2 | 2 | 1.819009444 | 1.48 | n/a | 3.4e-7 | 4.6e-4 |
      | O(5,3) | 30 | 32 | 1.819009450 | 1.48 | 9.7e-5 | 3.4e-7 | 4.6e-4 |
      | O(7,3) | 97 | 129 | 1.819009451 | 1.48 | 1.9e-7 | 3.4e-7 | 4.6e-4 |

      - The three orders give practically the same curve, as the thesis reports (`Thesis_Applications.tex`, right panel of `fig:MEMS_forcing`). The response hardens: the peak is 5.3e-6 above Ω.
      - With 15-digit matrices the predicted frequency error is 4.6e-4 × 5.3e-6 ≈ 2e-9.
      - Cost: the forcing monomials add directions. O(5,3) needs 32 jobs (11 without forcing) and O(7,3) needs 129 (26 without). At the legacy 19 s per job, O(7,3) is about 41 minutes one job at a time, against the 2 h 13 min reported in the thesis.
    - **Still to do on the host:** `scripts/order_convergence.jl configs/mems_forced_abaqus.toml --orders 3,5,7 --reference-config configs/mems_pysvk.toml`; record the table here.
- [ ] **P11.2** Exports:
  - the realified reduced dynamics (`realify`);
  - the observable polynomial at the monitor DOF (`observable_polynomial(W, free_index(node, dof))`);
  - everything written to `npz` and CSV.
  - `scripts/export_for_hbm.py` loads them into numpy arrays in the layout the user's HBM tool (pyhbm) needs. **Ask the user for that layout (§9 D4).**
  - *Done 2026-11-11, except the pyhbm adapter (open: decision D4):*
    - `real_rom(W, R; dofs)` (`MORFEStEP.jl`) applies MORFE's `realify` to the reduced dynamics and to the displacement and velocity of the chosen DOFs.
      - Real variables: `z_p = x_p + i y_p` for each master pair; for a forced ROM also `r₁ = c + i s`, which with the load of the config means `c = cos ω̂t`, `s = sin ω̂t`.
      - Result: one exponent table for all polynomials, `f` (2m × L) with `ẋ_p` and `ẏ_p`, `displacement` and `velocity` (one row per DOF), and `omega0` (the Ω the ROM was built at).
      - The number of master coordinates comes from the type `ReducedDynamics{ROM}`, because R also has rows for the forcing variables. If the realified displacements are not real (no conjugate symmetry), it throws an `ArgumentError`.
    - `export_real_rom(dir, W, R; dofs, labels)` writes `real_rom.npz`, `real_rom_f.csv` (exponents, then `dx1/dt`, `dy1/dt`, …) and `real_rom_obs.csv` (exponents, then `u_<label>`, `v_<label>`).
    - `run_rom.jl` writes them on every run, with the monitor DOF (label `node45_y` for MEMS).
    - Python: `nistep.realrom.RealROM.load(path)` with `rhs(q, t, omega, level)`, `displacement(...)` and `velocity(...)`; it also evaluates many points at once. `rhs` is the right-hand side of a real first-order ODE with `2m` states, ready for a time integrator or a harmonic balance solver.
    - Tests:
      - `julia/MORFEStEP/test/exports.jl` (89): the real polynomials equal the complex ones at random points; the orbits of `forced_response` solve the exported ODE; the backbone of the autonomous ROM; the CSV headers; the Python loader through PythonCall.
      - `python/tests/test_realrom.py` (3).
    - **Open (D4):** the call layout of pyhbm. `RealROM.rhs` is generic; an adapter needs pyhbm's interface.
    - **P11 status:** Julia suite 621 passed, Python suite 193 passed (2026-11-11).
- [ ] **P11.3** **Accept:** FRF curves (from the user's HBM tool, or BifurcationKit via the MORFE extension) agree with the PySVK reference and with the thesis MEMS figure (`Thesis_Applications.tex`, `sec:MEMS`) at O(5,3).
  - *Open (host only), 2026-11-11. Done without the host:*
    - **Change from the text above:** the curves come from `forced_response` (P11.1), not from an HBM tool or BifurcationKit. In complex normal form with one master pair, the steady states of the reduced dynamics are exactly the solutions of `Re G(ρ, ψ) = 0`, `ω̂ = Im G / ρ`. A harmonic balance of the same reduced dynamics converges to the same orbits, so no truncation in harmonics is involved. The real ROM of P11.2 is there for the user's HBM tool.
    - **Against the PySVK reference:** MEMS O(5,3), engine against exact closures 3.4e-7 of the largest detuning (table in P11.1); with 15-digit matrices 4.6e-4.
    - **Against the thesis figure** (`images/Results_MEMS/scalar_factorsordesn.jpg`, digitised from the image; the thesis plots max|U₄₅| against ω − ω₀). Our curve is from PySVK, O(5,3), node 45 along y. The thesis' ω₀ is not ours: one constant offset, fitted at u = 0.1, puts it at 1.8190000 (ours: Ω = 1.81900417).

      | u (node 45) | Width of the curve, thesis | Width, ours | Middle of the curve, thesis | Middle, ours (shifted) |
      |---|---|---|---|---|
      | 0.1 | 7.54e-6 | 7.39e-6 | 4.20e-6 | 4.20e-6 (fitted) |
      | 0.2 | 3.70e-6 | 3.67e-6 | 4.27e-6 | 4.27e-6 |
      | 0.5 | 1.41e-6 | 1.39e-6 | 4.77e-6 | 4.78e-6 |
      | 1.0 | 0.53e-6 | 0.55e-6 | 6.55e-6 | 6.58e-6 |

      - Peak (inset of the left panel): MORFE2.0 1.48 at 9.45e-6; thesis Abaqus at `d_ref = 20` 1.48 at 9.39e-6, at `d_ref = 10` 9.26e-6; **ours 1.480 at 9.43e-6**. Our curve lies between the thesis Abaqus curve and the MORFE2.0 reference, 0.5 % of the nonlinear shift (5.3e-6) from MORFE2.0. That is at the accuracy of reading the image: about ±0.03e-6, set by the axis alignment.
      - The thesis axis label is U₄₅;x, but node 45 moves along y in the first mode (P10.2); the peak amplitude 1.48 matches the y displacement.
    - **Still to do on the host:** P11.1's `order_convergence.jl` run with `configs/mems_forced_abaqus.toml`, then `run_rom.jl configs/mems_forced_abaqus.toml`; compare its `frf.csv` with the PySVK one (`run_rom.jl configs/mems_forced_pysvk.toml`). If the user provides the MORFE2.0 or legacy data (D5), compare with those instead of the digitised figure.

### P12 — Legacy regression, documentation, cleanup

- [ ] **P12.1** *(Recommended)* Formula-level legacy regression. Build a harness that imports `test/StEP_validation_with_complex_numbers_abaqus_G_terms.py` and `..._H_terms.py`, then monkeypatches:
  - `Duplicate_Input_File.InputFileModifier.create_modified_input_files` to capture the displacement and output name;
  - `Run_jobs.run_abaqus_job` to a no-op;
  - `Organize_the_matrix_right_order.MatrixProcessor.process_matrix_file` to compute `K_tg(captured U)` with the Analytic or PySVK backend;
  - `EmbeddedStiffnessMatrix.load_and_extract_non_embedded_matrix` to return the stored matrix.

  Feed the same `W_u` vectors. **Accept:** the legacy `sum_G` and `sum_H` per γ equal the new engine's thesis-convention sums (`Σ multiplicity · Ĝ/Ĥ` over the same decompositions) to ≤ 1e-9. No edits to legacy files; patch only at runtime.
- [ ] **P12.2** *(Optional)* Full legacy DPIM on the single-element deck with the PySVK stand-in, against the MORFE pipeline. Compare backbones; raw coefficients depend on the gauge, so use `RomComparison` `:gauge_invariant` or the backbones.
- [ ] **P12.3** Documentation: a `README_interface.md` covering install, configs, running on the host and through the spool, troubleshooting and the conventions. Update the repository `README.md` only if the user asks.
- [ ] **P12.4** Final review: `pytest -q` and `Pkg.test()` green; one stats summary per validated model saved under `runs/`.

---

## 7. Risk register

| # | Risk | Mitigation | Detection |
|---|---|---|---|
| R1 | Factor or sign errors (2×, 3×, 6×; RHS minus) | §4 is the single contract; the minus sign exists only in the Julia closure; exact analytic tests | P2.4, P3.2, P4.5, P5.2 |
| R2 | Accidental complex conjugation | §4.8 rules; tests with independent Re/Im parts would fail | P2.4 (b) |
| R3 | DOF order mismatch (inp BC ↔ assembly ↔ M/K ↔ Julia) | One `DofMap`; sorted nodes | P6.6 (a) PySVK versus repository matrices; P8.6 (a, b) |
| R4 | Stale or failed Abaqus outputs reused (legacy B1–B5) | Unique job directories, content keys, exit/.sta/element-count checks | P8.3 tests (inject a failing job) |
| R5 | Noise floor against nodal overlap (d_ref) | Exact polynomial identities let c be large; d_ref sweep; preflight det F; full-precision BC values | P8.6 (c), P10.1 |
| R6 | K ≠ UEL K_tg(0) breaks `−2K` cancellation | K always taken from the backend's own U=0 job | P8.6 (d); `debug` symmetry and noise checks |
| R7 | UEL F-bar flag uninitialised (B9) gives non-polynomial K_tg | Cubic-exactness test; fix only with user approval | P8.6 (c) |
| R8 | Job count or wall time explodes | Lowest-order probing, conjugate and proportional dedupe, precompiled library, parallel prefetch batches, cost estimate from PySVK dry runs | P5.3, P7.2, P10.4 |
| R9 | PythonCall environment issues on Windows (Store Python, offline CondaPkg, PyCall clash) | CondaPkg default; Null-backend fallback with a real Python; separate project env | P0.6 |
| R10 | MORFE alpha API changes | Pin version and tree hash in the Manifest; all MORFE calls in one file; VERIFY steps | P0.4, P4 |
| R11 | Non-bitwise reproducible MORFE vectors across passes | Tolerance-matched registry; final lazy pass | P9.3 |
| R12 | Memory from many sparse K_tg | Shared pattern, LRU and disk spill | P9.4 |
| R13 | Remote execution fragility (share latency, worker crash, licences) | Atomic rename protocol, heartbeat, timeouts and retries, replay backend for offline debugging | P8.5 tests (kill the worker mid-job) |
| R14 | Julia 1.12 KLU bug | Stay on 1.11.6 | P0.4 |
| R15 | Galerkin basis-mode ROM misleads (constraint forces) | Clearly labelled approximate; the closure route is the reference | P3 docs |
| R16 | Thesis errata copied into code | §3.3 list; implement from §4 | review |

---

## 8. Verification matrix (what proves the system works)

1. **Math primitives (exact):** the Analytic backend gives the engine `Ĝ`/`Ĥ` versus exact to ≤ 1e-9, for real and complex arguments and under all permutations (P2.4).
2. **StEP.tex basis mode:** the 2-DoF example gives `G̃_11 = [20,20]` and `H̃_111 = [10,10]`; the job count equals `(n²+n+2)/2` for n = 1..8 (P3.2).
3. **MORFE integration:** the engine-closure ROM equals the pure-Julia exact-closure ROM to ≤ 1e-7 (P5.2).
4. **FE plumbing without Abaqus:** PySVK K and M equal the repository `.mtx` matrices; the engine on PySVK equals the PySVK exact polarised forms (P6.6).
5. **Physics sanity:** ω₁ matches the thesis for MEMS and cantilever (P7.1); the PySVK ROM matches MORFEFerrite (P7.3).
6. **Abaqus correctness:** single element, Abaqus versus PySVK `K_tg(U)`; cubic-exactness third difference ≈ 0 (P8.6).
7. **Pipeline equivalence:** prefetch equals lazy; resume causes no duplicate jobs (P9).
8. **Real models:** order convergence; the Abaqus-EED ROM agrees with the PySVK and MORFEFerrite ROMs and the thesis figures (P10–P11).
9. **Legacy regression:** the legacy G/H sums equal the new engine's sums for identical inputs (P12.1).

Commands:

- Python: `.venv\Scripts\python -m pytest python/tests -q`
- Julia: `julia --project=julia/MORFEStEP -e "using Pkg; Pkg.test()"`
- Run: `julia --project=julia/MORFEStEP scripts/run_rom.jl configs/<model>.toml`

---

## 9. Decisions deferred (ask the user at the named step)

- **D1 (P8):** Where do Julia and Python run in production: on the Abaqus host (LocalRunner, simplest) or here, with jobs sent via a shared folder (SpoolRunner) or SSH? What OS is the host, and what is the share path?
- **D2 (P8):** Approve fixing UEL bugs B9 (`fbarFlag`) and B10 (`rho` integer) in **copies** of the Fortran files, keeping the originals?
- **D3 (P8):** How many Abaqus licence tokens are available, which sets `max_parallel`?
- **D4 (P11):** Which HBM tool and data layout should the exports target (pyhbm as in the thesis, or BifurcationKit inside Julia)?
- **D5 (P10):** Can the user provide legacy outputs or MORFE2.0 results (backbones, FRFs) for numeric comparison?

---

## 10. Out of scope (possible follow-ups)

- **Element-level tensor identification.** Abaqus already outputs *element* matrices. Global probe fields built from a node colouring (8 colours × 3 DOFs) plus about 300 pair fields could identify every element's `Ĝᵉ` and `Ĥᵉ` in a few hundred jobs, independent of order and of the number of masters. All later closures would then be exact and free. This is a big speed-up, but it goes beyond the EED-StEP as specified.
- Force-based (ED) StEP and RF-output cross-checks.
- Other FE codes (COMSOL, Ansys), other materials, hyper-reduction (thesis future work).

---

## Appendix A — Legacy → new module map

| Legacy (read-only reference) | New | Notes |
|---|---|---|
| `test/WriteInputfile_Class.py` | `nistep/inp_io.py`, `dofmap.py` | configurable clamped set; sorted; deduplicated |
| `test/Write_BC_SETS_Input_file.py:86-102` | `inp_io.prepare_base_inp` | per-node Nsets |
| `test/Duplicate_Input_File.py:40-94` | `inp_io.render_job_inp` | returns text; `.17e`; never overwrites |
| `test/Run_jobs.py` | `runners/local.py` | subprocess, checks, precompiled library |
| `test/Organize_the_matrix_right_order.py:101-134, 220-236` | `mtx_io.py`, `assembly.py` | vectorised; raises; keeps zeros |
| `test/EmbeddedStiffnessMatrix_Class.py` | `assembly.py` (elimination), `mtx_io.load_processed_ijv` | |
| `test/Compute_all_options_K_tg.py` (options 1/2/3, scale and sign rules) | `engine.py` + `registry.py` | generalised §4.3–4.5 |
| `test/StEP_validation_*_G_terms.py`, `*_H_terms.py` | `engine.quad`, `engine.cubic` (+ thesis-convention sums in tests) | returns Ĝ/Ĥ, not monomial coefficients |
| `test/eigenproblem_new.py` | MORFE `StructureModalDampingEigensolver` (+ scipy check in P7.1) | |
| `test/Compute_higher_order_terms.py`, `Homological_equation_solver.py`, `Multiindex_exponents.py`, `Resonance_handler.py`, `Xi_gamma_computer.py`, `Compute_velocity_mapping.py`, `Compute_OrderOneTerms.py` | **replaced by MORFE.jl** | kept only for P12 regression |

## Appendix B — Code skeletons

**Julia closures** (`julia/MORFEStEP/src/MORFEStEP.jl`):

```julia
function quadratic_term(engine::Py)
    f! = function (res, a, b)
        Threads.threadid() == 1 || error("Python closure called off main thread")
        out = engine.quad(collect(a), collect(b))        # Ĝ(a,b), internal-force sign (+)
        res .-= PyArray{ComplexF64,1}(out)               # MORFE: nonlinear terms on RHS
        return res
    end
    return MultilinearMap(f!; multiindex = (2, 0), multiplicity_external = 0,
                          fully_asymmetric = false)
end
# cubic_term: same with (res, a, b, c), engine.cubic, multiindex = (3, 0)

function force_term(F::Matrix{Float64})                  # columns: (F̂/2)·M·φ for r₁, r₂
    MultilinearMap((res, r) -> begin
        for j in axes(F, 2); iszero(r[j]) || axpy!(r[j], view(F, :, j), res); end
        res
    end, (0, 0), 1)
end
```

**Python engine API** (`python/nistep/engine.py`):

```python
class EEDEngine:
    def __init__(self, backend: TangentBackend, step: StepCfg, store: KtgStore): ...
    def K(self) -> csr_matrix: ...            # backend.stiffness0()  (== K_tg(0))
    def M(self) -> csr_matrix: ...
    def quad(self, a, b) -> np.ndarray: ...   # Ĝ(a,b), complex128 (N,)
    def cubic(self, a, b, c) -> np.ndarray: ...# Ĥ(a,b,c)
    def set_mode(self, mode: str) -> None: ... # "lazy" | "record"
    def pending_count(self) -> int: ...
    def execute_pending(self, max_parallel: int | None = None) -> int: ...
    def stats(self) -> dict: ...
```

**Config sketch** (`configs/mems.toml`):

```toml
[model]   inp = "../test/mems_abaqus_UEL_modified.inp"   clamped_nset = "Set-2"
[damping] alpha = 5e-7   beta = 0.0
[step]    d_ref = 20.0   eps = 0.24
[backend] kind = "abaqus"        # analytic | pysvk | replay | abaqus
[abaqus]  runner = "spool"   spool_dir = "<share>"   max_parallel = 4   timeout_s = 900
[rom]     masters = [1]   order = 5   forcing_order = 3   style = "complex_normal_form"   resonance_tol = 0.05
[forcing] enabled = true   amplitude = 2e-3
[output]  dir = "../runs/mems_O5_3"   monitor_node = 45   monitor_dof = 2
[cache]   dir = "../cache"   lru_items = 64
```

(Use proper multi-line TOML tables in the real file.)

## Appendix C — Abaqus facts used

- Deck: `*User Element, Type=U3, Nodes=8, Coordinates=3, Properties=3, Iproperties=4, Unsym`; `*UEL PROPERTY: E, ν, ρ, nInt=8, fbar=0, matID=1, nPost=12`; `*Step, nlgeom=YES, inc=1` / `*Static, direct` / `1., 1.`.
- BC block: `*Boundary` / `<clamped>, ENCASTRE` / `node<i>, d, d, value` for every free DOF.
- Output: `*Element Matrix Output, Elset=allUEL, File Name=<name>, Output File=User Defined, Stiffness=Yes` produces `<name>.mtx` with per-element blocks.
- UEL outputs: `AMATRX = K_tg` (or M in the mass variant) and `RHS = −F_int`. Linear-perturbation steps are rejected.
- The legacy command was `abaqus interactive double analysis ask_delete=off job=<j> user=uel_nlmech_pk1.for output_precision=full`, which took about 19 s per job including recompilation. The new design uses a precompiled library.
