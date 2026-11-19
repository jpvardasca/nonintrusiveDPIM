"""
MORFEStEP: thin glue between MORFE.jl (the DPIM solver) and the Python package `nistep`
(the FE side: tangent-stiffness backends and the EED-StEP closure engine).

Math conventions are in `CONVENTIONS.md` (PLAN.md §4). The two that matter here:

* The Python engine returns the polarised internal forces `Ĝ(a,b)` and `Ĥ(a,b,c)` with the
  **internal-force sign (+)**. MORFE keeps nonlinear terms on the right-hand side, so the
  closures in this file do `res .-= …`. This is the only place the minus sign is applied.
* MORFE vectors are complex and must never be conjugated: use `transpose`, never `'`.

All calls into Python must come from the main thread (run Julia with one thread).
"""
module MORFEStEP

using LinearAlgebra
using SparseArrays
using Arpack, LinearMaps   # Arpack for the sparse eigenproblem; both activate MORFE's Arpack extension
using MORFE
using PythonCall
import NPZ

export RunConfig, load_engine, run_config, to_julia_sparse, engine_stats,
       quadratic_term, cubic_term, force_term,
       exact_quadratic_term, exact_cubic_term, exact_terms, backend_exact_terms,
       build_matrices, build_model, modal_spectrum, build_spectral, resonance_config,
       forcing_matrix, harmonic_forcing, setup_rom,
       solve_rom, solve_rom_by_order, expansion_set, run_with_prefetch,
       rom_rel_diff, rom_rel_diff_by_degree,
       backbone_amplitudes, backbone_rel_diff, cubic_coefficient, rom_d_ref_sweep,
       monitor_index, physical_amplitude, forced_response, forced_response_amplitudes,
       forced_response_rel_diff, orbit_amplitude, export_rom, real_rom, export_real_rom

"Repository root (`nonintrusiveDPIM-main/`)."
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
"Directory holding the Python package `nistep`."
const PYTHON_DIR = joinpath(REPO_ROOT, "python")
"Models with at most this many DOFs use the dense eigensolver."
const DENSE_LIMIT = 200
"Keeps the last loaded engine alive (closures also hold a reference)."
const ENGINE_REF = Ref{Any}(nothing)

# ============================================================================== bridge

"""
    load_engine(config_path) -> Py

Put `python/` on Python's `sys.path`, import `nistep` and build the `EEDEngine` described
by the TOML file. The Python `Config` object is available as `engine.config`.
"""
function load_engine(config_path::AbstractString)
    sys = pyimport("sys")
    paths = [pyconvert(String, pystr(p)) for p in sys.path]
    PYTHON_DIR in paths || sys.path.insert(0, PYTHON_DIR)
    nistep = pyimport("nistep")
    engine = nistep.build_engine(abspath(config_path))
    ENGINE_REF[] = engine
    return engine
end

"""
    RunConfig

The settings Julia needs, read from the Python config (`engine.config`). Python's
`nistep/config.py` is the only place where defaults and validation live.
"""
struct RunConfig
    name::String
    path::String
    alpha::Float64
    beta::Float64
    masters::Vector{Int}       # 1-based mode numbers, ordered by frequency
    order::Int
    forcing_order::Int
    style::Symbol
    resonance_tol::Float64
    n_modes::Int               # 0 = automatic
    forcing_enabled::Bool
    forcing_amplitude::Float64
    forcing_mode::Int
    output_dir::String
    monitor_node::Union{Nothing, Int}
    monitor_dof::Union{Nothing, Int}
    max_parallel::Int
    prefetch::Bool             # run the tangent jobs of each order as one batch (P9)
end

_pystring(x::Py) = pyconvert(String, pystr(x))
_optional_int(x::Py) = pyis(x, pybuiltins.None) ? nothing : pyconvert(Int, x)

"""
    run_config(engine) -> RunConfig
"""
function run_config(engine::Py)
    cfg = engine.config
    pyis(cfg, pybuiltins.None) &&
        error("engine has no config; build it with load_engine(config_path)")
    return RunConfig(
        pyconvert(String, cfg.name),
        _pystring(cfg.path),
        pyconvert(Float64, cfg.damping.alpha),
        pyconvert(Float64, cfg.damping.beta),
        [pyconvert(Int, m) for m in cfg.rom.masters],
        pyconvert(Int, cfg.rom.order),
        pyconvert(Int, cfg.rom.forcing_order),
        Symbol(pyconvert(String, cfg.rom.style)),
        pyconvert(Float64, cfg.rom.resonance_tol),
        pyconvert(Int, cfg.rom.n_modes),
        pyconvert(Bool, cfg.forcing.enabled),
        pyconvert(Float64, cfg.forcing.amplitude),
        pyconvert(Int, cfg.forcing.mode),
        _pystring(cfg.output.dir),
        _optional_int(cfg.output.monitor_node),
        _optional_int(cfg.output.monitor_dof),
        pyconvert(Int, cfg.abaqus.max_parallel),
        pyconvert(Bool, cfg.use_prefetch))
end

"""
    to_julia_sparse(A::Py) -> SparseMatrixCSC{Float64, Int}

Convert a scipy sparse matrix. Explicit zeros are kept, so the pattern is stable.
"""
function to_julia_sparse(A::Py)
    csc = A.tocsc()
    csc.sum_duplicates()   # canonical format: sorted row indices, no duplicates
    m, n = pyconvert(Tuple{Int, Int}, csc.shape)
    colptr = Int.(PyArray(csc.indptr)) .+ 1
    rowval = Int.(PyArray(csc.indices)) .+ 1
    nzval = Vector{Float64}(PyArray(csc.data))
    return SparseMatrixCSC{Float64, Int}(m, n, colptr, rowval, nzval)
end

"""
    engine_stats(engine) -> Dict{String, Any}

Counters of the Python engine (jobs run, calls, cache hits, backend time, …).
"""
engine_stats(engine::Py) = pyconvert(Dict{String, Any}, engine.stats())

# ============================================================================ closures

function _check_thread()
    Threads.threadid() == 1 ||
        error("MORFEStEP: a Python closure was called off the main thread; run Julia with one thread")
    return nothing
end

# A fresh dense complex vector: MORFE passes views of W, and real states are allowed too.
_pyvec(a) = Vector{ComplexF64}(a)

"""
    quadratic_term(engine) -> MultilinearMap

`f!(res, a, b)` adds `−Ĝ(a, b)` to `res`, with `Ĝ` evaluated by the EED-StEP engine.
"""
function quadratic_term(engine::Py)
    f! = function (res, a, b)
        _check_thread()
        out = engine.quad(_pyvec(a), _pyvec(b))        # Ĝ(a,b), internal-force sign (+)
        res .-= PyArray{ComplexF64, 1}(out)            # MORFE: nonlinear terms on the RHS
        return res
    end
    return MultilinearMap(f!, (2, 0); fully_asymmetric = false)
end

"""
    cubic_term(engine) -> MultilinearMap

`f!(res, a, b, c)` adds `−Ĥ(a, b, c)` to `res`.
"""
function cubic_term(engine::Py)
    f! = function (res, a, b, c)
        _check_thread()
        out = engine.cubic(_pyvec(a), _pyvec(b), _pyvec(c))
        res .-= PyArray{ComplexF64, 1}(out)
        return res
    end
    return MultilinearMap(f!, (3, 0); fully_asymmetric = false)
end

"""
    force_term(F::Matrix{Float64}) -> MultilinearMap

External forcing: `res += Σ_j r[j] F[:, j]`, one column per external variable. MORFE calls
it with basis directions `r`, so index `r[j]`; never sum it.
"""
function force_term(F::Matrix{Float64})
    f! = function (res, r)
        for j in axes(F, 2)
            iszero(r[j]) || (res .+= r[j] .* view(F, :, j))
        end
        return res
    end
    return MultilinearMap(f!, (0, 0), 1)
end

# ------------------------------------------------------------ exact reference closures

"""
    exact_quadratic_term(I, J, K, V) -> MultilinearMap

Pure-Julia closure for `Ĝ(a,b)[I[e]] += V[e] a[J[e]] b[K[e]]` (1-based COO entries of a
tensor that is symmetric in `(J, K)`). Used as the reference for the engine closures.
"""
function exact_quadratic_term(I::Vector{Int}, J::Vector{Int}, K::Vector{Int},
        V::Vector{Float64})
    f! = function (res, a, b)
        @inbounds for e in eachindex(V)
            res[I[e]] -= V[e] * a[J[e]] * b[K[e]]
        end
        return res
    end
    return MultilinearMap(f!, (2, 0); fully_asymmetric = false)
end

"""
    exact_cubic_term(I, J, K, L, V) -> MultilinearMap
"""
function exact_cubic_term(I::Vector{Int}, J::Vector{Int}, K::Vector{Int}, L::Vector{Int},
        V::Vector{Float64})
    f! = function (res, a, b, c)
        @inbounds for e in eachindex(V)
            res[I[e]] -= V[e] * a[J[e]] * b[K[e]] * c[L[e]]
        end
        return res
    end
    return MultilinearMap(f!, (3, 0); fully_asymmetric = false)
end

_index(x::Py) = Int.(PyArray(x)) .+ 1
_values(x::Py) = Vector{Float64}(PyArray(x))

"""
    exact_terms(engine) -> (quad, cubic)

Pure-Julia exact closures built from the tensors of an **analytic** backend.
"""
function exact_terms(engine::Py)
    be = engine.backend
    kind = pyconvert(String, be.kind)
    kind == "analytic" ||
        throw(ArgumentError("exact_terms needs an analytic backend, got \"$kind\"; " *
                            "use backend_exact_terms for other backends"))
    G, H = be.G, be.H
    quad = exact_quadratic_term(_index(G.i), _index(G.j), _index(G.k), _values(G.v))
    cubic = exact_cubic_term(_index(H.i), _index(H.j), _index(H.k), _index(H.l),
        _values(H.v))
    return (quad, cubic)
end

"""
    backend_exact_terms(engine) -> (quad, cubic)

Closures that call the backend's own `exact_quad` / `exact_cubic` in Python (no EED-StEP).
Available when the backend knows its polarised forms (analytic, PySVK).
"""
function backend_exact_terms(engine::Py)
    be = engine.backend
    (pyis(be.exact_quad, pybuiltins.None) || pyis(be.exact_cubic, pybuiltins.None)) &&
        throw(ArgumentError("this backend has no exact polarised forms"))
    np = pyimport("numpy")
    q! = function (res, a, b)
        _check_thread()
        out = np.asarray(be.exact_quad(_pyvec(a), _pyvec(b)); dtype = np.complex128)
        res .-= PyArray{ComplexF64, 1}(out)
        return res
    end
    c! = function (res, a, b, c)
        _check_thread()
        out = np.asarray(be.exact_cubic(_pyvec(a), _pyvec(b), _pyvec(c));
            dtype = np.complex128)
        res .-= PyArray{ComplexF64, 1}(out)
        return res
    end
    return (MultilinearMap(q!, (2, 0); fully_asymmetric = false),
        MultilinearMap(c!, (3, 0); fully_asymmetric = false))
end

# =============================================================================== model

"""
    build_matrices(engine, alpha, beta) -> (K, C, M)

`K` is the backend's own `K_tg(0)`, `M` the mass matrix and `C = alpha M + beta K`, all
`SparseMatrixCSC{Float64, Int}` in the DOF order of the Python side.
"""
function build_matrices(engine::Py, alpha::Real, beta::Real)
    K = to_julia_sparse(engine.K())
    M = to_julia_sparse(engine.M())
    size(K) == size(M) || error("K and M have different sizes: $(size(K)) and $(size(M))")
    C = Float64(alpha) * M + Float64(beta) * K
    return K, C, M
end

"""
    build_model(engine, cfg = run_config(engine); terms = :engine, matrices = nothing,
                forcing = nothing)

The MORFE full-order model `K u + C u̇ + M ü = −g(u) − h(u) [+ f(t)]`.

`terms` selects the nonlinear closures: `:engine` (EED-StEP, the production route),
`:exact` (pure-Julia closures of an analytic backend) or `:backend_exact` (the backend's
exact forms through Python). Pass `matrices = (K, C, M)` to reuse matrices already built.

The model is autonomous unless `forcing = (F, Ω)` is given: then `f = F[:, 1] r₁ + F[:, 2] r₂`
with the external system `ṙ₁ = iΩ r₁`, `ṙ₂ = −iΩ r₂` (two forcing variables, after the
reduced coordinates). [`setup_rom`](@ref) builds `(F, Ω)` from the `[forcing]` section of
the config; this function does not read that section.
"""
function build_model(engine::Py, cfg::RunConfig = run_config(engine);
        terms::Symbol = :engine, matrices = nothing, forcing = nothing)
    K, C, M = matrices === nothing ? build_matrices(engine, cfg.alpha, cfg.beta) : matrices
    nl = if terms === :engine
        (quadratic_term(engine), cubic_term(engine))
    elseif terms === :exact
        exact_terms(engine)
    elseif terms === :backend_exact
        backend_exact_terms(engine)
    else
        throw(ArgumentError("terms must be :engine, :exact or :backend_exact, got :$terms"))
    end
    forcing === nothing && return NthOrderModel((K, C, M), nl)
    F, Ω = forcing
    size(F) == (size(K, 1), 2) ||
        throw(ArgumentError("the forcing matrix must be $(size(K, 1)) × 2, got $(size(F))"))
    Ω > 0 || throw(ArgumentError("the forcing frequency must be positive, got $Ω"))
    ext = ExternalSystem((im * Float64(Ω), -im * Float64(Ω)))
    return NthOrderModel((K, C, M), (nl..., force_term(Matrix{Float64}(F))), ext)
end

"""
    forcing_matrix(M, ϕ, amplitude) -> Matrix{Float64}

The `N × 2` forcing matrix of a harmonic modal load: both columns are `(amplitude / 2) M ϕ`.
With the forcing variables `r₁ = e^{iω̂t}`, `r₂ = e^{−iω̂t}` the force is
`amplitude · M ϕ · cos(ω̂t)`, the load of the thesis (`F̂ M Ỹᵘ cos(ω̂t)`).
"""
function forcing_matrix(M::AbstractMatrix, ϕ::AbstractVector, amplitude::Real)
    v = (Float64(amplitude) / 2) .* Vector{Float64}(M * ϕ)
    return hcat(v, v)
end

"""
    harmonic_forcing(M, spectrum, cfg) -> (F, Ω)

The load of the `[forcing]` section: `F = forcing_matrix(M, ϕ, amplitude)` with `ϕ` the
(real, mass-normalised) shape of mode `[forcing] mode`, and `Ω = Im λ` of the first master
mode, so the forcing is in 1:1 resonance with the master pair, as in the thesis. The ROM is
built at this `Ω`; [`forced_response`](@ref) then varies the forcing frequency around it.
"""
function harmonic_forcing(M::AbstractMatrix, sp, cfg::RunConfig)
    p, q = cfg.forcing_mode, cfg.masters[1]
    nev = length(sp.eigenvalues) ÷ 2
    p <= nev || throw(ArgumentError("forcing mode $p is not among the $nev modes computed; " *
                                    "raise [rom] n_modes"))
    ϕ = real.(sp.eigenmodes[:, 1, 2p - 1])
    Ω = imag(sp.eigenvalues[2q - 1])
    Ω > 0 || throw(ArgumentError("master mode $q is overdamped; harmonic forcing needs an " *
                                 "oscillating master mode"))
    return forcing_matrix(M, ϕ, cfg.forcing_amplitude), Ω
end

# ============================================================================ spectrum

_symmetrise(A) = (A + sparse(transpose(A))) / 2

"""
    modal_spectrum(K, C, M, alpha, beta, nev; dense = size(K, 1) <= DENSE_LIMIT) -> Spectrum

Spectrum of the proportionally damped structure `M ü + C u̇ + K u = 0`, `C = alpha M + beta K`,
from the `nev` lowest undamped modes `(K − ω² M) ϕ = 0`.

It follows MORFE's `StructureModalDampingEigensolver` path: mode `p` occupies the adjacent
entries `2p−1` (positive imaginary part) and `2p` (its conjugate); the modes are real and
mass-normalised; the left blocks are analytic. Two additions:

* a dense solver for small models (Arpack cannot handle tiny matrices);
* a fixed sign: the entry of largest magnitude of each mode is positive, so results do not
  depend on the eigensolver's arbitrary sign.

Only the eigenproblem uses the symmetrised `K` and `M`; the model keeps the originals.
"""
function modal_spectrum(K::AbstractMatrix, C::AbstractMatrix, M::AbstractMatrix,
        alpha::Real, beta::Real, nev::Integer; dense::Bool = size(K, 1) <= DENSE_LIMIT)
    n = size(K, 1)
    1 <= nev <= (dense ? n : n - 1) ||
        throw(ArgumentError("nev = $nev is out of range for a model with $n DOFs"))
    Ks, Ms = _symmetrise(K), _symmetrise(M)
    if dense
        F = eigen(Symmetric(Matrix(Ks)), Symmetric(Matrix(Ms)))
        ω2 = F.values[1:nev]
        Φ = F.vectors[:, 1:nev]
    else
        vals, vecs = eigs(Ks, Ms; nev = nev, which = :LM, sigma = 0.0)
        any(x -> abs(imag(x)) > 1e-10 * abs(real(x)), vals) &&
            error("modal_spectrum: complex ω² from Arpack; K or M is not symmetric")
        p = sortperm(real.(vals))
        ω2 = real.(vals[p])
        Φ = real.(vecs[:, p])
    end
    all(>(0), ω2) ||
        error("modal_spectrum: non-positive ω² found ($(minimum(ω2))); K must be positive " *
              "definite on the free DOFs (check the boundary conditions)")

    λ = Vector{ComplexF64}(undef, 2nev)
    Y = Array{ComplexF64, 3}(undef, n, 2, 2nev)
    for i in 1:nev
        ϕ = Φ[:, i]                          # real vector, so `dot` conjugates nothing
        ϕ ./= sqrt(dot(ϕ, Ms * ϕ))
        ϕ[argmax(abs.(ϕ))] < 0 && (ϕ .*= -1)
        re = -0.5 * (alpha + beta * ω2[i])
        disc = re^2 - ω2[i]
        if disc < 0
            λ[2i - 1] = complex(re, sqrt(-disc))
            λ[2i] = complex(re, -sqrt(-disc))
        else                                 # overdamped: two real eigenvalues
            λ[2i - 1] = complex(re + sqrt(disc), 0.0)
            λ[2i] = complex(re - sqrt(disc), 0.0)
        end
        for (col, l) in ((2i - 1, λ[2i - 1]), (2i, λ[2i]))
            Y[:, 1, col] .= ϕ
            Y[:, 2, col] .= l .* ϕ
        end
    end
    # Left blocks of a self-adjoint pencil, as in MORFE's structural path: φ₂ = ϕ, φ₁ = (conj(λ) M + C) ϕ.
    left = left_eigenmode_orders_from_slice((M, C, M), view(Y, :, 1, :), λ; apply = identity)
    solver = StructureModalDampingEigensolver(Int(nev), Float64(alpha), Float64(beta))
    return Spectrum(solver, λ, Y, left)
end

"Whole-spectrum conjugate involution of a `modal_spectrum`: entries `2p−1` and `2p` are paired."
function _pair_permutation(λ::AbstractVector)
    σ = collect(1:length(λ))
    for p in 1:(length(λ) ÷ 2)
        if imag(λ[2p - 1]) != 0
            σ[2p - 1], σ[2p] = 2p, 2p - 1
        end
    end
    return σ
end

"""
    build_spectral(model, cfg; dense = nothing, spectrum = nothing)
        -> (spectral::SpectralData, spectrum::Spectrum)

Compute the modal spectrum and select the master modes `cfg.masters`. Master mode `p`
gives the reduced coordinates `(z, z̄)` in that order, for each master in turn. Pass
`spectrum` (from an earlier call on a model with the same `K`, `C`, `M`) to skip the
eigenproblem, e.g. for the forced version of a model.
"""
function build_spectral(model::NthOrderModel, cfg::RunConfig;
        dense::Union{Nothing, Bool} = nothing, spectrum = nothing)
    if spectrum === nothing
        K, C, M = model.linear_terms
        n = size(K, 1)
        use_dense = dense === nothing ? n <= DENSE_LIMIT : dense
        nmax = use_dense ? n : n - 1
        nev = min(cfg.n_modes > 0 ? cfg.n_modes : maximum(cfg.masters) + 8, nmax)
        maximum(cfg.masters) <= nev ||
            throw(ArgumentError("master mode $(maximum(cfg.masters)) is not among the $nev " *
                                "modes that can be computed"))
        sp = modal_spectrum(K, C, M, cfg.alpha, cfg.beta, nev; dense = use_dense)
    else
        sp = spectrum
        2maximum(cfg.masters) <= length(sp.eigenvalues) ||
            throw(ArgumentError("master mode $(maximum(cfg.masters)) is not in the spectrum given"))
    end
    master = reduce(vcat, [[2p - 1, 2p] for p in cfg.masters])
    sd = SpectralData(model, sp; master = master,
        conjugate_permutation = _pair_permutation(sp.eigenvalues))
    return sd, sp
end

"""
    setup_rom(engine, cfg = run_config(engine); terms = :engine, matrices = nothing,
              forced = cfg.forcing_enabled, dense = nothing)
        -> (; model, spectral, spectrum, forcing)

The model and its spectral data as the config describes them. With `forced` (by default
`[forcing] enabled`) the model carries the harmonic load of [`harmonic_forcing`](@ref) and
`forcing` is `(F, Ω)`; otherwise `forcing` is `nothing`. To build a second model with the
same load (e.g. exact closures for a reference), pass `forcing` to [`build_model`](@ref)
and `spectrum` to [`build_spectral`](@ref).
"""
function setup_rom(engine::Py, cfg::RunConfig = run_config(engine); terms::Symbol = :engine,
        matrices = nothing, forced::Bool = cfg.forcing_enabled,
        dense::Union{Nothing, Bool} = nothing)
    mats = matrices === nothing ? build_matrices(engine, cfg.alpha, cfg.beta) : matrices
    model = build_model(engine, cfg; terms = terms, matrices = mats)
    spectral, sp = build_spectral(model, cfg; dense = dense)
    forced || return (; model, spectral, spectrum = sp, forcing = nothing)
    forcing = harmonic_forcing(mats[3], sp, cfg)
    model = build_model(engine, cfg; terms = terms, matrices = mats, forcing = forcing)
    spectral, _ = build_spectral(model, cfg; spectrum = sp)
    return (; model, spectral, spectrum = sp, forcing)
end

"""
    resonance_config(cfg) -> ResonanceConfig

`cfg.resonance_tol` is the absolute detuning threshold; 0 selects MORFE's default.
"""
function resonance_config(cfg::RunConfig)
    cfg.style === :graph && return ResonanceConfig(style = :graph)
    tol = cfg.resonance_tol > 0 ? cfg.resonance_tol : nothing
    if cfg.style === :real_normal_form
        rom = 2 * length(cfg.masters)
        cmap = [isodd(i) ? i + 1 : i - 1 for i in 1:rom]
        return ResonanceConfig(style = :real_normal_form, tol = tol, conjugacy_map = cmap)
    end
    return ResonanceConfig(style = cfg.style, tol = tol)
end

# ================================================================================ solve

"""
    solve_rom(model, spectral, cfg; order = cfg.order, verbose = false) -> (W, R)

Run MORFE's `parametrise` with the resonance settings of the config. `order` is a total
degree or a `MultiindexSet`. For a forced model a total degree `q` means the monomials of
[`expansion_set`](@ref) with the forcing degree capped at `[rom] forcing_order` (no cap
when that is 0), i.e. the thesis' O(q, forcing_order).
"""
function solve_rom(model::NthOrderModel, spectral::SpectralData, cfg::RunConfig;
        order = cfg.order, verbose::Bool = false)
    mset = order isa Integer ?
           expansion_set(model, spectral, order; forcing_order = _forcing_cap(cfg, order)) :
           order
    options = ParametrisationOptions(show_progress = verbose, verbose = verbose)
    return parametrise(model, spectral, mset;
        resonance = resonance_config(cfg), options = options)
end

"Forcing-degree cap at total degree `q`: `[rom] forcing_order`, or `q` when that is 0."
_forcing_cap(forcing_order::Integer, q::Integer) = forcing_order >= 1 ? Int(forcing_order) : Int(q)
_forcing_cap(cfg::RunConfig, q::Integer) = _forcing_cap(cfg.forcing_order, q)

_stat(engine::Py, key::AbstractString) = pyconvert(Int, engine.stats()[key])

"""
    solve_rom_by_order(model, spectral, cfg, engine; order = cfg.order) -> (W, R, counts)

Solve at total degree 2, 3, …, `order` in turn and record what each order cost. The engine
keeps every tangent matrix, so order `q` only runs the jobs that orders `< q` did not need.
`counts` is a vector of `(order, new_jobs, jobs_total, calls_quad, calls_cubic)`; the call
counters are cumulative and include the repeated lower orders.
"""
function solve_rom_by_order(model::NthOrderModel, spectral::SpectralData, cfg::RunConfig,
        engine::Py; order::Integer = cfg.order, verbose::Bool = false)
    order >= 2 || throw(ArgumentError("order must be >= 2"))
    counts = NamedTuple{(:order, :new_jobs, :jobs_total, :calls_quad, :calls_cubic),
        NTuple{5, Int}}[]
    prev = _stat(engine, "jobs_total")
    local W, R
    for q in 2:order
        W, R = solve_rom(model, spectral, cfg; order = q, verbose = verbose)
        now = _stat(engine, "jobs_total")
        push!(counts,
            (order = q, new_jobs = now - prev, jobs_total = now,
                calls_quad = _stat(engine, "calls_quad"),
                calls_cubic = _stat(engine, "calls_cubic")))
        prev = now
    end
    return W, R, counts
end

"""
    expansion_set(model, spectral, q; forcing_order = q) -> Int or MultiindexSet

The monomials of total degree `1 … q`. For an autonomous model this is just `q`. For a
model with an external (forcing) system, monomials whose degree in the forcing variables
exceeds `min(q, forcing_order)` are left out. The reduced coordinates come first in each
exponent vector, the forcing variables last. (`solve_rom` and `run_with_prefetch` read a
`forcing_order` of 0 as "no cap".)
"""
function expansion_set(::NthOrderModel{ORD, ORDP1, N_NL, N_EXT}, ::SpectralData{ORD, ROM},
        q::Integer; forcing_order::Integer = q) where {ORD, ORDP1, N_NL, N_EXT, ROM}
    q >= 1 || throw(ArgumentError("the order must be >= 1"))
    N_EXT == 0 && return Int(q)
    cap = min(Int(q), Int(forcing_order))
    full = all_multiindices_up_to(ROM + N_EXT, Int(q); min_degree = 1)
    return filter(α -> sum(α[i] for i in (ROM + 1):(ROM + N_EXT)) <= cap, full)
end

const PrefetchBatch = NamedTuple{
    (:order, :new_jobs, :jobs_total, :calls_quad, :calls_cubic, :seconds_jobs, :seconds_pass),
    Tuple{Int, Int, Int, Int, Int, Float64, Float64}}

"""
    run_with_prefetch(model, spectral, cfg, engine; order = cfg.order,
                      max_parallel = cfg.max_parallel, forcing_order = cfg.forcing_order,
                      verbose = false) -> (W, R, batches)

Solve the ROM with the tangent jobs of each order run as one batch (PLAN.md P9), so a
runner can execute them in parallel:

1. for `q = 2 … order`: a *record pass* at order `q`. The engine queues the jobs it lacks
   and returns zeros for those calls; the queue is then run, `max_parallel` jobs at a time.
2. a normal solve at `order`, in which every job is already in the cache.

One pass per order is enough. At order `q` MORFE evaluates the nonlinear terms only on
coefficients of order below `q`. The earlier passes ran their jobs, so those coefficients
are exact and the requests recorded at order `q` are the right ones. The order-`q`
coefficients of a record pass are wrong (zeros were returned) and are thrown away.

Step 2 runs in lazy mode: if a vector failed to match a recorded one within tolerance, the
missing job simply runs there, so the result is never wrong. Step 2 is skipped when the last
record pass had no incomplete call, because that pass is then already the full solve.

`batches` has one entry per order with `new_jobs`, `jobs_total`, the cumulative call
counters, `seconds_jobs` (time in the job batch) and `seconds_pass` (time in MORFE). A last
entry with `order = 0` describes step 2; its `new_jobs` is 0 unless a job was missed.

With a disk cache (`[cache] dir`) an interrupted run can simply be started again: finished
jobs are found in the cache and are not repeated.
"""
function run_with_prefetch(model::NthOrderModel, spectral::SpectralData, cfg::RunConfig,
        engine::Py; order::Integer = cfg.order, max_parallel::Integer = cfg.max_parallel,
        forcing_order::Integer = cfg.forcing_order, verbose::Bool = false)
    order >= 2 || throw(ArgumentError("order must be >= 2"))
    mset(q) = expansion_set(model, spectral, q; forcing_order = _forcing_cap(forcing_order, q))
    batches = PrefetchBatch[]
    entry(q, new, tj, tp) = (order = q, new_jobs = new, jobs_total = _stat(engine, "jobs_total"),
        calls_quad = _stat(engine, "calls_quad"), calls_cubic = _stat(engine, "calls_cubic"),
        seconds_jobs = tj, seconds_pass = tp)
    local W, R
    complete = false
    try
        for q in 2:order
            engine.set_mode("record")
            incomplete = _stat(engine, "calls_incomplete")
            tp = @elapsed W, R = solve_rom(model, spectral, cfg; order = mset(q))
            complete = _stat(engine, "calls_incomplete") == incomplete
            before = _stat(engine, "jobs_total")
            tj = @elapsed engine.execute_pending(max_parallel)
            push!(batches, entry(q, _stat(engine, "jobs_total") - before, tj, tp))
            verbose && println("  prefetch O$q: ", batches[end].new_jobs, " jobs")
        end
    finally
        engine.set_mode("lazy")
    end
    if !complete
        before = _stat(engine, "jobs_total")
        tp = @elapsed W, R = solve_rom(model, spectral, cfg; order = mset(order), verbose = verbose)
        missed = _stat(engine, "jobs_total") - before
        missed == 0 ||
            @warn "run_with_prefetch: $missed job(s) were not prefetched and ran one by one"
        push!(batches, entry(0, missed, 0.0, tp))
    end
    return W, R, batches
end

"""
    rom_rel_diff(W1, R1, W2, R2) -> (dR, dW)

`max|ΔR| / max|R|` and `max|ΔW| / max|W|` between two ROMs with the same monomial set.
"""
function rom_rel_diff(W1, R1, W2, R2)
    R1.poly.multiindex_set.exponents == R2.poly.multiindex_set.exponents ||
        throw(ArgumentError("the two ROMs use different monomial sets"))
    r1, r2 = R1.poly.coefficients, R2.poly.coefficients
    w1, w2 = W1.poly.coefficients, W2.poly.coefficients
    dR = maximum(abs, r1 .- r2) / max(maximum(abs, r2), floatmin())
    dW = maximum(abs, w1 .- w2) / max(maximum(abs, w2), floatmin())
    return dR, dW
end

_rel(x, y) = (d = maximum(abs, x .- y); d == 0 ? 0.0 : d / maximum(abs, y))

"""
    rom_rel_diff_by_degree(W1, R1, W2, R2) -> Vector{(degree, dR, dW)}

Like [`rom_rel_diff`](@ref), but separately for the monomials of each total degree:
`max|ΔR_d| / max|R_d|` and `max|ΔW_d| / max|W_d|`, with ROM 2 as the reference. Use this
for real models: there the nonlinear coefficients can be many orders of magnitude smaller
than the linear ones, and a single global ratio would hide their errors.
"""
function rom_rel_diff_by_degree(W1, R1, W2, R2)
    er, ew = R1.poly.multiindex_set.exponents, W1.poly.multiindex_set.exponents
    (er == R2.poly.multiindex_set.exponents && ew == W2.poly.multiindex_set.exponents) ||
        throw(ArgumentError("the two ROMs use different monomial sets"))
    r1, r2 = R1.poly.coefficients, R2.poly.coefficients
    w1, w2 = W1.poly.coefficients, W2.poly.coefficients
    dr, dw = [sum(e) for e in er], [sum(e) for e in ew]
    out = NamedTuple{(:degree, :dR, :dW), Tuple{Int, Float64, Float64}}[]
    for d in sort(unique(vcat(dr, dw)))
        ir, iw = findall(==(d), dr), findall(==(d), dw)
        push!(out,
            (degree = d,
                dR = isempty(ir) ? 0.0 : _rel(view(r1, :, ir), view(r2, :, ir)),
                dW = isempty(iw) ? 0.0 : _rel(view(w1, :, :, iw), view(w2, :, :, iw))))
    end
    return out
end

"""
    backbone_amplitudes(engine, spectrum, mode = 1; factor = 1.0, length = 101)

Reduced amplitudes `ρ` from 0 up to the value at which the linear mode reaches a physical
displacement of `factor * d_ref` at its largest DOF (`u ≈ 2 ρ ϕ cos θ`).
"""
function backbone_amplitudes(engine::Py, spectrum, mode::Integer = 1; factor::Real = 1.0,
        length::Integer = 101)
    d_ref = pyconvert(Float64, engine.step.d_ref)
    ϕmax = maximum(abs, view(spectrum.eigenmodes, :, 1, 2mode - 1))
    return range(0, factor * d_ref / (2ϕmax); length = length)
end

"""
    backbone_rel_diff(R1, R2; amplitudes = range(0, 1; length = 101)) -> Float64

Difference between the backbone curves (frequency against reduced amplitude) of two ROMs
with one master pair in complex normal form (MORFE's `normal_form_branch`), relative to the
largest frequency **shift** of ROM 2 over the amplitude range: `max|Δω| / max|ω₂(ρ) − ω₂(0)|`.
Pass a physically meaningful amplitude range, e.g. from [`backbone_amplitudes`](@ref).
"""
function backbone_rel_diff(R1, R2; amplitudes = range(0, 1; length = 101))
    b1 = normal_form_branch(R1; amplitudes = amplitudes)
    b2 = normal_form_branch(R2; amplitudes = amplitudes)
    diff = maximum(abs, b1.frequency .- b2.frequency)
    shift = maximum(abs, b2.frequency .- b2.frequency[1])
    diff == 0 && return 0.0
    return diff / (shift > 0 ? shift : maximum(abs, b2.frequency))
end

# ============================================================== physical amplitude

"""
    monitor_index(engine, cfg, ϕ) -> Int

1-based index, among the free DOFs, of the DOF whose response is reported: `[output]
monitor_node` and `monitor_dof` of the config when both are set and the backend has a mesh,
otherwise the DOF where the mode `ϕ` is largest.
"""
function monitor_index(engine::Py, cfg::RunConfig, ϕ::AbstractVector)
    if cfg.monitor_node !== nothing && cfg.monitor_dof !== nothing &&
       pyhasattr(engine.backend, "dofmap")
        return pyconvert(Int, engine.backend.dofmap.free_index(cfg.monitor_node, cfg.monitor_dof)) + 1
    end
    return argmax(abs.(ϕ))
end

"""
    physical_amplitude(W, dof, amplitudes; samples = 256) -> Vector{Float64}

Largest displacement of the free DOF `dof` over one period of the orbit `z = ρ e^{iθ}`,
`z̄ = ρ e^{−iθ}`, for each reduced amplitude `ρ` in `amplitudes` (one master pair):

    max over θ of | Σ_α W[dof, α] ρ^(α₁+α₂) e^{i(α₁−α₂)θ} |

At small `ρ` this is `2 ρ |ϕ[dof]|`. Use it to plot a backbone against a displacement, as
the thesis does, instead of against `ρ`.
"""
function physical_amplitude(W, dof::Integer, amplitudes; samples::Integer = 256)
    exps = W.poly.multiindex_set.exponents
    length(first(exps)) == 2 ||
        throw(ArgumentError("physical_amplitude needs a ROM with one master pair"))
    1 <= dof <= size(W.poly.coefficients, 1) || throw(ArgumentError("dof $dof is out of range"))
    c = view(W.poly.coefficients, dof, 1, :)
    θ = range(0, 2π; length = samples + 1)[1:samples]
    out = Float64[]
    u = zeros(ComplexF64, samples)
    for ρ in amplitudes
        fill!(u, 0)
        for (k, e) in enumerate(exps)
            a, b = Int(e[1]), Int(e[2])
            coef = c[k] * Float64(ρ)^(a + b)
            iszero(coef) && continue
            @. u += coef * cis((a - b) * θ)
        end
        push!(out, maximum(x -> abs(real(x)), u))
    end
    return out
end

# ================================================================= forced response

"""
    _forced_row(R, level) -> (n, k, c)

The first reduced equation of a forced ROM with one master pair and one forcing pair
(variables `z, z̄, r₁, r₂`): for each non-zero monomial `z^a z̄^b r₁^p r₂^q`, the powers
`n = a + b` and `k = p − q`, and the coefficient times `level^(p + q)`.
"""
function _forced_row(R, level::Real)
    exps = R.poly.multiindex_set.exponents
    length(first(exps)) == 4 ||
        throw(ArgumentError("this needs a ROM with one master pair and one forcing pair " *
                            "(4 reduced variables), got $(length(first(exps)))"))
    row = view(R.poly.coefficients, 1, :)
    n, k, c = Int[], Int[], ComplexF64[]
    for (m, e) in enumerate(exps)
        iszero(row[m]) && continue
        a, b, p, q = (Int(x) for x in e)
        (a - b) + (p - q) == 1 ||
            throw(ArgumentError("monomial $(Tuple(e)) of the first reduced equation has " *
                                "(a − b) + (p − q) ≠ 1: the ROM is not in complex normal form " *
                                "with the forcing in 1:1 resonance"))
        push!(n, a + b)
        push!(k, p - q)
        push!(c, row[m] * Float64(level)^(p + q))
    end
    return n, k, c
end

"""
    forced_response(R; amplitudes, level = 1.0, n_phase = 720)
        -> (; amplitude, phase, frequency, stable, sheet)

Steady-state responses of a forced ROM in complex normal form with one master pair and one
harmonic forcing pair (reduced variables `z, z̄, r₁, r₂`) as the forcing frequency `ω̂`
varies around the one the ROM was built at: the frequency response curve. The thesis
obtains the same curve with a harmonic balance of the reduced dynamics.

The forcing variables are `r₁ = level · e^{iω̂t}` and `r₂ = level · e^{−iω̂t}`; `level = 1`
is the load of the config, and another level needs no new solve because the reduced
dynamics are polynomial in `r`. Every monomial `z^a z̄^b r₁^p r₂^q` of the first reduced
equation has `(a − b) + (p − q) = 1`, so with `z = ρ e^{i(ω̂t + ψ)}`

    ρ̇ + iρ(ω̂ + ψ̇) = G(ρ, ψ) = Σ R₁[a, b, p, q] level^(p+q) ρ^(a+b) e^{−i(p−q)ψ}.

A steady state has `ρ̇ = ψ̇ = 0`, that is `Re G(ρ, ψ) = 0` and `ω̂ = Im G(ρ, ψ) / ρ`. For each
`ρ > 0` in `amplitudes` the phases `ψ ∈ [−π, π)` that solve the first equation are found
(sign changes on `n_phase` samples, then bisection), and the second gives `ω̂`. The curve is
traced by amplitude, so folds need no continuation.

One entry per point: `amplitude` (ρ), `phase` (ψ, how far the response leads the forcing),
`frequency` (ω̂), `stable` (from the Jacobian of `(ρ̇, ψ̇)`) and `sheet` (rank of the point by
frequency among the points with the same ρ; 1 is the lowest). For the displacement of a
DOF along the curve use [`orbit_amplitude`](@ref).
"""
function forced_response(R; amplitudes, level::Real = 1.0, n_phase::Integer = 720)
    n_phase >= 8 || throw(ArgumentError("n_phase must be at least 8"))
    n, k, c = _forced_row(R, level)
    terms = eachindex(c)
    G(ρ, ψ) = sum(c[i] * ρ^n[i] * cis(-k[i] * ψ) for i in terms; init = zero(ComplexF64))
    Gρ(ρ, ψ) = sum((n[i] == 0 ? zero(ComplexF64) : c[i] * n[i] * ρ^(n[i] - 1) * cis(-k[i] * ψ))
    for i in terms; init = zero(ComplexF64))
    Gψ(ρ, ψ) = sum(-im * k[i] * c[i] * ρ^n[i] * cis(-k[i] * ψ) for i in terms;
        init = zero(ComplexF64))
    out = (amplitude = Float64[], phase = Float64[], frequency = Float64[], stable = Bool[],
        sheet = Int[])
    ψs = range(-π, π; length = n_phase + 1)
    roots = Float64[]
    for ρ in Float64.(amplitudes)
        ρ > 0 || continue
        f(ψ) = real(G(ρ, ψ))
        empty!(roots)
        fa = f(ψs[1])
        for j in 1:n_phase                 # f(π) = f(−π): the last end point is not a new root
            a, b = ψs[j], ψs[j + 1]
            fb = f(b)
            if fa == 0
                push!(roots, a)
            elseif fa * fb < 0
                lo, hi, flo = a, b, fa
                for _ in 1:60
                    mid = (lo + hi) / 2
                    fm = f(mid)
                    fm == 0 && ((lo, hi) = (mid, mid); break)
                    if (fm < 0) == (flo < 0)
                        lo, flo = mid, fm
                    else
                        hi = mid
                    end
                end
                push!(roots, (lo + hi) / 2)
            end
            fa = fb
        end
        pts = sort!([(ψ, imag(G(ρ, ψ)) / ρ) for ψ in roots]; by = last)
        for (s, (ψ, ω)) in enumerate(pts)
            g, gρ, gψ = G(ρ, ψ), Gρ(ρ, ψ), Gψ(ρ, ψ)
            j11, j12 = real(gρ), real(gψ)                       # ρ̇ = Re G
            j21, j22 = imag(gρ) / ρ - imag(g) / ρ^2, imag(gψ) / ρ   # ψ̇ = Im G / ρ − ω̂
            push!(out.amplitude, ρ)
            push!(out.phase, ψ)
            push!(out.frequency, ω)
            push!(out.stable, j11 + j22 < 0 && j11 * j22 - j12 * j21 > 0)
            push!(out.sheet, s)
        end
    end
    return out
end

"Index of the monomial with exponents `e` in the ROM `X` (`W` or `R`), or `nothing`."
_monomial(X, e) = findfirst(x -> collect(Int, x) == collect(Int, e), X.poly.multiindex_set.exponents)

"""
    forced_response_amplitudes(R; factor = 1.5, length = 301, level = 1.0)

Reduced amplitudes `ρ` from 0 to `factor` times the linear resonance peak
`|R₁[r₁]| level / |Re λ|`, a range that holds the whole frequency response curve of a
weakly nonlinear, damped ROM (beyond its peak [`forced_response`](@ref) finds no points).
"""
function forced_response_amplitudes(R; factor::Real = 1.5, length::Integer = 301,
        level::Real = 1.0)
    _forced_row(R, level)                                    # checks the structure
    iλ, iF = _monomial(R, (1, 0, 0, 0)), _monomial(R, (0, 0, 1, 0))
    (iλ === nothing || iF === nothing) &&
        throw(ArgumentError("the ROM has no linear master or forcing monomial"))
    re = abs(real(R.poly.coefficients[1, iλ]))
    re > 0 || throw(ArgumentError("the master mode is undamped, so the forced response is " *
                                  "unbounded; give the amplitudes explicitly"))
    peak = abs(R.poly.coefficients[1, iF]) * Float64(level) / re
    return range(0, factor * peak; length = length)
end

"""
    forced_response_rel_diff(R1, R2; amplitudes, level = 1.0) -> (; diff, unmatched)

Difference between the frequency response curves of two forced ROMs
([`forced_response`](@ref) on the same amplitudes), relative to the largest detuning
`|ω̂ − Ω₀|` on the curve of ROM 2 (`Ω₀`: the forcing frequency the ROMs were built at).
Points are matched by amplitude and sheet; `unmatched` counts the amplitudes at which the
two curves have a different number of points (near the tip of a peak), which are left out.
"""
function forced_response_rel_diff(R1, R2; amplitudes, level::Real = 1.0)
    f1 = forced_response(R1; amplitudes = amplitudes, level = level)
    f2 = forced_response(R2; amplitudes = amplitudes, level = level)
    i3 = _monomial(R2, (0, 0, 1, 0))
    Ω0 = imag(R2.poly.coefficients[3, i3])
    diff, unmatched = 0.0, 0
    for ρ in unique(f2.amplitude) ∪ unique(f1.amplitude)
        a, b = findall(==(ρ), f1.amplitude), findall(==(ρ), f2.amplitude)
        if length(a) != length(b)
            unmatched += 1
            continue
        end
        diff = max(diff, maximum(abs.(f1.frequency[a] .- f2.frequency[b]); init = 0.0))
    end
    detuning = maximum(x -> abs(x - Ω0), f2.frequency; init = 0.0)
    return (diff = detuning > 0 ? diff / detuning : diff, unmatched = unmatched)
end

"""
    orbit_amplitude(W, dof, amplitude, phase; level = 1.0, samples = 256) -> Vector{Float64}

Largest displacement of the free DOF `dof` over one period of each forced orbit of
[`forced_response`](@ref): `z = ρ e^{iθ}`, `r₁ = level · e^{i(θ − ψ)}`, and `z̄`, `r₂` their
conjugates, for each pair `(ρ, ψ)` of `amplitude` and `phase`.
"""
function orbit_amplitude(W, dof::Integer, amplitude, phase; level::Real = 1.0,
        samples::Integer = 256)
    exps = W.poly.multiindex_set.exponents
    length(first(exps)) == 4 ||
        throw(ArgumentError("orbit_amplitude needs a ROM with one master pair and one forcing pair"))
    length(amplitude) == length(phase) ||
        throw(DimensionMismatch("amplitude and phase have different lengths"))
    1 <= dof <= size(W.poly.coefficients, 1) || throw(ArgumentError("dof $dof is out of range"))
    c = view(W.poly.coefficients, dof, 1, :)
    θ = range(0, 2π; length = samples + 1)[1:samples]
    u = zeros(ComplexF64, samples)
    out = Float64[]
    for (ρ, ψ) in zip(amplitude, phase)
        fill!(u, 0)
        for (m, e) in enumerate(exps)
            a, b, p, q = (Int(x) for x in e)
            coef = c[m] * Float64(ρ)^(a + b) * Float64(level)^(p + q) * cis(-(p - q) * ψ)
            iszero(coef) && continue
            h = (a - b) + (p - q)
            @. u += coef * cis(h * θ)
        end
        push!(out, maximum(x -> abs(real(x)), u))
    end
    return out
end

# ======================================================================== d_ref sweep

"""
    cubic_coefficient(R; pair = 1) -> ComplexF64

Coefficient of `z² z̄` in the reduced equation of `z`, for the master pair number `pair`
(reduced coordinates `2pair − 1` and `2pair`). In complex normal form with one master pair
its imaginary part is the slope of the backbone: `ω(ρ) ≈ Im λ + Im(coefficient) ρ²`.
The ROM must have order 3 or more.
"""
function cubic_coefficient(R; pair::Integer = 1)
    exps = R.poly.multiindex_set.exponents
    nvar = length(first(exps))
    1 <= pair <= nvar ÷ 2 || throw(ArgumentError("there is no master pair number $pair"))
    target = zeros(Int, nvar)
    target[2pair - 1], target[2pair] = 2, 1
    idx = findfirst(e -> collect(Int, e) == target, exps)
    idx === nothing &&
        throw(ArgumentError("the ROM has no monomial z² z̄ for master pair $pair (order below 3?)"))
    return ComplexF64(R.poly.coefficients[2pair - 1, idx])
end

const SweepRow = NamedTuple{
    (:d_ref, :status, :reason, :jobs, :coefficient, :change, :error, :dR),
    Tuple{Float64, Symbol, String, Int, ComplexF64, Float64, Float64, Float64}}

"""
    rom_d_ref_sweep(engine, d_refs; order = 3, reference = :auto, cfg = run_config(engine))
        -> Vector{SweepRow}

Solve the ROM at `order` once per `d_ref` and report how the result depends on it (PLAN.md
P10.1). Each solve uses `engine.with_d_ref(d_ref)`, so all of them share the backend and the
matrix store. At order 3 with one master pair a `d_ref` costs two tangent jobs, the same two
as in Python's `nistep.diagnostics.d_ref_sweep`.

`reference` is the ROM to compare with: `:auto` (the backend's exact closures if it has
them, otherwise none), `nothing`, or a tuple `(W, R)` solved elsewhere with the same
settings, e.g. from the PySVK twin of an Abaqus model.

Each row has
* `status`: `:ok` or `:failed` (a failed job; `reason` holds the first line of the error);
* `jobs`: tangent jobs run for this `d_ref`;
* `coefficient`: [`cubic_coefficient`](@ref) of the first master pair;
* `change`: `|coefficient − previous| / |previous|` against the previous row that ran;
* `error`: the same against the reference;
* `dR`: the largest per-degree difference of the reduced dynamics against the reference
  ([`rom_rel_diff_by_degree`](@ref), degrees 2 and up).
`change`, `error` and `dR` are `NaN` when there is nothing to compare with.
"""
function rom_d_ref_sweep(engine::Py, d_refs; order::Integer = 3, reference = :auto,
        cfg::RunConfig = run_config(engine))
    order >= 3 || throw(ArgumentError("order must be >= 3"))
    mats = build_matrices(engine, cfg.alpha, cfg.beta)
    ref = if reference === :auto
        be = engine.backend
        if pyis(be.exact_quad, pybuiltins.None)
            nothing
        else
            m = build_model(engine, cfg; terms = :backend_exact, matrices = mats)
            s, _ = build_spectral(m, cfg)
            solve_rom(m, s, cfg; order = order)
        end
    else
        reference
    end
    ref === nothing || ref isa Tuple ||
        throw(ArgumentError("reference must be :auto, nothing or a tuple (W, R)"))
    rows = SweepRow[]
    previous = nothing
    for d in d_refs
        eng = engine.with_d_ref(Float64(d))
        row = try
            model = build_model(eng, cfg; terms = :engine, matrices = mats)
            spectral, _ = build_spectral(model, cfg)
            W, R = solve_rom(model, spectral, cfg; order = order)
            coef = cubic_coefficient(R)
            change = previous === nothing ? NaN : abs(coef - previous) / abs(previous)
            err, dR = NaN, NaN
            if ref !== nothing
                Wr, Rr = ref
                cr = cubic_coefficient(Rr)
                err = abs(coef - cr) / abs(cr)
                dR = maximum(x -> x.dR, filter(x -> x.degree >= 2,
                    rom_rel_diff_by_degree(W, R, Wr, Rr)))
            end
            previous = coef
            (d_ref = Float64(d), status = :ok, reason = "", jobs = _stat(eng, "jobs_total"),
                coefficient = coef, change = change, error = err, dR = dR)
        catch exc
            exc isa PyException || rethrow()
            reason = first(split(sprint(showerror, exc), '\n'))
            (d_ref = Float64(d), status = :failed, reason = String(first(reason, 200)),
                jobs = _stat(eng, "jobs_total"), coefficient = complex(NaN, NaN), change = NaN,
                error = NaN, dR = NaN)
        end
        push!(rows, row)
    end
    return rows
end

# =============================================================================== export

_exponent_matrix(exps) = permutedims(reduce(hcat, [collect(Int, e) for e in exps]))

"""
    real_rom(W, R; dofs = Int[]) -> NamedTuple

The ROM in real variables (MORFE's `realify`), for time integration or harmonic balance in
any tool. Master pair `p` has the real coordinates `z_p = x_p + i y_p`. A forcing pair, if
there is one, has `r₁ = c + i s` (and `r₂` its conjugate): with the load of the config,
`c = cos ω̂t` and `s = sin ω̂t`; with `level` times that load, `c = level cos ω̂t`,
`s = level sin ω̂t`.

Fields:
* `variables`: `["x1", "y1", "x2", "y2", …]`, then `["c", "s"]` for a forced ROM;
* `exponents`: `L × length(variables)` integer matrix, one row per monomial;
* `f`: `2m × L` real matrix, the reduced dynamics `ẋ_p = Σ_l f[2p−1, l] mono_l`,
  `ẏ_p = Σ_l f[2p, l] mono_l`, where `mono_l = Π_j variables[j]^exponents[l, j]`;
* `dofs`, `displacement`, `velocity`: the displacement and velocity of the free DOFs `dofs`
  (1-based), one row per DOF, on the same monomials;
* `omega0`: the forcing frequency the ROM was built at (`NaN` without forcing).

`W` and `R` must have the conjugate symmetry of a real model (MORFE's default), otherwise
the realified coefficients are not real and an `ArgumentError` is thrown.
"""
_rom_size(::ReducedDynamics{ROM}) where {ROM} = ROM

function real_rom(W, R::ReducedDynamics; dofs = Int[])
    exps = R.poly.multiindex_set.exponents
    nvar = length(first(exps))
    nrow = _rom_size(R)            # master coordinates; R also has rows for the forcing
    (iseven(nrow) && (nvar == nrow || nvar == nrow + 2)) ||
        throw(ArgumentError("expected 2m master coordinates and 0 or 2 forcing variables"))
    m = nrow ÷ 2
    forced = nvar == nrow + 2
    npair = nvar ÷ 2
    conj_map = [isodd(i) ? i + 1 : i - 1 for i in 1:nvar]
    # realify orders the real variables as (x of every pair, y of every pair)
    order = reduce(vcat, [[k, npair + k] for k in 1:npair])
    names = reduce(vcat, [["x$p", "y$p"] for p in 1:m])
    forced && append!(names, ["c", "s"])

    rows = Dict{Vector{Int}, Vector{ComplexF64}}[]
    function add!(coeffs)
        poly = realify(DensePolynomial(Vector{ComplexF64}(coeffs), R.poly.multiindex_set), conj_map)
        d = Dict{Vector{Int}, Vector{ComplexF64}}()
        for (e, c) in zip(poly.multiindex_set.exponents, poly.coefficients)
            iszero(c) && continue
            key = [Int(e[j]) for j in order]
            d[key] = push!(get(d, key, ComplexF64[]), c)
        end
        push!(rows, d)
    end
    for p in 1:m
        add!(view(R.poly.coefficients, 2p - 1, :))
    end
    1 <= minimum(dofs; init = 1) && maximum(dofs; init = 1) <= size(W.poly.coefficients, 1) ||
        throw(ArgumentError("dofs out of range"))
    W.poly.multiindex_set.exponents == exps ||
        throw(ArgumentError("W and R use different monomial sets"))
    nlev = size(W.poly.coefficients, 2)
    for i in dofs, lev in 1:min(2, nlev)
        add!(view(W.poly.coefficients, i, lev, :))
    end
    keys_all = sort!(collect(union((keys(d) for d in rows)...)); by = e -> (sum(e), -e))
    index = Dict(k => l for (l, k) in enumerate(keys_all))
    L = length(keys_all)
    vals = zeros(ComplexF64, length(rows), L)
    for (r, d) in enumerate(rows), (k, cs) in d
        vals[r, index[k]] = sum(cs)
    end

    f = zeros(2m, L)
    for p in 1:m
        f[2p - 1, :] .= real.(vals[p, :])
        f[2p, :] .= imag.(vals[p, :])
    end
    obs = vals[(m + 1):end, :]
    scale = maximum(abs, obs; init = 0.0)
    maximum(x -> abs(imag(x)), obs; init = 0.0) <= 1e-8 * scale ||
        throw(ArgumentError("the realified displacements are not real: W lacks the conjugate symmetry"))
    obs = real.(obs)
    nd = length(dofs)
    step = min(2, nlev)
    displacement = nd == 0 ? zeros(0, L) : obs[1:step:end, :]
    velocity = nd == 0 || nlev < 2 ? zeros(0, L) : obs[2:step:end, :]
    omega0 = NaN
    if forced
        i3 = _monomial(R, [k == nrow + 1 ? 1 : 0 for k in 1:nvar])
        i3 === nothing || (omega0 = imag(R.poly.coefficients[nrow + 1, i3]))
    end
    return (variables = names, exponents = isempty(keys_all) ? zeros(Int, 0, nvar) :
                                            permutedims(reduce(hcat, keys_all)),
        f = f, dofs = collect(Int, dofs), displacement = displacement, velocity = velocity,
        omega0 = omega0)
end

"""
    export_real_rom(dir, W, R; dofs = Int[], labels = String[]) -> path

Write [`real_rom`](@ref) to `dir/real_rom.npz` (arrays `exponents`, `f`, `dofs`,
`displacement`, `velocity`, `omega0`, and the variable names joined by commas in
`variables`) and to two CSV files: `real_rom_f.csv` (exponents, then one column per
equation) and `real_rom_obs.csv` (exponents, then displacement and velocity of each DOF).
`labels` names the DOFs in the CSV headers (default `dof<i>`). Python: `nistep.realrom`.
"""
function export_real_rom(dir::AbstractString, W, R; dofs = Int[], labels = String[])
    mkpath(dir)
    rr = real_rom(W, R; dofs = dofs)
    isempty(labels) && (labels = ["dof$i" for i in rr.dofs])
    length(labels) == length(rr.dofs) || throw(ArgumentError("one label per DOF"))
    path = joinpath(dir, "real_rom.npz")
    NPZ.npzwrite(path,
        Dict{String, Any}("exponents" => rr.exponents, "f" => rr.f, "dofs" => rr.dofs,
            "displacement" => rr.displacement, "velocity" => rr.velocity,
            "omega0" => rr.omega0, "variables" => Vector{UInt8}(join(rr.variables, ","))))
    m = size(rr.f, 1) ÷ 2
    eqs = reduce(vcat, [["d$(rr.variables[2p - 1])/dt", "d$(rr.variables[2p])/dt"] for p in 1:m])
    _write_table(joinpath(dir, "real_rom_f.csv"), rr.variables, rr.exponents, eqs, rr.f)
    obsnames = vcat(["u_" * l for l in labels], isempty(rr.velocity) ? String[] : ["v_" * l for l in labels])
    _write_table(joinpath(dir, "real_rom_obs.csv"), rr.variables, rr.exponents, obsnames,
        vcat(rr.displacement, rr.velocity))
    return path
end

function _write_table(path, vars, exps, names, vals)
    open(path, "w") do io
        println(io, join(vcat(vars, names), ","))
        for l in axes(exps, 1)
            print(io, join(exps[l, :], ","))
            for r in axes(vals, 1)
                print(io, ",", repr(vals[r, l]))
            end
            println(io)
        end
    end
end

"""
    export_rom(dir, W, R, engine, cfg; spectrum = nothing, extra = Pair{String, Any}[])

Write the results of a run:

* `W.npz`: `W_re`, `W_im` (`N × ORD × L`) and `exponents` (`L × NVAR`);
* `R.npz`: `R_re`, `R_im` (`NVAR × L`), `exponents`, and the eigenvalues when given;
* `data/R_coefficients.csv`, `data/W.jls`, `data/R.jls`, `summary.txt` (MORFE's `save_rom`);
* `engine_stats.json`: counters of the Python engine, plus `jobs_per_order` when the
  `counts` of [`solve_rom_by_order`](@ref) or the `batches` of [`run_with_prefetch`](@ref)
  are passed.
"""
function export_rom(dir::AbstractString, W, R, engine::Py, cfg::RunConfig;
        spectrum = nothing, jobs_per_order = nothing, extra = Pair{String, Any}[])
    mkpath(dir)
    Wc, Rc = W.poly.coefficients, R.poly.coefficients
    NPZ.npzwrite(joinpath(dir, "W.npz"),
        Dict("W_re" => real.(Wc), "W_im" => imag.(Wc),
            "exponents" => _exponent_matrix(W.poly.multiindex_set.exponents)))
    rdata = Dict{String, Any}("R_re" => real.(Rc), "R_im" => imag.(Rc),
        "exponents" => _exponent_matrix(R.poly.multiindex_set.exponents))
    if spectrum !== nothing
        rdata["eigenvalues_re"] = real.(spectrum.eigenvalues)
        rdata["eigenvalues_im"] = imag.(spectrum.eigenvalues)
    end
    NPZ.npzwrite(joinpath(dir, "R.npz"), rdata)

    stats = engine.stats()
    if jobs_per_order !== nothing
        stats["jobs_per_order"] = pylist([pydict(Dict(String(k) => v for (k, v) in pairs(c)))
                                          for c in jobs_per_order])
    end
    json = pyimport("json")
    write(joinpath(dir, "engine_stats.json"),
        pyconvert(String, json.dumps(stats; indent = 2, sort_keys = true)))

    metadata = Pair{String, Any}[
        "name" => cfg.name,
        "config" => cfg.path,
        "n_dof" => size(Wc, 1),
        "masters" => cfg.masters,
        "order" => maximum(sum, R.poly.multiindex_set.exponents),
        "style" => cfg.style,
        "alpha" => cfg.alpha,
        "beta" => cfg.beta,
        "jobs_total" => pyconvert(Int, stats["jobs_total"]),
        "jobs_k0" => pyconvert(Int, stats["jobs_k0"]),
        "time_backend_s" => pyconvert(Float64, stats["time_backend_s"])
    ]
    append!(metadata, extra)
    save_rom(dir, W, R; metadata = metadata)
    return dir
end

end # module MORFEStEP
