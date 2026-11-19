# P4.5: the PythonCall bridge and the EED closures against exact closures.
#
# Fast way to run (reuses the project's Python environment):
#   julia --project=julia/MORFEStEP julia/MORFEStEP/test/runtests.jl
using Test
using LinearAlgebra
using SparseArrays
using Random
using MORFE
using MORFEStEP
using PythonCall

const CONFIGS = joinpath(MORFEStEP.REPO_ROOT, "configs")
config(name) = joinpath(CONFIGS, name * ".toml")
relerr(x, y) = maximum(abs, x .- y) / max(maximum(abs, y), floatmin())
cvec(rng, n) = randn(rng, n) .+ im .* randn(rng, n)

@testset "MORFEStEP" begin
    @testset "bridge" begin
        engine = load_engine(config("analytic_random"))
        cfg = run_config(engine)
        @test cfg.name == "analytic_random"
        @test cfg.masters == [1] && cfg.order == 5 && cfg.style === :complex_normal_form
        @test cfg.alpha == 0.001 && cfg.beta == 0.0
        @test isabspath(cfg.output_dir) && cfg.monitor_node === nothing

        K = to_julia_sparse(engine.K())
        M = to_julia_sparse(engine.M())
        @test K isa SparseMatrixCSC{Float64, Int} && size(K) == (50, 50)
        @test Matrix(K) == pyconvert(Matrix{Float64}, engine.K().toarray())
        @test Matrix(M) == pyconvert(Matrix{Float64}, engine.M().toarray())
        @test issorted(K.rowval[K.colptr[1]:(K.colptr[2] - 1)])

        Kb, C, Mb = build_matrices(engine, 0.3, 0.02)
        @test typeof(C) == typeof(Kb) == typeof(Mb)
        @test Matrix(C) ≈ 0.3 .* Matrix(M) .+ 0.02 .* Matrix(K) rtol = 1e-14

        stats = engine_stats(engine)
        @test stats["n_dof"] == 50 && stats["jobs_k0"] == 1
    end

    @testset "closures vs exact: $name" for name in
                                            ("analytic_two_dof", "analytic_chain", "analytic_random")
        engine = load_engine(config(name))
        cfg = run_config(engine)
        mats = build_matrices(engine, cfg.alpha, cfg.beta)
        m_eng = build_model(engine, cfg; terms = :engine, matrices = mats)
        m_ex = build_model(engine, cfg; terms = :exact, matrices = mats)
        m_be = build_model(engine, cfg; terms = :backend_exact, matrices = mats)
        n = m_eng.n_fom
        rng = MersenneTwister(2026)
        for trial in 1:3
            a, b, c = cvec(rng, n), cvec(rng, n), cvec(rng, n)
            # distinct complex arguments, called the way MORFE calls them
            for (k, args) in ((1, (a, b)), (2, (a, b, c)), (2, (a, a, c)), (1, (b, b)))
                r_eng = zeros(ComplexF64, n)
                r_ex = zeros(ComplexF64, n)
                r_be = zeros(ComplexF64, n)
                m_eng.nonlinear_terms[k].f!(r_eng, args...)
                m_ex.nonlinear_terms[k].f!(r_ex, args...)
                m_be.nonlinear_terms[k].f!(r_be, args...)
                @test relerr(r_eng, r_ex) <= 1e-9
                @test relerr(r_be, r_ex) <= 1e-12
            end
            # views of a 3-D array (what MORFE passes) and accumulation into a non-zero res
            Wfake = reshape(vcat(a, b), n, 1, 2)
            r0 = cvec(rng, n)
            r_eng = copy(r0)
            r_ex = copy(r0)
            m_eng.nonlinear_terms[1].f!(r_eng, view(Wfake, :, 1, 1), view(Wfake, :, 1, 2))
            m_ex.nonlinear_terms[1].f!(r_ex, a, b)
            @test relerr(r_eng, r_ex) <= 1e-9
            @test relerr(r_eng .- r0, r_ex .- r0) <= 1e-9
            # MORFE's own evaluation at a state (x, ẋ)
            x, v = cvec(rng, n), cvec(rng, n)
            for order in (2, 3)
                r_eng = zeros(ComplexF64, n)
                r_ex = zeros(ComplexF64, n)
                evaluate_nonlinear_terms!(r_eng, m_eng, order, (x, v))
                evaluate_nonlinear_terms!(r_ex, m_ex, order, (x, v))
                @test relerr(r_eng, r_ex) <= 1e-9
            end
        end
        # sign: the closure adds MINUS the internal force returned by Python
        a, b = cvec(rng, n), cvec(rng, n)
        r = zeros(ComplexF64, n)
        m_eng.nonlinear_terms[1].f!(r, a, b)
        g = pyconvert(Vector{ComplexF64}, engine.backend.exact_quad(a, b))
        @test relerr(r, -g) <= 1e-9
        # real state and real result vector
        xr = randn(rng, n)
        rr = zeros(n)
        m_eng.nonlinear_terms[1].f!(rr, xr, xr)
        @test relerr(rr, -real.(pyconvert(Vector{ComplexF64}, engine.backend.exact_quad(xr, xr)))) <= 1e-9
        @test pyconvert(Float64, engine.stats()["debug_max_rel_err_quad"]) <= 1e-9
        @test pyconvert(Float64, engine.stats()["debug_max_rel_err_cubic"]) <= 1e-9
    end

    @testset "two-DoF example values" begin
        engine = load_engine(config("analytic_two_dof"))
        model = build_model(engine)
        V1 = ComplexF64[1, 1]
        r = zeros(ComplexF64, 2)
        model.nonlinear_terms[1].f!(r, V1, V1)
        @test r ≈ ComplexF64[-20, -20] atol = 1e-11
        r .= 0
        model.nonlinear_terms[2].f!(r, V1, V1, V1)
        @test r ≈ ComplexF64[-10, -10] atol = 1e-11
    end

    @testset "force term" begin
        F = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        t = force_term(F)
        @test t.multiindex == (0, 0) && t.multiplicity_external == 1
        r = zeros(ComplexF64, 3)
        t.f!(r, ComplexF64[1, 0])
        @test r == ComplexF64[1, 3, 5]
        t.f!(r, ComplexF64[0, 2im])
        @test r == ComplexF64[1 + 4im, 3 + 8im, 5 + 12im]
    end

    @testset "modal spectrum" begin
        engine = load_engine(config("analytic_random"))
        cfg = run_config(engine)
        K, C, M = build_matrices(engine, 0.01, 0.002)
        nev = 6
        sd_ = modal_spectrum(K, C, M, 0.01, 0.002, nev; dense = true)
        sa_ = modal_spectrum(K, C, M, 0.01, 0.002, nev; dense = false)
        @test length(sd_.eigenvalues) == 2nev
        @test sd_.eigenvalues ≈ sa_.eigenvalues rtol = 1e-9
        @test sd_.eigenmodes ≈ sa_.eigenmodes rtol = 1e-7
        λ = sd_.eigenvalues
        for p in 1:nev
            ϕ = real.(sd_.eigenmodes[:, 1, 2p - 1])
            @test imag(λ[2p - 1]) > 0 && λ[2p] == conj(λ[2p - 1])
            @test dot(ϕ, M * ϕ) ≈ 1 rtol = 1e-10
            @test maximum(ϕ) == maximum(abs, ϕ)                      # sign convention
            ω2 = dot(ϕ, K * ϕ)
            @test norm(K * ϕ - ω2 * (M * ϕ)) <= 1e-8 * norm(K * ϕ)
            @test real(λ[2p - 1]) ≈ -0.5 * (0.01 + 0.002 * ω2) rtol = 1e-8
            @test abs2(λ[2p - 1]) ≈ ω2 rtol = 1e-8
            # quadratic eigenproblem residual
            @test norm((λ[2p - 1]^2 .* M .+ λ[2p - 1] .* C .+ K) * ϕ) <= 1e-7 * norm(K * ϕ)
        end
        # same eigenvalues as MORFE's own structural eigensolver
        model = NthOrderModel((K, C, M), exact_terms(engine))
        ref = spectrum(model, StructureModalDampingEigensolver(nev, 0.01, 0.002))
        @test sort(ref.eigenvalues; by = x -> (abs(x), -imag(x))) ≈
              sort(λ; by = x -> (abs(x), -imag(x))) rtol = 1e-9

        spectral, sp = build_spectral(model, cfg)
        @test master_eigenvalues(spectral) == sp.eigenvalues[1:2]
        @test MORFE.SpectralDecomposition.master_conjugate_permutation(spectral) == [2, 1]
    end

    @testset "ROM: engine closures vs exact closures ($name, order $order)" for (name, order) in
                                                                                 (("analytic_two_dof", 3), ("analytic_random", 3))
        engine = load_engine(config(name))
        cfg = run_config(engine)
        mats = build_matrices(engine, cfg.alpha, cfg.beta)
        m_eng = build_model(engine, cfg; terms = :engine, matrices = mats)
        m_ex = build_model(engine, cfg; terms = :exact, matrices = mats)
        spectral, sp = build_spectral(m_ex, cfg)
        W1, R1 = solve_rom(m_eng, spectral, cfg; order = order)
        W2, R2 = solve_rom(m_ex, spectral, cfg; order = order)
        dR, dW = rom_rel_diff(W1, R1, W2, R2)
        @test dR <= 1e-7 && dW <= 1e-7
        @test size(R1.poly.coefficients, 1) == 2
        @test size(W1.poly.coefficients)[1:2] == (m_eng.n_fom, 2)
        # linear part of the reduced dynamics = master eigenvalues
        exps = R1.poly.multiindex_set.exponents
        i10 = findfirst(e -> Tuple(e) == (1, 0), exps)
        @test R1.poly.coefficients[1, i10] ≈ sp.eigenvalues[1] rtol = 1e-12
        # the cubic backbone coefficient exists
        i21 = findfirst(e -> Tuple(e) == (2, 1), exps)
        @test abs(R1.poly.coefficients[1, i21]) > 0
        @test engine_stats(engine)["jobs_total"] > 0

        dir = mktempdir()
        export_rom(dir, W1, R1, engine, cfg; spectrum = sp)
        @test all(isfile, joinpath.(dir, ("W.npz", "R.npz", "engine_stats.json", "summary.txt")))
        @test isfile(joinpath(dir, "data", "R_coefficients.csv"))
    end

    include("physics.jl")      # conventions against a closed-form backbone
    include("end_to_end.jl")   # P5.2
    include("real_meshes.jl")  # P7.1, P7.2
    include("prefetch.jl")     # P9
    include("d_ref_sweep.jl")  # P10.1
    include("amplitude.jl")    # P10.2
    include("forcing.jl")      # P11.1
    include("exports.jl")      # P11.2 (uses forced_config of forcing.jl)
end
