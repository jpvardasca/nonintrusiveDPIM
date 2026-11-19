# P7: linear spectrum and closure ROMs on the real meshes, with the PySVK backend (the
# numpy stand-in for the Abaqus UEL). No Abaqus is needed.
#
# Included by runtests.jl (which defines `config`).

@testset "P7.1 spectrum: $name" for (name, ω_thesis) in
                                    (("mems_pysvk", 1.819006), ("cantilever_pysvk", 8.425818))
    engine = load_engine(config(name))
    cfg = run_config(engine)
    K, C, M = build_matrices(engine, cfg.alpha, cfg.beta)
    n = size(K, 1)
    @test n == (name == "mems_pysvk" ? 1521 : 2700)

    nev = 6
    sp = modal_spectrum(K, C, M, cfg.alpha, cfg.beta, nev)       # sparse path (Arpack)
    ω = [abs(sp.eigenvalues[2p - 1]) for p in 1:nev]

    # scipy reference on the same matrices
    spla = pyimport("scipy.sparse.linalg")
    np = pyimport("numpy")
    w2 = spla.eigsh(engine.K(); k = nev, M = engine.M(), sigma = 0.0, which = "LM",
        return_eigenvectors = false)
    ω_scipy = sqrt.(sort(pyconvert(Vector{Float64}, np.asarray(w2))))
    @test ω ≈ ω_scipy rtol = 1e-8

    # MORFE's own structural eigensolver
    model = build_model(engine, cfg; terms = :engine, matrices = (K, C, M))
    ref = spectrum(model, StructureModalDampingEigensolver(nev, cfg.alpha, cfg.beta))
    @test sort(abs.(ref.eigenvalues)) ≈ sort(abs.(sp.eigenvalues)) rtol = 1e-8

    # thesis value of the first natural frequency
    @test ω[1] ≈ ω_thesis rtol = 1e-4
    @test isapprox(real(sp.eigenvalues[1]), -cfg.alpha / 2; atol = 1e-12)
    @test engine_stats(engine)["jobs_total"] == 0        # the linear model needs no EED job
end

@testset "P7.2 closure ROM on MEMS with PySVK, order $order" for order in (3, 5)
    engine = load_engine(config("mems_pysvk"))
    cfg = run_config(engine)
    mats = build_matrices(engine, cfg.alpha, cfg.beta)
    m_eng = build_model(engine, cfg; terms = :engine, matrices = mats)
    m_ref = build_model(engine, cfg; terms = :backend_exact, matrices = mats)
    spectral, sp = build_spectral(m_eng, cfg)

    W, R, counts = solve_rom_by_order(m_eng, spectral, cfg, engine; order = order)
    Wr, Rr = solve_rom(m_ref, spectral, cfg; order = order)

    # the plan's criterion: relative to the largest coefficient
    dR, dW = rom_rel_diff(W, R, Wr, Rr)
    @test dR <= 1e-6 && dW <= 1e-6

    # stricter: degree by degree. The cubic normal-form coefficient of this model is about
    # 1e-8 of the two terms that cancel to form it, so force round-off of 1e-13 shows up as
    # about 1e-5 there. W itself is accurate to about 1e-7.
    byd = rom_rel_diff_by_degree(W, R, Wr, Rr)
    @test maximum(x -> x.dW, byd) <= 1e-5
    @test maximum(x -> x.dR, byd) <= 1e-3

    # backbone up to a mode amplitude of d_ref, relative to the frequency shift
    amps = backbone_amplitudes(engine, sp, 1)
    @test backbone_rel_diff(R, Rr; amplitudes = amps) <= 1e-4
    b = normal_form_branch(R; amplitudes = amps)
    @test b.frequency[1] ≈ abs(imag(sp.eigenvalues[1])) rtol = 1e-12
    @test b.frequency[end] > b.frequency[1]                # hardening

    stats = engine_stats(engine)
    @test stats["jobs_total"] == (order == 3 ? 2 : 11)
    @test [c.new_jobs for c in counts] == (order == 3 ? [2, 0] : [2, 0, 6, 3])
    # every prescribed displacement respects the overlap rule max|U| <= 2c = 0.48 d_ref
    @test stats["c"] == 0.24 * 20.0
end
