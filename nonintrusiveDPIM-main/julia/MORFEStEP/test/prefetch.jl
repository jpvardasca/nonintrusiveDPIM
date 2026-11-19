# P9: tangent jobs in per-order batches (prefetch), resume after an interruption, and the
# monomial set with a cap on the forcing degree.
#
# Included by runtests.jl (which defines `config`).

manifest_keys(engine) = pyconvert(Vector{String}, engine.store.manifest_keys())
engine_mode(engine) = pyconvert(String, engine.mode)

# everything needed for a solve with the EED closures of `engine`
function prefetch_setup(engine)
    cfg = run_config(engine)
    mats = build_matrices(engine, cfg.alpha, cfg.beta)
    model = build_model(engine, cfg; terms = :engine, matrices = mats)
    spectral, _ = build_spectral(model, cfg)
    return cfg, model, spectral
end

@testset "P9.3 prefetch equals lazy: $name, order $order" for (name, order) in
                                                              (("analytic_chain", 5), ("analytic_random", 6),
    ("single_element_pysvk", 5))
    lazy = load_engine(config(name))
    cfg, model, spectral = prefetch_setup(lazy)
    @test cfg.prefetch == false                      # "auto": off unless the backend is abaqus
    W, R, counts = solve_rom_by_order(model, spectral, cfg, lazy; order = order)

    pre = load_engine(config(name))                  # a fresh engine with an empty store
    cfg, model, spectral = prefetch_setup(pre)
    Wp, Rp, batches = run_with_prefetch(model, spectral, cfg, pre; order = order, max_parallel = 4)

    dR, dW = rom_rel_diff(Wp, Rp, W, R)
    @test dR <= 1e-12 && dW <= 1e-12
    @test maximum(x -> max(x.dR, x.dW), rom_rel_diff_by_degree(Wp, Rp, W, R)) <= 1e-9

    # same jobs, order by order; the final solve found every job in the cache
    per_order = filter(b -> b.order > 0, batches)
    @test [b.order for b in per_order] == collect(2:order)
    @test [b.new_jobs for b in per_order] == [c.new_jobs for c in counts]
    @test all(b.new_jobs == 0 for b in batches if b.order == 0)
    sl, sp_ = engine_stats(lazy), engine_stats(pre)
    @test sp_["jobs_total"] == sl["jobs_total"] > 0
    @test sp_["jobs_probe"] == sl["jobs_probe"] && sp_["jobs_pair"] == sl["jobs_pair"]
    @test sp_["store_puts"] == sp_["jobs_total"] + sp_["jobs_k0"]   # nothing ran twice
    @test sp_["batches"] == sum(cld(b.new_jobs, 4) for b in per_order)
    @test sp_["pending"] == 0 && engine_mode(pre) == "lazy"
    @test all(b.seconds_pass >= 0 && b.seconds_jobs >= 0 for b in batches)

    # a last record pass without missing jobs is already the full solve: no extra pass then
    W2, R2, again = run_with_prefetch(model, spectral, cfg, pre; order = order)
    @test all(b.order > 0 && b.new_jobs == 0 for b in again)
    @test R2.poly.coefficients == Rp.poly.coefficients
    @test W2.poly.coefficients == Wp.poly.coefficients

    dir = mktempdir()
    export_rom(dir, Wp, Rp, pre, cfg; jobs_per_order = per_order)
    txt = read(joinpath(dir, "engine_stats.json"), String)
    @test occursin("\"seconds_jobs\"", txt) && occursin("\"store_MB_per_matrix\"", txt)
end

@testset "P9.2 resume after an interruption" begin
    dir = mktempdir()
    body = """
           [step]
           d_ref = 1.0
           [backend]
           kind = "analytic"
           [analytic]
           model = "spring_chain"
           n = 20
           [damping]
           alpha = 0.001
           [rom]
           masters = [1, 2]
           order = 4
           resonance_tol = 0.01
           """
    path = joinpath(dir, "resume_chain.toml")
    write(path, body * """
                       [cache]
                       dir = "$(replace(joinpath(dir, "cache"), "\\" => "/"))"
                       lru_items = 4
                       """)
    path_clean = joinpath(dir, "clean_chain.toml")
    write(path_clean, body)
    order = 4

    # reference: the same model without a cache, solved in one go
    clean = load_engine(path_clean)
    cfg, model, spectral = prefetch_setup(clean)
    Wc, Rc = solve_rom(model, spectral, cfg; order = order)
    total = engine_stats(clean)["jobs_total"]
    @test total >= 8
    fail_after = total ÷ 2

    # first run: the backend fails half-way through the tangent jobs
    wrappers = pyimport("nistep.backends.wrappers")
    first = load_engine(path)
    first.backend = wrappers.ThrottledBackend(first.backend; fail_after = fail_after)
    cfg, model, spectral = prefetch_setup(first)
    @test_throws PyException run_with_prefetch(model, spectral, cfg, first; order = order,
        max_parallel = 2)
    @test engine_mode(first) == "lazy"               # the mode is restored after an error
    keys_before = manifest_keys(first)
    done_before = length(keys_before) - 1            # one key is K_tg(0)
    @test 1 <= done_before <= fail_after && allunique(keys_before)

    # second run, as a new process would do it: new engine, same cache folder
    second = load_engine(path)
    cfg, model, spectral = prefetch_setup(second)
    @test engine_stats(second)["n_directions"] > 0    # directions were loaded from disk
    W, R, batches = run_with_prefetch(model, spectral, cfg, second; order = order, max_parallel = 2)
    keys = manifest_keys(second)
    @test allunique(keys)                             # no job ran twice
    @test keys[1:length(keys_before)] == keys_before
    @test length(keys) == total + 1
    s = engine_stats(second)
    @test s["jobs_total"] == total - done_before && s["jobs_k0"] == 0
    @test s["store_items_memory"] <= 4 && s["store_items_disk"] == total + 1
    @test s["store_ram_estimate_MB"] <= s["store_ram_limit_MB"]
    dR, dW = rom_rel_diff(W, R, Wc, Rc)
    @test dR <= 1e-12 && dW <= 1e-12

    # third run: everything comes from the cache
    third = load_engine(path)
    cfg, model, spectral = prefetch_setup(third)
    W3, R3, b3 = run_with_prefetch(model, spectral, cfg, third; order = order)
    @test engine_stats(third)["jobs_total"] == 0 && engine_stats(third)["jobs_k0"] == 0
    @test manifest_keys(third) == keys
    @test R3.poly.coefficients == R.poly.coefficients
    @test W3.poly.coefficients == W.poly.coefficients
end

@testset "P9 monomial set with a forcing cap" begin
    engine = load_engine(config("analytic_two_dof"))
    cfg, model, spectral = prefetch_setup(engine)
    @test expansion_set(model, spectral, 4) === 4    # autonomous: the plain total degree
    @test_throws ArgumentError expansion_set(model, spectral, 0)
    @test_throws ArgumentError run_with_prefetch(model, spectral, cfg, engine; order = 1)

    # the same model with a harmonic forcing system (two forcing variables)
    K, C, M = model.linear_terms
    Ω = abs(master_eigenvalues(spectral)[1])
    F = zeros(2, 2)
    F[1, :] .= 0.5
    forced = NthOrderModel((K, C, M), (model.nonlinear_terms..., force_term(F)),
        ExternalSystem((im * Ω, -im * Ω)))
    fspectral, _ = build_spectral(forced, cfg)
    for (q, cap) in ((3, 1), (4, 2), (2, 5))
        mset = expansion_set(forced, fspectral, q; forcing_order = cap)
        exps = mset.exponents
        @test all(length(e) == 4 for e in exps)
        @test extrema(sum.(exps)) == (1, q)
        @test maximum(e[3] + e[4] for e in exps) == min(q, cap)
        @test is_downward_closed(mset)
        # count: Σ over forcing degree f ≤ cap of (monomials of 2 variables with degree f)
        # × (monomials of 2 variables with degree ≤ q − f), minus the constant
        n = sum((f + 1) * binomial(q - f + 2, 2) for f in 0:min(q, cap)) - 1
        @test length(exps) == n
    end
    @test expansion_set(forced, fspectral, 3).exponents ==
          all_multiindices_up_to(4, 3; min_degree = 1).exponents
end
