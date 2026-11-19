# P5.2: end-to-end ROMs on analytic models. The ROM built with the EED-StEP closures must
# equal the ROM built with the exact pure-Julia closures.
#
# Included by runtests.jl (which defines `config`).

@testset "end to end: $name, order $order" for name in ("analytic_chain", "analytic_random"),
    order in (5, 7)

    engine = load_engine(config(name))
    cfg = run_config(engine)
    mats = build_matrices(engine, cfg.alpha, cfg.beta)
    m_eng = build_model(engine, cfg; terms = :engine, matrices = mats)
    m_ex = build_model(engine, cfg; terms = :exact, matrices = mats)
    spectral, sp = build_spectral(m_eng, cfg)

    W, R, counts = solve_rom_by_order(m_eng, spectral, cfg, engine; order = order)
    Wx, Rx = solve_rom(m_ex, spectral, cfg; order = order)

    dR, dW = rom_rel_diff(W, R, Wx, Rx)
    @test dR <= 1e-7
    @test dW <= 1e-7
    byd = rom_rel_diff_by_degree(W, R, Wx, Rx)
    @test maximum(x -> x.dR, byd) <= 1e-7
    @test maximum(x -> x.dW, byd) <= 1e-7
    @test backbone_rel_diff(R, Rx; amplitudes = backbone_amplitudes(engine, sp, 1)) <= 1e-6

    # complex normal form on one pair: row 1 only has monomials z^(k+1) z̄^k
    exps = R.poly.multiindex_set.exponents
    row = R.poly.coefficients[1, :]
    @test all(e[1] - e[2] == 1 for (e, c) in zip(exps, row) if abs(c) > 0)
    @test maximum(sum, exps) == order

    # job bookkeeping: one entry per order, increments add up, nothing ran twice
    @test [c.order for c in counts] == collect(2:order)
    @test sum(c.new_jobs for c in counts) == counts[end].jobs_total
    stats = engine_stats(engine)
    @test stats["jobs_total"] == counts[end].jobs_total
    @test stats["store_puts"] == stats["jobs_total"] + stats["jobs_k0"]
    @test stats["debug_max_rel_err_quad"] <= 1e-8
    @test stats["debug_max_rel_err_cubic"] <= 1e-8

    # solving again is free: every tangent matrix is already in the store
    W2, R2 = solve_rom(m_eng, spectral, cfg; order = order)
    @test engine_stats(engine)["jobs_total"] == stats["jobs_total"]
    @test R2.poly.coefficients == R.poly.coefficients

    dir = mktempdir()
    export_rom(dir, W, R, engine, cfg; spectrum = sp, jobs_per_order = counts)
    txt = read(joinpath(dir, "engine_stats.json"), String)
    @test occursin("\"jobs_per_order\"", txt) && occursin("\"new_jobs\"", txt)
    @test isfile(joinpath(dir, "W.npz")) && isfile(joinpath(dir, "R.npz"))
end
