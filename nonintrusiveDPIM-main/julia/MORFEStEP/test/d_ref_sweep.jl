# P10.1: the ROM must not depend on d_ref. `rom_d_ref_sweep` solves it at several d_ref and
# reports the cubic normal-form coefficient (the slope of the backbone).
#
# Included by runtests.jl (which defines `config`).

@testset "P10.1 cubic coefficient (Duffing, closed form)" begin
    engine = load_engine(config("analytic_duffing"))
    cfg = run_config(engine)
    model = build_model(engine, cfg)
    spectral, _ = build_spectral(model, cfg)
    _, R3 = solve_rom(model, spectral, cfg; order = 3)
    k, m, k2, k3 = 1.0, 1.0, 0.3, 0.5
    ω0 = sqrt(k / m)
    expected = (9 * (k3 / m) * ω0^2 - 10 * (k2 / m)^2) / (24 * ω0^3) * 4 / m   # see physics.jl
    c = cubic_coefficient(R3)
    @test c isa ComplexF64
    @test imag(c) ≈ expected rtol = 1e-10
    @test abs(real(c)) <= 1e-12
    _, R5 = solve_rom(model, spectral, cfg; order = 5)
    @test cubic_coefficient(R5) ≈ c rtol = 1e-12          # higher orders do not change it
    _, R2 = solve_rom(model, spectral, cfg; order = 2)
    @test_throws ArgumentError cubic_coefficient(R2)
    @test_throws ArgumentError cubic_coefficient(R3; pair = 2)
    @test_throws ArgumentError cubic_coefficient(R3; pair = 0)
end

@testset "P10.1 d_ref sweep of the ROM: $name" for name in ("analytic_chain", "single_element_pysvk")
    tol = 1e-9          # for d_ref >= 0.5; at d_ref = 0.05 the round-off is about 100 times larger
    engine = load_engine(config(name))
    cfg = run_config(engine)
    d_refs = [0.05, 0.5, 1.0, 2.0]
    rows = rom_d_ref_sweep(engine, d_refs)
    @test [r.d_ref for r in rows] == d_refs
    @test all(r.status === :ok && r.reason == "" for r in rows)
    @test all(abs(r.coefficient) > 0 for r in rows)
    @test isnan(rows[1].change) && rows[2].change <= 1e-7
    @test all(r.change <= tol for r in rows[3:end])
    @test rows[1].error <= 1e-7 && rows[1].dR <= 1e-7
    @test all(r.error <= tol && r.dR <= tol for r in rows[2:end])
    @test rows[1].error > 10 * rows[end].error               # the round-off grows as d_ref shrinks
    @test all(r.jobs == rows[1].jobs > 0 for r in rows)     # every d_ref needs its own jobs
    # the engines share one store, and the base engine ran nothing itself
    s = engine_stats(engine)
    @test s["jobs_total"] == 0 && s["jobs_k0"] == 1
    @test s["store_puts"] == 1 + sum(r.jobs for r in rows)
    @test pyconvert(Float64, engine.step.d_ref) == 1.0      # unchanged

    # a repeat finds every matrix in the store
    again = rom_d_ref_sweep(engine, d_refs)
    @test engine_stats(engine)["store_puts"] == s["store_puts"]
    @test [r.coefficient for r in again] == [r.coefficient for r in rows]

    # the reference as a ROM solved elsewhere, and no reference at all
    mats = build_matrices(engine, cfg.alpha, cfg.beta)
    m_ref = build_model(engine, cfg; terms = :backend_exact, matrices = mats)
    sp_ref, _ = build_spectral(m_ref, cfg)
    ref = solve_rom(m_ref, sp_ref, cfg; order = 3)
    given = rom_d_ref_sweep(engine, d_refs[2:3]; reference = ref)
    @test [r.error for r in given] == [r.error for r in rows[2:3]]
    none = rom_d_ref_sweep(engine, d_refs[2:3]; reference = nothing)
    @test all(isnan(r.error) && isnan(r.dR) && r.status === :ok for r in none)
    @test none[2].change <= tol

    @test_throws ArgumentError rom_d_ref_sweep(engine, d_refs; order = 2)
    @test_throws ArgumentError rom_d_ref_sweep(engine, d_refs; reference = :exact)
end

@testset "P10.1 d_ref sweep records failed jobs" begin
    engine = load_engine(config("analytic_chain"))
    engine.K()                                               # K_tg(0) is in the store already
    wrappers = pyimport("nistep.backends.wrappers")
    engine.backend = wrappers.ThrottledBackend(engine.backend; fail_after = 2)
    rows = rom_d_ref_sweep(engine, [0.5, 1.0, 2.0])
    @test rows[1].status === :ok && rows[1].jobs == 2 && rows[1].error <= 1e-9
    @test all(r.status === :failed && isnan(real(r.coefficient)) for r in rows[2:3])
    @test all(occursin("simulated failure", r.reason) for r in rows[2:3])
    @test all(r.jobs == 0 for r in rows[2:3])
end
