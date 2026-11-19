# P11.1: the forced model (harmonic modal load, external system (iΩ, −iΩ)), its monomial
# set O(q, forcing order), and the frequency response curve of the reduced dynamics.
#
# Included by runtests.jl (which defines `config`).

"""
A copy of a repository config with forcing switched on, in a temporary folder. `inp` paths
of the repository configs are made absolute; `edits` are extra text replacements.
"""
function forced_config(name; amplitude, forcing_order = 3, order = nothing, edits = Pair[])
    txt = read(config(name), String)
    deck = replace(joinpath(MORFEStEP.REPO_ROOT, "test"), "\\" => "/")
    txt = replace(txt, "inp = \"../test/" => "inp = \"$deck/")
    order === nothing || (txt = replace(txt, r"\norder = \d+" => "\norder = $order"))
    txt = replace(txt,
        r"\norder = (\d+)" => SubstitutionString("\norder = \\1\nforcing_order = $forcing_order"))
    for e in edits
        txt = replace(txt, e)
    end
    path = joinpath(mktempdir(), name * "_forced.toml")
    write(path, txt * "\n[forcing]\nenabled = true\namplitude = $amplitude\nmode = 1\n")
    return path
end

# u'' + α u' + u + 0.3 u² + 0.5 u³ = F̂ cos(ω̂ t) by RK4 from (u0, v0); the largest |u| over
# the last of `periods` forcing periods.
function duffing_steady(α, F̂, ω̂, u0, v0; periods = 400, steps = 200)
    acc(t, u, v) = F̂ * cos(ω̂ * t) - α * v - u - 0.3u^2 - 0.5u^3
    h = 2π / ω̂ / steps
    t, u, v, umax = 0.0, u0, v0, 0.0
    for p in 1:periods, _ in 1:steps
        k1u, k1v = v, acc(t, u, v)
        k2u, k2v = v + h / 2 * k1v, acc(t + h / 2, u + h / 2 * k1u, v + h / 2 * k1v)
        k3u, k3v = v + h / 2 * k2v, acc(t + h / 2, u + h / 2 * k2u, v + h / 2 * k2v)
        k4u, k4v = v + h * k3v, acc(t + h, u + h * k3u, v + h * k3v)
        u += h / 6 * (k1u + 2k2u + 2k3u + k4u)
        v += h / 6 * (k1v + 2k2v + 2k3v + k4v)
        t += h
        p == periods && (umax = max(umax, abs(u)))
    end
    return umax
end

# displacement and velocity of DOF 1 at t = 0 on the ROM orbit (ρ, ψ): z = ρ e^{iψ}, r = (1, 1)
function orbit_state(W, ρ, ψ)
    z = ρ * cis(ψ)
    vals = (z, conj(z), 1.0 + 0im, 1.0 + 0im)
    exps = W.poly.multiindex_set.exponents
    mono(e) = prod(vals[j]^e[j] for j in 1:4)
    u = sum(W.poly.coefficients[1, 1, m] * mono(e) for (m, e) in enumerate(exps))
    v = sum(W.poly.coefficients[1, 2, m] * mono(e) for (m, e) in enumerate(exps))
    return real(u), real(v)
end

@testset "P11.1 forced Duffing: model, monomial set, conventions" begin
    path = forced_config("analytic_duffing"; amplitude = 0.01,
        edits = ["alpha = 0.0" => "alpha = 0.02", "debug = true" => "debug = false"])
    engine = load_engine(path)
    cfg = run_config(engine)
    @test cfg.forcing_enabled && cfg.forcing_amplitude == 0.01 && cfg.forcing_order == 3
    s = setup_rom(engine, cfg)
    F, Ω = s.forcing
    λ = s.spectrum.eigenvalues[1]
    @test F == [0.005 0.005]                      # (F̂/2) M ϕ with M = 1, ϕ = 1
    @test Ω == imag(λ)
    @test Ω ≈ sqrt(1 - 0.01^2) rtol = 1e-12
    @test s.model.external_system !== nothing
    @test collect(s.model.external_system.eigenvalues) == [im * Ω, -im * Ω]
    # without [forcing] the same engine gives the autonomous model
    @test setup_rom(engine, cfg; forced = false).forcing === nothing

    W, R = solve_rom(s.model, s.spectral, cfg; order = 5)
    exps = R.poly.multiindex_set.exponents
    @test length(exps) == 104 && all(length(e) == 4 for e in exps)   # O(5, 3)
    @test maximum(e[3] + e[4] for e in exps) == 3 && maximum(sum, exps) == 5
    # external rows of R: ṙ₁ = iΩ r₁, ṙ₂ = −iΩ r₂
    i3, i4 = MORFEStEP._monomial(R, (0, 0, 1, 0)), MORFEStEP._monomial(R, (0, 0, 0, 1))
    @test R.poly.coefficients[3:4, [i3, i4]] ≈ [im*Ω 0; 0 -im*Ω] atol = 1e-14
    # modal projection of the load: R₁[r₁] = −i F̂ / (4 Im λ) for a mass-normalised mode
    @test R.poly.coefficients[1, i3] ≈ -im * 0.01 / (4Ω) rtol = 1e-10
    # a forcing order of 0 means "no cap"
    @test MORFEStEP._forcing_cap(0, 5) == 5 && MORFEStEP._forcing_cap(3, 5) == 3

    # engine closures against the exact closures, same load
    ex = build_model(engine, cfg; terms = :exact, matrices = s.model.linear_terms,
        forcing = s.forcing)
    exs, _ = build_spectral(ex, cfg; spectrum = s.spectrum)
    We, Re = solve_rom(ex, exs, cfg; order = 5)
    dR, dW = rom_rel_diff(W, R, We, Re)
    @test dR <= 1e-12 && dW <= 1e-12
    @test engine_stats(engine)["jobs_total"] == 2           # the forcing needs no extra job

    # the reduced dynamics are polynomial in r: level 2 equals a load twice as large
    s2 = build_model(engine, cfg; terms = :exact, matrices = s.model.linear_terms,
        forcing = (2F, Ω))
    sd2, _ = build_spectral(s2, cfg; spectrum = s.spectrum)
    _, R2 = solve_rom(s2, sd2, cfg; order = 5)
    amps = range(0, 0.4; length = 41)
    a, b = forced_response(Re; amplitudes = amps, level = 2), forced_response(R2; amplitudes = amps)
    @test a.amplitude == b.amplitude && a.sheet == b.sheet
    @test a.frequency ≈ b.frequency rtol = 1e-12
    @test forced_response_rel_diff(R2, R2; amplitudes = amps) == (diff = 0.0, unmatched = 0)
    @test forced_response_rel_diff(Re, R2; amplitudes = amps).diff > 0.1  # half the load
end

@testset "P11.1 forced Duffing: frequency response against the full equation" begin
    path = forced_config("analytic_duffing"; amplitude = 0.01, order = 7,
        edits = ["alpha = 0.0" => "alpha = 0.02", "debug = true" => "debug = false"])
    engine = load_engine(path)
    cfg = run_config(engine)
    s = setup_rom(engine, cfg)
    W7, R7 = solve_rom(s.model, s.spectral, cfg)            # O(7, 3) from the config
    W3, R3 = solve_rom(s.model, s.spectral, cfg; order = 3)
    amps = forced_response_amplitudes(R7; length = 151)
    # linear peak: ρ = (F̂ / 4Ω) / (α / 2)
    @test last(amps) ≈ 1.5 * (0.01 / (4 * s.forcing[2])) / 0.01 rtol = 1e-10
    fr = forced_response(R7; amplitudes = amps)
    @test !isempty(fr.amplitude) && maximum(fr.amplitude) < last(amps)  # the curve closes
    @test maximum(fr.sheet) == 2                          # two points per amplitude below the peak
    @test fr.stable[1] && any(!, fr.stable)               # hardening: an unstable middle branch
    # hardening: the peak leans to frequencies above Ω
    @test fr.frequency[argmax(fr.amplitude)] > s.forcing[2] + 0.01

    # stable points near resonance: the ROM orbit, integrated with the full equation, stays
    # on the ROM's amplitude
    u7 = orbit_amplitude(W7, 1, fr.amplitude, fr.phase)
    picks = [i for i in eachindex(fr.amplitude) if fr.stable[i] && fr.sheet[i] == 1 &&
                                                   0.08 <= fr.amplitude[i] <= 0.23]
    picks = picks[round.(Int, range(1, length(picks); length = 4))]
    for i in picks
        u0, v0 = orbit_state(W7, fr.amplitude[i], fr.phase[i])
        ufom = duffing_steady(cfg.alpha, cfg.forcing_amplitude, fr.frequency[i], u0, v0)
        @test abs(ufom - u7[i]) / ufom <= 1.5e-3
    end
    # convergence with the order at the largest amplitude checked
    i = last(picks)
    u0, v0 = orbit_state(W7, fr.amplitude[i], fr.phase[i])
    ufom = duffing_steady(cfg.alpha, cfg.forcing_amplitude, fr.frequency[i], u0, v0)
    f3 = forced_response(R3; amplitudes = [fr.amplitude[i]])
    u3 = orbit_amplitude(W3, 1, f3.amplitude[1:1], f3.phase[1:1])[1]
    @test abs(u3 - ufom) > 2 * abs(u7[i] - ufom)

    # misuse
    @test_throws ArgumentError forced_response(R7; amplitudes = amps, n_phase = 4)
    @test_throws DimensionMismatch orbit_amplitude(W7, 1, [0.1, 0.2], [0.0])
    @test_throws ArgumentError orbit_amplitude(W7, 2, [0.1], [0.0])
    auto = setup_rom(engine, cfg; forced = false)
    Wa, Ra = solve_rom(auto.model, auto.spectral, cfg; order = 3)
    @test_throws ArgumentError forced_response(Ra; amplitudes = amps)
    @test_throws ArgumentError orbit_amplitude(Wa, 1, [0.1], [0.0])
end

@testset "P11.1 forced two-DoF: prefetch equals lazy" begin
    path = forced_config("analytic_two_dof"; amplitude = 0.05)
    lazy = load_engine(path)
    cfg = run_config(lazy)
    s = setup_rom(lazy, cfg)
    W, R, counts = solve_rom_by_order(s.model, s.spectral, cfg, lazy; order = 5)
    @test length(R.poly.multiindex_set.exponents) == 104

    pre = load_engine(path)
    sp = setup_rom(pre, cfg)
    Wp, Rp, batches = run_with_prefetch(sp.model, sp.spectral, cfg, pre; order = 5, max_parallel = 4)
    dR, dW = rom_rel_diff(Wp, Rp, W, R)
    @test dR <= 1e-12 && dW <= 1e-12
    per_order = filter(b -> b.order > 0, batches)
    @test [b.new_jobs for b in per_order] == [c.new_jobs for c in counts]
    @test all(b.new_jobs == 0 for b in batches if b.order == 0)
    @test engine_stats(pre)["jobs_total"] == engine_stats(lazy)["jobs_total"]
    # the engine against the exact closures
    ex = build_model(lazy, cfg; terms = :exact, matrices = s.model.linear_terms, forcing = s.forcing)
    exs, _ = build_spectral(ex, cfg; spectrum = s.spectrum)
    We, Re = solve_rom(ex, exs, cfg; order = 5)
    @test maximum(x -> max(x.dR, x.dW), rom_rel_diff_by_degree(W, R, We, Re)) <= 1e-9
    amps = forced_response_amplitudes(Re; length = 101)
    d = forced_response_rel_diff(R, Re; amplitudes = amps)
    @test d.diff <= 1e-9 && d.unmatched == 0
end

@testset "P11.1 forced MEMS (PySVK), O(3, 3)" begin
    path = forced_config("mems_pysvk"; amplitude = 2e-3, order = 3)
    engine = load_engine(path)
    cfg = run_config(engine)
    s = setup_rom(engine, cfg)
    F, Ω = s.forcing
    W, R = solve_rom(s.model, s.spectral, cfg)
    @test engine_stats(engine)["jobs_total"] == 2
    i3 = MORFEStEP._monomial(R, (0, 0, 1, 0))
    @test R.poly.coefficients[1, i3] ≈ -im * 2e-3 / (4Ω) rtol = 1e-8
    ex = build_model(engine, cfg; terms = :backend_exact, matrices = s.model.linear_terms,
        forcing = s.forcing)
    exs, _ = build_spectral(ex, cfg; spectrum = s.spectrum)
    We, Re = solve_rom(ex, exs, cfg)
    byd = filter(x -> x.degree >= 2, rom_rel_diff_by_degree(W, R, We, Re))
    @test maximum(x -> x.dR, byd) <= 1e-4 && maximum(x -> x.dW, byd) <= 1e-6
    amps = forced_response_amplitudes(Re; length = 201)
    d = forced_response_rel_diff(R, Re; amplitudes = amps)
    @test d.diff <= 1e-4 && d.unmatched <= 2
    fr = forced_response(Re; amplitudes = amps)
    dof = monitor_index(engine, cfg, real.(s.spectrum.eigenmodes[:, 1, 1]))
    u = orbit_amplitude(We, dof, fr.amplitude, fr.phase)
    @test fr.frequency[argmax(u)] > Ω                     # hardening, as in the thesis
end
