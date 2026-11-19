# Physics check of the conventions (signs, the factors 2 and 3, MORFE's multiplicities):
# a one-DOF oscillator  m ü + k u + k2 u² + k3 u³ = 0  has the closed-form backbone
#
#     ω = ω0 + (9 α3 ω0² − 10 α2²) / (24 ω0³) · a²,   α2 = k2/m, α3 = k3/m, ω0² = k/m
#
# where a is the amplitude of the fundamental harmonic (Nayfeh and Mook). In the ROM,
# u = 2 ρ ϕ cos θ + …, so a = 2 ρ ϕ with ϕ = 1/√m, and ω(ρ) = ω0 + Im(R[1, z² z̄]) ρ² + ….
#
# A wrong sign of the closures flips the cubic term; a wrong factor on the quadratic
# closure changes the α2² term. Included by runtests.jl (which defines `config`).

@testset "physics: Duffing backbone, nl_scale = $scale" for scale in (1.0, -1.0, 2.5)
    k, m, k2, k3 = 1.0, 1.0, 0.3 * scale, 0.5 * scale
    path = config("analytic_duffing")
    if scale != 1.0
        txt = replace(read(path, String), "nl_scale = 1.0" => "nl_scale = $scale")
        path = joinpath(mktempdir(), "duffing.toml")
        write(path, txt)
    end
    engine = load_engine(path)
    cfg = run_config(engine)
    ω0 = sqrt(k / m)
    expected = (9 * (k3 / m) * ω0^2 - 10 * (k2 / m)^2) / (24 * ω0^3) * 4 / m   # a² = 4 ρ² / m

    for terms in (:engine, :exact)
        model = build_model(engine, cfg; terms = terms)
        spectral, sp = build_spectral(model, cfg)
        @test sp.eigenvalues[1] ≈ im * ω0
        W, R = solve_rom(model, spectral, cfg; order = 3)
        exps = R.poly.multiindex_set.exponents
        c = R.poly.coefficients[1, findfirst(e -> Tuple(e) == (2, 1), exps)]
        @test imag(c) ≈ expected rtol = 1e-10
        @test abs(real(c)) <= 1e-12
        # backbone: hardening for scale > 0 (as long as 9 α3 ω0² > 10 α2²), softening for scale < 0
        b = normal_form_branch(R; amplitudes = [0.0, 0.1])
        @test sign(b.frequency[2] - b.frequency[1]) == sign(expected)
        # the manifold: u = ϕ (z + z̄) at first order, with ϕ = 1/√m > 0
        Wc = W.poly.coefficients
        ew = W.poly.multiindex_set.exponents
        @test Wc[1, 1, findfirst(e -> Tuple(e) == (1, 0), ew)] ≈ 1 / sqrt(m)
        # second order, by harmonic balance of u" + ω0² u + α2 u² = 0 with u = ϕ(z + z̄):
        # z z̄ term: u = −2 α2 ϕ² / ω0² ;   z² term: u = α2 ϕ² / (3 ω0²)
        @test Wc[1, 1, findfirst(e -> Tuple(e) == (1, 1), ew)] ≈ -2 * (k2 / m) / (m * ω0^2) atol = 1e-12
        @test Wc[1, 1, findfirst(e -> Tuple(e) == (2, 0), ew)] ≈ (k2 / m) / (3 * m * ω0^2) atol = 1e-12
    end
end
