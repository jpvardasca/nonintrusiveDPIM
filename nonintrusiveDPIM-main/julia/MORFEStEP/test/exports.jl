# P11.2: the ROM in real variables (realify) and its export for other tools.
#
# Included by runtests.jl after forcing.jl (uses `forced_config`).

# value of the complex polynomial with coefficients `c` (over the monomials of `X`) at `v`
cpoly(X, c, v) = sum(c[m] * prod(v[j]^e[j] for j in eachindex(v))
                     for (m, e) in enumerate(X.poly.multiindex_set.exponents))
# value of row `r` of a real ROM table at the real variables `v`
rpoly(rr, row, v) = sum(row[l] * prod(v[j]^rr.exponents[l, j] for j in eachindex(v))
                        for l in axes(rr.exponents, 1))

@testset "P11.2 real ROM of a forced model" begin
    path = forced_config("analytic_two_dof"; amplitude = 0.05)
    engine = load_engine(path)
    cfg = run_config(engine)
    s = setup_rom(engine, cfg)
    W, R = solve_rom(s.model, s.spectral, cfg; order = 5)
    rr = real_rom(W, R; dofs = [1, 2])
    @test rr.variables == ["x1", "y1", "c", "s"]
    @test size(rr.f, 1) == 2 && size(rr.exponents, 2) == 4
    @test size(rr.displacement) == (2, size(rr.exponents, 1)) == size(rr.velocity)
    @test rr.omega0 == s.forcing[2]
    @test maximum(sum(rr.exponents; dims = 2)) == 5

    # the real polynomials equal the complex ones at random points
    rng = Random.MersenneTwister(11)
    for _ in 1:5
        x, y, c, sn = 0.3 .* randn(rng, 4)
        z, r = complex(x, y), complex(c, sn)
        v = (z, conj(z), r, conj(r))
        rv = [x, y, c, sn]
        rhs = cpoly(R, view(R.poly.coefficients, 1, :), v)
        @test rpoly(rr, rr.f[1, :], rv) ≈ real(rhs) atol = 1e-12
        @test rpoly(rr, rr.f[2, :], rv) ≈ imag(rhs) atol = 1e-12
        for (k, i) in enumerate((1, 2))
            u = cpoly(W, view(W.poly.coefficients, i, 1, :), v)
            du = cpoly(W, view(W.poly.coefficients, i, 2, :), v)
            @test abs(imag(u)) <= 1e-12 * max(abs(u), 1e-30)
            @test rpoly(rr, rr.displacement[k, :], rv) ≈ real(u) atol = 1e-12
            @test rpoly(rr, rr.velocity[k, :], rv) ≈ real(du) atol = 1e-12
        end
    end

    # the orbits of forced_response solve the exported ODE: x + iy = ρ e^{i(ω̂t + ψ)}
    fr = forced_response(R; amplitudes = forced_response_amplitudes(R; length = 41))
    for i in round.(Int, range(1, length(fr.amplitude); length = 5)), t in (0.0, 0.7, 2.1)
        ρ, ψ, ω = fr.amplitude[i], fr.phase[i], fr.frequency[i]
        θ = ω * t + ψ
        rv = [ρ * cos(θ), ρ * sin(θ), cos(ω * t), sin(ω * t)]
        @test rpoly(rr, rr.f[1, :], rv) ≈ -ρ * ω * sin(θ) atol = 1e-10 * ρ * ω
        @test rpoly(rr, rr.f[2, :], rv) ≈ ρ * ω * cos(θ) atol = 1e-10 * ρ * ω
    end

    # files, and the Python loader
    dir = mktempdir()
    file = export_real_rom(dir, W, R; dofs = [2], labels = ["tip_x"])
    @test isfile(file) && isfile(joinpath(dir, "real_rom_f.csv")) &&
          isfile(joinpath(dir, "real_rom_obs.csv"))
    hdr = readline(joinpath(dir, "real_rom_obs.csv"))
    @test hdr == "x1,y1,c,s,u_tip_x,v_tip_x"
    @test readline(joinpath(dir, "real_rom_f.csv")) == "x1,y1,c,s,dx1/dt,dy1/dt"
    np = pyimport("numpy")
    rom = pyimport("nistep.realrom").RealROM.load(file)
    @test pyconvert(Vector{String}, pylist(rom.variables)) == rr.variables
    @test pyconvert(Bool, rom.forced) && pyconvert(Float64, rom.omega0) == rr.omega0
    q, t, ω = [0.2, -0.1], 0.9, 1.01 * rr.omega0
    rv = [q..., cos(ω * t), sin(ω * t)]
    py_rhs = pyconvert(Vector{Float64}, rom.rhs(np.array(q), t, ω))
    @test py_rhs ≈ [rpoly(rr, rr.f[1, :], rv), rpoly(rr, rr.f[2, :], rv)] rtol = 1e-12
    @test pyconvert(Vector{Float64}, rom.displacement(np.array(q), t, ω))[1] ≈
          rpoly(rr, rr.displacement[1, :], rv) rtol = 1e-12
    @test_throws ArgumentError real_rom(W, R; dofs = [3])
    @test_throws ArgumentError export_real_rom(dir, W, R; dofs = [1], labels = ["a", "b"])
end

@testset "P11.2 real ROM of an autonomous model" begin
    engine = load_engine(config("analytic_duffing"))
    cfg = run_config(engine)
    s = setup_rom(engine, cfg)
    @test s.forcing === nothing
    W, R = solve_rom(s.model, s.spectral, cfg; order = 5)
    rr = real_rom(W, R; dofs = [1])
    @test rr.variables == ["x1", "y1"] && isnan(rr.omega0)
    x, y = 0.2, -0.15
    z = complex(x, y)
    rhs = cpoly(R, view(R.poly.coefficients, 1, :), (z, conj(z)))
    @test rpoly(rr, rr.f[1, :], [x, y]) ≈ real(rhs) atol = 1e-13
    @test rpoly(rr, rr.f[2, :], [x, y]) ≈ imag(rhs) atol = 1e-13
    # the backbone: on the circle x + iy = ρ e^{iθ}, the frequency is ω(ρ) of normal_form_branch
    ρ = 0.3
    b = normal_form_branch(R; amplitudes = [ρ])
    θ = 0.4
    @test rpoly(rr, rr.f[2, :], [ρ * cos(θ), ρ * sin(θ)]) ≈ ρ * b.frequency[1] * cos(θ) rtol = 1e-12
end
