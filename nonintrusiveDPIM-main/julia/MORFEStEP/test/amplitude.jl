# P10: the monitor DOF and the physical amplitude of an orbit (used to plot backbones against
# a displacement, as the thesis does).
#
# Included by runtests.jl (which defines `config`).

@testset "P10 physical amplitude (Duffing)" begin
    engine = load_engine(config("analytic_duffing"))      # m = k = 1, k2 = 0.3, k3 = 0.5
    cfg = run_config(engine)
    model = build_model(engine, cfg)
    spectral, sp = build_spectral(model, cfg)
    ϕ = real.(sp.eigenmodes[:, 1, 1])
    @test monitor_index(engine, cfg, ϕ) == 1               # no mesh: the largest entry of ϕ

    # Order 2: u(θ) = 2ρ cos θ + ρ² (W₁₁ + 2 W₂₀ cos 2θ) with W₁₁ = −0.6, W₂₀ = 0.1 (see
    # physics.jl). The largest |u| is at θ = π: 2ρ + 0.4 ρ².
    W2, _ = solve_rom(model, spectral, cfg; order = 2)
    ρs = [0.0, 0.01, 0.1, 0.5]
    a = physical_amplitude(W2, 1, ρs)
    @test a[1] == 0
    @test a[2:end] ≈ [2ρ + 0.4ρ^2 for ρ in ρs[2:end]] rtol = 1e-12
    @test physical_amplitude(W2, 1, ρs; samples = 64) ≈ a rtol = 1e-12
    # small amplitudes: the linear mode, 2 ρ |ϕ|
    W5, _ = solve_rom(model, spectral, cfg; order = 5)
    @test physical_amplitude(W5, 1, [1e-6])[1] ≈ 2e-6 * abs(ϕ[1]) rtol = 1e-5
    @test_throws ArgumentError physical_amplitude(W5, 2, ρs)
    @test_throws ArgumentError physical_amplitude(W5, 0, ρs)
end

@testset "P10 monitor DOF on a mesh" begin
    base = load_engine(config("single_element_pysvk"))
    cfg = run_config(base)
    @test cfg.monitor_node === nothing
    ϕ = collect(1.0:12.0)
    @test monitor_index(base, cfg, ϕ) == 12                # nothing configured: largest |ϕ|
    node, dof = pyconvert(Tuple{Int, Int}, base.backend.dofmap.node_dof(7))   # free index 7, 0-based

    deck = replace(joinpath(MORFEStEP.REPO_ROOT, "test", "Meu_input_file.inp"), "\\" => "/")
    txt = replace(read(config("single_element_pysvk"), String),
        "inp = \"../test/Meu_input_file.inp\"" => "inp = \"$deck\"")
    path = joinpath(mktempdir(), "single_monitor.toml")
    write(path, txt * "\n[output]\nmonitor_node = $node\nmonitor_dof = $dof\n")
    engine = load_engine(path)
    cfg = run_config(engine)
    @test cfg.monitor_node == node && cfg.monitor_dof == dof
    @test monitor_index(engine, cfg, ϕ) == 8               # 1-based
end

@testset "P10 monitor DOF of the repository models" begin
    # The configured monitor DOF must be where the first mode is (almost) largest.
    for (name, node, dof) in (("mems_pysvk", 45, 2), ("cantilever_pysvk", 768, 1))
        engine = load_engine(config(name))
        cfg = run_config(engine)
        @test cfg.monitor_node == node && cfg.monitor_dof == dof
        model = build_model(engine, cfg)
        _, sp = build_spectral(model, cfg)
        ϕ = real.(sp.eigenmodes[:, 1, 1])
        @test abs(ϕ[monitor_index(engine, cfg, ϕ)]) >= 0.999 * maximum(abs, ϕ)
    end
end
