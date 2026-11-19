# Build a reduced-order model with MORFE.jl, with the nonlinear forces evaluated by the
# Python EED-StEP engine (package `nistep`).
#
# Usage (from the repository folder `nonintrusiveDPIM-main/`):
#
#   julia --project=julia/MORFEStEP scripts/run_rom.jl configs/<model>.toml [options]
#
# Options:
#   --order N      expansion order (default: [rom] order of the config)
#   --out DIR      output directory (default: [output] dir of the config, i.e. runs/<name>)
#   --reference    also solve with the backend's exact closures and report the differences
#                  (only for backends that know their polarised forms: analytic, pysvk)
#   --tol X        tolerance of the reference check (default 1e-4), applied to the relative
#                  difference of R and W per polynomial degree and to the backbone.
#                  Typical values: 1e-14 on analytic models; 1e-7 for W and up to 1e-5 for R
#                  on real meshes, where the normal-form coefficients are small differences
#                  of large terms (double-precision round-off, not a method error).
#   --no-per-order solve once at the final order instead of order by order
#                  (then engine_stats.json has no "jobs_per_order" table)
#   --prefetch     run the tangent jobs of each order as one batch before solving, so the
#                  runner can execute them in parallel ([abaqus] max_parallel at a time).
#                  This is the default for the abaqus backend ([engine] prefetch = "auto").
#   --no-prefetch  run each job at the moment the solver needs it
#   --verbose      show MORFE's setup banner and progress
#
# Outputs in the output directory: W.npz, R.npz, engine_stats.json, summary.txt,
# data/R_coefficients.csv, data/W.jls, data/R.jls, and the ROM in real variables with the
# monitor DOF: real_rom.npz (read it with python nistep.realrom.RealROM.load),
# real_rom_f.csv and real_rom_obs.csv.
#
# With [forcing] enabled the model is forced (load amplitude * M * phi, frequency Im(lambda)
# of the first master) and the order is O(order, [rom] forcing_order). For one master pair
# in complex normal form the frequency response curve is also written, to frf.csv: reduced
# amplitude rho, phase psi, forcing frequency omega, stable (0/1), sheet, and the largest
# displacement of the monitor DOF over the period. --reference then compares these curves.
#
# Run Julia with one thread: the Python engine must be called from the main thread.

using MORFE
using MORFEStEP
using PythonCall
using Printf

function parse_args(args)
    opts = Dict{String, Any}("order" => nothing, "out" => nothing, "reference" => false,
        "per_order" => true, "verbose" => false, "tol" => 1e-4, "prefetch" => nothing)
    config = nothing
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--order"
            opts["order"] = parse(Int, args[i + 1])
            i += 1
        elseif a == "--out"
            opts["out"] = args[i + 1]
            i += 1
        elseif a == "--tol"
            opts["tol"] = parse(Float64, args[i + 1])
            i += 1
        elseif a == "--reference"
            opts["reference"] = true
        elseif a == "--no-per-order"
            opts["per_order"] = false
        elseif a == "--prefetch"
            opts["prefetch"] = true
        elseif a == "--no-prefetch"
            opts["prefetch"] = false
        elseif a == "--verbose"
            opts["verbose"] = true
        elseif startswith(a, "--")
            error("unknown option $a")
        elseif config === nothing
            config = a
        else
            error("unexpected argument $a")
        end
        i += 1
    end
    config === nothing && error("usage: run_rom.jl <config.toml> [--order N] [--out DIR] " *
                                "[--reference] [--tol X] [--no-per-order] [--prefetch] " *
                                "[--no-prefetch] [--verbose]")
    return config, opts
end

function main(args)
    config, opts = parse_args(args)
    t_start = time()
    engine = load_engine(config)
    cfg = run_config(engine)
    order = something(opts["order"], cfg.order)
    outdir = something(opts["out"], cfg.output_dir)
    verbose = opts["verbose"]

    println("run_rom: ", cfg.name, "  (", cfg.path, ")")
    mats = build_matrices(engine, cfg.alpha, cfg.beta)
    setup = setup_rom(engine, cfg; matrices = mats)
    model, spectral, sp, forcing = setup.model, setup.spectral, setup.spectrum, setup.forcing
    n = model.n_fom
    @printf("  DOFs = %d, masters = %s, order = %d, style = %s\n", n, string(cfg.masters),
        order, string(cfg.style))
    for p in cfg.masters
        λ = sp.eigenvalues[2p - 1]
        @printf("  mode %d: λ = %.9g %+.9gi   (ω = %.9g)\n", p, real(λ), imag(λ), abs(λ))
    end
    if forcing !== nothing
        cap = cfg.forcing_order >= 1 ? min(cfg.forcing_order, order) : order
        @printf("  forcing: %g M ϕ%d cos(Ω t), Ω = %.9g; expansion O(%d, %d)\n",
            cfg.forcing_amplitude, cfg.forcing_mode, forcing[2], order, cap)
    end
    one_pair = length(cfg.masters) == 1 && cfg.style === :complex_normal_form

    println("  ", pyconvert(String, engine.store.memory_report()))

    t0 = time()
    counts = nothing
    prefetch = something(opts["prefetch"], cfg.prefetch) && order >= 2
    if prefetch
        W, R, counts = run_with_prefetch(model, spectral, cfg, engine; order = order,
            verbose = verbose)
        @printf("  prefetch: %d jobs at a time; %.2f s in job batches, %.2f s in the solver\n",
            cfg.max_parallel, sum(c -> c.seconds_jobs, counts), sum(c -> c.seconds_pass, counts))
        missed = sum(c -> c.new_jobs, filter(c -> c.order == 0, counts); init = 0)
        missed == 0 || println("  WARNING: ", missed, " job(s) ran outside the batches")
        counts = filter(c -> c.order > 0, counts)
    elseif opts["per_order"] && order >= 2
        W, R, counts = solve_rom_by_order(model, spectral, cfg, engine; order = order,
            verbose = verbose)
    else
        W, R = solve_rom(model, spectral, cfg; order = order, verbose = verbose)
    end
    t_solve = time() - t0
    stats = engine_stats(engine)
    @printf("  solved in %.2f s: %d tangent jobs (+ %d for K), backend time %.2f s\n", t_solve,
        stats["jobs_total"], stats["jobs_k0"], stats["time_backend_s"])
    if counts !== nothing
        println("  jobs per order: ", join(["O$(c.order): $(c.new_jobs)" for c in counts], ", "))
    end

    extra = Pair{String, Any}["time_solve_s" => round(t_solve; digits = 3),
        "prefetch" => prefetch]
    if counts !== nothing
        push!(extra, "jobs_per_order" => join(["$(c.order):$(c.new_jobs)" for c in counts], " "))
    end

    ok = true
    tol = opts["tol"]
    if opts["reference"]
        kind = pyconvert(String, engine.backend.kind)
        ref_terms = kind == "analytic" ? :exact : :backend_exact
        m_ref = build_model(engine, cfg; terms = ref_terms, matrices = mats, forcing = forcing)
        Wr, Rr = solve_rom(m_ref, spectral, cfg; order = order, verbose = false)
        byd = rom_rel_diff_by_degree(W, R, Wr, Rr)
        dR = maximum(x -> x.dR, byd)
        dW = maximum(x -> x.dW, byd)
        println("  reference (", ref_terms, " closures), relative difference per degree:")
        for x in byd
            @printf("    degree %d: R %.3e   W %.3e\n", x.degree, x.dR, x.dW)
        end
        push!(extra, "reference_terms" => ref_terms, "reference_dR" => dR, "reference_dW" => dW)
        if one_pair && forcing !== nothing
            amps = forced_response_amplitudes(Rr; length = 401)
            d = forced_response_rel_diff(R, Rr; amplitudes = amps)
            @printf("  frequency response: difference relative to the largest detuning = %.3e (%d amplitude(s) near the peak not matched)\n",
                d.diff, d.unmatched)
            push!(extra, "reference_frf" => d.diff, "reference_frf_unmatched" => d.unmatched)
            ok &= d.diff <= tol
        elseif one_pair
            amps = backbone_amplitudes(engine, sp, cfg.masters[1])
            dB = backbone_rel_diff(R, Rr; amplitudes = amps)
            b = normal_form_branch(R; amplitudes = amps)
            @printf("  backbone up to ρ = %.4g (mode amplitude d_ref): ω from %.9g to %.9g; ",
                last(amps), b.frequency[1], b.frequency[end])
            @printf("difference relative to the shift = %.3e\n", dB)
            push!(extra, "reference_backbone" => dB, "backbone_rho_max" => last(amps),
                "backbone_omega_0" => b.frequency[1], "backbone_omega_end" => b.frequency[end])
            ok &= dB <= tol
        end
        ok &= dR <= tol && dW <= tol
        println(ok ? "  reference check PASSED" : "  reference check FAILED", " (tolerance ", tol, ")")
    end

    push!(extra, "time_total_s" => round(time() - t_start; digits = 3))
    export_rom(outdir, W, R, engine, cfg; spectrum = sp, jobs_per_order = counts, extra = extra)
    # the ROM in real variables with the monitor DOF, for time integration or harmonic balance
    dof = monitor_index(engine, cfg, real.(sp.eigenmodes[:, 1, 2cfg.masters[1] - 1]))
    label = cfg.monitor_node === nothing ? "dof$dof" :
            "node$(cfg.monitor_node)_$("xyz"[cfg.monitor_dof])"
    try
        export_real_rom(outdir, W, R; dofs = [dof], labels = [label])
        println("  real form of the ROM (monitor ", label, "): real_rom.npz, real_rom_f.csv, real_rom_obs.csv")
    catch exc
        exc isa ArgumentError || rethrow()
        println("  no real form of the ROM: ", exc.msg)
    end
    if one_pair && forcing !== nothing && cfg.alpha + cfg.beta > 0
        fr = forced_response(R; amplitudes = forced_response_amplitudes(R; length = 401))
        u = orbit_amplitude(W, dof, fr.amplitude, fr.phase)
        open(joinpath(outdir, "frf.csv"), "w") do io
            println(io, "rho,phase,omega,stable,sheet,monitor_amplitude")
            for i in eachindex(u)
                @printf(io, "%.17g,%.17g,%.17g,%d,%d,%.17g\n", fr.amplitude[i], fr.phase[i],
                    fr.frequency[i], fr.stable[i], fr.sheet[i], u[i])
            end
        end
        if !isempty(u)
            i = argmax(u)
            @printf("  frequency response: %d points, peak %.6g at ω = %.9g (monitor DOF %d); frf.csv\n",
                length(u), u[i], fr.frequency[i], dof)
        end
    end
    engine.save_state()
    println("  results written to ", outdir)
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS) ? 0 : 1)
end
