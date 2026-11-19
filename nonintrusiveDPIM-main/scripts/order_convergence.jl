# Convergence of the ROM with the expansion order, and its cost (PLAN.md P10.2 to P10.4).
#
# For each order of a list the ROM is solved with the EED-StEP closures, on one engine, so
# the tangent matrices of the lower orders are reused. Per order it reports
#
#   * the new tangent jobs, the cumulative jobs and the time;
#   * the backbone of the first master pair: frequency and displacement of the monitor DOF
#     at the end of the amplitude range, and the change against the previous order;
#   * the difference against the ROM of the exact closures at the same order.
#
# With [forcing] enabled (PLAN.md P11) the orders are O(q, [rom] forcing_order) and the
# frequency response curve takes the place of the backbone: the table gives the peak (its
# forcing frequency and the displacement of the monitor DOF), and the differences between
# curves are relative to the largest detuning |ω̂ − Ω| on the curve (see
# MORFEStEP.forced_response_rel_diff). --amplitude is not used then: the curves run over the
# reduced amplitudes 0 … 1.5 × the linear peak.
#
# Usage (from the repository folder `nonintrusiveDPIM-main/`):
#
#   julia --project=julia/MORFEStEP scripts/order_convergence.jl configs/<model>.toml \
#         --orders 3,5,7 [options]
#
# Options:
#   --orders LIST        comma-separated orders, ascending (required)
#   --amplitude A        the backbones are compared for mode amplitudes of the monitor DOF
#                        from 0 to A (linear estimate 2 ρ |ϕ|). Default: [step] d_ref.
#   --reference-config F config of an exact twin (a pysvk config of the same deck) for the
#                        reference ROMs. Not needed for analytic and pysvk backends.
#   --no-reference       skip the reference ROMs
#   --prefetch / --no-prefetch   as in run_rom.jl (default: [engine] prefetch of the config)
#   --json FILE          save the table and the curves (backbones, or frequency responses
#                        with phase, stability and sheet for forced configs)
#
# The monitor DOF is [output] monitor_node / monitor_dof of the config; without them it is
# the DOF where the mode is largest.
#
# Run Julia with one thread: the Python engine must be called from the main thread.

using MORFE
using MORFEStEP
using PythonCall
using Printf

function parse_args(args)
    opts = Dict{String, Any}("orders" => nothing, "amplitude" => nothing,
        "reference_config" => nothing, "reference" => true, "prefetch" => nothing,
        "json" => nothing)
    config = nothing
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--orders"
            opts["orders"] = [parse(Int, x) for x in split(args[i + 1], ',')]
            i += 1
        elseif a == "--amplitude"
            opts["amplitude"] = parse(Float64, args[i + 1])
            i += 1
        elseif a == "--reference-config"
            opts["reference_config"] = args[i + 1]
            i += 1
        elseif a == "--no-reference"
            opts["reference"] = false
        elseif a == "--prefetch"
            opts["prefetch"] = true
        elseif a == "--no-prefetch"
            opts["prefetch"] = false
        elseif a == "--json"
            opts["json"] = args[i + 1]
            i += 1
        elseif startswith(a, "--")
            error("unknown option $a")
        elseif config === nothing
            config = a
        else
            error("unexpected argument $a")
        end
        i += 1
    end
    (config === nothing || opts["orders"] === nothing) &&
        error("usage: order_convergence.jl <config.toml> --orders 3,5,7 [--amplitude A] " *
              "[--reference-config F] [--no-reference] [--prefetch] [--no-prefetch] [--json FILE]")
    issorted(opts["orders"]) && allunique(opts["orders"]) && first(opts["orders"]) >= 2 ||
        error("--orders must be ascending, without repeats, and start at 2 or more")
    return config, opts
end

_num(x) = isnan(x) ? "        -" : @sprintf("%9.2e", x)
_json(x) = isnan(x) ? nothing : x

function main(args)
    config, opts = parse_args(args)
    engine = load_engine(config)
    cfg = run_config(engine)
    orders = opts["orders"]
    length(cfg.masters) == 1 && cfg.style === :complex_normal_form ||
        error("order_convergence.jl needs one master mode and style = \"complex_normal_form\"")
    println("order_convergence: ", cfg.name, "  (", cfg.path, ")")

    mats = build_matrices(engine, cfg.alpha, cfg.beta)
    setup = setup_rom(engine, cfg; matrices = mats)
    model, spectral, sp, forcing = setup.model, setup.spectral, setup.spectrum, setup.forcing
    forced = forcing !== nothing
    p = cfg.masters[1]
    ϕ = real.(sp.eigenmodes[:, 1, 2p - 1])
    dof = monitor_index(engine, cfg, ϕ)
    d_ref = pyconvert(Float64, engine.step.d_ref)
    A = something(opts["amplitude"], d_ref)
    ρs = range(0, A / (2 * abs(ϕ[dof])); length = 101)
    @printf("  DOFs = %d, mode %d: ω = %.9g, d_ref = %g\n", model.n_fom, p,
        abs(sp.eigenvalues[2p - 1]), d_ref)
    if forced
        @printf("  forcing: %g M ϕ%d cos(Ω t), Ω = %.9g, forcing order %d; monitor: free DOF %d\n",
            cfg.forcing_amplitude, cfg.forcing_mode, forcing[2], cfg.forcing_order, dof)
    else
        @printf("  monitor: free DOF %d (ϕ = %.4g); backbone up to a mode amplitude of %g (ρ = %.4g)\n",
            dof, ϕ[dof], A, last(ρs))
    end

    # the reference model: exact closures of this backend or of the twin
    ref = nothing
    if opts["reference"]
        if opts["reference_config"] !== nothing
            reng = load_engine(opts["reference_config"])
            rcfg = run_config(reng)
            # same load and same monomial sets (solved with `cfg`) as the main model
            rmodel = build_model(reng, rcfg; terms = :backend_exact, forcing = forcing)
            rspectral, _ = build_spectral(rmodel, rcfg)
            ref = (rmodel, rspectral, cfg)
        elseif !pyis(engine.backend.exact_quad, pybuiltins.None)
            rmodel = build_model(engine, cfg; terms = :backend_exact, matrices = mats,
                forcing = forcing)
            ref = (rmodel, spectral, cfg)
        else
            println("  no reference: this backend has no exact forms (use --reference-config)")
        end
    end

    prefetch = something(opts["prefetch"], cfg.prefetch)
    rows = Dict{String, Any}[]
    curves = Dict{String, Any}[]
    previous = nothing
    stat(k) = pyconvert(Float64, engine.stats()[k])
    for q in orders
        jobs0, tb0 = stat("jobs_total"), stat("time_backend_s")
        t = @elapsed begin
            if prefetch
                W, R, _ = run_with_prefetch(model, spectral, cfg, engine; order = q)
            else
                W, R = solve_rom(model, spectral, cfg; order = q)
            end
        end
        if forced
            q == first(orders) && (ρs = forced_response_amplitudes(R; length = 401))
            fr = forced_response(R; amplitudes = ρs)
            amp_all = orbit_amplitude(W, dof, fr.amplitude, fr.phase)
            ipk = argmax(amp_all)
            ω_end, amp_end = fr.frequency[ipk], amp_all[ipk]
            shift = maximum(x -> abs(x - forcing[2]), fr.frequency)
            change = previous === nothing ? NaN :
                     forced_response_rel_diff(R, previous; amplitudes = ρs).diff
            previous = R
            curve = Dict{String, Any}("order" => q, "rho" => fr.amplitude, "phase" => fr.phase,
                "omega" => fr.frequency, "stable" => fr.stable, "sheet" => fr.sheet,
                "amplitude" => amp_all)
        else
            b = normal_form_branch(R; amplitudes = ρs)
            ω = b.frequency
            amp = physical_amplitude(W, dof, ρs)
            ω_end, amp_end = ω[end], amp[end]
            shift = maximum(abs, ω .- ω[1])
            change = previous === nothing ? NaN : maximum(abs, ω .- previous) / shift
            previous = ω
            curve = Dict{String, Any}("order" => q, "rho" => collect(ρs), "omega" => collect(ω),
                "amplitude" => amp)
        end
        dR, dW, dB, t_ref = NaN, NaN, NaN, NaN
        if ref !== nothing
            rmodel, rspectral, rcfg = ref
            t_ref = @elapsed Wr, Rr = solve_rom(rmodel, rspectral, rcfg; order = q)
            byd = filter(x -> x.degree >= 2, rom_rel_diff_by_degree(W, R, Wr, Rr))
            dR = maximum(x -> x.dR, byd)
            dW = maximum(x -> x.dW, byd)
            dB = forced ? forced_response_rel_diff(R, Rr; amplitudes = ρs).diff :
                 backbone_rel_diff(R, Rr; amplitudes = ρs)
        end
        push!(rows,
            Dict{String, Any}("order" => q, "new_jobs" => Int(stat("jobs_total") - jobs0),
                "jobs_total" => Int(stat("jobs_total")), "seconds" => t,
                "seconds_backend" => stat("time_backend_s") - tb0, "seconds_reference" => _json(t_ref),
                "omega_end" => ω_end, "amplitude_end" => amp_end, "shift" => shift,
                "change" => _json(change), "dR" => _json(dR), "dW" => _json(dW),
                "backbone_diff" => _json(dB)))
        push!(curves, curve)
    end
    engine.save_state()

    println("  order  new jobs  jobs   time s  backend s        ω(end)     u(end)     change    R diff    W diff  backbone")
    for r in rows
        nan(x) = x === nothing ? NaN : x
        @printf("  %5d  %8d  %4d  %7.2f  %9.2f  %12.9f  %9.4g  %s %s %s %s\n", r["order"],
            r["new_jobs"], r["jobs_total"], r["seconds"], r["seconds_backend"], r["omega_end"],
            r["amplitude_end"], _num(nan(r["change"])), _num(nan(r["dR"])), _num(nan(r["dW"])),
            _num(nan(r["backbone_diff"])))
    end
    if forced
        println("  ω(end), u(end): forcing frequency and displacement of the monitor DOF at the peak of the frequency response.")
        println("  change: largest frequency difference of the curve against the previous order, relative to its largest detuning |ω̂ − Ω|.")
        println("  R diff, W diff: worst relative difference per polynomial degree against the exact-closure ROM;")
        println("  backbone: the same frequency difference against the exact-closure ROM (frequency response curves).")
    else
        println("  ω(end), u(end): frequency and largest displacement of the monitor DOF at the end of the range.")
        println("  change: largest difference of the backbone against the previous order, relative to its frequency shift.")
        println("  R diff, W diff: worst relative difference per polynomial degree against the exact-closure ROM;")
        println("  backbone: difference of the two backbones relative to the frequency shift.")
    end
    s = engine_stats(engine)
    @printf("  totals: %d tangent jobs (+ %d for K), backend time %.2f s; %s\n", s["jobs_total"],
        s["jobs_k0"], s["time_backend_s"], pyconvert(String, engine.store.memory_report()))

    if opts["json"] !== nothing
        json = pyimport("json")
        out = pydict(Dict{String, Any}("config" => cfg.path, "d_ref" => d_ref, "monitor_dof" => dof,
            "amplitude" => A, "prefetch" => prefetch, "rows" => pylist([pydict(r) for r in rows]),
            "forced" => forced,
            "curves" => pylist([pydict(Dict{String, Any}(k => (v isa AbstractVector ? pylist(v) : v)
                                    for (k, v) in c)) for c in curves])))
        write(opts["json"], pyconvert(String, json.dumps(out; indent = 1)))
        println("  table and curves written to ", opts["json"])
    end
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS) ? 0 : 1)
end
