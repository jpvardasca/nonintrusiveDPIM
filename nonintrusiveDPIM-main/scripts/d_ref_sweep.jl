# Find the usable range of d_ref for a model (PLAN.md P10.1).
#
# For an exactly cubic internal force the EED-StEP results do not depend on d_ref. What
# depends on it is the error: the rounding of the tangent matrices is divided by c (quadratic
# force) and by c² (cubic force), c = eps·d_ref, so a small d_ref amplifies it, and a large
# d_ref distorts the elements. This script evaluates, at each d_ref of a list,
#
#   1. the forces Ĝ(φ,φ) and Ĥ(φ,φ,φ) of the first master mode φ (two tangent jobs per d_ref),
#      with the smallest det F of the two prescribed states;
#   2. the ROM at order 3 and its cubic normal-form coefficient (no further jobs when there
#      is one master pair).
#
# Usage (from the repository folder `nonintrusiveDPIM-main/`):
#
#   julia --project=julia/MORFEStEP scripts/d_ref_sweep.jl configs/<model>.toml \
#         --d-refs 0.1,1,5,10,20,40 [options]
#
# Options:
#   --d-refs LIST       comma-separated values of d_ref (required)
#   --order N           order of the ROM in part 2 (default 3; 0 skips part 2)
#   --det-f-min X       refuse a d_ref whose prescribed states have min det F <= X
#                       (default 0.2; a negative number switches the refusal off)
#   --reference-config  config of an exact twin of the model, used for the "err" columns.
#                       Not needed for analytic, pysvk and abaqus backends: analytic and
#                       pysvk are exact themselves, and for abaqus the numpy element (PySVK)
#                       is built from the same deck.
#   --no-rom-reference  do not solve the reference ROM in part 2
#   --json FILE         save both tables as JSON
#
# With the abaqus backend, set [cache] dir in the config: then a second run of the sweep
# (or a later ROM run at one of these d_ref) reuses the jobs.
#
# Run Julia with one thread: the Python engine must be called from the main thread.

using MORFE
using MORFEStEP
using PythonCall
using Printf

function parse_args(args)
    opts = Dict{String, Any}("d_refs" => nothing, "order" => 3, "det_f_min" => 0.2,
        "reference_config" => nothing, "rom_reference" => true, "json" => nothing)
    config = nothing
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--d-refs"
            opts["d_refs"] = [parse(Float64, x) for x in split(args[i + 1], ',')]
            i += 1
        elseif a == "--order"
            opts["order"] = parse(Int, args[i + 1])
            i += 1
        elseif a == "--det-f-min"
            opts["det_f_min"] = parse(Float64, args[i + 1])
            i += 1
        elseif a == "--reference-config"
            opts["reference_config"] = args[i + 1]
            i += 1
        elseif a == "--no-rom-reference"
            opts["rom_reference"] = false
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
    (config === nothing || opts["d_refs"] === nothing) &&
        error("usage: d_ref_sweep.jl <config.toml> --d-refs a,b,c [--order N] [--det-f-min X] " *
              "[--reference-config other.toml] [--no-rom-reference] [--json FILE]")
    return config, opts
end

_num(x) = isnan(x) ? "         -" : @sprintf("%10.3e", x)
_json(x) = isnan(x) ? nothing : x    # JSON has no NaN

function main(args)
    config, opts = parse_args(args)
    engine = load_engine(config)
    cfg = run_config(engine)
    d_refs = opts["d_refs"]
    println("d_ref_sweep: ", cfg.name, "  (", cfg.path, ")")

    mats = build_matrices(engine, cfg.alpha, cfg.beta)
    model = build_model(engine, cfg; terms = :engine, matrices = mats)
    _, sp = build_spectral(model, cfg)
    p = cfg.masters[1]
    φ = real.(sp.eigenmodes[:, 1, 2p - 1])
    @printf("  DOFs = %d, mode %d: ω = %.9g, max|φ| = %.4g (mass-normalised)\n", model.n_fom, p,
        abs(sp.eigenvalues[2p - 1]), maximum(abs, φ))

    # ---- part 1: forces of the mode -------------------------------------------------
    diagnostics = pyimport("nistep.diagnostics")
    ref_engine = opts["reference_config"] === nothing ? nothing :
                 load_engine(opts["reference_config"])
    reference = ref_engine === nothing ? pybuiltins.None : ref_engine.backend
    det_f_min = opts["det_f_min"] < 0 ? pybuiltins.None : opts["det_f_min"]
    report = diagnostics.d_ref_sweep(engine, φ, pylist(d_refs); reference = reference,
        det_f_min = det_f_min)
    println(pyconvert(String, diagnostics.format_d_ref_sweep(report)))
    ran = [pyconvert(Float64, r["d_ref"]) for r in report["rows"]
           if pyconvert(String, r["status"]) == "ok"]

    # ---- part 2: the ROM ---------------------------------------------------------------
    rows = nothing
    order = opts["order"]
    if order >= 3 && !isempty(ran)
        ref = if !opts["rom_reference"]
            nothing
        elseif ref_engine !== nothing
            rcfg = run_config(ref_engine)
            m = build_model(ref_engine, rcfg; terms = :backend_exact)
            s, _ = build_spectral(m, rcfg)
            solve_rom(m, s, rcfg; order = order)
        else
            :auto
        end
        rows = rom_d_ref_sweep(engine, ran; order = order, reference = ref, cfg = cfg)
        println("ROM at order ", order, ": coefficient of z² z̄ (its imaginary part is the slope ",
            "of the backbone, ω ≈ Im λ + Im(coefficient) ρ²)")
        println("     d_ref  jobs      Re(coef)       Im(coef)     change      error     dR max")
        for r in rows
            if r.status === :ok
                @printf("%10.4g %5d %14.6e %14.6e %s %s %s\n", r.d_ref, r.jobs,
                    real(r.coefficient), imag(r.coefficient), _num(r.change), _num(r.error),
                    _num(r.dR))
            else
                @printf("%10.4g %5d   FAILED: %s\n", r.d_ref, r.jobs, r.reason)
            end
        end
        println("'change' compares a row with the previous one that ran; 'error' and 'dR max' ",
            "compare with the ROM of the exact closures.")
    end
    engine.save_state()

    if opts["json"] !== nothing
        out = pydict()
        out["config"] = cfg.path
        out["forces"] = report
        if rows !== nothing
            out["rom_order"] = order
            out["rom"] = pylist([pydict(Dict{String, Any}(
                                     "d_ref" => r.d_ref, "status" => String(r.status),
                                     "reason" => r.reason, "jobs" => r.jobs,
                                     "coefficient_re" => _json(real(r.coefficient)),
                                     "coefficient_im" => _json(imag(r.coefficient)),
                                     "change" => _json(r.change), "error" => _json(r.error),
                                     "dR" => _json(r.dR)))
                                 for r in rows])
        end
        json = pyimport("json")
        write(opts["json"], pyconvert(String, json.dumps(out; indent = 2)))
        println("  tables written to ", opts["json"])
    end
    failed = any(pyconvert(String, r["status"]) == "failed" for r in report["rows"]) ||
             (rows !== nothing && any(r -> r.status === :failed, rows))
    return !failed
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS) ? 0 : 1)
end
