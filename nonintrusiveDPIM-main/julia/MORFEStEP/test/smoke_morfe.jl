# P0.5 — MORFE smoke test: README Duffing example (2 DOF, order 5, complex normal form).
using MORFE

K = [2.0 -1.0; -1.0 2.0]
M = [1.0 0.0; 0.0 1.0]
C = 0.001 * M
cubic = MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), (3, 0);
    fully_asymmetric = false)
model = NthOrderModel((K, C, M), (cubic,))
spec = spectrum(model)
sd = SpectralData(model, spec; master = master_by_sorting(2))
W, R = parametrise(model, sd, 5;
    resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05))

Rc = R.poly.coefficients
ex = R.poly.multiindex_set.exponents
println("typeof(R.poly.coefficients) = ", typeof(Rc), " size ", size(Rc))
println("typeof(W.poly.coefficients) = ", typeof(W.poly.coefficients), " size ", size(W.poly.coefficients))
println("exponents type ", typeof(ex))
for (j, e) in enumerate(eachcol(ex isa AbstractMatrix ? ex : reduce(hcat, ex)))
    any(!iszero, Rc[:, j]) && println("  exponent ", Tuple(e), "  R = ", Rc[:, j])
end
# backbone term: z1^2 z2 in row 1 (exponent (2,1))
j21 = findfirst(e -> Tuple(e) == (2, 1), collect(eachcol(ex isa AbstractMatrix ? ex : reduce(hcat, ex))))
@assert j21 !== nothing "no (2,1) monomial"
@assert abs(Rc[1, j21]) > 1e-8 "cubic backbone term is zero"
println("P0.5 OK: cubic backbone coefficient R[1,(2,1)] = ", Rc[1, j21])
