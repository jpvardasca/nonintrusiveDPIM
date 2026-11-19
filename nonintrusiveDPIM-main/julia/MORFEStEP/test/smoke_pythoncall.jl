# P0.6 — PythonCall smoke test (CondaPkg-managed Python).
using PythonCall
sys = pyimport("sys")
np = pyimport("numpy")
sp = pyimport("scipy")
println("python   ", pyconvert(String, sys.version))
println("exe      ", pyconvert(String, sys.executable))
println("numpy    ", pyconvert(String, np.__version__), "   scipy ", pyconvert(String, sp.__version__))
v = pyconvert(Vector{ComplexF64}, np.array([1 + 2im, 3 - 4im]))
@assert v == ComplexF64[1 + 2im, 3 - 4im]
# zero-copy wrap of a numpy complex array
a = np.arange(4) .* pycomplex(0, 1)
pa = PyArray{ComplexF64,1}(a)
@assert collect(pa) == ComplexF64[0, 1im, 2im, 3im]
# Julia -> numpy round trip of a complex vector without conjugation
x = ComplexF64[1 + 1im, 2 - 3im]
y = pyconvert(Vector{ComplexF64}, np.asarray(x) * 2)
@assert y == 2 .* x
println("P0.6 OK: ", v)
