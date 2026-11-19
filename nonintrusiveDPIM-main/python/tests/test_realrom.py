import numpy as np
import pytest

from nistep.realrom import RealROM

# Linear forced oscillator in complex normal form, z' = lam z + a r1 with lam = -zeta + i w,
# a = -i A, z = x + i y, r1 = c + i s:
#   x' = -zeta x - w y + A s,   y' = w x - zeta y - A c;   displacement u = 2 phi x.
ZETA, W, A, PHI = 0.01, 1.3, 0.2, 0.7


def write_rom(path, forced=True):
    if forced:
        names = "x1,y1,c,s"
        exps = np.eye(4, dtype=np.int64)
        f = np.array([[-ZETA, -W, 0.0, A], [W, -ZETA, -A, 0.0]])
        disp = np.array([[2 * PHI, 0.0, 0.0, 0.0]])
        omega0 = W
    else:
        names = "x1,y1"
        exps = np.array([[1, 0], [0, 1], [3, 0]], dtype=np.int64)  # + a cubic x^3 term
        f = np.array([[-ZETA, -W, 0.5], [W, -ZETA, 0.0]])
        disp = np.array([[2 * PHI, 0.0, 0.0]])
        omega0 = np.nan
    np.savez(path, variables=np.frombuffer(names.encode(), dtype=np.uint8), exponents=exps,
             f=f, dofs=np.array([7]), displacement=disp, velocity=np.zeros((0, exps.shape[0])),
             omega0=np.float64(omega0))
    return path


def test_forced_rhs_and_observables(tmp_path):
    rom = RealROM.load(write_rom(tmp_path / "real_rom.npz"))
    assert rom.variables == ("x1", "y1", "c", "s") and rom.forced and rom.n_states == 2
    assert rom.omega0 == W and list(rom.dofs) == [7]
    q, t, om = np.array([0.3, -0.1]), 0.8, 1.25
    c, s = np.cos(om * t), np.sin(om * t)
    expect = [-ZETA * q[0] - W * q[1] + A * s, W * q[0] - ZETA * q[1] - A * c]
    assert np.allclose(rom.rhs(q, t, om), expect, rtol=1e-14)
    # the forcing level scales c and s
    expect2 = [-ZETA * q[0] - W * q[1] + 2 * A * s, W * q[0] - ZETA * q[1] - 2 * A * c]
    assert np.allclose(rom.rhs(q, t, om, level=2.0), expect2, rtol=1e-14)
    assert np.allclose(rom.displacement(q, t, om), [2 * PHI * q[0]])
    assert rom.velocity(q, t, om).shape == (0,)
    # many points at once
    Q = np.array([[0.3, 0.1, -0.2], [-0.1, 0.4, 0.0]])
    T = np.array([0.8, 1.0, 2.0])
    batch = rom.rhs(Q, T, om)
    assert batch.shape == (2, 3)
    for k in range(3):
        assert np.allclose(batch[:, k], rom.rhs(Q[:, k], T[k], om), rtol=1e-14)
    with pytest.raises(ValueError, match="omega"):
        rom.rhs(q, t)
    with pytest.raises(ValueError, match="state"):
        rom.rhs(np.zeros(3), t, om)


def test_autonomous(tmp_path):
    rom = RealROM.load(write_rom(tmp_path / "real_rom.npz", forced=False))
    assert not rom.forced and np.isnan(rom.omega0)
    q = np.array([0.5, 0.2])
    assert np.allclose(rom.rhs(q), [-ZETA * 0.5 - W * 0.2 + 0.5 * 0.5**3, W * 0.5 - ZETA * 0.2])


def test_inconsistent_file(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, variables=np.frombuffer(b"x1,y1", dtype=np.uint8), exponents=np.eye(2, dtype=np.int64),
             f=np.ones((3, 2)), dofs=np.array([1]), displacement=np.ones((1, 2)),
             velocity=np.zeros((0, 2)), omega0=np.float64(np.nan))
    with pytest.raises(ValueError, match="inconsistent"):
        RealROM.load(p)
