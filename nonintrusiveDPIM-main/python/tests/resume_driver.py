"""Driver of the kill-and-resume test (``test_resume.py``); run as a separate process.

    python resume_driver.py <config.toml> <number of basis vectors> <output.npz>

Evaluates all polarised forms of a fixed random basis with the engine described by the
config (which has a disk cache and a slow backend) and writes them to the output file.
"""

import sys

import numpy as np

import nistep
from nistep.basis_mode import polarised_forms


def basis(n_dof: int, n_vec: int) -> np.ndarray:
    return np.random.default_rng(5).normal(size=(n_dof, n_vec))


def forms_as_arrays(G: dict, H: dict) -> dict:
    arrays = {"G_%d_%d" % k: v for k, v in G.items()}
    arrays.update({"H_%d_%d_%d" % k: v for k, v in H.items()})
    return arrays


def main(config: str, n_vec: str, out: str) -> int:
    engine = nistep.build_engine(config)
    G, H = polarised_forms(engine, basis(engine.n, int(n_vec)), prefetch=True, max_parallel=2)
    stats = engine.stats()
    np.savez(out, jobs_total=np.int64(stats["jobs_total"]), jobs_k0=np.int64(stats["jobs_k0"]),
             **forms_as_arrays(G, H))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
