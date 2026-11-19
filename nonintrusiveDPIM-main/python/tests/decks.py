"""Repository decks and matrices used by the FE tests (read-only legacy files in ``test/``)."""

from functools import lru_cache
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[2] / "test"

# name -> (deck, processed stiffness, processed mass, nodes, elements, clamped, free DOFs)
DECKS = {
    "single": ("Meu_input_file.inp", "Meu_input_file_stiffness_matrix_E_210e9.mtx",
               "Meu_input_file_mass_matrix_E_210e9.mtx", 8, 1, 4, 12),
    "mems": ("mems_abaqus_UEL_modified.inp", "Memes_240_elements_stiffness_matrix.mtx",
             "Memes_240_elements_mass_matrix.mtx", 525, 240, 18, 1521),
    "cantilever": ("Cantilever_Beam_500_elements_UEL_modified.inp",
                   "Cantilever_Beam_500_elements_stiffness_matrix.mtx",
                   "Cantilever_Beam_500_elements_mass_matrix.mtx", 918, 500, 18, 2700),
    "cc": ("Clamped_Clamped_beam_200_elements_UEL_modified.inp",
           "Clamped_clamped_Beam_200_elements_stiffness_matrix.mtx",
           "Clamped_clamped_Beam_200_elements_mass_matrix.mtx", 606, 200, 12, 1782),
}


def deck_path(name: str) -> Path:
    return LEGACY / DECKS[name][0]


@lru_cache(maxsize=None)
def model(name: str):
    from nistep.inp_io import read_inp

    return read_inp(deck_path(name))


@lru_cache(maxsize=None)
def backend(name: str):
    from nistep.backends.pysvk import PySVKBackend

    return PySVKBackend(model(name))
