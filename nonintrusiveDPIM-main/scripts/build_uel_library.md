# Setting up the Abaqus machine

This guide prepares the machine that runs Abaqus so it can serve tangent stiffness matrices
to the ROM pipeline. Do it once per machine. Steps marked **VERIFY** depend on your Abaqus
installation and have not been tested yet; note what you find in `PLAN.md` (P8).

## 1. Prerequisites

- [ ] A full Abaqus/Standard installation. The Learning Edition cannot run user subroutines.
- [ ] A Fortran compiler linked to Abaqus (Intel Fortran, plus Visual Studio on Windows).
- [ ] `abaqus verify -user_std` passes.
- [ ] A copy of this repository with `test/` and `module/` side by side. The UEL sources in
      `test/` include files from `../module/`.
- [ ] Python 3.8 or newer for the worker (only the standard library is used). If the machine
      has no Python, try `abaqus python scripts/abaqus_worker.py ...` (**VERIFY**: Abaqus
      2024 ships Python 3.10; older releases ship Python 2.7, which will not work).

## 2. Build the two UEL libraries (recommended)

A precompiled library avoids recompiling the Fortran for every job (the legacy pipeline
spent most of its 19 s per job on that).

Run from inside `test/`, so the `include '../module/...'` lines resolve:

```bat
cd <repo>\test
mkdir C:\nistep\uel_lib\stiffness C:\nistep\uel_lib\mass

abaqus make library=uel_nlmech_pk1.for directory=C:\nistep\uel_lib\stiffness
abaqus make library=uel_nlmech_pk1_mass_matrix.for directory=C:\nistep\uel_lib\mass
```

- [ ] **VERIFY** that each folder now holds the library (`standardU.dll` on Windows,
      `libstandardU.so` on Linux).
- [ ] Keep the two libraries in **separate folders**, and keep the word `mass` in the name of
      the mass folder. Each job selects its library through an `abaqus_v6.env` file in the job
      directory containing `usub_lib_dir = r'<folder>'`.

### Fallback: no library

If `abaqus make` does not work, write single-file copies of the sources (the originals are
not changed) and let Abaqus compile them per job:

```bat
python scripts\make_flat_uel.py test\uel_nlmech_pk1.for C:\nistep\uel_flat\uel_flat.for
python scripts\make_flat_uel.py test\uel_nlmech_pk1_mass_matrix.for C:\nistep\uel_flat\uel_mass_flat.for
```

Then use `user_source` / `mass_source` in the config (or `--stiffness-source` /
`--mass-source` for the worker) instead of the library folders.

## 3. Choose how jobs reach Abaqus

**A. Everything on the Abaqus machine** (`runner = "local"`). Install Julia 1.11 and the
project there (see `PLAN.md` P0), set `user_lib_dir` and `mass_lib_dir` in the config, and run
the pipeline on that machine.

**B. Pipeline on another PC, Abaqus behind a shared folder** (`runner = "spool"`). Both
machines must see one folder (a network share). On the Abaqus machine start the worker and
leave it running:

```bat
python scripts\abaqus_worker.py --spool \\server\share\nistep_spool --work C:\nistep\work ^
    --stiffness-lib C:\nistep\uel_lib\stiffness --mass-lib C:\nistep\uel_lib\mass --parallel 2
```

- `--parallel` is the number of Abaqus jobs run at once. Each job needs licence tokens.
- `--work` should be a short local path (Windows limits path lengths).
- If Abaqus is not started by the command `abaqus`, add `--command <launcher>` as the
  **last** option.
- Stop the worker with Ctrl+C, or by creating a file named `stop` in the spool folder.

On the other PC set, in the config: `runner = "spool"`, `spool_dir = "<the same folder>"` and
`max_parallel` equal to the worker's `--parallel`.

## 4. Smoke test (PLAN.md P8.6)

Edit `configs/single_element_abaqus.toml`, then run on the PC that drives the pipeline:

```bat
python scripts\abaqus_smoke_test.py configs\single_element_abaqus.toml
python scripts\abaqus_smoke_test.py configs\mems_abaqus.toml
```

Six jobs run per deck. The report compares Abaqus with the numpy element built from the same
deck. What to look for:

| Check | Meaning | If it fails |
|---|---|---|
| `a_K0_vs_pysvk`, `b_KU_vs_pysvk` | Abaqus and the numpy element give the same tangent | Wrong library, wrong deck, or low output precision. The value shows how many digits agree. |
| `c_cubic_exactness` | `K_tg` is exactly quadratic along a line | The UEL is not polynomial. The likely cause is the uninitialised `fbarFlag` (bug B9 in `PLAN.md`). **Ask before changing the Fortran.** |
| `mass_vs_pysvk` | The mass library is the one being used | The mass job picked up the stiffness library, or the density was truncated (bug B10). |
| `d_K0_vs_processed` | Agreement with the matrices stored in the repository | The deck or the properties differ from the ones that produced those files. |

Record in `PLAN.md` (P8.6): the values printed, the seconds per job, and the Abaqus version.

## 5. Things that go wrong

- **`cannot start 'abaqus'`**: the launcher is not on `PATH`. Give its full path in the
  config (`command = "C:/SIMULIA/Commands/abaqus.bat"`) or with `--command`.
- **`status file lacks 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY'`**: the job ran but did not
  finish. The error message ends with the tail of the `.msg`, `.dat` and `.log` files, and the
  job folder is kept under the work folder. (**VERIFY** that your Abaqus version writes this
  exact sentence in `<job>.sta`.)
- **`the element matrices look like mass matrices ... wrong UEL library?`** (or the reverse):
  the two library folders are swapped. The check looks at the matrix values: a stiffness
  matrix maps a rigid translation to zero, a mass matrix does not.
- **`matrix type STIFFNESS, expected MASS`** (or the reverse): the label comes from the
  `*Element Matrix Output` request in the job deck (`Stiffness=Yes` or `Mass=Yes`), not
  from the library. The job decks set it for you, so this message points to an old copy of
  `nistep`: run `git pull`. (Found on the first host run, 2026-11-22.)
- **`worker down`**: the worker is not running, or the two machines do not see the same
  folder.
- **Licence errors**: lower `--parallel` / `max_parallel`.
