"""Run one Abaqus job in its own directory and check that it really succeeded.

Standard library only, Python >= 3.8, and no package-relative imports: this file is used
both by ``nistep.runners.local`` and by the stand-alone host script
``scripts/abaqus_worker.py`` (which may run under an old Python on the Abaqus machine).

A job counts as successful only if **all** of these hold (legacy bugs B3 and B4):

1. the Abaqus command exits with code 0;
2. ``<job>.sta`` contains the success marker;
3. the element-matrix file ``<matrix_file>.mtx`` exists;
4. it holds exactly ``n_elements`` element blocks.

Nothing from an earlier attempt is ever reused: stale outputs are deleted before a run.
"""

import os
import shutil
import subprocess
import time

SUCCESS_MARKER = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"
ELEMENT_MARKER = "** ELEMENT NUMBER"
LOG_SUFFIXES = (".msg", ".dat", ".log", ".sta")


def build_command(command, name, user_source=None, extra_args=()):
    """Argument list of the Abaqus call for job ``name`` (run inside the job directory)."""
    cmd = list(command) + ["job=" + name, "input=" + name + ".inp", "double=both",
                           "output_precision=full", "interactive", "ask_delete=OFF", "cpus=1"]
    if user_source:
        cmd.append("user=" + str(user_source))
    cmd.extend(extra_args)
    return cmd


def write_job_files(job_dir, name, inp_text, lib_dir=None):
    """Write ``<name>.inp`` and, with a precompiled UEL library, ``abaqus_v6.env``.

    ``inp_text`` is text (written as UTF-8) or bytes (written unchanged). Nothing here may
    depend on the locale encoding: the repository decks contain non-ASCII characters in
    comments, and the default encoding on Windows (cp1252) cannot write them.
    """
    os.makedirs(job_dir, exist_ok=True)
    data = inp_text if isinstance(inp_text, bytes) else inp_text.encode("utf-8")
    with open(os.path.join(job_dir, name + ".inp"), "wb") as fh:
        fh.write(data)
    env_file = os.path.join(job_dir, "abaqus_v6.env")
    if lib_dir:
        with open(env_file, "w", encoding="utf-8") as fh:
            fh.write("usub_lib_dir = r'%s'\n" % lib_dir)
    elif os.path.exists(env_file):
        os.remove(env_file)


def clean_job_dir(job_dir, name, matrix_file):
    """Delete outputs of an earlier attempt, so a stale file can never be read."""
    if not os.path.isdir(job_dir):
        return
    for fn in os.listdir(job_dir):
        if fn.startswith(name + ".") or fn == matrix_file + ".mtx" or fn.endswith(".lck"):
            try:
                os.remove(os.path.join(job_dir, fn))
            except OSError:
                pass


def count_elements(mtx_path):
    n = 0
    with open(mtx_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.lstrip().upper().startswith(ELEMENT_MARKER):
                n += 1
    return n


def tail(path, n_lines=25):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])


def log_tail(job_dir, name, output="", n_lines=25):
    parts = []
    if output and output.strip():
        parts.append("--- command output ---\n" + "\n".join(output.splitlines()[-n_lines:]))
    for suffix in LOG_SUFFIXES:
        text = tail(os.path.join(job_dir, name + suffix), n_lines)
        if text.strip():
            parts.append("--- %s%s ---\n%s" % (name, suffix, text))
    return "\n".join(parts)


def _kill_tree(proc):
    """Kill the process and its children (Abaqus starts solver processes of its own)."""
    try:
        if os.name == "nt":
            subprocess.call(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            import signal

            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def run_job(job_dir, name, inp_text, matrix_file, n_elements, command=("abaqus",), lib_dir=None,
            user_source=None, timeout_s=900.0, extra_args=(), success_marker=SUCCESS_MARKER,
            env=None):
    """Run the job once. Returns a dict with ``status`` ("ok" or "failed"), ``reason``,
    ``returncode``, ``wall_s``, ``mtx_path`` and ``log_tail``. Never raises for a failed job."""
    job_dir = str(job_dir)
    clean_job_dir(job_dir, name, matrix_file)
    write_job_files(job_dir, name, inp_text, lib_dir)
    cmd = build_command(command, name, user_source, extra_args)
    mtx_path = os.path.join(job_dir, matrix_file + ".mtx")
    result = {"status": "failed", "reason": "", "returncode": None, "wall_s": 0.0,
              "mtx_path": mtx_path, "log_tail": "", "command": " ".join(cmd)}
    t0 = time.time()
    output = ""
    try:
        kwargs = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        # shell=True on Windows because "abaqus" is a .bat file there
        proc = subprocess.Popen(cmd, cwd=job_dir, shell=(os.name == "nt"), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, **kwargs)
        try:
            output, _ = proc.communicate(timeout=timeout_s)
            result["returncode"] = proc.returncode
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            try:
                output, _ = proc.communicate(timeout=30)
            except (subprocess.TimeoutExpired, OSError, ValueError):
                output = ""
            result["reason"] = "timeout after %g s" % timeout_s
    except OSError as exc:
        result["reason"] = "cannot start %r: %s" % (cmd[0], exc)
    result["wall_s"] = time.time() - t0

    if not result["reason"]:
        sta = os.path.join(job_dir, name + ".sta")
        if result["returncode"] != 0:
            result["reason"] = "exit code %s" % result["returncode"]
        elif not os.path.isfile(sta):
            result["reason"] = "no status file %s.sta" % name
        elif success_marker.upper() not in tail(sta, 10 ** 9).upper():
            result["reason"] = "status file lacks %r" % success_marker
        elif not os.path.isfile(mtx_path):
            result["reason"] = "no element-matrix file %s.mtx" % matrix_file
        else:
            n = count_elements(mtx_path)
            if n != n_elements:
                result["reason"] = "matrix file has %d element blocks, expected %d" % (n, n_elements)
    if result["reason"]:
        result["log_tail"] = log_tail(job_dir, name, output)
    else:
        result["status"] = "ok"
    return result


def run_job_with_retries(retries=1, **kwargs):
    """Call :func:`run_job` up to ``retries + 1`` times. Adds ``attempts`` to the result."""
    result = None
    for attempt in range(1, int(retries) + 2):
        result = run_job(**kwargs)
        result["attempts"] = attempt
        if result["status"] == "ok":
            break
    return result


def remove_job_dir(job_dir):
    shutil.rmtree(str(job_dir), ignore_errors=True)
