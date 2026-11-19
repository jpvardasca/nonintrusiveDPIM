"""P8.5: SpoolRunner (client) and scripts/abaqus_worker.py (host side), with the mock abaqus.

The worker runs as a separate process, exactly as it would on the Abaqus host; the
"shared folder" is a temporary directory.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from decks import backend, model
from nistep.backends.abaqus import AbaqusBackend
from nistep.runners.base import RunnerError
from nistep.runners.spool import SpoolRunner

WORKER = Path(__file__).resolve().parents[2] / "scripts" / "abaqus_worker.py"
MOCK = [sys.executable, "-m", "nistep.runners.mock_abaqus"]


class WorkerProcess:
    def __init__(self, tmp_path, mode="", parallel=2, extra=(), utf8=True):
        self.spool = tmp_path / "spool"
        self.work = tmp_path / "hostwork"
        env = dict(os.environ)
        env["NISTEP_MOCK_MODE"] = mode
        flags = []
        if not utf8:  # the host's Python as it is by default on Windows: locale encoding
            env.pop("PYTHONUTF8", None)
            flags = ["-X", "utf8=0"]
        self.log = open(tmp_path / "worker.log", "w")
        self.proc = subprocess.Popen(
            [sys.executable, *flags, str(WORKER), "--spool", str(self.spool), "--work", str(self.work),
             "--mass-lib", str(tmp_path / "uel" / "mass"), "--parallel", str(parallel),
             "--poll", "0.05", "--heartbeat", "0.2", "--idle-exit", "120", *extra,
             "--command", MOCK[0], *flags, *MOCK[1:]],
            env=env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 30
        while not (self.spool / "worker.alive").exists():
            assert self.proc.poll() is None, "worker exited at start-up"
            assert time.time() < deadline, "worker did not start"
            time.sleep(0.05)

    def stop(self):
        if self.proc.poll() is None:
            (self.spool / "stop").write_text("stop\n")
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.log.close()

    def kill(self):
        self.proc.kill()
        self.proc.wait(timeout=10)
        self.log.close()


@pytest.fixture
def worker(tmp_path):
    w = WorkerProcess(tmp_path)
    yield w
    w.stop()


def client(spool, **kw):
    kw.setdefault("poll_s", 0.05)
    kw.setdefault("heartbeat_stale_s", 20.0)
    kw.setdefault("retries", 0)
    return SpoolRunner(spool, **kw)


def rand_u(be, seed, scale=0.05):
    return np.random.default_rng(seed).uniform(-scale, scale, size=be.n_dof)


def test_spool_roundtrip(worker, tmp_path):
    runner = client(worker.spool, max_parallel=2)
    ab = AbaqusBackend(model("single"), runner)
    ref = backend("single")
    Us = [rand_u(ab, s) for s in range(4)]
    Ks = ab.tangent_many(Us)
    for U, K in zip(Us, Ks):
        assert np.array_equal(K.data, ref.tangent(U).data)
    assert np.array_equal(ab.mass().data, ref.mass().data)  # mass library on the host
    assert runner.n_jobs_run == 5
    # nothing is left behind in the shared folder
    for sub in ("inbox", "running", "outbox"):
        assert not list((worker.spool / sub).iterdir()), sub
    assert not list(worker.work.iterdir())  # successful job directories removed on the host
    assert not list(worker.spool.rglob("*.tmp"))


def test_spool_roundtrip_without_utf8_mode_on_the_host(tmp_path):
    """The deck (with its UTF-8 'ν') must reach the host's Abaqus byte for byte."""
    w = WorkerProcess(tmp_path, utf8=False, extra=("--keep", "all"))
    try:
        ab = AbaqusBackend(model("single"), client(w.spool))
        U = rand_u(ab, 1)
        assert np.array_equal(ab.tangent(U).data, backend("single").tangent(U).data)
        decks = list(w.work.glob("*/*.inp"))
        assert len(decks) == 1 and b"\xce\xbd" in decks[0].read_bytes()
    finally:
        w.stop()


def test_spool_failure_reports_host_log(tmp_path):
    w = WorkerProcess(tmp_path, mode="fail")
    try:
        ab = AbaqusBackend(model("single"), client(w.spool))
        with pytest.raises(RunnerError) as exc:
            ab.stiffness0()
        assert "exit code 1" in exc.value.reason and "mock failure" in exc.value.log_tail
        assert exc.value.work_dir and Path(exc.value.work_dir).is_dir()  # kept on the host
        assert not list((w.spool / "outbox").iterdir())  # the failed result was consumed
    finally:
        w.stop()


def test_worker_down_is_detected(tmp_path):
    runner = client(tmp_path / "spool", heartbeat_stale_s=1.0)
    ab = AbaqusBackend(model("single"), runner)
    t0 = time.time()
    with pytest.raises(RunnerError, match="worker down"):
        ab.stiffness0()
    assert time.time() - t0 < 15
    assert not list((tmp_path / "spool" / "inbox").iterdir())  # unclaimed jobs withdrawn


def test_worker_killed_mid_job_then_restarted(tmp_path):
    w = WorkerProcess(tmp_path, mode="sleep=30", parallel=1)
    runner = client(w.spool, heartbeat_stale_s=2.0)
    ab = AbaqusBackend(model("single"), runner)
    job = ab.make_job(np.zeros(ab.n_dof))
    import threading

    errors = []
    t = threading.Thread(target=lambda: errors.append(_run_expect_error(runner, job)))
    t.start()
    deadline = time.time() + 30
    while not list((w.spool / "running").glob("*.json")):
        assert time.time() < deadline, "the worker never claimed the job"
        time.sleep(0.05)
    w.kill()  # crash while the job is running
    t.join(timeout=30)
    assert errors and "worker down" in errors[0]
    assert list((w.spool / "running").glob("*.json"))  # the claimed job is still there
    # a new worker puts the job back in the inbox and runs it; the result is then reused
    w2 = WorkerProcess(tmp_path, parallel=1)
    try:
        deadline = time.time() + 60
        while not (w2.spool / "outbox" / f"{job.name}.json").exists():
            assert time.time() < deadline, "the restarted worker did not finish the job"
            time.sleep(0.1)
        res = runner.run([job])
        assert runner.n_reused == 1 and "** ELEMENT NUMBER" in res[0].mtx_text
    finally:
        w2.stop()


def _run_expect_error(runner, job):
    try:
        runner.run([job])
    except RunnerError as exc:
        return str(exc)
    return "no error"


def test_stop_file_ends_worker(tmp_path):
    w = WorkerProcess(tmp_path)
    (w.spool / "stop").write_text("stop\n")
    assert w.proc.wait(timeout=20) == 0
    w.log.close()
    assert "done, 0 ok, 0 failed" in (tmp_path / "worker.log").read_text()
