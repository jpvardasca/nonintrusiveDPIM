#!/usr/bin/env python
"""Abaqus worker: runs on the Abaqus host and serves jobs sent through a shared folder.

Standard library only; works with Python 3.8 or newer. Start it from the copy of this
repository on the host (it loads ``python/nistep/runners/abaqus_exec.py`` by path)::

    python scripts/abaqus_worker.py --spool <shared folder> --work C:\\nistep\\work
        --stiffness-lib <dir of the compiled stiffness UEL>
        --mass-lib <dir of the compiled mass UEL> --parallel 2

The Abaqus command defaults to ``abaqus``. To use another one, give it as the LAST option:
``--command <path to the abaqus launcher>``.

Without precompiled libraries, pass the Fortran sources instead (slower: Abaqus compiles
them for every job)::

        --stiffness-source <path>/uel_flat.for --mass-source <path>/uel_mass_flat.for

Protocol (see ``python/nistep/runners/spool.py``):

* the client writes ``inbox/<job>.inp`` and then ``inbox/<job>.json``;
* the worker claims a job by renaming the ``.json`` into ``running/``, runs Abaqus in
  ``<work>/<job>/``, then writes ``outbox/<job>.mtx`` and, last, ``outbox/<job>.json``;
* ``worker.alive`` is rewritten every few seconds (heartbeat);
* a file named ``stop`` in the spool folder makes the worker exit after the current jobs.

Stop it with Ctrl+C. Jobs found in ``running/`` at start-up (left by a crash) are put back
in the inbox.
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python", "nistep", "runners"))
import abaqus_exec  # noqa: E402  (stdlib-only module, loaded without importing nistep)


def atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


class Worker:
    def __init__(self, args):
        self.args = args
        self.spool = os.path.abspath(args.spool)
        self.inbox = os.path.join(self.spool, "inbox")
        self.running = os.path.join(self.spool, "running")
        self.outbox = os.path.join(self.spool, "outbox")
        self.work = os.path.abspath(args.work)
        for d in (self.inbox, self.running, self.outbox, self.work):
            os.makedirs(d, exist_ok=True)
        self.lib = {"stiffness": args.stiffness_lib, "mass": args.mass_lib}
        self.source = {"stiffness": args.stiffness_source, "mass": args.mass_source}
        self.stop = threading.Event()
        self.active = 0
        self.lock = threading.Lock()
        self.n_done = 0
        self.n_failed = 0

    # ---------------------------------------------------------------- heartbeat
    def heartbeat_loop(self):
        path = os.path.join(self.spool, "worker.alive")
        counter = 0
        while not self.stop.is_set():
            counter += 1
            try:
                atomic_write(path, "%d %.3f pid=%d\n" % (counter, time.time(), os.getpid()))
            except OSError:
                pass
            self.stop.wait(self.args.heartbeat)

    # --------------------------------------------------------------------- jobs
    def recover(self):
        """Put jobs left in running/ by a crashed worker back in the inbox."""
        for fn in sorted(os.listdir(self.running)):
            if fn.endswith(".inp"):
                os.replace(os.path.join(self.running, fn), os.path.join(self.inbox, fn))
        for fn in sorted(os.listdir(self.running)):
            if fn.endswith(".json"):
                os.replace(os.path.join(self.running, fn), os.path.join(self.inbox, fn))

    def claim(self, limit):
        """Claim at most ``limit`` jobs (oldest first) and return their names."""
        try:
            metas = [fn for fn in os.listdir(self.inbox) if fn.endswith(".json")]
        except OSError:
            return []
        metas.sort(key=lambda fn: os.path.getmtime(os.path.join(self.inbox, fn))
                   if os.path.exists(os.path.join(self.inbox, fn)) else 0.0)
        claimed = []
        for fn in metas:
            if len(claimed) >= limit:
                break
            name = fn[:-5]
            try:  # atomic: only one worker wins
                os.replace(os.path.join(self.inbox, fn), os.path.join(self.running, fn))
            except OSError:
                continue
            try:
                os.replace(os.path.join(self.inbox, name + ".inp"),
                           os.path.join(self.running, name + ".inp"))
            except OSError:
                self.finish(name, {"status": "failed", "reason": "job deck missing in inbox",
                                   "log_tail": "", "wall_s": 0.0, "attempts": 0}, counted=False)
                continue
            claimed.append(name)
        return claimed

    def process(self, name):
        try:
            with open(os.path.join(self.running, name + ".json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            with open(os.path.join(self.running, name + ".inp"), "rb") as fh:
                inp_text = fh.read()  # bytes: the deck reaches Abaqus exactly as it was sent
            library = meta.get("library", "stiffness")
            job_dir = os.path.join(self.work, name)
            result = abaqus_exec.run_job_with_retries(
                retries=int(meta.get("retries", self.args.retries)), job_dir=job_dir, name=name,
                inp_text=inp_text, matrix_file=meta["matrix_file"],
                n_elements=int(meta["n_elements"]), command=self.args.command,
                lib_dir=self.lib.get(library), user_source=self.source.get(library),
                timeout_s=float(meta.get("timeout_s", self.args.timeout)))
            if result["status"] == "ok":
                tmp = os.path.join(self.outbox, name + ".mtx.tmp")
                shutil.copyfile(result["mtx_path"], tmp)
                os.replace(tmp, os.path.join(self.outbox, name + ".mtx"))
                if self.args.keep != "all":
                    abaqus_exec.remove_job_dir(job_dir)
            else:
                result["work_dir"] = job_dir
                if self.args.keep == "none":
                    abaqus_exec.remove_job_dir(job_dir)
        except Exception as exc:  # never let one job kill the worker
            result = {"status": "failed", "reason": "worker error: %r" % (exc,), "log_tail": "",
                      "wall_s": 0.0, "attempts": 0}
        self.finish(name, result)

    def finish(self, name, result, counted=True):
        out = dict((k, result.get(k)) for k in
                   ("status", "reason", "returncode", "wall_s", "attempts", "log_tail", "work_dir"))
        out["name"] = name
        atomic_write(os.path.join(self.outbox, name + ".json"), json.dumps(out))  # written last
        for suffix in (".json", ".inp"):
            try:
                os.remove(os.path.join(self.running, name + suffix))
            except OSError:
                pass
        with self.lock:
            if counted:
                self.active -= 1
            if result["status"] == "ok":
                self.n_done += 1
            else:
                self.n_failed += 1
        print("%s  %-14s %-6s %6.1f s  %s" % (time.strftime("%H:%M:%S"), name, result["status"],
                                              result.get("wall_s") or 0.0, result.get("reason") or ""))
        sys.stdout.flush()

    # --------------------------------------------------------------------- loop
    def run(self):
        self.recover()
        beat = threading.Thread(target=self.heartbeat_loop, daemon=True)
        beat.start()
        print("abaqus_worker: spool=%s work=%s parallel=%d command=%s"
              % (self.spool, self.work, self.args.parallel, " ".join(self.args.command)))
        sys.stdout.flush()
        idle_since = time.time()
        try:
            with ThreadPoolExecutor(max_workers=self.args.parallel) as pool:
                while not self.stop.is_set():
                    if os.path.exists(os.path.join(self.spool, "stop")):
                        break
                    with self.lock:
                        free = self.args.parallel - self.active
                    names = self.claim(free) if free > 0 else []
                    for name in names:
                        with self.lock:
                            self.active += 1
                        pool.submit(self.process, name)
                    with self.lock:
                        busy = self.active > 0
                    if names or busy:
                        idle_since = time.time()
                    elif self.args.once or (self.args.idle_exit and
                                            time.time() - idle_since > self.args.idle_exit):
                        break
                    time.sleep(self.args.poll)
        except KeyboardInterrupt:
            print("abaqus_worker: interrupted")
        finally:
            self.stop.set()
        print("abaqus_worker: done, %d ok, %d failed" % (self.n_done, self.n_failed))
        return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Serve Abaqus jobs from a shared spool folder.")
    p.add_argument("--spool", required=True, help="shared folder (inbox/, running/, outbox/)")
    p.add_argument("--work", required=True, help="local folder for the job directories (short path)")
    p.add_argument("--command", nargs=argparse.REMAINDER, default=None,
                   help="Abaqus executable and leading arguments (default: abaqus). "
                        "Must be the LAST option: everything after it is the command.")
    p.add_argument("--stiffness-lib", default=None, help="directory of the compiled stiffness UEL")
    p.add_argument("--mass-lib", default=None, help="directory of the compiled mass UEL")
    p.add_argument("--stiffness-source", default=None, help="Fortran source used as user= (no library)")
    p.add_argument("--mass-source", default=None, help="Fortran source of the mass UEL")
    p.add_argument("--parallel", type=int, default=1, help="concurrent Abaqus jobs (licence tokens)")
    p.add_argument("--timeout", type=float, default=900.0, help="seconds per attempt (default)")
    p.add_argument("--retries", type=int, default=1, help="extra attempts after a failure (default)")
    p.add_argument("--poll", type=float, default=0.5, help="seconds between inbox scans")
    p.add_argument("--heartbeat", type=float, default=2.0, help="seconds between heartbeats")
    p.add_argument("--keep", choices=("failed", "all", "none"), default="failed",
                   help="which job directories to keep")
    p.add_argument("--once", action="store_true", help="exit when the inbox is empty")
    p.add_argument("--idle-exit", type=float, default=0.0,
                   help="exit after this many idle seconds (0 = never)")
    args = p.parse_args(argv)
    if not args.command:
        args.command = ["abaqus"]
    if args.parallel < 1:
        p.error("--parallel must be >= 1")
    return Worker(args).run()


if __name__ == "__main__":
    sys.exit(main())
