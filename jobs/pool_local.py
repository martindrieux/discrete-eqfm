"""Job pool for a shared 8-GPU node (fal-h100): trains and evaluates the tasks of a TSV file, one GPU slot each.

    python jobs/pool_local.py jobs/tasks_conf.tsv          (in the background; log: $DEQ_ROOT/pool.log)

Tasks file: name <tab> per_gpu <tab> train command <tab> eval command; commands run from the repository root with
{run} replaced by the run directory. `per_gpu` tasks share one GPU (small toy models). A task is done when
runs/NAME/DONE exists; training resumes from last.pt (train.resume=true), so a stopped task loses little.

Rules of the node (shared with other users):
    - at least 4 GPUs stay unused: this pool holds at most 8 - 4 - (GPUs used by others) GPUs, and never more than 4;
    - when another user starts a new GPU process, every task of this pool is stopped at once (SIGTERM: checkpoint,
      exit 3). Stopped tasks start again only when the rule above leaves room.
"""
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ME = os.environ.get("USER", "martin")
LOG = os.path.join(os.environ.get("DEQ_ROOT", ROOT), "pool.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def gpu_state():
    """(index -> set of foreign pids, set of indices with any process)."""
    idx = dict(l.split(", ") for l in subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"], capture_output=True, text=True).stdout.split("\n") if l)
    apps = [l.split(", ") for l in subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.split("\n") if l]
    foreign, busy = {}, set()
    for uuid, pid in apps:
        i = int(idx[uuid])
        busy.add(i)
        user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if user and user != ME:
            foreign.setdefault(i, set()).add(int(pid))
    return foreign, busy


def read_tasks(path):
    tasks = []
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            name, per_gpu, train, ev = line.rstrip("\n").split("\t")
            tasks.append({"name": name, "per_gpu": int(per_gpu), "train": train, "eval": ev})
    return tasks


def main():
    path = sys.argv[1]
    running = {}                 # name -> (Popen, gpu)
    seen = None                  # foreign pids at the previous poll
    log(f"pool started on {path}")
    while True:
        tasks = read_tasks(path)                 # re-read: tasks can be added while the pool runs
        foreign, busy = gpu_state()
        pids = set().union(*foreign.values()) if foreign else set()
        if seen is not None and pids - seen and running:
            log(f"new process of another user ({sorted(pids - seen)}): stopping all {len(running)} tasks")
            for name, (p, g) in running.items():
                os.killpg(p.pid, signal.SIGTERM)
            for name, (p, g) in running.items():
                try:
                    p.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                log(f"{name}: stopped (exit {p.returncode})")
            running = {}
            seen = pids
            time.sleep(30)
            continue
        seen = pids
        for name, (p, g) in list(running.items()):
            if p.poll() is not None:
                done = os.path.exists(f"{ROOT}/runs/{name}/DONE")
                log(f"{name}: exited {p.returncode} on GPU {g}" + (", done" if done else ", NOT done"))
                if not done:
                    open(f"{ROOT}/runs/{name}/FAILED", "a").close()
                del running[name]
        todo = [t for t in tasks if t["name"] not in running and not os.path.exists(f"{ROOT}/runs/{t['name']}/DONE")
                and not os.path.exists(f"{ROOT}/runs/{t['name']}/FAILED")]
        if not todo and not running:
            log("all tasks finished")
            return
        mine = {g for _, g in running.values()}
        cap = min(4, 8 - 4 - len(set(foreign)))          # GPUs this pool may hold
        free = [i for i in range(8) if i not in busy and i not in mine]
        for t in todo:
            load = {g: sum(1 for n, (_, gg) in running.items() if gg == g) for g in mine}
            share = [g for g in sorted(mine) if g not in foreign and load[g] < t["per_gpu"]
                     and all(next(x for x in tasks if x["name"] == n)["per_gpu"] == t["per_gpu"]
                             for n, (_, gg) in running.items() if gg == g)]
            if share:
                g = share[0]
            elif len(mine) < cap and free:
                g = free.pop(0)
            else:
                continue
            os.makedirs(f"{ROOT}/runs/{t['name']}", exist_ok=True)
            run = f"runs/{t['name']}"
            cmd = (f"set -e; {t['train'].format(run=run)}; {t['eval'].format(run=run)}; touch {run}/DONE")
            p = subprocess.Popen(["bash", "-c", cmd], cwd=ROOT, start_new_session=True,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)},
                                 stdout=open(f"{ROOT}/{run}/stdout.log", "a"), stderr=subprocess.STDOUT)
            running[t["name"]] = (p, g)
            mine.add(g)
            log(f"{t['name']}: started on GPU {g}")
        time.sleep(30)


if __name__ == "__main__":
    main()
