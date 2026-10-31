"""Resume-safe job queue: runs the engine / baseline jobs listed for a named batch with N parallel workers.
Usage: python completion/run_queue.py <batch> [--workers 4] [--threads 3]
A job whose result file already exists is skipped, so the queue can be stopped and restarted at any time (or after the laptop sleeps)."""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
os.chdir(PROJ)
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

PY = os.path.join("venv", "Scripts", "python.exe")
EXPS = ("E480", "E60", "E30")
REGS = ("hybrid", "causal", "forced")
FOLDS = (0, 1, 2)
CFG = dict(epochs=30, n_train=300)


def rid(exp, arch, regime, variant="full", fold=0, K=3, seed=None, epochs=30, augment=False, blind=False, blocked=False, batch=1, lr=1e-3):
    seed = 1000 + 10 * fold + K if seed is None else seed
    return (f"{exp}_{arch}_{regime}_{variant}{'_aug' if augment else ''}{'_blind' if blind else ''}{'_blk' if blocked else ''}"
            f"{'_b%d' % batch if batch > 1 else ''}{'_lr%g' % lr if lr != 1e-3 else ''}_f{fold}of{K}_s{seed}_ep{epochs}")


def neural(exp, arch, regime, fold, variant="full", blocked=False, ablate=True, batch=1, lr=1e-3):
    r = rid(exp, arch, regime, variant, fold, epochs=CFG["epochs"], blocked=blocked, batch=batch, lr=lr)
    cmd = ([PY, "-u", "completion/engine.py", "run", exp, regime, str(fold), "--arch", arch, "--variant", variant, "--epochs", str(CFG["epochs"]),
            "--n-train", str(CFG["n_train"])] + (["--blocked"] if blocked else []) + ([] if ablate else ["--no-ablate"])
           + (["--batch", str(batch)] if batch > 1 else []) + (["--lr", str(lr)] if lr != 1e-3 else []))
    return dict(name=r, out=f"completion/results/runs/{r}.json", cmd=cmd)


def reverse(exp, arch, regime, fold):
    """Reverse-direction transfer (completion/reverse_transfer.py). The '--tag engine.py run' argument is ignored by the script; it only makes older queue runners count the job."""
    r = rid(exp, arch, regime, "full", fold, epochs=CFG["epochs"])
    cmd = [PY, "-u", "completion/reverse_transfer.py", "run", exp, regime, str(fold), "--arch", arch, "--epochs", str(CFG["epochs"]), "--n-train", str(CFG["n_train"]), "--tag", "engine.py run"]
    return dict(name=r, out=f"completion/results/runs/{r}.json", cmd=cmd)


def reverse_baseline(exp, regime, fold):
    o = f"completion/results/baselines_{exp}_{regime}_f{fold}of3_n{CFG['n_train']}.json"
    return dict(name=f"base_{exp}_{regime}_f{fold}", out=o, cmd=[PY, "-u", "completion/reverse_transfer.py", "baseline", exp, regime, str(fold), "--n-train", str(CFG["n_train"]), "--tag", "engine.py run"])


def baseline(exp, regime, fold, blocked=False):
    o = f"completion/results/baselines_{exp}_{regime}_f{fold}of3_n{CFG['n_train']}{'_blk' if blocked else ''}.json"
    return dict(name=f"base_{exp}_{regime}_f{fold}{'_blk' if blocked else ''}", out=o,
                cmd=[PY, "-u", "completion/b_baselines.py", exp, regime, str(fold), "--n-train", str(CFG["n_train"])] + (["--blocked"] if blocked else []))


def severity(regime, head, fold, seed=None, augment=False):
    seed = 2000 + 10 * fold + 3 if seed is None else seed
    r = f"SEV_{regime}_{head}{'_aug' if augment else ''}_f{fold}of3_s{seed}_ep30"
    return dict(name=r, out=f"completion/results/severity/{r}.json",
                cmd=[PY, "-u", "completion/severity_study.py", regime, head, str(fold)] + ([] if seed == 2000 + 10 * fold + 3 else ["--seed", str(seed)]) + (["--augment"] if augment else []))


def batch(name):
    J = []
    if name == "S":       # severity / rules / Grad-CAM (Australia 30 m): capsule vs dense head, hybrid regime, 3 folds, + a second seed for stability (TRIMMED: forced regime dropped)
        J += [severity("hybrid", h, f) for h in ("capsule", "dense") for f in FOLDS]
        J += [severity("hybrid", "capsule", 0, seed=9), severity("hybrid", "dense", 0, seed=9)]
        return J
    if name == "BL":      # baselines re-run WITH the clipped-linear and per-pixel-MLP rungs (S4); the first linear-only files were archived
        return [baseline(e, r, f) for e in EXPS for r in REGS for f in FOLDS]
    if name == "K":       # spatial-context ablation: kernel 1 (per pixel) / 5 (default local) / 11, hybrid
        return [neural(e, a, "hybrid", f) for e in EXPS for f in FOLDS for a in ("local1", "local11")]
    if name == "OPT":     # optimisation robustness (a DEVIATION from the frozen recipe, labelled as such): batch 8, lr 3e-3
        return [neural(e, a, "hybrid", f, batch=8, lr=3e-3) for e in EXPS for f in FOLDS for a in ("local", "clstm") if not (a == "clstm" and e != "E60")]   # TRIMMED: clstm on E60 only
    if name == "FX":      # ConvLSTM with the future rain as SIDE channels (forced regime only): tests the input-layout explanation of the clstm forced deficit (code review F4)
        return [neural(e, "clstmf", "forced", f) for e in ("E60", "E480") for f in FOLDS]      # TRIMMED: no 30 m
    if name == "FXR":     # RETRY (post hoc, disclosed) of side-channel runs whose frozen-recipe seed ended in a DEAD NETWORK (guard): same fold, seed + 1; reported separately, never merged
        return [dict(name=f"{e}_clstmf_forced_retry_f{f}", out=f"completion/results/runs/{rid(e, 'clstmf', 'forced', 'full', f, seed=1000 + 10 * f + 4, epochs=30)}.json",
                     cmd=[PY, "-u", "completion/engine.py", "run", e, "forced", str(f), "--arch", "clstmf", "--epochs", "30", "--n-train", "300", "--seed", str(1000 + 10 * f + 4)])
                for e, f in (("E60", 1),)]
    if name == "CG":      # POST HOC (disclosed): original model (v1) runs that the dead-network guard aborted under the frozen recipe, finished WITHOUT the guard to obtain a score (labelled '_noguard'; never merged into a headline group)
        J = []
        for e in EXPS:
            for f in FOLDS:
                base = neural(e, "v1", "hybrid", f)
                lf = f"completion/logs/runs/{base['name']}.log"
                if not os.path.exists(base["out"]) and os.path.exists(lf) and "DEAD NETWORK after epoch " in open(lf, errors="replace").read():
                    r = base["name"].replace("_full_", "_full_noguard_", 1)
                    J.append(dict(name=r, out=f"completion/results/runs/{r}.json", cmd=base["cmd"] + ["--no-guard"]))
        return J
    if name == "FXG":     # DIAGNOSTIC (post hoc): the side-channel run whose frozen seed the guard aborted (E60 fold 1, epoch 13), finished WITHOUT the guard - was it a real collapse or a false alarm / transient?
        r = "E60_clstmf_forced_full_noguard_f1of3_s1013_ep30"
        return [dict(name=r, out=f"completion/results/runs/{r}.json",
                     cmd=[PY, "-u", "completion/engine.py", "run", "E60", "forced", "1", "--arch", "clstmf", "--epochs", "30", "--n-train", "300", "--no-guard"])]
    if name == "FXR2":    # THIRD attempt (post hoc, disclosed) for the same fold after the retry with another seed also collapsed: seed + 2 and a halved learning rate (5e-4)
        return [dict(name=f"{e}_clstmf_forced_retry2_f{f}", out=f"completion/results/runs/{rid(e, 'clstmf', 'forced', 'full', f, seed=1000 + 10 * f + 5, epochs=30, lr=5e-4)}.json",
                     cmd=[PY, "-u", "completion/engine.py", "run", e, "forced", str(f), "--arch", "clstmf", "--epochs", "30", "--n-train", "300", "--seed", str(1000 + 10 * f + 5), "--lr", "0.0005"])
                for e, f in (("E60", 1),)]
    # ---- the cells that the first plan TRIMMED (marked above) were reinstated on 20 September 19:40 as separate batches, so that nothing in the plan is dropped:
    if name == "LC":      # learning curve of the local CNN: how many training windows are needed? (n_train 25/50/100/600 besides the frozen 300; hybrid; a distinct seed per size so the run ids differ)
        return [dict(name=f"{e}_local_hybrid_n{n}_f{f}", out=f"completion/results/runs/{rid(e, 'local', 'hybrid', 'full', f, seed=4000 + 100 * k + 10 * f + 3, epochs=30)}.json",
                     cmd=[PY, "-u", "completion/engine.py", "run", e, "hybrid", str(f), "--arch", "local", "--epochs", "30", "--n-train", str(n), "--seed", str(4000 + 100 * k + 10 * f + 3), "--no-ablate"])
                for e in EXPS for f in FOLDS for k, n in enumerate((25, 50, 100, 600))]
    if name == "LCB":     # the same learning curve for the baseline ladder (lookup table, per-pixel MLP, linear models)
        return [dict(name=f"base_{e}_hybrid_f{f}_n{n}", out=f"completion/results/baselines_{e}_hybrid_f{f}of3_n{n}.json",
                     cmd=[PY, "-u", "completion/b_baselines.py", e, "hybrid", str(f), "--n-train", str(n)]) for e in EXPS for f in FOLDS for n in (25, 50, 100, 600)]
    if name == "EP":      # training length: the local CNN for 90 instead of 30 epochs (is it under-trained?)
        return [dict(name=f"{e}_local_hybrid_ep90_f{f}", out=f"completion/results/runs/{rid(e, 'local', 'hybrid', 'full', f, epochs=90)}.json",
                     cmd=[PY, "-u", "completion/engine.py", "run", e, "hybrid", str(f), "--arch", "local", "--epochs", "90", "--n-train", "300", "--no-ablate"])
                for e in EXPS for f in FOLDS]
    if name == "EXS":     # is the strict-split failure of the plain CNN on Pakistan (fold 0, E480) a property of the data (flood onset never seen) or of one training run? 4 more seeds + the batch-8 / lr 3e-3 recipe
        J = [dict(name=f"E480_local_hybrid_blk_seed{sd}_f0", out=f"completion/results/runs/{rid('E480', 'local', 'hybrid', 'full', 0, seed=sd, epochs=30, blocked=True)}.json",
                  cmd=[PY, "-u", "completion/engine.py", "run", "E480", "hybrid", "0", "--arch", "local", "--epochs", "30", "--n-train", "300", "--blocked", "--seed", str(sd), "--no-ablate"]) for sd in (11, 12, 13, 14)]
        r = rid("E480", "local", "hybrid", "full", 0, epochs=30, blocked=True, batch=8, lr=3e-3)
        J.append(dict(name=r, out=f"completion/results/runs/{r}.json",
                      cmd=[PY, "-u", "completion/engine.py", "run", "E480", "hybrid", "0", "--arch", "local", "--epochs", "30", "--n-train", "300", "--blocked", "--batch", "8", "--lr", "0.003", "--no-ablate"]))
        return J
    if name == "LC2":     # code review round 2b F1: the learning curve fixed 30 epochs, so the number of optimiser steps grew with n. These cells hold the STEPS equal while n changes:
        #                      n = 600 for 15 epochs (9,000 steps, same as n = 300 for 30 epochs) and n = 300 for 60 epochs (18,000 steps, same as n = 600 for 30 epochs)
        return [dict(name=f"{e}_local_hybrid_n{n}_ep{ep}_f{f}", out=f"completion/results/runs/{rid(e, 'local', 'hybrid', 'full', f, seed=4000 + 100 * k + 10 * f + 3, epochs=ep)}.json",
                     cmd=[PY, "-u", "completion/engine.py", "run", e, "hybrid", str(f), "--arch", "local", "--epochs", str(ep), "--n-train", str(n), "--seed", str(4000 + 100 * k + 10 * f + 3), "--no-ablate"])
                for e in EXPS for f in FOLDS for k, (n, ep) in ((4, (600, 15)), (5, (300, 60)))]
    if name == "EPC":     # fairness check for the recurrence claim: the ConvLSTM ALSO trained for 90 epochs (E60 and E480; the local CNN gained +0.01..+0.02 from 90 epochs)
        return [dict(name=f"{e}_clstm_hybrid_ep90_f{f}", out=f"completion/results/runs/{rid(e, 'clstm', 'hybrid', 'full', f, epochs=90)}.json",
                     cmd=[PY, "-u", "completion/engine.py", "run", e, "hybrid", str(f), "--arch", "clstm", "--epochs", "90", "--n-train", "300", "--no-ablate"])
                for e in ("E60", "E480") for f in FOLDS]
    if name == "RB":      # reverse-direction transfers (EXPLORATORY): baseline ladder, all regimes
        return [reverse_baseline(e, r, f) for e in ("R480", "R60", "R30") for r in REGS for f in FOLDS]
    if name == "RN":      # reverse-direction transfers: local CNN in all three regimes, ConvLSTM in the hybrid regime
        return ([reverse(e, "local", r, f) for e in ("R480", "R60", "R30") for r in REGS for f in FOLDS]
                + [reverse(e, "clstm", "hybrid", f) for e in ("R480", "R60", "R30") for f in FOLDS])
    if name == "SX":      # severity under the CAUSAL regime (no look-ahead rain in the rule/capsule inputs; closes code-review finding F8)
        return [severity("causal", h, f) for h in ("capsule", "dense") for f in FOLDS]
    if name == "SXF":     # severity under the FORCED regime (rain of the forecast window known)
        return [severity("forced", h, f) for h in ("capsule", "dense") for f in FOLDS]
    if name == "AX":      # the 30 m recurrent model in the causal and forced regimes (slowest cells of the matrix)
        return [neural("E30", "clstm", r, f) for r in ("causal", "forced") for f in FOLDS]
    if name == "BX":      # training-time ablations of the local CNN at 30 m
        return [neural("E30", "local", "hybrid", f, v) for f in FOLDS for v in ("nodepthhist", "nohist", "norain", "norainhist", "noterrain")]
    if name == "EX":      # strict space + time hold-out for the ConvLSTM at 30 m
        return [neural("E30", "clstm", "hybrid", f, blocked=True) for f in FOLDS]
    if name == "FXX":     # side-channel ConvLSTM at 30 m
        return [neural("E30", "clstmf", "forced", f) for f in FOLDS]
    if name == "OPTX":    # optimisation robustness for the ConvLSTM at 30 m and 480 m
        return [neural(e, "clstm", "hybrid", f, batch=8, lr=3e-3) for e in ("E480",) for f in FOLDS]      # E30 omitted: same storm pair as E60
    if name == "S2":      # severity with augmentation (does it help the capsule / dense head?)
        return [severity("hybrid", h, f, augment=True) for h in ("capsule", "dense") for f in FOLDS]
    if name == "F":       # seed repeats of the cheap local model (separates seed noise from fold effects): 4 extra seeds, hybrid
        return [dict(name=f"{e}_local_hybrid_seed{sd}_f{f}", out=f"completion/results/runs/{rid(e, 'local', 'hybrid', 'full', f, seed=sd, epochs=30)}.json",
                     cmd=[PY, "-u", "completion/engine.py", "run", e, "hybrid", str(f), "--arch", "local", "--epochs", "30", "--n-train", "300", "--seed", str(sd), "--no-ablate"])
                for e in EXPS for f in FOLDS for sd in (11, 12, 13, 14)]
    if name == "A":       # local + clstm on the matrix (the baselines are batch BL)
        J += [neural(e, "local", r, f) for e in EXPS for r in REGS for f in FOLDS]
        # TRIMMED: the 30 m recurrent model is run in the hybrid regime only (the slowest cells; E30 repeats the E60 storm pair at a finer resolution)
        J += [neural(e, "clstm", r, f) for e in EXPS for r in REGS for f in FOLDS if not (e == "E30" and r != "hybrid")]
    elif name == "B":     # training-time ablations with the cheap local model (TRIMMED: E60 and E480, hybrid regime only)
        for e in ("E60", "E480"):
            for f in FOLDS:
                for v in ("nodepthhist", "nohist", "norain", "norainhist", "noterrain"):
                    J.append(neural(e, "local", "hybrid", f, v))
    elif name == "B2":    # training-time ablations with the recurrent model (hybrid only)
        J += [neural(e, "clstm", "hybrid", f, v) for e in EXPS for f in FOLDS for v in ("nodepthhist", "norain")]
    elif name == "C":     # the original model under the same folds / recipe (hybrid)
        J += [neural(e, "v1", "hybrid", f) for e in EXPS for f in FOLDS]
    elif name == "E":     # strict space + time hold-out
        J += [baseline(e, "hybrid", f, blocked=True) for e in EXPS for f in FOLDS]
        J += [neural(e, a, "hybrid", f, blocked=True) for e in EXPS for f in FOLDS for a in ("local", "clstm") if not (a == "clstm" and e != "E60")]   # TRIMMED: clstm on E60 only
    elif name == "E2":    # added after the strict hold-out exposed a failure of the local CNN on Pakistan: the ConvLSTM under the same strict split (E480)
        J += [neural("E480", "clstm", "hybrid", f, blocked=True) for f in FOLDS]
    elif name == "D":     # heavy comparator (653k parameters; does capacity matter?), all folds and experiments, hybrid (reinstated: it was first planned as one fold per experiment)
        J += [neural(e, "v2lin", "hybrid", f) for e in ("E60", "E480") for f in FOLDS]      # E30 omitted: same storm pair as E60 (not independent evidence)
    else:
        raise SystemExit("unknown batch " + name)
    return J


def count_running():
    """Number of study jobs currently alive (one venv launcher process per job), whoever started them."""
    import subprocess as sp
    cmd = ("@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.ExecutablePath -like '*venv*' -and "
           "$_.CommandLine -match 'engine.py run|severity_study.py|b_baselines.py|reverse_transfer.py' }).Length")
    try:
        out = sp.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out or 0)
    except Exception:
        print("[warning] could not count running jobs; assuming the cap is reached", flush=True)
        return 999            # never 0: an unreadable count must not let the runner start every queued job at once (code review round 2b, F3)


def log_active(j, minutes=12):
    lf = f"completion/logs/runs/{j['name']}.log"
    return os.path.exists(lf) and (time.time() - os.path.getmtime(lf)) < minutes * 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    jobs = []
    for b in a.batches:
        jobs += batch(b)
    def aborted(j):
        """A job whose last log shows a dead-network abort is a RECORDED abort (all 8 so far came from the old check that looked at the 12 earliest, near-static held-out windows and were false alarms):
        it is not re-run automatically; the batches CG / FXG re-run chosen ones with the guard disabled."""
        lf = f"completion/logs/runs/{j['name']}.log"
        if "--no-guard" in j["cmd"] or not os.path.exists(lf):
            return False
        with open(lf, errors="replace") as fh:
            return "DEAD NETWORK after epoch " in fh.read()

    todo = [j for j in jobs if not os.path.exists(j["out"]) and not aborted(j)]
    n_ab = sum(1 for j in jobs if not os.path.exists(j["out"]) and aborted(j))
    print(f"{len(jobs)} jobs in batches {a.batches}: {len(jobs) - len(todo) - n_ab} already done, {n_ab} previously aborted by the dead-network guard (recorded, not re-run), {len(todo)} to run", flush=True)
    os.makedirs("completion/logs/runs", exist_ok=True)
    running, failed = [], []
    env = dict(os.environ, TF_THREADS=str(a.threads))
    while todo or running:
        while todo and max(count_running(), sum(1 for it in running if it[1].poll() is None)) < a.workers:
            j = todo.pop(0)
            if os.path.exists(j["out"]) or log_active(j):        # finished meanwhile, or already running from an earlier runner
                continue
            lf = open(f"completion/logs/runs/{j['name']}.log", "w")
            running.append((j, subprocess.Popen(j["cmd"], stdout=lf, stderr=subprocess.STDOUT, env=env), lf, time.time()))
            print(f"[start] {j['name']}  ({len(todo)} queued)", flush=True)
        time.sleep(5)
        for item in list(running):
            j, p, lf, t0 = item
            if p.poll() is not None:
                lf.close()
                running.remove(item)
                ok = os.path.exists(j["out"])
                print(f"[{'done' if ok else 'FAILED'}] {j['name']}  rc={p.returncode}  {(time.time() - t0) / 60:.1f} min", flush=True)
                if not ok:
                    failed.append(j["name"])
    print("QUEUE FINISHED; failed:", failed, flush=True)


if __name__ == "__main__":
    main()
