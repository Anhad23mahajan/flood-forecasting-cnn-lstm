"""Reproducibility check: re-run one stored training job from scratch (same seed, same code) into a scratch folder and compare with the stored result.
Purpose: shows how much of a result's last digits is run-to-run noise on this machine (TensorFlow CPU threading is not bit-deterministic), independent of the seed repeats.
Usage: python completion/repro_check.py [exp] [arch] [regime] [fold]     (default: E480 local hybrid 0; ~5 minutes; writes completion/results/repro_check.json)
Run it only when a queue slot is free (it is one more training job)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402  (chdirs to the project root)

exp, arch, regime, fold = (sys.argv[1:5] + ["E480", "local", "hybrid", "0"][len(sys.argv[1:5]):])[:4]
fold = int(fold)
seed = 1000 + 10 * fold + 3
rid = f"{exp}_{arch}_{regime}_full_f{fold}of3_s{seed}_ep30"
stored = json.load(open(os.path.join("completion", "results", "runs", rid + ".json")))
E.RUNS = tempfile.mkdtemp(prefix="repro_")
new = E.run(exp, regime, fold, K=3, epochs=30, seed=None, n_train=300, n_test_ablate=False, arch=arch, log=lambda s: print(s, flush=True))
res = {"id": rid, "stored_test_skill": stored["test"]["skill"], "rerun_test_skill": new["test"]["skill"],
       "stored_heldout_skill": stored["heldout"]["skill"], "rerun_heldout_skill": new["heldout"]["skill"],
       "abs_diff_test": abs(stored["test"]["skill"] - new["test"]["skill"]), "abs_diff_heldout": abs(stored["heldout"]["skill"] - new["heldout"]["skill"])}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join("completion", "results", "repro_check.json"), "w"), indent=1)
