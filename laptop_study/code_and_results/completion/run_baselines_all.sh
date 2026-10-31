#!/bin/bash
cd /c/Dev/flood-forecast-demo
PYV=venv/Scripts/python.exe
SELFTEST_EXPS=E480,E60,E30,R480 $PYV -u completion/engine.py selftest > completion/logs/selftest_full.log 2>&1
for e in E480 E60 E30; do for r in hybrid causal forced; do for f in 0 1 2; do
  [ -f completion/results/baselines_${e}_${r}_f${f}of3.json ] || $PYV -u completion/b_baselines.py $e $r $f > completion/logs/baseline_${e}_${r}_f${f}.log 2>&1
done; done; done
echo ALLDONE > completion/logs/baselines_done.flag
