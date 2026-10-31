#!/bin/bash
# Wait until every v1 job of batch C has either produced a result or ended (log older than 8 minutes, no result), then queue the no-guard reruns (batch CG) for the aborted ones.
cd /c/Dev/flood-forecast-demo || exit 1
while true; do
  n=0; pending=0
  for e in E480 E60 E30; do
    for f in 0 1 2; do
      s=$((1000 + 10 * f + 3))
      id="${e}_v1_hybrid_full_f${f}of3_s${s}_ep30"
      if [ -f "completion/results/runs/${id}.json" ]; then n=$((n+1)); continue; fi
      lf="completion/logs/runs/${id}.log"
      if [ -f "$lf" ] && grep -q "DEAD NETWORK after epoch" "$lf"; then n=$((n+1)); continue; fi
      pending=$((pending+1))
    done
  done
  if [ "$pending" -eq 0 ]; then break; fi
  sleep 120
done
echo "all v1 jobs decided at $(date +%H:%M); starting CG"
venv/Scripts/python.exe -u completion/run_queue.py CG --workers 4 --threads 3
echo "CG runner ended at $(date +%H:%M)"
