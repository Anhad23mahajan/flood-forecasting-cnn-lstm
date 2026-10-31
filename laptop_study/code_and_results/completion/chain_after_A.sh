#!/bin/bash
cd /c/Dev/flood-forecast-demo
until grep -q "QUEUE FINISHED" completion/logs/queue_A.log; do sleep 30; done
venv/Scripts/python.exe -u completion/run_queue.py BL K S E B FX OPT C S2 B2 D F --workers 4 --threads 3 > completion/logs/queue_rest.log 2>&1
