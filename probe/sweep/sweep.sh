#!/bin/bash
# Runs each line of a config file: "<name> <gpus> <args...>", one after another; appends results to sweep.tsv.
TORCHRUN=/home/seunghoo/workspace/noise/.venv/bin/torchrun
while read -r name gpus args; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  [ -f logs/sweep/$name/result.json ] && continue
  $TORCHRUN --standalone --nproc_per_node=$gpus train/val/run.py --estimator probe $args --output logs/sweep/$name \
    > logs/sweep_$name.out 2>&1
  printf '%s\t%s\t%s\n' "$name" "$args" "$(tr -d '\n ' < logs/sweep/$name/result.json 2>/dev/null)" >> sweep.tsv
done < "$1"
