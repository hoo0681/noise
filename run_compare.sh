#!/bin/bash
# Waits for the GPUs to be free of other training runs, then runs the three track 1 configurations one after another.
TORCHRUN=/home/seunghoo/workspace/noise/.venv/bin/torchrun
busy() { pgrep -f "train/(val|speedrun)/run.py|analysis/|experiments/" | grep -v $$ > /dev/null; }
for name in cutbi cutbir cutbir_full; do
  while busy; do sleep 20; done
  case $name in
    cutbi) args="--estimator cutbi" ;;
    cutbir) args="--estimator cutbir" ;;
    cutbir_full) args="--estimator cutbir --route-k 2048" ;;
  esac
  echo "$(date +%T) start $name: $args"
  $TORCHRUN --standalone --nproc_per_node=8 train/val/run.py $args --output logs/$name > logs_$name.out 2>&1
  echo "$(date +%T) done $name: $(cat logs/$name/result.json 2>/dev/null | tr -d '\n ')"
done
