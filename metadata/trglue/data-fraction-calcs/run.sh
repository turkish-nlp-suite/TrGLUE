#!/usr/bin/env bash
set -euo pipefail

tasks=("cola" "sst2" "qqp" "mnli" "qnli" "stsb" "rte" "mrpc")
fractions=(0.4 0.5 0.6 0.7 0.8 0.9 1.0)
seeds=(1 2 3)

mkdir -p results

for task in "${tasks[@]}"; do
  out_csv="results/${task}.csv"
  echo "frac,metric,seed" > "$out_csv"
  for frac in "${fractions[@]}"; do
    for seed in "${seeds[@]}"; do
      # Compute subset size using total train size. If you don't know it at runtime,
      # you can pass --max_train_samples via a lookup table you prepare per task.
      # Here we illustrate with proportional sampling using datasets' select in code.
      python3 run_with_fraction.py \
        --task_name "$task" \
        --model_name_or_path dbmdz/bert-base-turkish-cased \
        --seed "$seed" \
        --train_fraction "$frac" \
        --num_train_epochs 3 \
        --learning_rate "$( [[ "$task" =~ ^(stsb|rte|mrpc)$ ]] && echo 3e-5 || echo 2e-5 )" \
        --per_device_train_batch_size "$( [[ "$task" =~ ^(stsb|rte|mrpc)$ ]] && echo 16 || echo 128 )" \
        --per_device_eval_batch_size "$( [[ "$task" =~ ^(stsb|rte|mrpc)$ ]] && echo 16 || echo 128 )" \
        --output_dir tmp_${task}_${frac}_seed${seed} \
        --do_train --do_eval --save_strategy no \
        --report_to none > /dev/null

      # After run, read eval JSON written by Trainer (eval_results.json) and extract the metric
      metric=$(python3 - <<'PY'
import json, sys, glob
import os
files = glob.glob("tmp_*/eval_results_*.json") + glob.glob("tmp_*/all_results.json") + glob.glob("tmp_*/eval_results.json")
val = None
for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
        # pick best available key
        for k in ["eval_matthews_correlation","eval_accuracy","eval_f1","eval_spearmanr","eval_pearson","eval_combined_score"]:
            if k in d:
                val = d[k]; break
        if val is not None:
            break
    except Exception:
        pass
print(val if val is not None else "nan")
PY
)
      echo "${frac},${metric},${seed}" >> "$out_csv"
      rm -rf tmp_${task}_${frac}_seed${seed}
    done
  done
done
