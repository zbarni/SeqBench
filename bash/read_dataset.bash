#!/bin/bash

# Activate your venv!

root=$(dirname $(dirname $0))
echo "Detected workflow_root: $root"

# ========== CONFIG

config=$root"/examples/configs/onehot_raw.yaml"

# ========== SCRIPT

export PYTHONPATH=$root/src:$PYTHONPATH

python3 -B $root/src/seqbench/read_dataset.py --config $config
