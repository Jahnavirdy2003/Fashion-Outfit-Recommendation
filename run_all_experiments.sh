#!/bin/bash

# Backup any existing experiments before starting fresh
if [ -d "experiments" ] && [ "$(ls -A experiments 2>/dev/null)" ]; then
    mkdir -p experiments/backups
    backup_name="experiments/backups/backup_$(date +%Y%m%d_%H%M%S).zip"
    echo "Backing up existing experiments to $backup_name..."
    zip -r "$backup_name" experiments/ -x "experiments/backups/*"
    echo "Backup saved! Clearing old experiments (keeping backups)..."
    # Remove everything except backups folder
    find experiments -maxdepth 1 -mindepth 1 ! -name "backups" -exec rm -rf {} +
fi
echo "Running sanity check first..."
python src/experiment.py --name sanity_check --outfits 100 --epochs 1 --lr 1e-4

if [ $? -ne 0 ]; then
    echo "SANITY CHECK FAILED! Fix errors before running full experiments."
    exit 1
fi

echo "Sanity check passed! Cleaning up..."
rm -rf experiments/sanity_check/

echo "Starting all experiments..."
echo "=========================="

echo "[1/6] Baseline..."
python src/experiment.py --name exp1_baseline_5k_unfrozen --outfits 5000 --epochs 5 --lr 1e-4

echo "[2/6] Frozen backbone..."
python src/experiment.py --name exp2_frozen_backbone --outfits 5000 --epochs 5 --lr 1e-4 --freeze

echo "[3/6] More data (10k)..."
python src/experiment.py --name exp3_more_data_10k --outfits 10000 --epochs 5 --lr 1e-4

echo "[4/6] Lower learning rate..."
python src/experiment.py --name exp4_lower_lr_1e5 --outfits 5000 --epochs 5 --lr 1e-5

echo "[5/6] Higher dropout..."
python src/experiment.py --name exp5_high_dropout --outfits 5000 --epochs 5 --lr 1e-4 --dropout 0.5

echo "[6/6] Best combo (frozen + 10k + low LR)..."
python src/experiment.py --name exp6_best_combo --outfits 10000 --epochs 5 --lr 1e-5 --freeze

echo "=========================="
echo "All experiments done! Generating comparison..."
python src/experiment.py --compare

echo "Check the experiments/ folder for all results!"