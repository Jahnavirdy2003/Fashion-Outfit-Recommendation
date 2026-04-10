#!/bin/bash
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