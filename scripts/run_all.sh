#!/bin/bash
# Run all experiments for Social HRL project.
# Usage: bash scripts/run_all.sh
#
# Outputs are isolated per suite under outputs/suites/<suite_id>/.

set -e

SUITE_ID="${SUITE_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
SUITE_DIR="${SUITE_DIR:-outputs/suites/${SUITE_ID}}"
RUNS_DIR="${RUNS_DIR:-${SUITE_DIR}/runs}"
TRANSFER_DIR="${TRANSFER_DIR:-${SUITE_DIR}/transfer}"
PLOTS_DIR="${PLOTS_DIR:-${SUITE_DIR}/plots}"

SEEDS="${SEEDS:-42 123 7}"
TIMESTEPS="${TIMESTEPS:-1000000}"
TRANSFER_TIMESTEPS="${TRANSFER_TIMESTEPS:-500000}"
MIN_SOURCE_SUCCESS="${MIN_SOURCE_SUCCESS:-0.5}"
DEVICE="${DEVICE:-cpu}"  # CPU is optimal for these small models + MiniGrid
TRAIN_FLAGS="${TRAIN_FLAGS:---no-wandb}"

mkdir -p "$RUNS_DIR" "$TRANSFER_DIR" "$PLOTS_DIR"

echo "============================================"
echo "Social HRL — Full Experiment Suite"
echo "Suite: $SUITE_ID"
echo "Suite dir: $SUITE_DIR"
echo "Seeds: $SEEDS"
echo "Timesteps per run: $TIMESTEPS"
echo "Transfer timesteps: $TRANSFER_TIMESTEPS"
echo "Min source success: $MIN_SOURCE_SUCCESS"
echo "Device: $DEVICE"
echo "Train flags: $TRAIN_FLAGS"
echo "============================================"

# =============================================
# Experiment 1: Original 4-way on corridor
# =============================================
echo ""
echo ">>> Experiment 1: Original 4-way corridor comparison"
echo ""

for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"

    python scripts/train.py --mode flat --corridor --seed "$SEED" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode continuous --corridor --seed "$SEED" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode discrete --corridor --seed "$SEED" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode social --seed "$SEED" \
        --eval-comm-ablation \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    wait
done

echo ">>> Experiment 1 complete."

# =============================================
# Experiment 2: Harder environments
# =============================================
echo ""
echo ">>> Experiment 2: Harder environments"
echo ""

for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"

    python scripts/train.py --mode flat --seed "$SEED" \
        --env "MiniGrid-KeyCorridorS6R3-v0" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode discrete --seed "$SEED" \
        --env "MiniGrid-KeyCorridorS6R3-v0" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode flat --seed "$SEED" \
        --env "MiniGrid-MultiRoom-N6-v0" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode discrete --seed "$SEED" \
        --env "MiniGrid-MultiRoom-N6-v0" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    wait
done

echo ">>> Experiment 2 complete."

# =============================================
# Experiment 3: Social enhancements
# =============================================
echo ""
echo ">>> Experiment 3: Social mode enhancements"
echo ""

for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"

    python scripts/train.py --mode social --seed "$SEED" \
        --intrinsic-anneal \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode social --seed "$SEED" \
        --listener-reward 0.1 \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode social --seed "$SEED" \
        --corridor-width 1 \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode social --seed "$SEED" \
        --asymmetric-info \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    wait
done

echo ">>> Experiment 3 complete."

# =============================================
# Experiment 4: Algorithm comparison
# =============================================
echo ""
echo ">>> Experiment 4: Algorithm comparison"
echo ""

for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"

    python scripts/train.py --mode option_critic --corridor --seed "$SEED" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode sac_continuous --corridor --seed "$SEED" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode discrete --corridor --seed "$SEED" \
        --intrinsic-anneal \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    wait
done

echo ">>> Experiment 4 complete."

# =============================================
# Experiment 5: Transfer experiments
# =============================================
echo ""
echo ">>> Experiment 5: Transfer (suite-local, valid discrete sources only)"
echo ""

python scripts/evaluate_transfer.py --run-all \
    --source-mode discrete \
    --source-task-family corridor_single_agent \
    --same-family --corridor-size 15 \
    --total-timesteps "$TRANSFER_TIMESTEPS" --device $DEVICE \
    --base-dir "$RUNS_DIR" \
    --min-source-success "$MIN_SOURCE_SUCCESS" \
    --protocol-name same_family_discrete_full_freeze \
    --output-dir "$TRANSFER_DIR/same_family_discrete_full_freeze"

python scripts/evaluate_transfer.py --run-all \
    --source-mode discrete \
    --source-task-family keycorridor \
    --source-env "MiniGrid-KeyCorridorS3R2-v0" \
    --transfer-env "MiniGrid-KeyCorridorS4R3-v0" \
    --total-timesteps "$TRANSFER_TIMESTEPS" --device $DEVICE \
    --base-dir "$RUNS_DIR" \
    --min-source-success "$MIN_SOURCE_SUCCESS" \
    --protocol-name keycorridor_discrete_full_freeze \
    --output-dir "$TRANSFER_DIR/keycorridor_discrete_full_freeze"

python scripts/evaluate_transfer.py --run-all \
    --source-mode discrete \
    --source-task-family keycorridor \
    --source-env "MiniGrid-KeyCorridorS3R2-v0" \
    --transfer-env "MiniGrid-KeyCorridorS4R3-v0" \
    --no-freeze-encoder \
    --total-timesteps "$TRANSFER_TIMESTEPS" --device $DEVICE \
    --base-dir "$RUNS_DIR" \
    --min-source-success "$MIN_SOURCE_SUCCESS" \
    --protocol-name keycorridor_discrete_finetune_encoder \
    --output-dir "$TRANSFER_DIR/keycorridor_discrete_finetune_encoder"

echo ">>> Experiment 5 complete."

# =============================================
# Generate plots
# =============================================
echo ""
echo ">>> Generating comparison plots"
python scripts/plot_results.py \
    --experiment-dir "$RUNS_DIR" \
    --output-dir "$PLOTS_DIR"

echo ""
echo "============================================"
echo "All experiments complete!"
echo "Suite results in $SUITE_DIR"
echo "============================================"
