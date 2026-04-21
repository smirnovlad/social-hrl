#!/bin/bash
# Run all experiments for Social HRL project.
# Usage: bash scripts/run_all.sh
#
# Outputs are isolated per suite under outputs/suites/<suite_id>/.
#
# Tuned for M1 Max (10 cores). Each wave launches at most 4 jobs
# to avoid over-subscribing cores (each job uses num_envs threads).

set -e

# Pin BLAS / OpenMP threads to 1 per process. We run up to 4 PyTorch jobs
# in parallel, and each defaults to using every core for BLAS, producing
# 40-way contention on tiny matmuls. Pinning removes that and roughly 2x's
# throughput for this workload.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SUITE_ID="${SUITE_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
SUITE_DIR="${SUITE_DIR:-outputs/suites/${SUITE_ID}}"
RUNS_DIR="${RUNS_DIR:-${SUITE_DIR}/runs}"
TRANSFER_DIR="${TRANSFER_DIR:-${SUITE_DIR}/transfer}"
PLOTS_DIR="${PLOTS_DIR:-${SUITE_DIR}/plots}"

SEEDS="${SEEDS:-42 123 7}"
TIMESTEPS="${TIMESTEPS:-700000}"
# Corridor experiments saturate fast (100% success by ~50K in smoke tests),
# so 1M is mostly burning compute on a flat line. 400K leaves plenty of
# headroom while cutting ~60% of wall-clock on those runs.
CORRIDOR_TIMESTEPS="${CORRIDOR_TIMESTEPS:-300000}"
SWEEP_TIMESTEPS="${SWEEP_TIMESTEPS:-500000}"     # shorter for vocab sweep
TRANSFER_TIMESTEPS="${TRANSFER_TIMESTEPS:-500000}"
MIN_SOURCE_SUCCESS="${MIN_SOURCE_SUCCESS:-0.5}"
DEVICE="${DEVICE:-cpu}"  # CPU beats MPS for these tiny models + MiniGrid
TRAIN_FLAGS="${TRAIN_FLAGS:---no-wandb}"

mkdir -p "$RUNS_DIR" "$TRANSFER_DIR" "$PLOTS_DIR"

echo "============================================"
echo "Social HRL — Full Experiment Suite"
echo "Suite: $SUITE_ID"
echo "Suite dir: $SUITE_DIR"
echo "Seeds: $SEEDS"
echo "Timesteps per run: $TIMESTEPS"
echo "Corridor timesteps: $CORRIDOR_TIMESTEPS"
echo "Sweep timesteps: $SWEEP_TIMESTEPS"
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
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode continuous --corridor --seed "$SEED" \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode discrete --corridor --seed "$SEED" \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    # Main social run — no flags. Keep this as the headline number so the
    # result isn't conflated with variants and the run_slug is canonical.
    python scripts/train.py --mode social --seed "$SEED" \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    wait

    # Communication ablation — runs with the flag so training also eval'd
    # with messages zeroed. Tagged as a variant in its run_slug so it does
    # not collide with the main social run above.
    python scripts/train.py --mode social --seed "$SEED" \
        --eval-comm-ablation \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
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

    # Wave 1: KeyCorridor S3R2 + S6R3
    python scripts/train.py --mode flat --seed "$SEED" \
        --env "MiniGrid-KeyCorridorS3R2-v0" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode discrete --seed "$SEED" \
        --env "MiniGrid-KeyCorridorS3R2-v0" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode flat --seed "$SEED" \
        --env "MiniGrid-KeyCorridorS6R3-v0" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode discrete --seed "$SEED" \
        --env "MiniGrid-KeyCorridorS6R3-v0" \
        --total-timesteps "$TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    wait

    # Wave 2: MultiRoom
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

    # Wave 1 (4 jobs — social is heavier, keep to 4)
    python scripts/train.py --mode social --seed "$SEED" \
        --intrinsic-anneal \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode social --seed "$SEED" \
        --listener-reward 0.1 \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode social --seed "$SEED" \
        --corridor-width 1 \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode social --seed "$SEED" \
        --asymmetric-info \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    wait

    # Wave 2
    python scripts/train.py --mode social --seed "$SEED" \
        --rendezvous-bonus 0.3 \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode social --seed "$SEED" \
        --num-obstacles 5 \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
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
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode sac_continuous --corridor --seed "$SEED" \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    python scripts/train.py --mode discrete --corridor --seed "$SEED" \
        --intrinsic-anneal \
        --total-timesteps "$CORRIDOR_TIMESTEPS" --device $DEVICE \
        --output-root "$RUNS_DIR" $TRAIN_FLAGS &

    wait
done

echo ">>> Experiment 4 complete."

# =============================================
# Experiment 5: Vocabulary size / message length sweep (RQ3)
# =============================================
# Trimmed grid: K in {3,10,25}, L in {1,3} = 6 points per mode
# Uses SWEEP_TIMESTEPS (500K) — enough to see goal quality trends
echo ""
echo ">>> Experiment 5: Vocab size / message length sweep"
echo ""

for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"

    # Discrete sweep — skip (K=10, L=3) (the default; already in Exp 1).
    # Running it here would produce the same run_slug and clobber metrics.
    for VOCAB in 3 10 25; do
        for MLEN in 1 3; do
            if [ "$VOCAB" = "10" ] && [ "$MLEN" = "3" ]; then
                continue
            fi
            python scripts/train.py --mode discrete --corridor --seed "$SEED" \
                --vocab-size "$VOCAB" --message-length "$MLEN" \
                --total-timesteps "$SWEEP_TIMESTEPS" --device $DEVICE \
                --output-root "$RUNS_DIR" $TRAIN_FLAGS &
        done
    done
    wait

    # Social sweep — same dedup as discrete sweep.
    for VOCAB in 3 10 25; do
        for MLEN in 1 3; do
            if [ "$VOCAB" = "10" ] && [ "$MLEN" = "3" ]; then
                continue
            fi
            python scripts/train.py --mode social --seed "$SEED" \
                --vocab-size "$VOCAB" --message-length "$MLEN" \
                --total-timesteps "$SWEEP_TIMESTEPS" --device $DEVICE \
                --output-root "$RUNS_DIR" $TRAIN_FLAGS &
        done
    done
    wait
done

echo ">>> Experiment 5 complete."

# =============================================
# Experiment 6: Transfer experiments
# =============================================
echo ""
echo ">>> Experiment 6: Transfer (suite-local, valid sources only)"
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

# Social-to-single-agent transfer (RQ2)
python scripts/evaluate_transfer.py --run-all \
    --source-mode social \
    --source-task-family corridor_social \
    --same-family --corridor-size 15 \
    --total-timesteps "$TRANSFER_TIMESTEPS" --device $DEVICE \
    --base-dir "$RUNS_DIR" \
    --min-source-success "$MIN_SOURCE_SUCCESS" \
    --protocol-name same_family_social_full_freeze \
    --output-dir "$TRANSFER_DIR/same_family_social_full_freeze"

echo ">>> Experiment 6 complete."

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
