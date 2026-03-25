#!/bin/bash
# Run all experiments for Social HRL project
# Usage: bash scripts/run_all.sh

set -e

SEEDS="42 123 456"
TIMESTEPS=1000000
ENV="MiniGrid-KeyCorridorS3R2-v0"

echo "============================================"
echo "Social HRL — Full Experiment Suite"
echo "Environment: $ENV"
echo "Seeds: $SEEDS"
echo "Timesteps per run: $TIMESTEPS"
echo "============================================"

# =============================================
# Experiment 0: Sanity checks (shorter runs)
# =============================================
echo ""
echo ">>> Experiment 0: Sanity checks (100K steps)"
echo ""

python scripts/train.py \
    --mode flat --seed 42 \
    --total-timesteps 100000 \
    --env "$ENV" \
    --output-dir outputs/exp0_flat_sanity

python scripts/train.py \
    --mode continuous --seed 42 \
    --total-timesteps 100000 \
    --env "$ENV" \
    --output-dir outputs/exp0_continuous_sanity

echo ""
echo ">>> Sanity checks done. Review outputs/exp0_*/metrics.json before proceeding."
echo ">>> Press Enter to continue to full experiments, or Ctrl+C to abort."
read -r

# =============================================
# Experiment 1: Three-way comparison
# =============================================
echo ""
echo ">>> Experiment 1: Full three-way comparison"
echo ""

for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"

    # Condition (a): Flat PPO baseline
    python scripts/train.py \
        --mode flat --seed "$SEED" \
        --total-timesteps "$TIMESTEPS" \
        --env "$ENV" \
        --output-dir "outputs/exp1_flat_seed${SEED}" &

    # Condition (b): HRL continuous goals
    python scripts/train.py \
        --mode continuous --seed "$SEED" \
        --total-timesteps "$TIMESTEPS" \
        --env "$ENV" \
        --output-dir "outputs/exp1_continuous_seed${SEED}" &

    # Condition (c): HRL discrete bottleneck
    python scripts/train.py \
        --mode discrete --seed "$SEED" \
        --total-timesteps "$TIMESTEPS" \
        --env "$ENV" \
        --output-dir "outputs/exp1_discrete_seed${SEED}" &

    wait  # Wait for all 3 to finish before next seed
done

echo ""
echo ">>> Experiment 1 complete."

# =============================================
# Generate plots
# =============================================
echo ""
echo ">>> Generating comparison plots"
python scripts/plot_results.py --experiment-dir outputs/

echo ""
echo "============================================"
echo "All experiments complete!"
echo "Results in outputs/"
echo "============================================"
