# Social Interaction as Regularization for Hierarchical RL

ML8103 Sequential Decision Making — MBZUAI Spring 2026 — Option 6

---

## Problem

Hierarchical Reinforcement Learning (HRL) uses a two-level architecture: a high-level **manager** sets subgoals for a low-level **worker**. This promises temporal abstraction — the manager plans at a coarse level ("go to the blue door") while the worker handles fine-grained actions.

The core failure mode is **goal collapse**: subgoals degenerate to trivial solutions — either one step away from the current state (making hierarchy useless) or identical to the final goal (providing no decomposition). Existing regularization techniques help but don't address the fundamental question: what makes a goal representation "good"?

## Core Hypothesis

**Multi-agent coordination pressure can regularize HRL goal representations.** If agents must communicate their subgoals to coordinate on shared tasks, the communication channel imposes structure on goal space. Trivial goals would be communicatively useless — "go one step left" provides no information a partner couldn't infer. Meaningful coordination requires meaningful abstraction.

This is inspired by theories of cognitive evolution: abstraction, compositionality, and language co-emerged in social species as tools for coordination (Tomasello, 2009).

## Background

- **HIRO** (Nachum et al., NeurIPS 2018): Uses full states as goals. No information loss but high-dimensional, prone to trivial goals, hard to scale.
- **Feudal Networks** (Vezhnevets et al., ICML 2017): Learns latent goal spaces. Compact but prone to collapse. Uses dilated LSTM for the manager.
- **MADDPG** (Lowe et al., NeurIPS 2017): Multi-agent training framework with centralized training, decentralized execution. Backbone for adding communication.

Neither HIRO nor FeudalNets leverages multi-agent structure to shape goals.

---

## Hypotheses

### H1: Discrete bottleneck prevents goal collapse
Forcing goals through a discrete communication channel (Gumbel-Softmax) acts as an information bottleneck that prevents degenerate goal representations. With limited vocabulary (K tokens × L positions), the manager can't waste capacity on near-identical trivial goals.

**Test**: Compare continuous goals (baseline) vs. discrete goals (bottleneck only, single agent). Measure goal entropy and coverage.

### H2: Social pressure produces better goals than bottleneck alone
Adding a partner agent who reads the same discrete messages creates additional pressure for goals to be abstract and interpretable. The partner can't use A's private representation — messages must encode shared concepts (landmarks, task phases) to be useful for coordination.

**Test**: Compare discrete-bottleneck-only (single agent) vs. discrete-with-communication (two agents). Measure goal entropy, coverage, and coordination success.

### H3: Socially-trained goal representations transfer to new tasks
If social pressure shapes goals into genuinely meaningful abstractions (not just coordination artifacts), then a manager trained with communication should produce better subgoals even when deployed solo on new tasks.

**Test**: Freeze managers from each condition. Train fresh workers on new Minigrid layouts. Measure worker sample efficiency under each frozen manager.

### H4: Channel capacity has a sweet spot
Too small a vocabulary → agents can't communicate enough to coordinate. Too large → no compression pressure, no benefit over continuous goals. There exists an optimal bottleneck size that maximizes goal quality.

**Test**: Sweep vocab size K ∈ {3, 5, 10, 25} and message length L ∈ {1, 2, 3}. Plot goal quality metrics vs. channel capacity K^L.

---

## Experiments

### Experiment 0 — Sanity Checks
- Flat PPO baseline on target Minigrid task (100K steps)
- Single-agent HRL with continuous goals (100K steps)
- **Verify**: flat PPO does NOT easily solve the task (otherwise HRL isn't needed)
- **Verify**: goal collapse actually happens in continuous mode (low goal entropy)

### Experiment 1 — Three-Way Comparison (tests H1, H2)

| Condition | Mode | Goals | Communication | Tests |
|-----------|------|-------|---------------|-------|
| Flat PPO | `flat` | N/A | No | Baseline |
| (a) HRL Continuous | `continuous` | g ∈ R^16 | No | Shows collapse |
| (b) HRL Discrete | `discrete` | m ∈ {1,...,10}^3 | No | Tests H1 |
| (c) HRL Social | `social` | m ∈ {1,...,10}^3 | Yes (2 agents) | Tests H2 |

Per condition: 3 seeds, 1M steps. Metrics: goal entropy, coverage, temporal extent, task success rate.

**Expected**: (c) > (b) > (a) on goal quality. All HRL conditions ≥ flat on task success.

### Experiment 2 — Bottleneck Sweep (tests H4)
Take condition (c). Sweep K ∈ {3, 5, 10, 25}, L ∈ {1, 2, 3}. 12 configs × 3 seeds = 36 runs.

**Expected**: Sweet spot around K=10, L=2-3. Plot as heatmap.

### Experiment 3 — Transfer (tests H3)
Freeze trained managers from (a), (b), (c). Train fresh worker on new Minigrid layout. Compare learning curves.

**Expected**: Socially-trained manager (c) → fastest worker convergence on new task.

---

## Architecture

```
Mode: flat
  Encoder(obs) → Policy → action           (standard PPO)

Mode: continuous
  Encoder(obs) → Manager (every c=10 steps) → goal g ∈ R^16
  Encoder(obs) + g → Worker → action
  Worker intrinsic reward: -||φ(s) - g||₂

Mode: discrete
  Encoder(obs) → Manager → g ∈ R^16
  g → Sender MLP → Gumbel-Softmax → message m ∈ {1,...,K}^L
  m → Decoder MLP → ĝ → Worker → action

Mode: social
  Same as discrete, but two agents in shared environment.
  Agent A's message m_A sent to Agent B's manager (and vice versa).
  Each manager input: own observation + partner's message.
  Training: MAPPO with shared critic.
```

## Multi-Agent Environment

Two HRL agents in a shared Minigrid. Each has its own goal location.
A narrow corridor creates coordination pressure — agents block each other
if they try to pass simultaneously. Communicating intended subgoals
enables them to avoid deadlock and coordinate timing.

Coordination bonus: both agents get extra reward if both reach their goals.

---

## Key References

- Vezhnevets, A. et al. (2017). Feudal Networks for Hierarchical Reinforcement Learning. *ICML*.
- Nachum, O. et al. (2018). Data-Efficient Hierarchical Reinforcement Learning. *NeurIPS*.
- Bacon, P.-L. et al. (2017). The Option-Critic Architecture. *AAAI*.
- Lowe, R. et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. *NeurIPS*.
- Foerster, J. et al. (2018). Learning with Opponent-Learning Awareness. *AAMAS*.
- Tomasello, M. (2009). The cultural origins of human cognition. *Harvard University Press*.
- Chen, L. et al. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. *NeurIPS*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.
- Chaabouni, R. et al. (2020). Compositionality and Generalization in Emergent Languages. *ACL*.
