"""Goal quality metrics for evaluating HRL agents.

Metrics:
- Goal entropy: diversity of goals used
- Goal coverage: fraction of distinct goals per episode
- Temporal extent: average duration of each subgoal
- Message usage stats (for discrete mode)
"""

import numpy as np
from collections import Counter


def goal_entropy(messages):
    """Compute entropy of message distribution.

    Args:
        messages: List of numpy arrays, each (L,) with token indices.
    Returns:
        entropy: Float, higher = more diverse goals.
    """
    if not messages:
        return 0.0

    # Convert each message to a tuple for hashing
    msg_tuples = [tuple(m) for m in messages]
    counts = Counter(msg_tuples)
    total = sum(counts.values())

    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    return entropy


def goal_coverage(messages):
    """Fraction of unique messages out of total messages.

    Args:
        messages: List of numpy arrays.
    Returns:
        coverage: Float in [0, 1]. 1.0 = every message is unique.
    """
    if not messages:
        return 0.0

    msg_tuples = [tuple(m) for m in messages]
    unique = len(set(msg_tuples))
    total = len(msg_tuples)

    return unique / total


def temporal_extent(goals, threshold=0.1):
    """Average number of steps before the goal changes meaningfully.

    For continuous goals, measures L2 distance between consecutive goals.
    For discrete goals, measures exact token changes.

    Args:
        goals: numpy array of shape (T, D) for continuous or (T, L) for discrete.
        threshold: L2 threshold for continuous goals.
    Returns:
        avg_extent: Average steps between meaningful goal changes.
    """
    if len(goals) < 2:
        return len(goals)

    change_points = []
    for t in range(1, len(goals)):
        if goals.dtype in [np.int32, np.int64]:
            # Discrete: any token change counts
            changed = not np.array_equal(goals[t], goals[t-1])
        else:
            # Continuous: L2 distance threshold
            changed = np.linalg.norm(goals[t] - goals[t-1]) > threshold

        if changed:
            change_points.append(t)

    if not change_points:
        return len(goals)

    # Average gap between changes
    gaps = [change_points[0]]
    for i in range(1, len(change_points)):
        gaps.append(change_points[i] - change_points[i-1])

    return np.mean(gaps)


def message_usage_stats(messages, vocab_size=10, message_length=3):
    """Detailed stats on how the discrete vocabulary is used.

    Args:
        messages: List of numpy arrays, each (L,) with token indices.
        vocab_size: K.
        message_length: L.
    Returns:
        Dict with per-position token distributions and overall stats.
    """
    if not messages:
        return {}

    messages_arr = np.array(messages)  # (N, L)

    stats = {
        'total_messages': len(messages),
        'unique_messages': len(set(tuple(m) for m in messages)),
        'entropy': goal_entropy(messages),
        'coverage': goal_coverage(messages),
    }

    # Per-position entropy
    for pos in range(message_length):
        counts = Counter(messages_arr[:, pos].tolist())
        total = sum(counts.values())
        probs = np.array([counts.get(i, 0) / total for i in range(vocab_size)])
        pos_entropy = -np.sum(probs * np.log(probs + 1e-10))
        stats[f'position_{pos}_entropy'] = pos_entropy
        stats[f'position_{pos}_used_tokens'] = sum(1 for p in probs if p > 0.01)

    return stats


def compute_all_metrics(messages=None, continuous_goals=None, vocab_size=10, message_length=3):
    """Compute all available goal quality metrics.

    Args:
        messages: List of discrete messages (for discrete mode).
        continuous_goals: numpy array (T, D) of continuous goals (for continuous mode).
        vocab_size: K (for discrete mode).
        message_length: L (for discrete mode).
    Returns:
        Dict of all metrics.
    """
    metrics = {}

    if messages:
        metrics['entropy'] = goal_entropy(messages)
        metrics['coverage'] = goal_coverage(messages)
        metrics['temporal_extent'] = temporal_extent(np.array(messages))
        metrics.update(message_usage_stats(messages, vocab_size, message_length))

    if continuous_goals is not None:
        # Discretize continuous goals into bins for entropy
        bins = 20
        discretized = np.digitize(continuous_goals,
                                   np.linspace(continuous_goals.min(), continuous_goals.max(), bins))
        msg_like = [tuple(row) for row in discretized]
        counts = Counter(msg_like)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])

        metrics['entropy'] = -np.sum(probs * np.log(probs + 1e-10))
        metrics['coverage'] = len(counts) / total
        metrics['temporal_extent'] = temporal_extent(continuous_goals)
        metrics['goal_std'] = continuous_goals.std(axis=0).mean()

    return metrics
