"""Goal quality metrics for evaluating HRL agents.

Metrics:
- Goal entropy: diversity of goals used
- Goal coverage: fraction of distinct goals per episode
- Temporal extent: average duration of each subgoal
- Message usage stats (for discrete mode)
- Topographic similarity: correlation between message distance and state distance
- Message-state mutual information
- Listener accuracy probe
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


def topographic_similarity(messages, states, n_samples=5000):
    """Compute topographic similarity between messages and states.

    Measures whether similar states produce similar messages (compositionality).
    Uses Spearman rank correlation between Hamming distance of messages
    and L2 distance of states across sampled pairs.

    Args:
        messages: List of numpy arrays, each (L,) with token indices.
        states: List of numpy arrays, each (D,) encoder features.
        n_samples: Number of random pairs to sample.
    Returns:
        rho: Spearman correlation coefficient. Higher = more compositional.
    """
    if len(messages) < 10 or len(states) < 10:
        return 0.0

    try:
        from scipy.stats import spearmanr
    except ImportError:
        return 0.0

    messages_arr = np.array(messages)
    states_arr = np.array(states)

    n = len(messages_arr)
    n_samples = min(n_samples, n * (n - 1) // 2)

    # Sample random pairs
    idx_i = np.random.randint(0, n, size=n_samples)
    idx_j = np.random.randint(0, n, size=n_samples)
    # Avoid self-pairs
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    if len(idx_i) < 10:
        return 0.0

    # Hamming distance for messages
    msg_dist = np.sum(messages_arr[idx_i] != messages_arr[idx_j], axis=1).astype(float)

    # L2 distance for states
    state_dist = np.linalg.norm(states_arr[idx_i] - states_arr[idx_j], axis=1)

    # Check for zero variance
    if msg_dist.std() < 1e-10 or state_dist.std() < 1e-10:
        return 0.0

    rho, _ = spearmanr(msg_dist, state_dist)
    return float(rho) if not np.isnan(rho) else 0.0


def message_state_mutual_information(messages, states, n_bins=20):
    """Compute mutual information I(message; state).

    Discretizes states via PCA to 2D then histogram binning.

    Args:
        messages: List of numpy arrays, each (L,) with token indices.
        states: List of numpy arrays, each (D,) encoder features.
        n_bins: Number of bins per dimension for state discretization.
    Returns:
        mi: Mutual information in nats. Higher = messages carry more state info.
    """
    if len(messages) < 50 or len(states) < 50:
        return 0.0

    messages_arr = np.array(messages)
    states_arr = np.array(states)

    # Reduce states to 2D for tractable binning
    try:
        from sklearn.decomposition import PCA
        if states_arr.shape[1] > 2:
            pca = PCA(n_components=2)
            states_2d = pca.fit_transform(states_arr)
        else:
            states_2d = states_arr
    except ImportError:
        # Manual: take first 2 principal components via SVD
        states_centered = states_arr - states_arr.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(states_centered, full_matrices=False)
            states_2d = U[:, :2] * S[:2]
        except np.linalg.LinAlgError:
            return 0.0

    # Discretize states into bins
    n = len(messages_arr)
    state_bins = np.zeros(n, dtype=int)
    for dim in range(2):
        col = states_2d[:, dim]
        edges = np.linspace(col.min() - 1e-10, col.max() + 1e-10, n_bins + 1)
        digitized = np.digitize(col, edges) - 1
        digitized = np.clip(digitized, 0, n_bins - 1)
        state_bins = state_bins * n_bins + digitized

    # Message indices
    msg_tuples = [tuple(m) for m in messages_arr]
    unique_msgs = list(set(msg_tuples))
    msg_to_idx = {m: i for i, m in enumerate(unique_msgs)}
    msg_bins = np.array([msg_to_idx[m] for m in msg_tuples])

    # Compute MI = H(M) + H(S) - H(M,S)
    def _entropy(labels):
        counts = Counter(labels.tolist())
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        return -np.sum(probs * np.log(probs + 1e-10))

    h_m = _entropy(msg_bins)
    h_s = _entropy(state_bins)

    # Joint
    joint = msg_bins * (state_bins.max() + 1) + state_bins
    h_ms = _entropy(joint)

    mi = h_m + h_s - h_ms
    return max(0.0, float(mi))  # MI is non-negative


def listener_accuracy_probe(messages, states, vocab_size=10, message_length=3,
                            test_fraction=0.2):
    """Train a linear probe to predict state from message.

    Measures how much information about the state is encoded in the message.

    Args:
        messages: List of numpy arrays, each (L,) with token indices.
        states: List of numpy arrays, each (D,) encoder features.
        vocab_size: K.
        message_length: L.
        test_fraction: Fraction of data for testing.
    Returns:
        r_squared: R^2 score on test set. Higher = messages encode more state info.
    """
    if len(messages) < 100 or len(states) < 100:
        return 0.0

    messages_arr = np.array(messages)
    states_arr = np.array(states)

    # One-hot encode messages
    n = len(messages_arr)
    msg_features = np.zeros((n, message_length * vocab_size))
    for i in range(n):
        for pos in range(message_length):
            token = int(messages_arr[i, pos])
            if 0 <= token < vocab_size:
                msg_features[i, pos * vocab_size + token] = 1.0

    # Train/test split
    n_test = max(10, int(n * test_fraction))
    n_train = n - n_test
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    X_train, X_test = msg_features[train_idx], msg_features[test_idx]
    y_train, y_test = states_arr[train_idx], states_arr[test_idx]

    try:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return max(0.0, float(score))
    except ImportError:
        # Manual ridge regression: w = (X^T X + alpha I)^-1 X^T y
        try:
            XtX = X_train.T @ X_train + 1.0 * np.eye(X_train.shape[1])
            Xty = X_train.T @ y_train
            w = np.linalg.solve(XtX, Xty)
            y_pred = X_test @ w
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean(axis=0)) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-10)
            return max(0.0, float(r2))
        except np.linalg.LinAlgError:
            return 0.0


def compute_all_metrics(messages=None, continuous_goals=None, vocab_size=10,
                        message_length=3, states=None):
    """Compute all available goal quality metrics.

    Args:
        messages: List of discrete messages (for discrete mode).
        continuous_goals: numpy array (T, D) of continuous goals (for continuous mode).
        vocab_size: K (for discrete mode).
        message_length: L (for discrete mode).
        states: List of encoder feature vectors corresponding to each message.
    Returns:
        Dict of all metrics.
    """
    metrics = {}

    if messages:
        metrics['entropy'] = goal_entropy(messages)
        metrics['coverage'] = goal_coverage(messages)
        metrics['temporal_extent'] = temporal_extent(np.array(messages))
        metrics.update(message_usage_stats(messages, vocab_size, message_length))

        # New semantic quality metrics (require states)
        if states and len(states) == len(messages):
            metrics['topographic_similarity'] = topographic_similarity(
                messages, states
            )
            metrics['mutual_information'] = message_state_mutual_information(
                messages, states
            )
            metrics['listener_accuracy'] = listener_accuracy_probe(
                messages, states, vocab_size, message_length
            )

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
