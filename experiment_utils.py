"""Helpers for experiment naming, metadata, and run discovery."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


KNOWN_MODES = (
    "sac_continuous",
    "option_critic",
    "continuous",
    "discrete",
    "social",
    "flat",
)

MODE_LABELS = {
    "flat": "Flat PPO",
    "continuous": "HRL Continuous",
    "discrete": "HRL Discrete",
    "social": "HRL Social",
    "option_critic": "Option-Critic",
    "sac_continuous": "SAC Continuous",
}


def slugify(value):
    """Convert a string into a stable filesystem-friendly slug."""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "unknown"


def format_number_token(value):
    """Format a float for stable inclusion in slugs."""
    return str(value).replace("-", "neg").replace(".", "p")


def env_slug(env_name):
    """Create a compact slug for a Gym env name."""
    name = str(env_name or "unknown-env")
    name = re.sub(r"^MiniGrid-", "", name)
    name = re.sub(r"-v\d+$", "", name)
    return slugify(name)


def short_env_label(env_name):
    """Human-readable short label for an env name."""
    name = str(env_name or "Unknown")
    name = re.sub(r"^MiniGrid-", "", name)
    name = re.sub(r"-v\d+$", "", name)
    return name


def canonical_mode(mode, use_sac=False, use_option_critic=False):
    """Normalize mode naming across scripts."""
    if use_option_critic:
        return "option_critic"
    if use_sac:
        return "sac_continuous"
    return mode


def infer_task_family(mode, env_name, use_corridor=False):
    """Infer the task family used by a run."""
    if mode == "social":
        return "corridor_social"
    if use_corridor:
        return "corridor_single_agent"
    if env_name and env_name.startswith("MiniGrid-KeyCorridor"):
        return "keycorridor"
    if env_name and env_name.startswith("MiniGrid-MultiRoom"):
        return "multiroom"
    return env_slug(env_name)


def effective_env_name(mode, env_name, use_corridor=False, corridor_size=11, corridor_width=3):
    """Return the effective environment name used by the trainer."""
    if mode == "social":
        return f"TwoAgentCorridor-S{corridor_size}-W{corridor_width}-v0"
    if use_corridor:
        return f"SingleAgentCorridor-S{corridor_size}-W{corridor_width}-v0"
    return env_name


def build_condition_label(metadata):
    """Build a human-readable condition label from run metadata."""
    base = MODE_LABELS.get(metadata["mode"], metadata["mode"].replace("_", " ").title())

    if metadata["task_family"] == "corridor_single_agent":
        label = f"{base} (corridor)"
    elif metadata["task_family"] == "corridor_social":
        label = f"{base} (social corridor)"
    else:
        label = f"{base} ({short_env_label(metadata['env_name'])})"

    variants = []
    if metadata.get("corridor_width", 3) != 3:
        variants.append(f"w={metadata['corridor_width']}")
    if metadata.get("intrinsic_anneal"):
        variants.append("anneal")
    listener_reward = metadata.get("listener_reward_coef", 0.0)
    if listener_reward > 0:
        variants.append(f"listener={listener_reward:g}")
    if metadata.get("asymmetric_info"):
        variants.append("asymmetric")

    if variants:
        label += f" [{', '.join(variants)}]"
    return label


def build_run_metadata(
    mode,
    seed,
    env_name,
    use_corridor=False,
    corridor_width=3,
    corridor_size=11,
    intrinsic_anneal=False,
    listener_reward_coef=0.0,
    asymmetric_info=False,
    use_sac=False,
    use_option_critic=False,
):
    """Create the metadata dict shared by train/plot/transfer scripts."""
    mode = canonical_mode(mode, use_sac=use_sac, use_option_critic=use_option_critic)
    task_family = infer_task_family(mode, env_name, use_corridor=use_corridor)
    effective_name = effective_env_name(
        mode,
        env_name,
        use_corridor=use_corridor,
        corridor_size=corridor_size,
        corridor_width=corridor_width,
    )
    env_name_for_slug = effective_name if task_family.startswith("corridor") else env_name

    condition_parts = [
        f"mode-{slugify(mode)}",
        f"task-{slugify(task_family)}",
        f"env-{env_slug(env_name_for_slug)}",
    ]
    if task_family.startswith("corridor"):
        condition_parts.append(f"size-{corridor_size}")
        condition_parts.append(f"width-{corridor_width}")
    if intrinsic_anneal:
        condition_parts.append("anneal")
    if listener_reward_coef > 0:
        condition_parts.append(f"listener-{format_number_token(listener_reward_coef)}")
    if asymmetric_info:
        condition_parts.append("asymmetric")

    condition_id = "__".join(condition_parts)
    metadata = {
        "mode": mode,
        "seed": int(seed),
        "env_name": effective_name,
        "source_env_name": env_name,
        "task_family": task_family,
        "use_corridor": bool(use_corridor),
        "corridor_size": int(corridor_size),
        "corridor_width": int(corridor_width),
        "listener_reward_coef": float(listener_reward_coef),
        "intrinsic_anneal": bool(intrinsic_anneal),
        "asymmetric_info": bool(asymmetric_info),
        "use_sac": bool(mode == "sac_continuous"),
        "use_option_critic": bool(mode == "option_critic"),
        "condition_id": condition_id,
    }
    metadata["condition_label"] = build_condition_label(metadata)
    metadata["run_slug"] = f"{condition_id}__seed-{seed}"
    return metadata


def write_json(path, payload):
    """Write JSON with stable formatting."""
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))


def load_json(path):
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_run_record(run_dir):
    """Load a run record from a metadata-backed run directory."""
    run_dir = Path(run_dir)
    info = load_json(run_dir / "run_info.json")
    record = dict(info)
    record["run_dir"] = str(run_dir)
    record["returns_path"] = str(run_dir / "returns.npy")
    record["metrics_path"] = str(run_dir / "metrics.json")
    record["config_path"] = str(run_dir / "config.yaml")
    record["checkpoint_path"] = str(run_dir / "final.pt")
    if (run_dir / "metrics.json").exists():
        record["metrics"] = load_json(run_dir / "metrics.json")
    return record


def load_legacy_run_record(run_dir):
    """Best-effort record loading for old outputs without run_info.json."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_root = run_dir.parent.name
    match = re.match(r"^(.*)_seed(\d+)$", run_root)
    if not match:
        return None

    condition_id = match.group(1)
    seed = int(match.group(2))

    mode = condition_id
    for candidate in KNOWN_MODES:
        if condition_id.startswith(candidate):
            mode = candidate
            break

    env_name = (
        config.get("env", {}).get("effective_name")
        or config.get("env", {}).get("name")
        or "unknown-env"
    )
    task_family = config.get("env", {}).get("task_family")
    if task_family is None:
        task_family = infer_task_family(mode, config.get("env", {}).get("name"))

    record = {
        "mode": mode,
        "seed": seed,
        "env_name": env_name,
        "source_env_name": config.get("env", {}).get("name"),
        "task_family": task_family,
        "use_corridor": bool(config.get("env", {}).get("use_corridor", "_corridor" in condition_id)),
        "corridor_size": int(config.get("env", {}).get("corridor_size", 11)),
        "corridor_width": int(config.get("env", {}).get("corridor_width", 3)),
        "listener_reward_coef": float(config.get("communication", {}).get("listener_reward_coef", 0.0)),
        "intrinsic_anneal": bool(config.get("worker", {}).get("intrinsic_anneal", "anneal" in condition_id)),
        "asymmetric_info": bool(config.get("env", {}).get("asymmetric_info", "asymm" in condition_id)),
        "use_sac": bool(config.get("sac", {}).get("enabled", mode == "sac_continuous")),
        "use_option_critic": bool(config.get("manager", {}).get("use_option_critic", mode == "option_critic")),
        "condition_id": condition_id,
        "condition_label": condition_id.replace("_", " "),
        "run_slug": run_root,
        "run_dir": str(run_dir),
        "returns_path": str(run_dir / "returns.npy"),
        "metrics_path": str(run_dir / "metrics.json"),
        "config_path": str(config_path),
        "checkpoint_path": str(run_dir / "final.pt"),
    }
    if (run_dir / "metrics.json").exists():
        record["metrics"] = load_json(run_dir / "metrics.json")
    return record


def discover_runs(base_dir, allow_legacy=True):
    """Discover run records under a directory."""
    base = Path(base_dir)
    metadata_files = sorted(base.rglob("run_info.json"))
    if metadata_files:
        return [load_run_record(path.parent) for path in metadata_files]

    if not allow_legacy:
        return []

    records = []
    for config_path in sorted(base.rglob("config.yaml")):
        record = load_legacy_run_record(config_path.parent)
        if record is not None:
            records.append(record)
    return records


def latest_records(records, key_fields):
    """Keep the latest record for each key tuple."""
    latest = {}
    for record in sorted(records, key=lambda item: item["run_dir"]):
        key = tuple(record[field] for field in key_fields)
        latest[key] = record
    return latest
