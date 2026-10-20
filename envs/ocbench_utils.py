"""OCBench dataset loading and environment helpers."""

from __future__ import annotations

import glob
import os

import numpy as np
from envs.env_utils import (
    OCBENCH_ACTION_NORM_CLIP,
    OCBENCH_ACTION_NORM_PERCENTILE,
    transform_dataset_rewards,
)
from utils.datasets import Dataset

DATASET_KEYS = (
    "observations",
    "actions",
    "rewards",
    "masks",
    "terminals",
)
OCBENCH_ENV_FAMILIES = {"block", "chamber", "switch", "hanoi", "bowling"}


class ActionNormalizer:
    """Percentile action normalizer used by the OCBench reference implementation."""

    def __init__(self, q_low, q_high, clip, gripper_dims=(-1,)):
        self.q_low = np.asarray(q_low, dtype=np.float32).copy()
        self.q_high = np.asarray(q_high, dtype=np.float32).copy()
        self.clip = float(clip)
        self.gripper_dims = np.zeros_like(self.q_low, dtype=bool)
        self.gripper_dims[np.asarray(gripper_dims, dtype=np.int64)] = True
        self.q_low[self.gripper_dims] = -1.0
        self.q_high[self.gripper_dims] = 1.0
        self.center = (self.q_low + self.q_high) / 2
        self.half_range = (self.q_high - self.q_low) / 2
        self.valid = self.half_range > 0
        self.action_low = np.full_like(self.q_low, -self.clip)
        self.action_high = np.full_like(self.q_high, self.clip)
        self.action_low[self.gripper_dims] = -1.0
        self.action_high[self.gripper_dims] = 1.0

    @classmethod
    def fit(cls, actions, percentile, clip):
        q_low = np.percentile(actions, percentile, axis=0).astype(np.float32)
        q_high = np.percentile(actions, 100 - percentile, axis=0).astype(np.float32)
        return cls(q_low, q_high, clip)

    def normalize(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        normalized = np.zeros_like(actions, dtype=np.float32)
        normalized[..., self.valid] = (
            actions[..., self.valid] - self.center[self.valid]
        ) / self.half_range[self.valid]
        return np.clip(normalized, self.action_low, self.action_high)

    def unnormalize(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, self.action_low, self.action_high)
        return actions * self.half_range + self.center


def canonical_ocbench_env_name(env_name):
    """Translate the legacy environment spelling to the current OCBench ID."""
    visual_prefix = "visual-" if env_name.startswith("visual-") else ""
    base_name = env_name.removeprefix("visual-")
    family, separator, task = base_name.partition("-")
    if separator and task.startswith("mjwarp-"):
        task = task.removeprefix("mjwarp-")
    return f"{visual_prefix}{family}-{task}" if separator else env_name


def is_ocbench_env_name(env_name):
    """Return whether a legacy or current OCBench name is used."""
    base_name = canonical_ocbench_env_name(env_name).removeprefix("visual-")
    family, separator, task = base_name.partition("-")
    return bool(
        separator and family in OCBENCH_ENV_FAMILIES and not task.startswith("cpu-")
    )


def parse_ocbench_env_spec(env_name):
    """Resolve a legacy or current OCBench environment name."""
    if not is_ocbench_env_name(env_name):
        raise ValueError(f"Expected an OCBench environment name, got {env_name!r}.")

    import ocbench

    family, backend, env_kwargs, max_episode_steps = ocbench.parse_env_spec(
        canonical_ocbench_env_name(env_name)
    )
    if backend != "mjwarp":
        raise ValueError(f"Expected an OCBench environment name, got {env_name!r}.")
    return family, env_kwargs, max_episode_steps


def make_ocbench_env(family, env_kwargs, nworld):
    """Construct a batched OCBench environment."""
    from ocbench.mjwarp import make_env

    return make_env(family, env_kwargs, nworld)


def get_dataset_paths(dataset_dir, num_datasets=None):
    """Return sorted training shards from an OCBench dataset directory."""
    dataset_dir = os.path.expanduser(dataset_dir)
    paths = sorted(
        path
        for path in glob.glob(os.path.join(dataset_dir, "*.npz"))
        if not path.endswith("-val.npz")
    )
    if not paths:
        raise FileNotFoundError(
            f"No OCBench .npz dataset shards found in {dataset_dir}."
        )
    if num_datasets is not None:
        if num_datasets < 1:
            raise ValueError(
                f"ocbench_num_datasets must be positive, got {num_datasets}."
            )
        if len(paths) < num_datasets:
            raise FileNotFoundError(
                f"Requested {num_datasets} OCBench shards from {dataset_dir}, but found only {len(paths)}."
            )
        paths = paths[:num_datasets]
    return paths


def fit_action_normalizer(dataset_path):
    """Fit the reference OCBench normalization on the initial shard."""
    with np.load(dataset_path) as data:
        actions = np.asarray(data["actions"], dtype=np.float32)
    return ActionNormalizer.fit(
        actions, OCBENCH_ACTION_NORM_PERCENTILE, OCBENCH_ACTION_NORM_CLIP
    )


def _episode_ranges(data):
    rewards = np.asarray(data["rewards"])
    terminals = np.asarray(data["terminals"]) > 0
    terminal_idxs = np.flatnonzero(terminals)
    if len(terminal_idxs) == 0 or terminal_idxs[-1] != len(rewards) - 1:
        terminal_idxs = np.concatenate([terminal_idxs, [len(rewards) - 1]])
    initial_idxs = np.concatenate([[0], terminal_idxs[:-1] + 1])
    ranges = list(zip(initial_idxs, terminal_idxs + 1))
    return ranges


def _load_npz(path):
    with np.load(path) as data:
        missing = [key for key in DATASET_KEYS if key not in data]
        if missing:
            raise KeyError(
                f"OCBench dataset {path} is missing required keys: {missing}."
            )
        arrays = {key: np.asarray(data[key], dtype=np.float32) for key in DATASET_KEYS}
        dataset = dict(arrays)

        if "next_observations" in data:
            dataset["next_observations"] = np.asarray(
                data["next_observations"], dtype=np.float32
            )
        else:
            if "observation_interval" not in data:
                raise KeyError(
                    f"OCBench dataset {path} has neither next_observations nor observation_interval."
                )
            observation_interval = int(data["observation_interval"])
            if observation_interval != 1:
                raise ValueError(
                    "Sparse OCBench observations are not supported by this transition-based dataset sampler; "
                    f"got observation_interval={observation_interval} in {path}."
                )

            all_ranges = _episode_ranges(arrays)
            observation_offsets = np.concatenate(
                [[0], np.cumsum([end - start + 1 for start, end in all_ranges])]
            )
            if observation_offsets[-1] != len(arrays["observations"]):
                raise ValueError(
                    f"Invalid observation/episode boundaries in OCBench dataset {path}."
                )
            observation_ranges = list(
                zip(observation_offsets[:-1], observation_offsets[1:])
            )
            dataset["observations"] = np.concatenate(
                [
                    arrays["observations"][start : end - 1]
                    for start, end in observation_ranges
                ],
                axis=0,
            )
            dataset["next_observations"] = np.concatenate(
                [
                    arrays["observations"][start + 1 : end]
                    for start, end in observation_ranges
                ],
                axis=0,
            )
    return dataset


def load_ocbench_dataset(
    dataset_path,
    *,
    action_normalizer=None,
    action_clip_eps=None,
    sparse=False,
    reward_scale=1.0,
    reward_bias=0.0,
):
    """Load and transform one OCBench shard and its optional validation companion."""
    dataset_path = os.path.expanduser(dataset_path)
    train = _load_npz(dataset_path)
    val_path = dataset_path.replace(".npz", "-val.npz")
    val = _load_npz(val_path) if os.path.exists(val_path) else None

    for dataset in (train, val):
        if dataset is None:
            continue
        if action_normalizer is not None:
            dataset["actions"] = action_normalizer.normalize(dataset["actions"])
        elif action_clip_eps is not None:
            dataset["actions"] = np.clip(
                dataset["actions"], -1 + action_clip_eps, 1 - action_clip_eps
            )

    train = Dataset.create(**train)
    val = Dataset.create(**val) if val is not None else None
    train = transform_dataset_rewards(
        train,
        sparse=sparse,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
    )
    val = transform_dataset_rewards(
        val,
        sparse=sparse,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
    )
    return train, val
