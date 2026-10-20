"""Batched policy evaluation for OCBench environments."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from envs.ocbench_utils import make_ocbench_env, parse_ocbench_env_spec
from utils.evaluation import _prepare_actor


def evaluate_ocbench(
    agent,
    env_name,
    *,
    action_normalizer=None,
    num_eval_episodes=30,
    num_video_episodes=0,
    video_frame_skip=25,
    guidance_weight=None,
    rejection_sampling=1,
):
    """Evaluate all requested episodes concurrently in one OCBench environment."""
    num_episodes = num_eval_episodes + num_video_episodes
    if num_eval_episodes < 1 or num_video_episodes < 0 or video_frame_skip < 1:
        raise ValueError(
            "Require positive eval episodes/frame skip and nonnegative video episodes."
        )

    family, env_kwargs, max_episode_steps = parse_ocbench_env_spec(env_name)
    env = make_ocbench_env(family, env_kwargs, num_episodes)
    try:
        reset_seeds = np.random.SeedSequence(
            np.random.randint(0, 2**32)
        ).generate_state(num_episodes, dtype=np.uint32)
        observations, _ = env.reset(seeds=reset_seeds)
        actor_fn = _prepare_actor(agent, guidance_weight, rejection_sampling)

        action_dim = env.single_action_space.shape[-1]
        action_low = env.single_action_space.low
        action_high = env.single_action_space.high
        horizon = (
            int(agent.config.get("horizon_length", 1))
            if agent.config.get("action_chunking", False)
            else 1
        )

        done = np.zeros(num_episodes, dtype=bool)
        steps = np.zeros(num_episodes, dtype=np.int32)
        returns = np.zeros(num_episodes, dtype=np.float32)
        final_info = [None] * num_episodes
        trajectories = [defaultdict(list) for _ in range(num_eval_episodes)]
        renders = [[] for _ in range(num_video_episodes)]
        action_queues = [[] for _ in range(num_episodes)]

        while not np.all(done):
            need_actions = [
                idx
                for idx in range(num_episodes)
                if not done[idx] and not action_queues[idx]
            ]
            if need_actions:
                actions = np.atleast_2d(
                    np.asarray(
                        actor_fn(observations=observations[need_actions]),
                        dtype=np.float32,
                    )
                )
                chunks = actions.reshape(len(need_actions), horizon, action_dim)
                if action_normalizer is not None:
                    chunks = action_normalizer.unnormalize(
                        chunks.reshape(-1, action_dim)
                    ).reshape(chunks.shape)
                chunks = np.clip(chunks, action_low, action_high)
                for local_idx, world_idx in enumerate(need_actions):
                    action_queues[world_idx].extend(chunks[local_idx])

            step_actions = np.zeros((num_episodes, action_dim), dtype=np.float32)
            for idx in np.flatnonzero(~done):
                step_actions[idx] = action_queues[idx].pop(0)

            previous_observations = observations.copy()
            next_observations, rewards, terminated, truncated, info = env.step(
                step_actions
            )
            active = ~done
            steps[active] += 1
            returns[active] += rewards[active]
            newly_done = active & (
                terminated | truncated | (steps >= max_episode_steps)
            )

            for idx in np.flatnonzero(active):
                transition_info = {
                    "success": bool(info["success"][idx]),
                    "failure": bool(info["failure"][idx]),
                    "healthy": bool(info["healthy"][idx]),
                }
                if idx < num_eval_episodes:
                    trajectories[idx]["observation"].append(previous_observations[idx])
                    trajectories[idx]["next_observation"].append(next_observations[idx])
                    trajectories[idx]["action"].append(step_actions[idx])
                    trajectories[idx]["reward"].append(rewards[idx])
                    trajectories[idx]["done"].append(bool(newly_done[idx]))
                    trajectories[idx]["info"].append(transition_info)
                elif steps[idx] % video_frame_skip == 0 or newly_done[idx]:
                    renders[idx - num_eval_episodes].append(
                        env.render_world(idx).copy()
                    )

            for idx in np.flatnonzero(newly_done):
                final_info[idx] = {
                    "success": bool(info["success"][idx]),
                    "failure": bool(info["failure"][idx]),
                    "healthy": bool(info["healthy"][idx]),
                }
                action_queues[idx] = []

            done |= newly_done
            observations = next_observations

        stats = {
            "episode.return": float(np.mean(returns[:num_eval_episodes])),
            "episode.length": float(np.mean(steps[:num_eval_episodes])),
        }
        for key in ("success", "failure", "healthy"):
            stats[key] = float(
                np.mean([final_info[idx][key] for idx in range(num_eval_episodes)])
            )
        trajectories = [
            {key: np.asarray(value) for key, value in traj.items()}
            for traj in trajectories
        ]
        renders = [np.asarray(render) for render in renders]
        return stats, trajectories, renders
    finally:
        env.close()


def eval_with_test_time_guidance_ocbench(
    agent,
    env_name,
    *,
    action_normalizer,
    num_eval_episodes,
    num_video_episodes,
    video_frame_skip=25,
    rejection_sampling,
    guidance_weights,
):
    """Evaluate an OCBench policy over the standard test-time guidance grid."""
    weights = [float(weight) for weight in guidance_weights]
    results = {}
    renders = []
    for weight in weights:
        info, _, cur_renders = evaluate_ocbench(
            agent,
            env_name,
            action_normalizer=action_normalizer,
            num_eval_episodes=num_eval_episodes,
            num_video_episodes=num_video_episodes,
            video_frame_skip=video_frame_skip,
            guidance_weight=weight,
            rejection_sampling=rejection_sampling,
        )
        results[weight] = info
        renders.extend(cur_renders)

    best_weight = max(weights, key=lambda weight: results[weight]["episode.return"])
    metrics = {
        "evaluation/episode.return": results[best_weight]["episode.return"],
        "evaluation/success": max(results[weight]["success"] for weight in weights),
        "evaluation/episode_length": min(
            results[weight]["episode.length"] for weight in weights
        ),
        "evaluation/best_guidance_weight": best_weight,
    }
    for weight in weights:
        prefix = f"evaluation_guidance_weight_{weight}"
        metrics[f"{prefix}/episode_return"] = results[weight]["episode.return"]
        metrics[f"{prefix}/success"] = results[weight]["success"]
        metrics[f"{prefix}/episode_length"] = results[weight]["episode.length"]
    return metrics, renders
