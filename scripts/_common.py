"""Shared environment and agent configurations and launcher setup for the paper results."""

import argparse
import itertools
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate import SbatchGenerator  # noqa: E402

OGBENCH_FAMILIES = ("cube-single", "cube-double", "scene", "puzzle-3x3")
OCBENCH_TASKS = (
    ("block-lite-single-task1-v0", 1),
    ("block-lite-double-task1-v0", 3),
    ("block-lite-double-task2-v0", 3),
    ("chamber-lite-easy-task1-v0", 1),
    ("chamber-lite-easy-task2-v0", 1),
    ("chamber-lite-easy-task3-v0", 1),
)


def common_agent_config(
    suite, *, value_hidden_dims_key="value_network_kwargs.hidden_dims"
):
    if suite == "ogbench":
        config = {
            "agent.expectile": 0.9,
            "agent.action_chunking": True,
            "agent.horizon_length": 5,
            f"agent.{value_hidden_dims_key}": "(128,128)",
        }
    elif suite == "ocbench":
        dims = str((1024,) * 8)
        config = {
            "agent.batch_size": 1024,
            "agent.actor_hidden_dims": dims,
            "agent.mlp_class": "resmlp",
            "agent.discount": 0.999,
            "agent.expectile": 0.5,
            "agent.action_chunking": True,
            "agent.horizon_length": 3,
            f"agent.{value_hidden_dims_key}": dims,
        }
    return config


def ogbench_configs():
    for family, task, seed in itertools.product(
        OGBENCH_FAMILIES, range(1, 6), range(10)
    ):
        config = {
            "env_name": f"{family}-play-singletask-task{task}-v0",
            "seed": seed,
            "offline_steps": 500_000,
            "eval_episodes": 30,
            "eval_vecenv_size": 10,
            "eval_interval": 50_000,
            "save_interval": 100_000,
            "sparse": True,
        }
        yield config


def ocbench_configs(dataset_root):
    for env_name, shards in OCBENCH_TASKS:
        dataset_dir = dataset_root / "lite" / env_name
        for seed in range(1, 11):
            config = {
                "env_name": env_name,
                "seed": seed,
                "ocbench_dataset_dir": str(dataset_dir),
                "ocbench_num_datasets": shards,
                # Keep OCBench's original sparse 0/1 rewards.
                "sparse": False,
                "offline_steps": 1_000_000,
                "eval_episodes": 500,
                "eval_vecenv_size": 1,
                "eval_interval": 100_000,
                "save_interval": 250_000,
                "dataset_replace_interval": 10_000,
            }
            yield config


def build_generator(get_config, suite, run_group, *, dataset_root=None, debug=False):
    dataset_root = (
        Path(
            dataset_root
            or os.environ.get("OCBENCH_DATA_DIR", "~/.cache/ocbench/datasets")
        )
        .expanduser()
        .resolve()
    )
    configs = list(
        ogbench_configs() if suite == "ogbench" else ocbench_configs(dataset_root)
    )
    generator = SbatchGenerator(
        prefix=(
            "MUJOCO_GL=egl",
            "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "uv run python main.py",
        ),
        j=1,
        ram_gb=50,
        job_name=run_group,
        time="36:00:00",
    )
    generator.add_common_prefix({"wandb_run_group": run_group})
    for config in configs[:1] if debug else configs:
        config.update(get_config(suite, config["env_name"]))
        if debug:
            config.update(
                {
                    "debug": True,
                    "offline_steps": 10,
                    "log_interval": 5,
                    "eval_interval": 10,
                    "eval_episodes": 1,
                    "eval_vecenv_size": 1,
                    "save_interval": 10,
                }
            )
        generator.add_run(config)
    return generator


def write_launchers(get_config, run_group):
    parser = argparse.ArgumentParser(
        description="Generate paper experiment commands without executing them."
    )
    parser.add_argument("--suite", choices=("ogbench", "ocbench", "all"), default="all")
    parser.add_argument("--ocbench-data-dir", default=None)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Write sequential local runners instead of SLURM submission wrappers.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate one ten-update smoke run per suite.",
    )
    args = parser.parse_args()
    output_dir = REPO_ROOT / "sbatch"
    output_dir.mkdir(exist_ok=True)
    for suite in ("ogbench", "ocbench") if args.suite == "all" else (args.suite,):
        group = f"{run_group}_{suite}" + ("_debug" if args.debug else "")
        assert len(group) <= 70
        generator = build_generator(
            get_config,
            suite,
            group,
            dataset_root=args.ocbench_data_dir,
            debug=args.debug,
        )
        render = generator.generate_local_str if args.local else generator.generate_str
        output = output_dir / f"{group}{'_local' if args.local else ''}.sh"
        output.write_text(render())
        print(
            f"Wrote {len(generator.commands)} commands to {output}; inspect before running."
        )
