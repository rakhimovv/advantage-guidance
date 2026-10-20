"""Generate the ten-seed edp runs for OGBench and OCBench."""

from _common import common_agent_config, write_launchers

run_group = "edp"


def get_config(suite, env_name):
    config = common_agent_config(suite)
    if suite == "ogbench":
        config.update(
            {
                "agent": "agents/edp.py",
                "guidance_weights": "1",
            }
        )
        config["agent.bc_weight"] = {
            "cube-single": 100.0,
            "cube-double": 10.0,
            "scene": 30.0,
            "puzzle-3x3": 30.0,
        }[env_name.split("-play-")[0]]
    elif suite == "ocbench":
        config.update(
            {
                "agent": "agents/edp.py",
                "guidance_weights": "1",
                "agent.actor_lr": 1e-4,
                "agent.critic_lr": 1e-4,
                "agent.value_lr": 1e-4,
                "agent.denoise_steps": 10,
            }
        )
        config["agent.bc_weight"] = {
            "block-lite-single-task1-v0": 1.0,
            "block-lite-double-task1-v0": 1.0,
            "block-lite-double-task2-v0": 1.0,
            "chamber-lite-easy-task1-v0": 1.0,
            "chamber-lite-easy-task2-v0": 1.0,
            "chamber-lite-easy-task3-v0": 10.0,
        }[env_name]
    return config


if __name__ == "__main__":
    write_launchers(get_config, run_group)
