"""Generate the ten-seed fawr runs for OGBench and OCBench."""

from _common import common_agent_config, write_launchers

run_group = "fawr"


def get_config(suite, env_name):
    config = common_agent_config(suite)
    if suite == "ogbench":
        config.update(
            {
                "agent": "agents/iql_diffusion.py",
                "guidance_weights": "1",
                "agent.actor_loss": "awr",
            }
        )
        config["agent.alpha"] = {
            "cube-single": 0.3,
            "cube-double": 0.3,
            "scene": 10.0,
            "puzzle-3x3": 0.3,
        }[env_name.split("-play-")[0]]
    elif suite == "ocbench":
        config.update(
            {
                "agent": "agents/iql_diffusion.py",
                "guidance_weights": "1",
                "agent.lr": 1e-4,
                "agent.actor_loss": "awr",
                "agent.denoise_steps": 10,
            }
        )
        config["agent.alpha"] = {
            "block-lite-single-task1-v0": 3.0,
            "block-lite-double-task1-v0": 3.0,
            "block-lite-double-task2-v0": 0.3,
            "chamber-lite-easy-task1-v0": 1.0,
            "chamber-lite-easy-task2-v0": 1.0,
            "chamber-lite-easy-task3-v0": 1.0,
        }[env_name]
    return config


if __name__ == "__main__":
    write_launchers(get_config, run_group)
