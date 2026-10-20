"""Generate the ten-seed fql runs for OGBench and OCBench."""

from _common import common_agent_config, write_launchers

run_group = "fql"


def get_config(suite, env_name):
    config = common_agent_config(suite)
    if suite == "ogbench":
        config.update(
            {
                "agent": "agents/fql.py",
                "guidance_weights": "1",
                "agent.critic_loss_type": "iql",
            }
        )
        config["agent.alpha"] = {
            "cube-single": 1000.0,
            "cube-double": 100.0,
            "scene": 300.0,
            "puzzle-3x3": 300.0,
        }[env_name.split("-play-")[0]]
    elif suite == "ocbench":
        config.update(
            {
                "agent": "agents/fql.py",
                "guidance_weights": "1",
                "agent.lr": 1e-4,
                "agent.critic_loss_type": "iql",
                "agent.flow_steps": 10,
            }
        )
        config["agent.alpha"] = {
            "block-lite-single-task1-v0": 300.0,
            "block-lite-double-task1-v0": 100.0,
            "block-lite-double-task2-v0": 100.0,
            "chamber-lite-easy-task1-v0": 100.0,
            "chamber-lite-easy-task2-v0": 100.0,
            "chamber-lite-easy-task3-v0": 100.0,
        }[env_name]
    return config


if __name__ == "__main__":
    write_launchers(get_config, run_group)
