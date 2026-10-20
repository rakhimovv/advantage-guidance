"""Generate the ten-seed rcp_a runs for OGBench and OCBench."""

from _common import common_agent_config, write_launchers

run_group = "rcp_a"


def get_config(suite, env_name):
    config = common_agent_config(suite)
    if suite == "ogbench":
        config.update(
            {
                "agent": "agents/rcp.py",
                "guidance_weights": "1",
                "agent.rcp_type": "advantage",
            }
        )
    elif suite == "ocbench":
        config.update(
            {
                "agent": "agents/rcp.py",
                "guidance_weights": "1",
                "agent.actor_lr": 1e-4,
                "agent.critic_lr": 1e-4,
                "agent.value_lr": 1e-4,
                "agent.denoise_steps": 10,
                "agent.rcp_type": "advantage",
            }
        )
    return config


if __name__ == "__main__":
    write_launchers(get_config, run_group)
