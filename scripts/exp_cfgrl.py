"""Generate the ten-seed cfgrl runs for OGBench and OCBench."""

from _common import common_agent_config, write_launchers

run_group = "cfgrl"


def get_config(suite, env_name):
    config = common_agent_config(suite)
    if suite == "ogbench":
        config.update(
            {
                "agent": "agents/cfgrl.py",
                "guidance_weights": "0,0.5,1,2,3,5,10,15",
                "agent.adv_threshold": 0.0,
                "agent.denoise_steps": 10,
            }
        )
    elif suite == "ocbench":
        config.update(
            {
                "agent": "agents/cfgrl.py",
                "guidance_weights": "0,0.5,1,1.5,2,3,5,10,15",
                "agent.actor_lr": 1e-4,
                "agent.critic_lr": 1e-4,
                "agent.value_lr": 1e-4,
                "agent.denoise_steps": 10,
            }
        )
    return config


if __name__ == "__main__":
    write_launchers(get_config, run_group)
