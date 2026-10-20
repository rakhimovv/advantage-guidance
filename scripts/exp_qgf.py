"""Generate the ten-seed qgf runs for OGBench and OCBench."""

from _common import common_agent_config, write_launchers

run_group = "qgf"


def get_config(suite, env_name):
    config = common_agent_config(suite)
    if suite == "ogbench":
        config.update(
            {
                "agent": "agents/qgf.py",
                "guidance_weights": "0,0.002,0.004,0.01,0.02,0.04,0.08,0.16,0.32",
                "agent.denoise_steps": 10,
            }
        )
    elif suite == "ocbench":
        config.update(
            {
                "agent": "agents/qgf.py",
                "guidance_weights": "0,0.002,0.004,0.01,0.02,0.04,0.08,0.16,0.32",
                "agent.bc_lr": 1e-4,
                "agent.critic_lr": 1e-4,
                "agent.value_lr": 1e-4,
                "agent.denoise_steps": 10,
                "agent.denoised_action_approx": "one_euler_step_approx",
                "agent.apply_jacobian": False,
            }
        )
    return config


if __name__ == "__main__":
    write_launchers(get_config, run_group)
