"""Generate the ten-seed advantage_guidance runs for OGBench and OCBench."""

from _common import common_agent_config, write_launchers

run_group = "advantage_guidance"


def get_config(suite, env_name):
    config = common_agent_config(suite, value_hidden_dims_key="value_hidden_dims")
    if suite == "ogbench":
        config.update(
            {
                "agent": "agents/advantage_guidance.py",
                "guidance_weights": "0,0.75,1,1.25,1.5,2,2.5,4",
                "agent.batch_size": 256,
                "agent.num_imagined_actions": 7,
                "agent.adv_gaussian_std": 0.1,
                "agent.residual_parameterization": True,
                "agent.uncond_loss_weight": 0.1,
                "agent.cg_positive_adv": 1.0,
                "agent.cg_negative_adv": -1.0,
            }
        )
    elif suite == "ocbench":
        config.update(
            {
                "agent": "agents/advantage_guidance.py",
                "guidance_weights": "0,0.5,1,1.5,2,3,4",
                "agent.actor_lr": 1e-4,
                "agent.critic_lr": 1e-4,
                "agent.value_lr": 1e-4,
                "agent.actor_layer_norm": True,
                "agent.value_layer_norm": True,
                "agent.adaln_condition_hidden_dims": "(128,128)",
                "agent.denoise_steps": 10,
                "agent.num_imagined_actions": 7,
                "agent.residual_parameterization": True,
                "agent.uncond_loss_weight": 0.1,
                "agent.adv_norm_eps": 1e-8,
                "agent.adv_gaussian_std": 0.01,
                "agent.cg_positive_adv": 1.0,
                "agent.cg_negative_adv": -1.0,
            }
        )
        if env_name == "block-lite-single-task1-v0":
            config["offline_steps"] = 100_000
    return config


if __name__ == "__main__":
    write_launchers(get_config, run_group)
