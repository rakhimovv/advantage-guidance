import copy

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.iql import IQLAgent
from typing_extensions import override
from utils.activation import get_activation
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import MLP_CLASSES, ActorFlowField, Value


class IQLDiffusionAgent(IQLAgent):
    """
    IQL agent with diffusion-based actor.

    For the actor loss, grad_log_pi is approximiated with the diffusion loss,
    which is the ELBO of the likelihood.
    """

    def _get_batch_actions(self, batch):
        """Flatten an action chunk, or select the single action at t=0."""
        if self.config.get("action_chunking", False):
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def value_loss(self, batch, grad_params):
        """Compute IQL expectile regression on the configured action representation."""
        batch_actions = self._get_batch_actions(batch)
        valid_w = batch["valid"][..., -1]
        q1, q2 = self.network.select("target_critic")(
            batch["observations"], actions=batch_actions
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select("value")(batch["observations"], params=grad_params)
        value_loss = (
            self.expectile_loss(q - v, q - v, self.config["expectile"]) * valid_w
        ).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute an n-step IQL critic target for single actions or action chunks."""
        H = self.config.get("horizon_length", 1)
        batch_actions = self._get_batch_actions(batch)
        valid_w = batch["valid"][..., -1]
        next_v = self.network.select("value")(batch["next_observations"][..., -1, :])
        target_q = (
            batch["rewards"][..., -1]
            + (self.config["discount"] ** H) * batch["masks"][..., -1] * next_v
        )

        q1, q2 = self.network.select("critic")(
            batch["observations"], actions=batch_actions, params=grad_params
        )
        critic_loss = (((q1 - target_q) ** 2 + (q2 - target_q) ** 2) * valid_w).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": target_q.mean(),
            "q_max": target_q.max(),
            "q_min": target_q.min(),
        }

    @override
    def actor_loss(self, batch, grad_params, rng=None):
        if rng is None:
            rng = self.rng
        rng, eps_rng, time_rng = jax.random.split(rng, 3)

        v = self.network.select("value")(batch["observations"])
        batch_actions = self._get_batch_actions(batch)
        valid_w = batch["valid"][..., -1]
        q1, q2 = self.network.select(
            "target_critic" if self.config["target_extraction"] else "critic"
        )(batch["observations"], actions=batch_actions)
        q = jnp.minimum(q1, q2)
        adv = q - v
        exp_a = jnp.exp(adv * self.config["alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        # compute the diffusion loss
        x1 = batch_actions
        x0 = jax.random.normal(eps_rng, x1.shape)
        t = (
            jax.random.randint(
                time_rng, (x1.shape[0],), 0, self.config["denoise_steps"] + 1
            ).astype(jnp.float32)
            / self.config["denoise_steps"]
        )
        tv = t[..., None]
        x_t = x0 * (1 - tv) + x1 * tv
        vel = x1 - x0

        pred_vel = self.network.select("actor")(
            batch["observations"], x_t, t, params=grad_params
        )
        diffusion_loss = (vel - pred_vel) ** 2

        # weight the diffusion loss
        loss = (diffusion_loss.mean(axis=-1) * exp_a * valid_w).mean()

        return loss, {
            "actor_loss": loss,
            "diffusion_loss": diffusion_loss.mean(),
            "q_mean": q.mean(),
            "q_abs_mean": jnp.abs(q).mean(),
            "adv": adv.mean(),
            "exp_a": exp_a.mean(),
        }

    @override
    def sample_actions(
        self,
        observations,
        seed=None,
        rejection_sampling=1,
    ):
        assert rejection_sampling == 1

        full_action_dim = self.config["action_dim"] * (
            self.config.get("horizon_length", 1)
            if self.config.get("action_chunking", False)
            else 1
        )
        x = jax.random.normal(seed, (observations.shape[0], full_action_dim))
        dt = 1.0 / self.config["denoise_steps"]

        def step(x, t):
            ti = jnp.ones((x.shape[0],)) * (t / self.config["denoise_steps"])
            v = self.network.select("actor")(
                observations, x, ti, params=self.network.params
            )
            x = x + v * dt
            return x, None

        actions, _ = jax.lax.scan(
            step,
            x,
            jnp.arange(self.config["denoise_steps"]),
            length=self.config["denoise_steps"],
        )
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        config["action_dim"] = action_dim

        H = config.get("horizon_length", 1)
        if config.get("action_chunking", False):
            ex_full_actions = jnp.concatenate([ex_actions] * H, axis=-1)
        else:
            ex_full_actions = ex_actions

        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["value"] = encoder_module()
            encoders["critic"] = encoder_module()
            encoders["actor"] = encoder_module()

        activation_fn = get_activation(config["activation"])
        print(f"Using activation function: {activation_fn}")

        # Define networks.
        value_def = Value(
            network_class=MLP_CLASSES[config["mlp_class"]],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
            },
            num_ensembles=1,
            encoder=encoders.get("value"),
        )
        critic_def = Value(
            network_class=MLP_CLASSES[config["mlp_class"]],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
            },
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )
        actor_def = ActorFlowField(
            config["actor_hidden_dims"],
            ex_full_actions.shape[-1],
            mlp_kwargs=dict(
                activation=activation_fn, layer_norm=config["actor_layer_norm"]
            ),
            mlp_class=MLP_CLASSES[config["mlp_class"]],
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_full_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_full_actions),
            ),
            actor=(
                actor_def,
                (ex_observations, ex_full_actions, jnp.zeros(ex_actions.shape[0])),
            ),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config(variant=None):
    config = ml_collections.ConfigDict(
        dict(
            agent_name="iql_diffusion",
            # Common hyperparameters.
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            mlp_class="mlp",
            value_network_kwargs=dict(
                hidden_dims=(512, 512, 512, 512),
                layer_norm=True,
            ),
            activation="gelu",
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            # RL hyperparameters.
            discount=0.99,
            tau=0.005,
            expectile=0.9,  # IQL expectile.
            horizon_length=1,
            action_chunking=False,
            denoise_steps=10,  # Number of denoising steps for the diffusion actor.
            # IQL-diffusion-specific hyperparameters.
            actor_loss="awr",  # Actor loss type: "awr" or "ddpgbc".
            alpha=1.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            target_extraction=True,  # Use target critic (True) or online critic (False) for AWR advantages.
        )
    )
    return config
