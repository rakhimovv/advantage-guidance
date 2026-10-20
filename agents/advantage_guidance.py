from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.common import aggregate_q, get_flat_batch
from utils.activation import get_activation
from utils.flax_utils import TrainState, expectile_loss, target_update
from utils.networks import MLP_CLASSES, ConditionalAndUncondFlowField, Value


class AdvantageGuidanceAgent(flax.struct.PyTreeNode):
    """Advantage Guidance. https://arxiv.org/pdf/2610.19783

    An RL algorithm that trains with supervised learning objective and use contrastive
    diffusion guidance at inference time for advantage maximization.

    Training:
    Trains an advantage-conditioned policy and makes conditioning signals effective by
    (1) for each state, augment its dataset action with K imagined samples from an unconditional
    policy (i.e. reference BC flow) and (2) train the conditional policy as a residual flow
    on top of the reference unconditional BC flow.

    Inference:
    Contrastive guidance integrates v_uncond + guidance_weight * (v_cond(y_pos) -
    v_cond(y_neg)), where y_pos and y_neg are fixed advantage labels.
    """

    support_guidance = True

    rng: Any
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def _aggregate_q(self, qs):
        return aggregate_q(qs, self.config)

    def _advantage_normalization_stats(self, grouped_advantage):
        """Return per-state moments used to normalize candidate advantages."""
        return (
            grouped_advantage.mean(axis=1, keepdims=True),
            grouped_advantage.std(axis=1, keepdims=True),
        )

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------

    def critic_loss(self, batch, critic_params=None):
        """IQL critic loss on dataset transitions."""
        horizon = self.config.get("horizon_length", 1)
        actions, next_obs, rewards, masks, valid = get_flat_batch(batch, self.config)
        next_v = self.value(next_obs)
        target_q = rewards + self.config["discount"] ** horizon * masks * next_v
        qs = self.critic(batch["observations"], actions, params=critic_params)
        loss = (((qs - target_q[None]) ** 2) * valid).mean()
        return loss, {"critic_loss": loss, "q": qs[0].mean()}

    def value_loss(self, batch, value_params=None):
        """IQL expectile value loss on dataset actions."""
        actions, _, _, _, valid = get_flat_batch(batch, self.config)
        qs = self.target_critic(batch["observations"], actions)
        q = self._aggregate_q(qs)
        v = self.value(batch["observations"], params=value_params)
        loss = (expectile_loss(q - v, self.config["expectile"]) * valid).mean()
        return loss, {"value_loss": loss, "v": v.mean()}

    def _sample_augmented_actions(self, observations, seed, num_samples):
        """Draw K unconditional BC actions per state, in state-major order."""
        action_dim = self.config["action_dim"]
        horizon = self.config.get("horizon_length", 1)
        if self.config.get("action_chunking", False):
            action_dim *= horizon

        if num_samples == 0:
            # Keep concatenation valid when training with dataset actions only.
            return jnp.empty((0, action_dim), dtype=jnp.float32)

        # Each state has K consecutive candidates in the flattened sampling batch.
        observations = jnp.repeat(observations, num_samples, axis=0)

        actions = jax.random.normal(seed, (observations.shape[0], action_dim))
        dt = 1.0 / self.config["denoise_steps"]

        def step(actions, step_index):
            times = jnp.full(
                (actions.shape[0],),
                step_index / self.config["denoise_steps"],
            )
            velocity = self.actor(
                observations,
                actions,
                times,
                method="unconditional",
            )
            return actions + velocity * dt, None

        actions, _ = jax.lax.scan(
            step,
            actions,
            jnp.arange(self.config["denoise_steps"]),
            length=self.config["denoise_steps"],
        )
        actions = jnp.clip(actions, -1, 1)
        # Treat augmented actions as training targets, without backpropagating through sampling.
        return jax.lax.stop_gradient(actions)

    def actor_loss(self, batch, actor_params=None, rng=None):
        """Unconditional BC and advantage-conditioned flow-matching losses."""
        if rng is None:
            rng = self.rng
        (
            sample_rng,
            uncond_eps_rng,
            uncond_time_rng,
            cond_eps_rng,
            cond_time_rng,
            label_rng,
        ) = jax.random.split(rng, 6)

        observations = batch["observations"]
        batch_size = observations.shape[0]
        num_samples = self.config["num_imagined_actions"]
        if self.config.get("action_chunking", False):
            real_actions = jnp.reshape(batch["actions"], (batch_size, -1))
        else:
            real_actions = batch["actions"][..., 0, :]

        # Train the unconditional BC velocity on real dataset actions.
        uncond_noise = jax.random.normal(uncond_eps_rng, real_actions.shape)
        uncond_times = (
            jax.random.randint(
                uncond_time_rng,
                (batch_size,),
                0,
                self.config["denoise_steps"] + 1,
            ).astype(jnp.float32)
            / self.config["denoise_steps"]
        )
        uncond_noised_actions = (
            uncond_noise * (1 - uncond_times[:, None])
            + real_actions * uncond_times[:, None]
        )
        uncond_target_velocity = real_actions - uncond_noise
        trainable_unconditional_velocity = self.actor(
            observations,
            uncond_noised_actions,
            uncond_times,
            params=actor_params,
            method="unconditional",
        )
        unconditional_loss = (
            (uncond_target_velocity - trainable_unconditional_velocity) ** 2
        ).mean(axis=-1)

        # Train the conditional flow on dataset and imagined actions.
        # The unconditional flow outputs here has a stop_gradient applied
        imagined_actions = self._sample_augmented_actions(
            observations, sample_rng, num_samples
        )
        repeated_observations = jnp.concatenate(
            [observations, jnp.repeat(observations, num_samples, axis=0)], axis=0
        )
        # Layout: B dataset actions, followed by B groups of K imagined actions.
        concated_actions = jnp.concatenate([real_actions, imagined_actions], axis=0)

        # V depends only on the state, so evaluate each state once instead of
        # forwarding the K + 1 repeated copies through the value network.
        v_single = self.value(observations)
        v = jnp.concatenate(
            [v_single, jnp.repeat(v_single, num_samples, axis=0)], axis=0
        )
        critic = self.target_critic if self.config["target_extraction"] else self.critic
        q = self._aggregate_q(critic(repeated_observations, concated_actions))
        advantage = q - v

        # Regroup as (B, K + 1) to normalize advantage within each state.
        grouped_advantage = jnp.concatenate(
            [
                advantage[:batch_size, None],
                advantage[batch_size:].reshape(batch_size, num_samples),
            ],
            axis=1,
        )
        state_adv_mean, state_adv_std = self._advantage_normalization_stats(
            grouped_advantage
        )
        normalized_adv = (grouped_advantage - state_adv_mean) / (
            state_adv_std + self.config["adv_norm_eps"]
        )
        # Restore the dataset-first layout to align advantage labels with actions.
        normalized_adv = jnp.concatenate(
            [normalized_adv[:, 0], normalized_adv[:, 1:].reshape(-1)]
        )

        # Sample advantage labels from a Gaussian distribution.
        gaussian_adv_labels = normalized_adv + self.config[
            "adv_gaussian_std"
        ] * jax.random.normal(label_rng, normalized_adv.shape)
        cond_noise = jax.random.normal(cond_eps_rng, concated_actions.shape)
        cond_times = (
            jax.random.randint(
                cond_time_rng,
                (concated_actions.shape[0],),
                0,
                self.config["denoise_steps"] + 1,
            ).astype(jnp.float32)
            / self.config["denoise_steps"]
        )
        cond_noised_actions = (
            cond_noise * (1 - cond_times[:, None])
            + concated_actions * cond_times[:, None]
        )
        cond_target_velocity = concated_actions - cond_noise
        conditional_velocity, _unconditional_velocity = self.actor(
            repeated_observations,
            gaussian_adv_labels,
            cond_noised_actions,
            cond_times,
            params=actor_params,
        )
        conditional_loss = ((cond_target_velocity - conditional_velocity) ** 2).mean(
            axis=-1
        )

        loss = (
            conditional_loss.mean()
            + self.config["uncond_loss_weight"] * unconditional_loss.mean()
        )

        # diagnostics stats
        real_advantage = advantage[:batch_size]
        info = {
            "actor_loss": loss,
            "actor_cond_loss": conditional_loss.mean(),
            "actor_uncond_loss": unconditional_loss.mean(),
            "actor_q": q.mean(),
            "adv_mean": advantage.mean(),
            "adv_min": advantage.min(),
            "adv_max": advantage.max(),
            "gaussian_adv_max": gaussian_adv_labels.max(),
            "gaussian_adv_min": gaussian_adv_labels.min(),
            "adv_real_mean": real_advantage.mean(),
            "state_adv_std_mean": state_adv_std.mean(),
            "state_adv_std_min": state_adv_std.min(),
            "state_adv_std_max": state_adv_std.max(),
        }
        if num_samples > 0:
            imagined_advantage = advantage[batch_size:].reshape(batch_size, num_samples)
            info.update(
                {
                    "adv_imagined_mean": imagined_advantage.mean(),
                    "adv_imagined_minus_real": imagined_advantage.mean()
                    - real_advantage.mean(),
                    "max_imagined_adv_minus_real_adv": (
                        imagined_advantage.max(axis=1) - real_advantage
                    ).mean(),
                }
            )
        return loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Combined validation loss; ``grad_params`` refers to actor parameters."""
        if rng is None:
            rng = self.rng
        actor_params = self.actor.params if grad_params is None else grad_params
        critic_loss, critic_info = self.critic_loss(batch)
        value_loss, value_info = self.value_loss(batch)
        actor_loss, actor_info = self.actor_loss(batch, actor_params, rng=rng)
        return critic_loss + value_loss + actor_loss, {
            **critic_info,
            **value_info,
            **actor_info,
        }

    @jax.jit
    def update(self, batch):
        """Update the actor, IQL critic, target critic, and value network."""
        new_rng, actor_rng = jax.random.split(self.rng)
        new_critic, critic_info = self.critic.apply_loss_fn(
            loss_fn=lambda params: self.critic_loss(batch, params)
        )
        new_target_critic = target_update(
            self.critic, self.target_critic, self.config["tau"]
        )
        new_value, value_info = self.value.apply_loss_fn(
            loss_fn=lambda params: self.value_loss(batch, params)
        )
        new_actor, actor_info = self.actor.apply_loss_fn(
            loss_fn=lambda params: self.actor_loss(batch, params, rng=actor_rng)
        )
        return self.replace(
            rng=new_rng,
            actor=new_actor,
            critic=new_critic,
            target_critic=new_target_critic,
            value=new_value,
        ), {**critic_info, **value_info, **actor_info}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnames=["rejection_sampling"])
    def sample_actions(
        self,
        observations: jnp.ndarray,
        *,
        seed: Any,
        guidance_weight: float = 1.0,
        rejection_sampling: int = 1,
    ) -> jnp.ndarray:
        """Sample with contrastive guidance (CFG).

        At each Euler step, use v_uncond + guidance_weight * (v_cond(y_pos) -
        v_cond(y_neg)). Advantage labels use the same per-state normalized scale
        as training. With rejection_sampling > 1, return the highest-Q candidate.
        """
        has_batch_dim = observations.ndim == 2
        observations = observations if has_batch_dim else observations[None]
        batch_size = observations.shape[0]
        if rejection_sampling > 1:
            observations = jnp.repeat(observations, rejection_sampling, axis=0)

        action_dim = self.config["action_dim"]
        if self.config.get("action_chunking", False):
            action_dim *= self.config.get("horizon_length", 1)
        actions = jax.random.normal(seed, (observations.shape[0], action_dim))
        positive_labels = jnp.full(
            (actions.shape[0],),
            self.config["cg_positive_adv"],
            dtype=jnp.float32,
        )
        negative_labels = jnp.full(
            (actions.shape[0],),
            self.config["cg_negative_adv"],
            dtype=jnp.float32,
        )
        dt = 1.0 / self.config["denoise_steps"]

        def step(actions, step_index):
            times = jnp.full(
                (actions.shape[0],),
                step_index / self.config["denoise_steps"],
            )
            unconditional_velocity = self.actor(
                observations,
                actions,
                times,
                method="unconditional",
            )
            contrastive_velocity = self.actor(
                observations,
                positive_labels,
                negative_labels,
                actions,
                times,
                method="conditional_contrast",
            )
            velocity = unconditional_velocity + guidance_weight * contrastive_velocity
            return actions + velocity * dt, None

        actions, _ = jax.lax.scan(
            step,
            actions,
            jnp.arange(self.config["denoise_steps"]),
            length=self.config["denoise_steps"],
        )
        actions = jnp.clip(actions, -1, 1)

        if rejection_sampling > 1:
            q = self._aggregate_q(self.critic(observations, actions))
            q = q.reshape(batch_size, rejection_sampling)
            actions = actions.reshape(batch_size, rejection_sampling, -1)
            actions = actions[jnp.arange(batch_size), jnp.argmax(q, axis=1)]
        if not has_batch_dim:
            actions = actions[0]
        return actions

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config,
    ):
        """Initialize the advantage-conditioned actor and IQL value functions."""
        assert config["num_imagined_actions"] >= 0

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        action_dim = ex_actions.shape[-1]
        horizon = config.get("horizon_length", 1)
        if config.get("action_chunking", False):
            ex_full_actions = jnp.concatenate([ex_actions] * horizon, axis=-1)
        else:
            ex_full_actions = ex_actions

        config = dict(config)
        config["action_dim"] = action_dim
        activation = get_activation(config["activation"])
        if config["mlp_class"] not in MLP_CLASSES:
            raise ValueError(f"Invalid mlp_class: {config['mlp_class']}")
        mlp_class = MLP_CLASSES[config["mlp_class"]]

        actor_kwargs = dict(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=ex_full_actions.shape[-1],
            mlp_kwargs={
                "activation": activation,
                "layer_norm": config["actor_layer_norm"],
            },
            mlp_class=mlp_class,
        )
        actor_def = ConditionalAndUncondFlowField(
            **actor_kwargs,
            condition_hidden_dims=config["adaln_condition_hidden_dims"],
            residual_parameterization=config["residual_parameterization"],
            # use bfloat16 for matmul for better efficiency
            dtype=jnp.bfloat16,
        )
        ex_labels = jnp.zeros((ex_actions.shape[0],), dtype=jnp.float32)
        ex_times = jnp.zeros((ex_actions.shape[0],), dtype=jnp.float32)
        actor_params = actor_def.init(
            actor_key,
            ex_observations,
            ex_labels,
            ex_full_actions,
            ex_times,
        )["params"]
        actor = TrainState.create(
            actor_def,
            actor_params,
            tx=optax.adam(learning_rate=config["actor_lr"]),
        )

        value_kwargs = {
            "hidden_dims": config["value_hidden_dims"],
            "layer_norm": config["value_layer_norm"],
            "activation": activation,
        }
        critic_def = Value(
            network_class=mlp_class,
            network_kwargs=value_kwargs,
            num_ensembles=config["num_qs"],
        )
        critic_params = critic_def.init(critic_key, ex_observations, ex_full_actions)[
            "params"
        ]
        critic = TrainState.create(
            critic_def,
            critic_params,
            tx=optax.adam(learning_rate=config["critic_lr"]),
        )
        target_critic = TrainState.create(critic_def, critic_params)

        value_def = Value(
            network_class=mlp_class,
            network_kwargs=value_kwargs,
            num_ensembles=1,
        )
        value_params = value_def.init(value_key, ex_observations)["params"]
        value = TrainState.create(
            value_def,
            value_params,
            tx=optax.adam(learning_rate=config["value_lr"]),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            value=value,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    """Default network, IQL, and advantage-guidance hyperparameters."""
    return ml_collections.ConfigDict(
        dict(
            agent_name="advantage_guidance",
            # Common hyperparameters.
            batch_size=256,
            actor_lr=3e-4,
            critic_lr=3e-4,
            value_lr=3e-4,
            mlp_class="mlp",
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=True,
            adaln_condition_hidden_dims=(
                128,
                128,
            ),  # AdaLN for the conditional advantage inputs
            value_hidden_dims=(128, 128),
            value_layer_norm=True,
            activation="gelu",
            # RL critic hyperparameters.
            num_qs=2,
            q_aggregation="min",
            discount=0.99,
            expectile=0.9,
            tau=0.005,
            target_extraction=True,  # Use the target critic to compute advantage.
            # N-step returns & action chunking.
            horizon_length=5,
            action_chunking=True,
            # Actor hyperparameters
            denoise_steps=10,
            uncond_loss_weight=0.1,  # loss weight for the unconditional flow.
            # Advantage Guidance hyperparameters.
            num_imagined_actions=7,  # K generated actions plus one dataset action per state.
            residual_parameterization=True,  # Use residual parameterization for the conditional flow.
            adv_norm_eps=1e-8,  # Stabilize per-state advantage standardization.
            adv_gaussian_std=0.01,  # Noise std added to normalized advantage labels.
            # Advantage labels (normalized) for contrastive guidance at inference.
            cg_positive_adv=1.0,
            cg_negative_adv=-1.0,
        )
    )
