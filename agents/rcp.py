"""Reward-conditioned policies with a flow-matching actor and IQL critic."""

from functools import partial
from typing import Any, Optional

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.common import aggregate_q, get_flat_batch
from utils.activation import get_activation
from utils.flax_utils import TrainState, expectile_loss, target_update
from utils.networks import (
    MLP_CLASSES,
    ContinuousConditionedFlowField,
    Value,
)


class RCPAgent(flax.struct.PyTreeNode):
    """RCP-R or RCP-A with a continuously conditioned flow actor.

    The original RCP paper labels actions with Monte Carlo reward-to-go (RCP-R)
    or Monte Carlo advantage (RCP-A). This offline adaptation uses the same IQL
    critic as the repository's other extraction agents: RCP-R uses ``Q(s, a)``
    and RCP-A uses ``Q(s, a) - V(s)``. The actor objective remains unweighted
    conditional behavior cloning, implemented as conditional flow matching.
    """

    support_guidance = False

    rng: Any
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    target_mean: Any
    target_var: Any
    target_updates: Any
    config: dict = flax.struct.field(pytree_node=False)

    def _aggregate_q(self, qs):
        return aggregate_q(qs, self.config)

    def critic_loss(self, batch, critic_params=None):
        """Train the standard action-chunked IQL Q function."""
        horizon = self.config.get("horizon_length", 1)
        actions, next_obs, rewards, masks, valid = get_flat_batch(batch, self.config)
        next_v = self.value(next_obs)
        target_q = rewards + self.config["discount"] ** horizon * masks * next_v
        qs = self.critic(batch["observations"], actions, params=critic_params)
        loss = (((qs - target_q[None]) ** 2) * valid).mean()
        return loss, {
            "critic_loss": loss,
            "q": qs[0].mean(),
            "target_q": target_q.mean(),
        }

    def value_loss(self, batch, value_params=None):
        """Train IQL V by expectile regression onto the target twin-Q."""
        actions, _, _, _, valid = get_flat_batch(batch, self.config)
        q = self._aggregate_q(self.target_critic(batch["observations"], actions))
        v = self.value(batch["observations"], params=value_params)
        loss = (expectile_loss(q - v, self.config["expectile"]) * valid).mean()
        return loss, {
            "value_loss": loss,
            "v": v.mean(),
            "adv": (q - v).mean(),
        }

    def _condition_labels(self, observations, actions):
        """Return IQL return or advantage labels for conditional cloning."""
        critic = self.target_critic if self.config["target_extraction"] else self.critic
        q = self._aggregate_q(critic(observations, actions))
        if self.config["rcp_type"] == "return":
            labels = q
        elif self.config["rcp_type"] == "advantage":
            labels = q - self.value(observations)
        else:
            raise ValueError(
                f"Invalid rcp_type: {self.config['rcp_type']!r}; expected 'return' or 'advantage'"
            )
        return jax.lax.stop_gradient(labels), q

    def actor_loss(self, batch, actor_params=None, rng=None):
        """Unweighted RCP maximum likelihood as a conditional flow loss."""
        if rng is None:
            rng = self.rng
        noise_rng, time_rng = jax.random.split(rng)
        actions, _, _, _, valid = get_flat_batch(batch, self.config)
        observations = batch["observations"]
        labels, q = self._condition_labels(observations, actions)

        noise = jax.random.normal(noise_rng, actions.shape)
        times = jax.random.uniform(time_rng, (actions.shape[0],))
        noised_actions = noise * (1.0 - times[:, None]) + actions * times[:, None]
        target_velocity = actions - noise
        predicted_velocity = self.actor(
            observations,
            labels,
            noised_actions,
            times,
            params=actor_params,
        )
        per_example_loss = jnp.square(predicted_velocity - target_velocity).mean(
            axis=-1
        )
        loss = (per_example_loss * valid).sum() / (valid.sum() + 1e-8)
        return loss, {
            "actor_loss": loss,
            "flow_mse": per_example_loss.mean(),
            "actor_q": q.mean(),
            "label_mean": labels.mean(),
            "label_std": labels.std(),
            "label_min": labels.min(),
            "label_max": labels.max(),
            "target_mean": self.target_mean,
            "target_std": jnp.sqrt(jnp.maximum(self.target_var, 0.0)),
        }

    def _updated_target_stats(self, batch):
        """Approximate p*(Z) proportional to p_data(Z) exp(Z / beta)."""
        actions, _, _, _, valid = get_flat_batch(batch, self.config)
        labels, _ = self._condition_labels(batch["observations"], actions)
        logits = labels / self.config["target_beta"]
        logits = logits - logits.max()
        unnormalized_weights = jnp.exp(logits) * valid
        normalizer = unnormalized_weights.sum()
        weights = jnp.where(
            normalizer > 0,
            unnormalized_weights / jnp.maximum(normalizer, 1e-8),
            jnp.full_like(labels, 1.0 / labels.shape[0]),
        )
        batch_mean = jnp.sum(weights * labels)
        batch_var = jnp.sum(weights * jnp.square(labels - batch_mean))

        rate = self.config["target_stats_rate"]
        mean = jnp.where(
            self.target_updates == 0,
            batch_mean,
            (1.0 - rate) * self.target_mean + rate * batch_mean,
        )
        var = jnp.where(
            self.target_updates == 0,
            batch_var,
            (1.0 - rate) * self.target_var + rate * batch_var,
        )
        return mean, var

    @jax.jit
    def update(self, batch):
        """Update the flow actor, IQL critic/value, and target-value moments."""
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
        target_mean, target_var = self._updated_target_stats(batch)
        return self.replace(
            rng=new_rng,
            actor=new_actor,
            critic=new_critic,
            target_critic=new_target_critic,
            value=new_value,
            target_mean=target_mean,
            target_var=target_var,
            target_updates=self.target_updates + 1,
        ), {**critic_info, **value_info, **actor_info}

    @jax.jit
    def total_loss(self, batch, grad_params=None, rng=None):
        """Compute losses for validation without changing agent state."""
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

    @partial(jax.jit, static_argnames=["rejection_sampling"])
    def sample_actions(
        self,
        observations,
        *,
        seed,
        rejection_sampling=1,
    ):
        """Integrate the flow at the paper's deterministic mu + sigma target."""
        has_batch_dim = observations.ndim == 2
        observations = observations if has_batch_dim else observations[None]
        batch_size = observations.shape[0]
        if rejection_sampling > 1:
            observations = jnp.repeat(observations, rejection_sampling, axis=0)

        action_dim = self.config["action_dim"]
        if self.config.get("action_chunking", False):
            action_dim *= self.config.get("horizon_length", 1)
        actions = jax.random.normal(seed, (observations.shape[0], action_dim))
        target = self.target_mean + self.config["target_std_scale"] * jnp.sqrt(
            jnp.maximum(self.target_var, 0.0)
        )
        conditions = jnp.full((observations.shape[0],), target)
        dt = 1.0 / self.config["denoise_steps"]

        def step(current_actions, step_index):
            times = jnp.full(
                (current_actions.shape[0],),
                step_index / self.config["denoise_steps"],
            )
            velocity = self.actor(observations, conditions, current_actions, times)
            return current_actions + velocity * dt, None

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

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config,
    ):
        """Initialize the conditional flow actor and standard IQL networks."""
        if config["rcp_type"] not in ("return", "advantage"):
            raise ValueError("rcp_type must be 'return' or 'advantage'")
        if config["target_beta"] <= 0:
            raise ValueError("target_beta must be positive")
        if config["mlp_class"] not in MLP_CLASSES:
            raise ValueError(f"Invalid mlp_class: {config['mlp_class']}")
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
        mlp_class = MLP_CLASSES[config["mlp_class"]]
        activation = get_activation(config["activation"])
        actor_def = ContinuousConditionedFlowField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=ex_full_actions.shape[-1],
            mlp_kwargs={
                "activation": activation,
                "layer_norm": config["actor_layer_norm"],
            },
            condition_hidden_dims=config["condition_hidden_dims"],
            layer_norm_affine=config["actor_layer_norm_affine"],
            mlp_class=mlp_class,
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
            **config["value_network_kwargs"],
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
            target_mean=jnp.asarray(0.0, dtype=jnp.float32),
            target_var=jnp.asarray(1.0, dtype=jnp.float32),
            target_updates=jnp.asarray(0, dtype=jnp.int32),
            config=flax.core.FrozenDict(**config),
        )


def get_config(variant: Optional[str] = None):
    """Return sparse OGBench defaults for RCP-R/RCP-A."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name="rcp",
            rcp_type="return",
            batch_size=256,
            actor_lr=3e-4,
            critic_lr=3e-4,
            value_lr=3e-4,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=True,
            actor_layer_norm_affine=True,
            mlp_class="mlp",
            condition_hidden_dims=(128, 128),
            value_network_kwargs=dict(
                hidden_dims=(512, 512, 512, 512), layer_norm=True
            ),
            num_qs=2,
            q_aggregation="min",
            discount=0.99,
            expectile=0.9,
            tau=0.005,
            target_extraction=True,
            activation="gelu",
            horizon_length=5,
            action_chunking=True,
            denoise_steps=10,
            target_beta=1.0,
            target_stats_rate=0.005,
            target_std_scale=1.0,
            dataset_action_clip_eps=None,
        )
    )
    if variant in (None, "return", "rcp_r"):
        return config
    if variant in ("advantage", "rcp_a"):
        config["rcp_type"] = "advantage"
        return config
    raise ValueError(f"Unknown RCP variant: {variant}")
