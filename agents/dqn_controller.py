"""Masked DQN controller for CARL-D detector adaptation decisions."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CARLDAction(IntEnum):
    NO_OP = 0
    INCREASE_REPLAY_WEIGHT = 1
    REFINE_BUFFER = 2
    FREEZE_BACKBONE = 3
    EXPAND_BACKBONE = 4
    EXPAND_FPN = 5
    EXPAND_ROI = 6
    DECREASE_REPLAY_WEIGHT = 7


ACTION_NAMES: Dict[int, str] = {
    int(action): action.name.lower() for action in CARLDAction
}
EXPANSION_ACTION_GROUPS = {
    int(CARLDAction.EXPAND_BACKBONE): "backbone",
    int(CARLDAction.EXPAND_FPN): "fpn",
    int(CARLDAction.EXPAND_ROI): "roi",
}


STATE_NAMES = (
    "seen_domain_ap50",
    "old_domain_ap50",
    "forgetting",
    "rare_class_ap50",
    "rpn_recall",
    "recognition_loss_ema",
    "localization_loss_ema",
    "buffer_fill_ratio",
    "buffer_capacity_ratio",
    "replay_loss_weight_ratio",
    "parameter_growth_ratio",
    "task_progress",
    "backbone_expansion_ratio",
    "fpn_expansion_ratio",
    "roi_expansion_ratio",
    "structural_expansion_cooldown_ratio",
)


CONTROLLER_SCHEMA_VERSION = 2
FEEDBACK_EMA_METRICS = (
    "seen_domain_ap50",
    "old_domain_ap50",
    "current_domain_ap50",
    "worst_domain_ap50",
    "forgetting",
    "rare_class_ap50",
)
REWARD_WEIGHT_NAMES = (
    "current_ap_gain",
    "old_ap_gain",
    "rare_ap_gain",
    "forgetting_delta",
    "parameter_delta",
    "invalid_action",
)
_FEEDBACK_SCHEMA_MARKER = "_carl_d_controller_feedback_schema"


def _unit_interval(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass(frozen=True)
class ControllerState:
    """Named, ordered schema for the controller's 16 normalized inputs."""

    seen_domain_ap50: float
    old_domain_ap50: float
    forgetting: float
    rare_class_ap50: float
    rpn_recall: float
    recognition_loss_ema: float
    localization_loss_ema: float
    buffer_fill_ratio: float
    buffer_capacity_ratio: float
    replay_loss_weight_ratio: float
    parameter_growth_ratio: float
    task_progress: float
    backbone_expansion_ratio: float
    fpn_expansion_ratio: float
    roi_expansion_ratio: float
    structural_expansion_cooldown_ratio: float

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, Any]) -> "ControllerState":
        recognition_loss = max(0.0, float(metrics.get("recognition_loss_ema", 0.0)))
        localization_loss = max(0.0, float(metrics.get("localization_loss_ema", 0.0)))
        expansion = metrics.get("expansion_ratios", {})
        if not isinstance(expansion, Mapping):
            raise TypeError("controller expansion_ratios must be a mapping")
        values = (
            metrics.get("seen_domain_ap50", 0.0),
            metrics.get("old_domain_ap50", 0.0),
            metrics.get("forgetting", 0.0),
            metrics.get("rare_class_ap50", 0.0),
            metrics.get("rpn_recall", 0.0),
            recognition_loss / (1.0 + recognition_loss),
            localization_loss / (1.0 + localization_loss),
            metrics.get("buffer_fill_ratio", 0.0),
            metrics.get("buffer_capacity_ratio", 0.0),
            metrics.get("replay_loss_weight_ratio", 0.0),
            metrics.get("parameter_growth_ratio", 0.0),
            metrics.get("task_progress", 0.0),
            expansion.get("backbone", 0.0),
            expansion.get("fpn", 0.0),
            expansion.get("roi", 0.0),
            metrics.get("structural_expansion_cooldown_ratio", 0.0),
        )
        return cls(*(_unit_interval(value) for value in values))

    def as_array(self) -> np.ndarray:
        values = tuple(getattr(self, name) for name in STATE_NAMES)
        return np.asarray(values, dtype=np.float32)


def encode_controller_state(metrics: Mapping[str, Any]) -> np.ndarray:
    """Convert named detector/resource measurements to a stable DQN vector."""
    return ControllerState.from_metrics(metrics).as_array()


@dataclass
class DQNTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray

    def __post_init__(self) -> None:
        self.action = int(self.action)
        self.reward = float(self.reward)
        self.done = bool(self.done)
        self.state = np.asarray(self.state, dtype=np.float32).copy()
        self.next_state = np.asarray(self.next_state, dtype=np.float32).copy()
        self.next_action_mask = np.asarray(self.next_action_mask, dtype=np.bool_).copy()
        expected_state = (len(STATE_NAMES),)
        expected_mask = (len(CARLDAction),)
        if (
            self.state.shape != expected_state
            or self.next_state.shape != expected_state
        ):
            raise ValueError(f"DQN states must have shape {expected_state}")
        if not np.isfinite(self.state).all() or not np.isfinite(self.next_state).all():
            raise ValueError("DQN states must contain only finite values")
        if self.next_action_mask.shape != expected_mask:
            raise ValueError(f"DQN next-action masks must have shape {expected_mask}")
        if int(self.action) not in ACTION_NAMES:
            raise ValueError(f"Unknown DQN transition action: {self.action}")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.copy(),
            "action": int(self.action),
            "reward": float(self.reward),
            "next_state": self.next_state.copy(),
            "done": bool(self.done),
            "next_action_mask": self.next_action_mask.copy(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "DQNTransition":
        return cls(
            state=np.asarray(state["state"], dtype=np.float32),
            action=int(state["action"]),
            reward=float(state["reward"]),
            next_state=np.asarray(state["next_state"], dtype=np.float32),
            done=bool(state["done"]),
            next_action_mask=np.asarray(state["next_action_mask"], dtype=np.bool_),
        )


class MaskedReplayMemory:
    """Serializable replay memory that retains next-action masks."""

    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items: Deque[DQNTransition] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, transition: DQNTransition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[DQNTransition]:
        if batch_size > len(self._items):
            raise ValueError("batch_size exceeds stored transitions")
        return self._rng.sample(list(self._items), int(batch_size))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "items": [item.state_dict() for item in self._items],
            "random_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("DQN replay capacity differs from checkpoint")
        self._items.clear()
        for item in state["items"]:
            self._items.append(DQNTransition.from_state_dict(item))
        self._rng.setstate(state["random_state"])


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        dimensions = [int(state_dim), *map(int, hidden_dims), int(action_dim)]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
        output_layer = nn.Linear(dimensions[-2], dimensions[-1])
        # All actions begin with exactly the same value.  This removes a random
        # architectural bias before the online controller has observed data;
        # CARLDAction.NO_OP is used as the explicit greedy tie break.
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class CARLDDQNController:
    """DQN with valid-action selection and valid-action Bellman targets."""

    checkpoint_type = "carl_d_masked_dqn"

    def __init__(
        self,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        replay_capacity: int = 10_000,
        batch_size: int = 8,
        learning_starts: int = 8,
        gradient_steps: int = 4,
        hidden_dims: Sequence[int] = (64, 64),
        epsilon_start: float = 0.25,
        epsilon_end: float = 0.05,
        exploration_steps: int = 100,
        feedback_ema_alpha: float = 0.3,
        reward_weights: Optional[Mapping[str, float]] = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ):
        self.device = torch.device(
            device if str(device) != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.gradient_steps = int(gradient_steps)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.exploration_steps = max(1, int(exploration_steps))
        self.feedback_ema_alpha = float(feedback_ema_alpha)
        if (
            not math.isfinite(self.feedback_ema_alpha)
            or not 0.0 < self.feedback_ema_alpha <= 1.0
        ):
            raise ValueError("feedback_ema_alpha must be in the interval (0, 1]")
        self.reward_weights = {
            "current_ap_gain": 1.0,
            "old_ap_gain": 1.0,
            "rare_ap_gain": 0.5,
            "forgetting_delta": 0.25,
            "parameter_delta": 0.1,
            "invalid_action": 1.0,
            **dict(reward_weights or {}),
        }
        unknown = set(self.reward_weights) - set(REWARD_WEIGHT_NAMES)
        if unknown:
            raise ValueError(f"Unknown reward weights: {sorted(unknown)}")

        self.state_dim = len(STATE_NAMES)
        self.action_dim = len(CARLDAction)
        torch_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            self.q_network = QNetwork(self.state_dim, self.action_dim, hidden_dims).to(
                self.device
            )
            self.target_network = QNetwork(
                self.state_dim, self.action_dim, hidden_dims
            ).to(self.device)
        finally:
            torch.random.set_rng_state(torch_state)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(), lr=float(learning_rate)
        )
        self.replay = MaskedReplayMemory(replay_capacity, seed=seed + 1)
        self._rng = np.random.default_rng(seed + 2)
        self.decision_steps = 0
        self.training_updates = 0
        self.action_history: list[Dict[str, Any]] = []
        self.reward_history: list[Dict[str, Any]] = []
        self._feedback_ema_by_task: Dict[int, Dict[str, float]] = {}
        self._feedback_cache: Dict[tuple[Any, ...], Dict[str, Any]] = {}

    @property
    def epsilon(self) -> float:
        fraction = min(self.decision_steps / self.exploration_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    @staticmethod
    def _feedback_cache_key(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
        """Identify one validation boundary without changing report metrics."""
        boundary_fields = ("task_id", "optimizer_step", "epoch")
        if any(name in metrics for name in boundary_fields):
            return (
                "boundary",
                *(int(metrics.get(name, -1)) for name in boundary_fields),
            )
        # This fallback is primarily useful to callers that exercise the
        # controller without trainer bookkeeping fields. Reusing exactly the
        # same measurement remains idempotent.
        return (
            "measurement",
            int(metrics.get("task_id", -1)),
            *(
                None if name not in metrics else float(metrics[name])
                for name in FEEDBACK_EMA_METRICS
            ),
        )

    def prepare_feedback_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        """Return an EMA-filtered copy for controller state and reward input.

        The input mapping is never mutated. Results are cached per validation
        boundary, making this method safe to call from ``select_action`` and
        ``observe`` for the same probe. Each task has an independent EMA so a
        new domain cannot overwrite the feedback baseline of the previous one.
        """
        if int(metrics.get(_FEEDBACK_SCHEMA_MARKER, -1)) == CONTROLLER_SCHEMA_VERSION:
            return deepcopy(dict(metrics))

        cache_key = self._feedback_cache_key(metrics)
        cached = self._feedback_cache.get(cache_key)
        if cached is not None:
            return deepcopy(cached)

        task_id = int(metrics.get("task_id", -1))
        task_ema = self._feedback_ema_by_task.setdefault(task_id, {})
        prepared = deepcopy(dict(metrics))
        alpha = self.feedback_ema_alpha
        for name in FEEDBACK_EMA_METRICS:
            if name not in metrics:
                continue
            raw_value = float(metrics[name])
            if not math.isfinite(raw_value):
                raise ValueError(f"Controller feedback metric {name!r} must be finite")
            previous = task_ema.get(name)
            filtered = (
                raw_value
                if previous is None
                else alpha * raw_value + (1.0 - alpha) * previous
            )
            task_ema[name] = float(filtered)
            prepared[name] = float(filtered)
        prepared[_FEEDBACK_SCHEMA_MARKER] = CONTROLLER_SCHEMA_VERSION
        self._feedback_cache[cache_key] = deepcopy(prepared)
        return prepared

    def action_mask(
        self,
        valid_action_ids: Iterable[int],
        *,
        allow_empty: bool = False,
    ) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=np.bool_)
        for action_id in valid_action_ids:
            action_id = int(action_id)
            if action_id not in ACTION_NAMES:
                raise ValueError(f"Unknown CARL-D action ID: {action_id}")
            mask[action_id] = True
        if not mask.any() and not allow_empty:
            raise ValueError("At least one action must be valid")
        return mask

    def select_action(
        self,
        metrics: Mapping[str, Any],
        valid_action_ids: Iterable[int],
        *,
        deterministic: bool = False,
    ) -> tuple[int, str]:
        feedback_metrics = self.prepare_feedback_metrics(metrics)
        state = encode_controller_state(feedback_metrics)
        mask = self.action_mask(valid_action_ids)
        valid = np.flatnonzero(mask)
        selection_epsilon = 0.0 if deterministic else self.epsilon
        with torch.no_grad():
            q_values = self.q_network(
                torch.as_tensor(state, device=self.device).unsqueeze(0)
            )[0]
        if not deterministic and self._rng.random() < selection_epsilon:
            action = int(self._rng.choice(valid))
            selection_mode = "epsilon_random"
        else:
            tensor_mask = torch.as_tensor(mask, device=self.device)
            masked_q_values = q_values.masked_fill(~tensor_mask, -torch.inf)
            maximum = masked_q_values.max()
            tied = torch.nonzero(masked_q_values == maximum, as_tuple=False).flatten()
            no_op = int(CARLDAction.NO_OP)
            action = no_op if bool((tied == no_op).any()) else int(tied.min().item())
            selection_mode = (
                "deterministic_greedy" if deterministic else "epsilon_greedy"
            )
        self.decision_steps += 1
        self.action_history.append(
            {
                "decision_step": int(self.decision_steps),
                "task_id": int(feedback_metrics.get("task_id", -1)),
                "optimizer_step": int(feedback_metrics.get("optimizer_step", -1)),
                "action_id": action,
                "action_name": ACTION_NAMES[action],
                "epsilon": selection_epsilon,
                "q_values": [
                    float(value) for value in q_values.detach().cpu().tolist()
                ],
                "valid_action_mask": [bool(value) for value in mask.tolist()],
                "selection_mode": selection_mode,
                "execution": None,
                "reward": None,
                "reward_components": None,
            }
        )
        return action, ACTION_NAMES[action]

    def annotate_last_action_execution(
        self,
        execution: Mapping[str, Any],
        *,
        action: Optional[int] = None,
    ) -> None:
        """Attach the trainer's execution result to its selected decision.

        Call this immediately after the trainer executes the returned action.
        The explicit check prevents diagnostics from being silently associated
        with the wrong transition.
        """
        expected_action = int(
            execution.get("action_id", action if action is not None else -1)
        )
        if expected_action not in ACTION_NAMES:
            raise ValueError("Execution annotation has no valid CARL-D action ID")
        for decision in reversed(self.action_history):
            if decision.get("execution") is not None:
                continue
            if int(decision["action_id"]) != expected_action:
                raise ValueError(
                    "Execution annotation action does not match the latest "
                    "unannotated controller decision"
                )
            decision["execution"] = deepcopy(dict(execution))
            return
        raise RuntimeError("No unannotated controller decision is available")

    def compute_reward(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
        *,
        invalid_action: bool = False,
    ) -> tuple[float, Dict[str, float]]:
        previous_feedback = self.prepare_feedback_metrics(previous)
        current_feedback = self.prepare_feedback_metrics(current)
        current_ap_gain = float(current_feedback["current_domain_ap50"]) - float(
            previous_feedback["current_domain_ap50"]
        )
        old_ap_gain = float(current_feedback["old_domain_ap50"]) - float(
            previous_feedback["old_domain_ap50"]
        )
        rare_ap_gain = float(current_feedback.get("rare_class_ap50", 0.0)) - float(
            previous_feedback.get("rare_class_ap50", 0.0)
        )
        forgetting_delta = max(
            0.0,
            float(current_feedback.get("forgetting", 0.0))
            - float(previous_feedback.get("forgetting", 0.0)),
        )
        parameter_delta = max(
            0.0,
            float(current_feedback.get("parameter_growth_ratio", 0.0))
            - float(previous_feedback.get("parameter_growth_ratio", 0.0)),
        )
        terms = {
            "current_ap_gain": current_ap_gain,
            "old_ap_gain": old_ap_gain,
            "rare_ap_gain": rare_ap_gain,
            "forgetting_delta": forgetting_delta,
            "parameter_delta": parameter_delta,
            "invalid_action": float(bool(invalid_action)),
        }
        contributions = {
            "current_ap_gain_contribution": (
                self.reward_weights["current_ap_gain"] * current_ap_gain
            ),
            "old_ap_gain_contribution": (
                self.reward_weights["old_ap_gain"] * old_ap_gain
            ),
            "rare_ap_gain_contribution": (
                self.reward_weights["rare_ap_gain"] * rare_ap_gain
            ),
            "forgetting_delta_contribution": (
                -self.reward_weights["forgetting_delta"] * forgetting_delta
            ),
            "parameter_delta_contribution": (
                -self.reward_weights["parameter_delta"] * parameter_delta
            ),
            "invalid_action_contribution": (
                -self.reward_weights["invalid_action"] * float(bool(invalid_action))
            ),
        }
        reward = sum(contributions.values())
        components = {
            **terms,
            **contributions,
            "reward_total": float(reward),
        }
        return float(reward), components

    def observe(
        self,
        previous_metrics: Mapping[str, Any],
        action: int,
        current_metrics: Mapping[str, Any],
        *,
        next_state_metrics: Optional[Mapping[str, Any]] = None,
        done: bool,
        next_valid_action_ids: Iterable[int],
        invalid_action: bool = False,
    ) -> float:
        action = int(action)
        if action not in ACTION_NAMES:
            raise ValueError(f"Unknown CARL-D action ID: {action}")
        previous_feedback = self.prepare_feedback_metrics(previous_metrics)
        current_feedback = self.prepare_feedback_metrics(current_metrics)
        reward, components = self.compute_reward(
            previous_feedback, current_feedback, invalid_action=invalid_action
        )
        # A terminal state has no successor decision and therefore may have no
        # valid action. Non-terminal transitions must retain at least one valid
        # action for the masked Bellman maximum.
        next_mask = self.action_mask(next_valid_action_ids, allow_empty=bool(done))
        successor_metrics = self.prepare_feedback_metrics(
            current_feedback if next_state_metrics is None else next_state_metrics
        )
        transition = DQNTransition(
            state=encode_controller_state(previous_feedback),
            action=action,
            reward=reward,
            next_state=encode_controller_state(successor_metrics),
            done=bool(done),
            next_action_mask=next_mask,
        )
        self.replay.append(transition)
        matched_decision: Optional[Dict[str, Any]] = None
        for decision in reversed(self.action_history):
            if decision.get("reward") is not None:
                continue
            matched_decision = decision
            break
        if (
            matched_decision is not None
            and int(matched_decision["action_id"]) != action
        ):
            raise RuntimeError(
                "Observed action does not match the latest unrewarded "
                "controller decision"
            )
        if matched_decision is not None:
            matched_decision["reward"] = float(reward)
            matched_decision["reward_components"] = deepcopy(components)
        self.reward_history.append(
            {
                "decision_step": (
                    None
                    if matched_decision is None
                    else int(matched_decision["decision_step"])
                ),
                "task_id": (
                    int(previous_feedback.get("task_id", -1))
                    if matched_decision is None
                    else int(matched_decision["task_id"])
                ),
                "action_id": action,
                "action_name": ACTION_NAMES[action],
                **components,
            }
        )
        if len(self.replay) >= max(self.learning_starts, self.batch_size):
            for _ in range(self.gradient_steps):
                self._learn_once()
        return reward

    def _bellman_targets(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_q_values: torch.Tensor,
        next_masks: torch.Tensor,
    ) -> torch.Tensor:
        no_valid = ~next_masks.any(dim=1)
        if torch.any(no_valid & ~dones):
            raise ValueError("A non-terminal transition has no valid next action")
        safe_masks = next_masks.clone()
        safe_masks[no_valid, int(CARLDAction.NO_OP)] = True
        masked_q = next_q_values.masked_fill(~safe_masks, -torch.inf)
        max_next = masked_q.max(dim=1).values
        return rewards + self.gamma * (~dones).float() * max_next

    def _learn_once(self) -> float:
        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]), device=self.device
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]), device=self.device
        )
        dones = torch.as_tensor(
            [item.done for item in batch], dtype=torch.bool, device=self.device
        )
        next_masks = torch.as_tensor(
            np.stack([item.next_action_mask for item in batch]),
            dtype=torch.bool,
            device=self.device,
        )

        predicted = self.q_network(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states)
            targets = self._bellman_targets(rewards, dones, next_q, next_masks)
        loss = F.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        with torch.no_grad():
            for target, online in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target.lerp_(online, self.tau)
        self.training_updates += 1
        return float(loss.item())

    def get_action_distribution(self) -> Dict[str, int]:
        return dict(Counter(item["action_name"] for item in self.action_history))

    def get_mean_reward_by_action(self) -> Dict[str, Optional[float]]:
        rewards: Dict[int, list[float]] = defaultdict(list)
        for item in self.reward_history:
            action_id = int(item["action_id"])
            rewards[action_id].append(float(item["reward_total"]))
        return {
            ACTION_NAMES[action_id]: (
                None
                if not rewards[action_id]
                else float(sum(rewards[action_id]) / len(rewards[action_id]))
            )
            for action_id in range(self.action_dim)
        }

    def get_action_diagnostics(self) -> Dict[str, Any]:
        """Return report-ready selection, execution, and reward diagnostics."""
        return {
            "action_distribution": self.get_action_distribution(),
            "mean_reward_by_action": self.get_mean_reward_by_action(),
            "decisions": deepcopy(self.action_history),
            "rewards": deepcopy(self.reward_history),
        }

    def _controller_schema(self) -> Dict[str, Any]:
        return {
            "version": CONTROLLER_SCHEMA_VERSION,
            "state_names": list(STATE_NAMES),
            "action_names": [
                ACTION_NAMES[action_id] for action_id in range(self.action_dim)
            ],
            "feedback_ema_metrics": list(FEEDBACK_EMA_METRICS),
            "reward_weight_names": list(REWARD_WEIGHT_NAMES),
        }

    def _validate_checkpoint_schema(self, state: Mapping[str, Any]) -> None:
        if state.get("checkpoint_type") != self.checkpoint_type:
            raise ValueError("Checkpoint is not a CARL-D masked DQN state")
        schema = state.get("controller_schema")
        if not isinstance(schema, Mapping):
            raise ValueError(
                "Controller checkpoint predates the replay-intensity action/state "
                "schema and cannot be resumed"
            )
        expected = self._controller_schema()
        for field, expected_value in expected.items():
            observed = schema.get(field)
            if observed != expected_value:
                raise ValueError(
                    f"Incompatible CARL-D controller {field}: "
                    f"checkpoint={observed!r}, expected={expected_value!r}"
                )

        saved_alpha = float(state.get("feedback_ema_alpha", float("nan")))
        if not math.isclose(
            saved_alpha, self.feedback_ema_alpha, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "Controller feedback EMA alpha differs from the checkpoint"
            )
        saved_weights = state.get("reward_weights")
        if not isinstance(saved_weights, Mapping) or set(saved_weights) != set(
            REWARD_WEIGHT_NAMES
        ):
            raise ValueError("Controller reward schema differs from the checkpoint")
        for name in REWARD_WEIGHT_NAMES:
            if not math.isclose(
                float(saved_weights[name]),
                float(self.reward_weights[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Controller reward weight {name!r} differs from the checkpoint"
                )

        replay_state = state.get("replay")
        if not isinstance(replay_state, Mapping):
            raise ValueError("Controller checkpoint has no valid replay state")
        for index, item in enumerate(replay_state.get("items", ())):
            transition = DQNTransition.from_state_dict(item)
            if transition.state.shape != (self.state_dim,):
                raise ValueError(
                    f"Replay transition {index} has an incompatible state shape"
                )
            if transition.next_state.shape != (self.state_dim,):
                raise ValueError(
                    f"Replay transition {index} has an incompatible next-state shape"
                )
            if transition.action not in ACTION_NAMES:
                raise ValueError(f"Replay transition {index} has an unknown action ID")
            if transition.next_action_mask.shape != (self.action_dim,):
                raise ValueError(
                    f"Replay transition {index} has an incompatible action mask"
                )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_type": self.checkpoint_type,
            "controller_schema": self._controller_schema(),
            "feedback_ema_alpha": self.feedback_ema_alpha,
            "reward_weights": dict(self.reward_weights),
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "decision_steps": self.decision_steps,
            "training_updates": self.training_updates,
            "rng_state": self._rng.bit_generator.state,
            "action_history": self.action_history,
            "reward_history": self.reward_history,
            "feedback_ema_by_task": deepcopy(self._feedback_ema_by_task),
            "feedback_cache": deepcopy(self._feedback_cache),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._validate_checkpoint_schema(state)
        self.q_network.load_state_dict(state["q_network"])
        self.target_network.load_state_dict(state["target_network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.decision_steps = int(state["decision_steps"])
        self.training_updates = int(state["training_updates"])
        self._rng.bit_generator.state = state["rng_state"]
        self.action_history = deepcopy(list(state["action_history"]))
        self.reward_history = deepcopy(list(state["reward_history"]))
        self._feedback_ema_by_task = {
            int(task_id): {
                str(name): float(value) for name, value in task_metrics.items()
            }
            for task_id, task_metrics in state["feedback_ema_by_task"].items()
        }
        self._feedback_cache = deepcopy(dict(state["feedback_cache"]))


def create_dqn_controller(config: Mapping[str, Any], device: str) -> CARLDDQNController:
    controller = config.get("controller", {})
    return CARLDDQNController(
        learning_rate=controller.get("learning_rate", 3e-4),
        gamma=controller.get("gamma", 0.99),
        tau=controller.get("tau", 0.01),
        replay_capacity=controller.get("replay_capacity", 10_000),
        batch_size=controller.get("batch_size", 8),
        learning_starts=controller.get("learning_starts", 8),
        gradient_steps=controller.get("gradient_steps", 4),
        hidden_dims=controller.get("hidden_dims", (64, 64)),
        epsilon_start=controller.get("epsilon_start", 0.25),
        epsilon_end=controller.get("epsilon_end", 0.05),
        exploration_steps=controller.get("exploration_steps", 100),
        feedback_ema_alpha=controller.get("feedback_ema_alpha", 0.3),
        reward_weights=config.get("reward", {}),
        device=device,
        seed=config.get("seed", 0),
    )
