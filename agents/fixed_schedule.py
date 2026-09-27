"""Non-learning bookkeeping for the CARL-FS ablation."""

from collections import Counter
from copy import deepcopy


class FixedScheduleController:
    def __init__(self):
        self.decision_steps = 0
        self.training_updates = 0
        self.events = []

    def record(self, event):
        self.events.append(event.to_dict())

    def get_action_distribution(self):
        return dict(Counter(event["action_name"] for event in self.events))

    def get_mean_reward_by_action(self):
        return {}

    def get_action_diagnostics(self):
        return {"decisions": []}

    def state_dict(self):
        return {"algorithm": "fixed_schedule", "decision_steps": self.decision_steps,
                "events": deepcopy(self.events)}

    def load_state_dict(self, state):
        if state.get("algorithm") != "fixed_schedule":
            raise ValueError("CARL-FS requires a fixed-schedule controller checkpoint")
        self.decision_steps = int(state["decision_steps"])
        self.events = deepcopy(state["events"])
