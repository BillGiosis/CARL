"""Synchronized phase measurements for a single training process."""

import json
import os
import shutil
import subprocess
import time
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import torchvision


def _git(*args):
    if not shutil.which("git"):
        return None
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


class TrainingProfile:
    def __init__(self, path, trainer, resumed_from=None):
        self.path = Path(path).expanduser()
        self.device = trainer.device
        status = _git("status", "--porcelain")
        self.data = {
            "status": "incomplete", "run_id": trainer.run_timestamp,
            "policy": "fixed_schedule" if trainer.fixed_schedule else "dqn",
            "seed": trainer.config.get("seed", 0), "config": trainer.config,
            "commit": _git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None,
            "resumed_from": resumed_from, "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "cuda": torch.version.cuda, "amp": trainer.use_amp,
            "compile_dynamic": trainer.compile_dynamic,
            "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
            "dataset_fingerprint": trainer.dataset_fingerprint,
            "phases": {name: {"seconds": 0.0, "calls": 0} for name in
                       ("validation", "dqn_selection", "dqn_update", "action_execution")},
            "actions": [],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(self.data, stream, indent=2)

    def _write(self):
        with NamedTemporaryFile("w", dir=self.path.parent, delete=False, encoding="utf-8") as stream:
            temp = Path(stream.name)
            try:
                json.dump(self.data, stream, indent=2, allow_nan=False)
                stream.flush()
                os.replace(temp, self.path)
            finally:
                temp.unlink(missing_ok=True)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _instrument(self, stack, obj, name, phase, after=None):
        method = getattr(obj, name)

        @wraps(method)
        def timed(*args, **kwargs):
            self._sync()
            start = time.perf_counter()
            result = method(*args, **kwargs)
            self._sync()
            counter = self.data["phases"][phase]
            counter["seconds"] += time.perf_counter() - start
            counter["calls"] += 1
            if after is not None:
                after(result)
            return result

        if name in vars(obj):
            stack.callback(setattr, obj, name, vars(obj)[name])
        else:
            stack.callback(delattr, obj, name)
        setattr(obj, name, timed)

    def train(self, trainer):
        controller = trainer.controller
        previous_updates = controller.training_updates

        def record_action(event):
            decision = {} if trainer.fixed_schedule else controller.action_history[-1]
            self.data["actions"].append({
                "decision_step": controller.decision_steps + int(trainer.fixed_schedule),
                "valid_action_mask": decision.get("valid_action_mask"),
                "selected_action": decision.get("action_name", event.action_name),
                **{key: getattr(event, key) for key in (
                    "task_id", "applies_to_task_id", "epoch", "optimizer_step",
                    "action_name", "changed", "invalid", "duration_epochs", "added_parameters",
                    "replay_loss_weight_before", "replay_loss_weight_after",
                )},
            })

        with ExitStack() as stack:
            self._instrument(stack, trainer, "evaluate_domains", "validation")
            self._instrument(stack, trainer, "execute_action", "action_execution", record_action)
            if not trainer.fixed_schedule:
                self._instrument(stack, controller, "select_action", "dqn_selection")
                self._instrument(stack, controller, "_learn_once", "dqn_update")
            self._sync()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            start = time.perf_counter()
            result = trainer.train()
            self._sync()
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
        updates = controller.training_updates - previous_updates
        if updates != self.data["phases"]["dqn_update"]["calls"]:
            raise RuntimeError("Measured DQN updates disagree with controller bookkeeping")
        self.data.update(status="training_complete", training_seconds=seconds,
                         peak_allocated_bytes=peak, optimizer_updates_dqn=updates)
        self._write()
        return result, seconds

    def finish(self, test, checkpoint, report):
        self.data.update(status="complete", final_test_map50=test["equal_domain_map50"],
                         checkpoint=str(checkpoint), report=str(report))
        self._write()
