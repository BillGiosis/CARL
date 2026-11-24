"""Run one matched CARL / CARL-FS timing pair in separate GPU processes."""

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

import yaml


ROOT = Path(__file__).resolve().parents[1]


def pair_summary(profiles):
    full, fixed = profiles
    configs = [deepcopy(p["config"]) for p in profiles]
    for config in configs:
        del config["controller"]["algorithm"], config["logging"]["save_dir"]
    if configs[0] != configs[1]:
        raise ValueError("Effective detector/training configurations differ")
    for key in ("device", "gpu", "seed", "commit", "dirty", "dataset_fingerprint", "torch", "torchvision", "cuda"):
        if full[key] != fixed[key]:
            raise ValueError(f"Profile mismatch in {key}")
    if full["phases"]["validation"]["calls"] != fixed["phases"]["validation"]["calls"]:
        raise ValueError("Validation schedules differ")
    lines = ["| Policy | n | Training h | Peak allocated GiB | Validation min | DQN selection s | DQN updates s | Actions s | Test mAP50 (%) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for profile in profiles:
        phases = profile["phases"]
        row = [profile["training_seconds"]/3600, profile["peak_allocated_bytes"]/2**30,
               phases["validation"]["seconds"]/60, phases["dqn_selection"]["seconds"],
               phases["dqn_update"]["seconds"], phases["action_execution"]["seconds"],
               100*profile["final_test_map50"]]
        lines.append(f'| {profile["policy"]} | 1 | ' + " | ".join(f"{value:.3f}" for value in row) + " |")
    for field, scale, unit in (("training_seconds", 3600, "h"), ("peak_allocated_bytes", 2**30, "GiB")):
        diff = full[field] - fixed[field]
        lines.append(f"\nCARL minus FS: {diff/scale:+.3f} {unit} ({100*diff/fixed[field]:+.2f}% of FS).")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/carl_d.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/profile_pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text())
    dest = args.output.resolve() / f"seed_{args.seed}"
    subprocess.run([sys.executable, "-c",
                    "import sys, torch; d=torch.device(sys.argv[1]); "
                    "assert d.type=='cuda' and torch.cuda.is_available(), 'CUDA required for paired measurements'; "
                    "print(torch.ones(1,device=d).item(), torch.cuda.get_device_name(d), flush=True)",
                    args.device], check=True)
    profiles = []
    for policy in ("dqn", "fixed_schedule"):
        directory = dest / policy
        directory.mkdir(parents=True, exist_ok=True)
        config = deepcopy(base)
        config["seed"] = args.seed
        config["controller"]["algorithm"] = policy
        config["logging"]["save_dir"] = str(directory)
        config_path = directory / "config.yaml"
        if config_path.exists() and yaml.safe_load(config_path.read_text()) != config:
            raise ValueError(f"Different configuration in {directory}. Choose a new --output.")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        complete = []
        for candidate in sorted(directory.glob("attempt-*/profile.json")):
            previous = json.loads(candidate.read_text())
            if previous["status"] == "complete":
                complete.append(candidate)
        if len(complete) > 1:
            raise ValueError(f"Ambiguous complete profiles in {directory}")
        if complete:
            path = complete[0]
        else:
            attempt = Path(mkdtemp(prefix="attempt-", dir=directory))
            path = attempt / "profile.json"
            subprocess.run([
                sys.executable, str(ROOT / "experiments/train_carld.py"), "--config", str(config_path),
                "--device", args.device, "--checkpoint", str(attempt / "final.pt"),
                "--output", str(attempt / "results.txt"), "--profile-output", str(path),
            ], check=True, env={**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(attempt / "inductor"),
                                "TRITON_CACHE_DIR": str(attempt / "triton")})
        profile = json.loads(path.read_text())
        if (profile["status"] != "complete" or profile["policy"] != policy or profile["resumed_from"]
                or profile["seed"] != args.seed or profile["device"] != args.device
                or not profile["peak_allocated_bytes"]):
            raise ValueError(f"Invalid full-GPU profile: {path}")
        profiles.append(profile)
        print(f"{policy}: {path}", flush=True)
    table = pair_summary(profiles)
    (dest / "overhead.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
