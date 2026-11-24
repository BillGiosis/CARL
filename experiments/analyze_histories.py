"""Join CARL test reports to their training histories and summarize repetitions."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


VARIANTS = ("CARL", "CARL-FS", "CARL-NA", "CARL-NB", "CARL-ND", "CARL-NR")
SCORE = r"([01]\.\d+)"


def parse(path):
    text = path.read_text(encoding="utf-8-sig")
    checkpoint = re.search(r"^Checkpoint: (.+)$", text, re.M)
    if not checkpoint:
        raise ValueError(f"No checkpoint identifier in {path}")
    key = checkpoint[1].strip().replace("\\", "/").rsplit("/", 1)[-1]
    training = "TRAINING RESULTS" in text.splitlines()[1]
    test = text.partition("\nFINAL TEST PERFORMANCE\n")[2] if training else text
    test = test.split("\nAP PROTOCOL COMPARISON\n")[0]
    rows = {}

    def scalar(section, pattern, metric, split):
        match = re.search(pattern, section, re.M)
        if match:
            rows[(split, metric, "", "")] = match[1]

    scalar(test, r"^(?:CLAD-D 2023 Equal-Domain mAP50|Equal-Domain mAP50): " + SCORE,
           "mAP50", "test")
    classes = re.search(r"(?:^Per-Class Equal-Domain AP50:|^PER-CLASS EQUAL-DOMAIN COCO 101-POINT AP50)\n(?:=+\n)?((?:[^\n]+\n)+)", test, re.M)
    if classes:
        for name, value in re.findall(r"^ *([^:\n]+): " + SCORE, classes[1], re.M):
            rows[("test", "class_AP50", "", name.strip())] = value
    for domain, value in re.findall(r"^ *Domain (\d+) \([^\n]+\) - mAP50: " + SCORE, test, re.M):
        rows[("test", "domain_mAP50", "", domain)] = value
    for group, value in re.findall(r"^(Day|Night|Highway|City|Country) - mAP50: " + SCORE, text, re.M):
        rows[("test", "environment_mAP50", "", group)] = value
    if training:
        for field in ("Forgetting", "Plasticity", "Backward Transfer"):
            scalar(text, "^" + field + r": (-?\d+\.\d+)$", field, "validation")
        matrix = {}
        for task, line in re.findall(r"^After Task (\d+): (.+)$", text, re.M):
            for domain, value in re.findall(r"D(\d+)=" + SCORE, line):
                matrix[(task, domain)] = value
                rows[("validation", "domain_mAP50", task, domain)] = value
        for domain in ("1", "2", "3"):
            if (domain, domain) in matrix and ("4", domain) in matrix:
                delta = float(matrix["4", domain]) - float(matrix[domain, domain])
                rows[("validation", "final_minus_acquisition_pp", "4", domain)] = f"{100*delta:.4f}"
        for task, block in re.findall(r"^Task (\d+):\n((?:  .+\n)+)", text, re.M):
            weights = re.search(r"Replay Loss Weight: ([\d.]+) -> ([\d.]+)", block)
            if weights:
                for field, value in zip(("weight_before", "weight_after"), weights.groups()):
                    rows[("training", field, task, "")] = value
        task_lines = re.findall(r"^  Task (\d+): (.+)$", text.partition("Actions Per Task:")[2], re.M)
        for task, line in task_lines:
            selected = dict(re.findall(r"(\w+)=(\d+)", line))
            for action in ("no_op", "increase_replay_weight", "decrease_replay_weight", "refine_buffer",
                           "freeze_backbone", "expand_backbone", "expand_fpn", "expand_roi"):
                rows[("training", "action_count", task, action)] = selected.get(action, "0")
        for action, value in re.findall(r"^  (\w+): changed=(\d+)", text, re.M):
            rows[("training", "changed_executions", "", action)] = value
        for name in ("Base Model Parameters", "Final Model Parameters", "Maximum Model Parameters"):
            scalar(text.replace(",", ""), "^" + name + r": (\d+)", name, "training")
    seed = re.search(r"^Seed: (\d+)$", text, re.M)
    return key, training, seed[1] if seed else "", rows


def analyze(root, output):
    # Auxiliary reports are considered only if a curated variant names their checkpoint.
    supplemental = defaultdict(list)
    for path in sorted((root / "Dokimes").glob("results_*.txt")):
        checkpoint = re.search(r"^Checkpoint: (.+)$", path.read_text(), re.M)
        if checkpoint:
            supplemental[Path(checkpoint[1].strip()).name].append(path)
    data = []
    summary = []
    discrepancies = []
    for variant in VARIANTS:
        groups = defaultdict(list)
        for path in sorted((root / variant).glob("*.txt")):
            groups[parse(path)[0]].append(path)
        aggregates = defaultdict(list)
        for key, paths in groups.items():
            metrics = {}
            seed = ""
            for path in paths + supplemental[key]:
                _, training, reported_seed, values = parse(path)
                if seed and reported_seed and seed != reported_seed:
                    raise ValueError(f"Conflicting seeds for {key}")
                seed = reported_seed or seed
                for coord, value in values.items():
                    if coord in metrics and float(metrics[coord]) != float(value):
                        if coord[:2] == ("test", "mAP50") or path in paths:
                            raise ValueError(f"Conflicting {coord} for {key} in {path}")
                        discrepancies.append((variant, key, *coord, metrics[coord], value,
                                              str(path.relative_to(root))))
                    metrics.setdefault(coord, value)
                    data.append((variant, key, str(path.relative_to(root)), "training" if training else "test",
                                 reported_seed, *coord, value))
            for coord, value in metrics.items():
                aggregates[coord].append(float(value))
        for coord, values in aggregates.items():
            summary.append((variant, *coord, len(values), mean(values), stdev(values) if len(values)>1 else None))
    if not data:
        raise ValueError(f"No CARL variant reports found under {root}")
    output.mkdir(parents=True, exist_ok=True)
    axes = ("split", "metric", "task", "domain_class_or_action")
    for filename, header, rows in (
        ("source_metrics.csv", ("variant", "checkpoint", "report", "layout", "printed_seed", *axes, "value"), data),
        ("aggregate.csv", ("variant", *axes, "n_checkpoints", "mean", "sample_sd"), summary),
        ("discrepancies.csv", ("variant", "checkpoint", *axes, "curated_value", "auxiliary_value", "auxiliary_report"), discrepancies),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    table = ["| Method | Runs | Final test mAP50 (%), mean ± SD | Validation forgetting (pp) |",
             "|---|---:|---:|---:|"]
    for variant, split, metric, _, _, n, avg, sd in summary:
        if (split, metric) == ("test", "mAP50"):
            forget = next((f"{100*r[6]:.2f} (n={r[5]})" for r in summary
                           if r[:3] == (variant, "validation", "Forgetting")), "unavailable")
            spread = f" ± {100*sd:.2f}" if sd is not None else ""
            table.append(f"| {variant} | {n} | {100*avg:.2f}{spread} | {forget} |")
    table.append(f"\n{len(discrepancies)} auxiliary metric discrepancies. Curated final-test values take precedence. See discrepancies.csv.")
    (output / "summary.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Results directory containing CARL variant folders and optional Dokimes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output)
