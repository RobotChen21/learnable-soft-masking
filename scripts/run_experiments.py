#!/usr/bin/env python3
"""Run the 2 datasets x 2 model variants from one canonical configuration."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiments.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("test", "train"), default="test")
    parser.add_argument("--dataset", choices=("all", "esconv", "annomi"), default="all")
    parser.add_argument("--variant", choices=("all", "hard", "soft"), default="all")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs" / "benchmark")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected(value, choices):
    return choices if value == "all" else (value,)


def build_command(args, config, dataset_name, variant_name):
    dataset = config[dataset_name]
    shared = config["shared"]
    fusion = config["variants"][variant_name]
    seed = args.seed if args.seed is not None else shared["seed"]
    run_dir = args.output_root / dataset_name / variant_name / f"seed-{seed}"
    command = [
        sys.executable, "main.py", "--mode", args.mode, "--model", "roberta-hg",
        "--dataset", dataset["dataset"], "--expert_fusion", fusion,
        "--prior_strength_init", str(shared["prior_strength_init"]),
        "--erc_temperature", str(shared["erc_temperature"]), "--hg_dim", str(shared["hg_dim"]),
        "--exclude_others", str(shared["exclude_others"]), "--erc_mixed", "1", "--seed", str(seed),
        "--output_dir", str(run_dir / "checkpoints"), "--logging_dir", str(run_dir / "logs"),
    ]
    if args.mode == "test":
        checkpoint = args.checkpoint_root / dataset_name / variant_name / "best.pth"
        command += ["--checkpoint_path", str(checkpoint), "--batch_size", "1", "--save_cases", "0"]
        return command, checkpoint, run_dir
    command += [
        "--lr", str(shared["lr"]), "--total_epochs", str(dataset["total_epochs"]),
        "--total_steps", str(dataset["total_steps"]), "--eval_steps", str(dataset["eval_steps"]),
        "--save_steps", str(dataset["save_steps"]), "--batch_size", str(shared["batch_size"]),
        "--warmup", str(shared["warmup"]), "--weight_decay", str(shared["weight_decay"]),
        "--keep_only_best_checkpoint", "1", "--save_cases", "0",
    ]
    return command, None, run_dir


def main():
    args = parse_args()
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    commands, missing = [], []
    for dataset_name in selected(args.dataset, ("esconv", "annomi")):
        preprocessed = REPO_ROOT / "data" / f"{dataset_name}_preprocessed"
        if not all((preprocessed / f"{split}.pkl").is_file() for split in ("train", "valid", "test")):
            missing.append(f"preprocessed data: {preprocessed}")
        for variant_name in selected(args.variant, ("hard", "soft")):
            command, checkpoint, run_dir = build_command(args, config, dataset_name, variant_name)
            if checkpoint is not None and not checkpoint.is_file():
                missing.append(f"checkpoint: {checkpoint}")
            commands.append((dataset_name, variant_name, command, run_dir))
    if missing and not args.dry_run:
        print("Required artifacts are missing:", file=sys.stderr)
        for item in dict.fromkeys(missing):
            print(f"  - {item}", file=sys.stderr)
        print("See checkpoints/README.md and the Data section in README.md.", file=sys.stderr)
        return 2
    summary_rows = []
    for index, (dataset_name, variant_name, command, run_dir) in enumerate(commands, 1):
        print(f"\n[{index}/{len(commands)}] {args.mode}: {dataset_name}/{variant_name}", flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            if args.mode == "test":
                reports = list((run_dir / "logs").glob("result_*.json"))
                if not reports:
                    raise FileNotFoundError(f"No result report was written under {run_dir / 'logs'}")
                report = max(reports, key=lambda path: path.stat().st_mtime_ns)
                with report.open(encoding="utf-8") as handle:
                    metrics = json.load(handle)
                summary_rows.append({
                    "dataset": dataset_name,
                    "variant": variant_name,
                    **{key: value for key, value in metrics.items() if key != "confusion matrix"},
                    "report": str(report.relative_to(REPO_ROOT)),
                })
    if summary_rows:
        args.output_root.mkdir(parents=True, exist_ok=True)
        json_path = args.output_root / "summary.json"
        csv_path = args.output_root / "summary.csv"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_rows, handle, ensure_ascii=False, indent=2)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nCombined results: {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
