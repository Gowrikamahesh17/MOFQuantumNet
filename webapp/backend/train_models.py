"""CLI to (re)build every persisted baseline model artifact.

    python -m webapp.backend.train_models --all
    python -m webapp.backend.train_models --dataset large --task classification

Small dataset is classification-only (config.py::PipelineConfig enforces the same rule);
this script mirrors that rather than attempting a regression artifact that can't exist.
"""

import argparse
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from webapp.backend.model_store import train_and_save_baselines

DATASET_TASKS = [
    ("small", "classification"),
    ("large", "classification"),
    ("large", "regression"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist baseline models for the Predict console")
    parser.add_argument("--all", action="store_true", help="Train every (dataset, task) combination")
    parser.add_argument("--dataset", choices=["small", "large"])
    parser.add_argument("--task", choices=["classification", "regression"])
    args = parser.parse_args()

    if args.all:
        targets = DATASET_TASKS
    elif args.dataset and args.task:
        if args.dataset == "small" and args.task == "regression":
            raise SystemExit("small dataset has no continuous PLD — classification only")
        targets = [(args.dataset, args.task)]
    else:
        parser.error("pass --all, or both --dataset and --task")
        return

    for dataset, task in targets:
        start = time.time()
        print(f"Training {dataset}/{task} ...")
        metadata = train_and_save_baselines(dataset, task)
        elapsed = time.time() - start
        for name, info in metadata["models"].items():
            print(f"  {name}: {info['metrics']}")
        print(f"  done in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
