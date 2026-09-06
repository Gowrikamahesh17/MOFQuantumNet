"""CLI to (re)build every persisted GNN artifact.

    python -m webapp.backend.train_gnns --all
    python -m webapp.backend.train_gnns --dataset large --task classification

Real time cost, not simulated: ~35-40s (small dataset) to ~9 minutes (large dataset,
9 runs — 3 architectures x 3 graph variants), matching the README's own estimate for
this same underlying training loop.
"""

import argparse
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles default to cp1252, which can't print the "τ" in graph-variant
    # names like "BH-30 (τ=0.3)" — crashed a real training run's summary print after
    # the artifacts were already saved. UTF-8 stdout avoids that regardless of platform.
    sys.stdout.reconfigure(encoding="utf-8")

from webapp.backend.gnn_store import train_and_save_gnns
from webapp.backend.train_models import DATASET_TASKS


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist GNN models for the Predict console")
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
        print(f"Training GNNs for {dataset}/{task} (this trains 9 runs internally — 3 architectures x 3 graph variants) ...")
        metadata = train_and_save_gnns(dataset, task)
        elapsed = time.time() - start
        for name, info in metadata["models"].items():
            print(f"  {name} (best variant: {info['variant']}): {info['metrics']}")
        print(f"  done in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
