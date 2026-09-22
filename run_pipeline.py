"""Unified entry point for the two-stage Dino-L3R pipeline."""
from __future__ import annotations

import argparse
import json

from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("reproduce", "verify", "train"),
        default="reproduce",
        help="Default: reproduce GSP-DINO and OCRE with locked checkpoints.",
    )
    parser.add_argument("--config", default="configs/flare22_dino_l3r.toml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from pipeline.stages import (
        reproduce,
        train_pipeline,
        verify_final_assets,
    )

    config = load_config(args.config)
    if args.command == "train":
        result = train_pipeline(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.command == "verify":
        result = {"status": "verified", "checkpoints": verify_final_assets(config)}
    else:
        result = reproduce(config, dry_run=args.dry_run, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
