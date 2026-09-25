from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ocre import build_ocre_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu")
        model, layout = build_ocre_from_checkpoint(checkpoint)
        model.eval()
        with torch.inference_mode():
            output = model(
                torch.zeros(1, 22, 16, 32, 32),
                torch.zeros(1, dtype=torch.long),
            )
        if output.shape != (1, 1, 16, 32, 32):
            raise RuntimeError(f"Unexpected output shape for {path}: {output.shape}")
        if not torch.isfinite(output).all():
            raise RuntimeError(f"Non-finite output for {path}")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"{path}: layout={layout} parameters={parameters} status=ok")


if __name__ == "__main__":
    main()
