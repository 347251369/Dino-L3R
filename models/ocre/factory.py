from __future__ import annotations

from collections.abc import Mapping

import torch

from .model import OCRE


def _state_dict(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint)

    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a valid model state")

    return state


def build_ocre_from_checkpoint(
    checkpoint: Mapping,
) -> tuple[torch.nn.Module, str]:
    state = _state_dict(checkpoint)

    try:
        base_channels = int(state["enc1.conv1.weight"].shape[0])
    except KeyError as error:
        raise KeyError("Checkpoint is missing enc1.conv1.weight") from error

    if not any(key.startswith("enc4.") for key in state):
        raise RuntimeError("The checkpoint does not match the OCRE architecture")

    model = OCRE(base_channels=base_channels)
    model.load_state_dict(state, strict=True)

    return model