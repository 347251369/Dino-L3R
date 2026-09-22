"""Two-stage GSP -> OCRE training and reproduction pipeline."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from utils.checkpoints import sha256_file, verify_checkpoint
from utils.config import require, resolve_path
from utils.runtime import require_count, run_command

MODEL_STAGES = ("gsp_dino", "ocre")


def _python(config: dict[str, Any]) -> str:
    return str(require(config, "project.python"))


def _path(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config, require(config, key))


def _environment(config: dict[str, Any]) -> dict[str, str]:
    root = Path(config["_project_root"])
    return {
        "PYTHONPATH": os.pathsep.join(
            (str(root), str(resolve_path(config, require(config, "dependencies.dinov3_package"))))
        ),
        "CUDA_VISIBLE_DEVICES": str(require(config, "project.device")).split(":")[-1],
        "CLIP_MODEL_DIR": str(require(config, "dependencies.clip_model")),
        "SEG_DATASET": str(require(config, "dataset.name")),
    }


def _run(config: dict[str, Any], command: list[str], dry_run: bool) -> None:
    run_command(
        command,
        cwd=config["_project_root"],
        dry_run=dry_run,
        env=_environment(config),
    )


def _options(
    config: dict[str, Any],
    section: str,
    names: tuple[str | tuple[str, str], ...],
) -> list[str]:
    result = []
    for item in names:
        option, key = (item, item) if isinstance(item, str) else item
        result.extend((f"--{option}", str(require(config, f"{section}.{key}"))))
    return result


def _optional_options(
    config: dict[str, Any], section: str, names: tuple[str, ...]
) -> list[str]:
    values = config.get(section, {})
    result = []
    for name in names:
        if name in values:
            result.extend((f"--{name}", str(values[name])))
    return result


def _verify_split_and_data(config: dict[str, Any]) -> None:
    split_path = _path(config, "dataset.split_manifest")
    with split_path.open(encoding="utf-8") as handle:
        split = json.load(handle)
    expected = str(require(config, "dataset.split_sha256"))
    if split.get("split_sha256") != expected:
        raise RuntimeError(f"Split SHA mismatch: {split.get('split_sha256')} != {expected}")
    for key, count_key in (
        ("dataset.train_images", "dataset.train_cases"),
        ("dataset.train_labels", "dataset.train_cases"),
        ("dataset.test_images", "dataset.test_cases"),
        ("dataset.test_labels", "dataset.test_cases"),
    ):
        require_count(_path(config, key), "*.nii.gz", int(require(config, count_key)))


def _verify_stage(config: dict[str, Any], stage: str) -> str:
    checkpoint = _path(config, f"{stage}.checkpoint")
    expected = config.get(stage, {}).get("checkpoint_sha256")
    if expected:
        verify_checkpoint(checkpoint, str(expected))
    elif not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {stage} checkpoint: {checkpoint}")
    if bool(config.get(stage, {}).get("require_train_predictions", True)):
        require_count(
            _path(config, f"{stage}.train_predictions"),
            "*.nii.gz",
            int(require(config, "dataset.train_cases")),
        )
    require_count(
        _path(config, f"{stage}.test_predictions"),
        "*.nii.gz",
        int(require(config, "dataset.test_cases")),
    )
    checksum = sha256_file(checkpoint)
    print(f"[pipeline] locked {stage} sha256={checksum}", flush=True)
    return checksum


def verify_final_assets(config: dict[str, Any]) -> dict[str, str]:
    _verify_split_and_data(config)
    return {stage: _verify_stage(config, stage) for stage in MODEL_STAGES}


def _train_gsp(config: dict[str, Any], dry_run: bool) -> None:
    resume = _path(config, "gsp_dino.resume")
    command = [
        _python(config), "-m", "models.gsp_dino.train",
        "--train_images", str(_path(config, "dataset.train_images")),
        "--train_labels", str(_path(config, "dataset.train_labels")),
        "--dino_weights", str(_path(config, "dependencies.dino_weights")),
        "--clip_dir", str(_path(config, "dependencies.clip_model")),
        "--latest_path", str(resume),
        "--best_path", str(_path(config, "gsp_dino.checkpoint")),
        "--report_path", str(_path(config, "gsp_dino.report")),
        *_options(config, "gsp_dino", (
            "epochs", "samples_per_epoch", "batch_size", ("num_workers", "workers"),
            "lr_backbone", "lr_backbone_early", "lr_backbone_middle",
            "lr_backbone_late", "lr_head", "weight_decay", "balance_cap",
            "class_weight_cap", "background_weight", "label_smoothing",
            "augmentation_profile", "context_mm", "seed",
        )),
        *_optional_options(config, "gsp_dino", (
            "sampling_mode", "organ_sampling_probability", "loss_ce_weight",
            "loss_dice_weight", "loss_tversky_weight", "loss_presence_weight",
            "tversky_alpha", "tversky_beta", "trainable_block_start",
            "view_mode", "local_view_size",
        )),
        "--amp",
    ]
    if resume.is_file():
        command.extend(("--resume", str(resume)))
    gsp = config.get("gsp_dino", {})
    for name in ("depth_context_adapter", "presence_head", "spatial_view_context"):
        if bool(gsp.get(name, False)):
            command.append(f"--{name}")
    if bool(require(config, "gsp_dino.no_validation")):
        command.append("--no_validation")
    _run(config, command, dry_run)


def _infer_gsp(
    config: dict[str, Any], images: Path, output: Path, checksum: str, dry_run: bool
) -> None:
    gsp = config.get("gsp_dino", {})
    command = [
        _python(config), "-m", "models.gsp_dino.infer",
        "--images_dir", str(images),
        "--out_dir", str(output),
        "--checkpoint", str(_path(config, "gsp_dino.checkpoint")),
        "--expected_sha256", checksum,
        "--dino_weights", str(_path(config, "dependencies.dino_weights")),
        "--clip_dir", str(_path(config, "dependencies.clip_model")),
        "--batch_size", str(gsp.get("inference_batch_size", 4)),
        "--smooth_sigma", str(require(config, "gsp_dino.smooth_sigma")),
        "--context_mm", str(require(config, "gsp_dino.context_mm")),
        "--amp",
    ]
    for name in ("view_mode", "local_view_size", "trainable_block_start"):
        if name in gsp:
            command.extend((f"--{name}", str(gsp[name])))
    for name in ("depth_context_adapter", "presence_head", "spatial_view_context"):
        if bool(gsp.get(name, False)):
            command.append(f"--{name}")
    _run(config, command, dry_run)


def _train_ocre(config: dict[str, Any], dry_run: bool) -> None:
    ocre = config.get("ocre", {})
    command = [
        _python(config), "-m", "models.ocre.train",
        "--train_images", str(_path(config, "dataset.train_images")),
        "--train_labels", str(_path(config, "dataset.train_labels")),
        "--coarse_masks", str(_path(config, "gsp_dino.train_predictions")),
        "--cache_dir", str(_path(config, "ocre.cache")),
        "--save_path", str(_path(config, "ocre.checkpoint")),
        "--patch_size", *map(str, require(config, "ocre.patch_size")),
        *_options(config, "ocre", (
            "base_channels", "epochs", "samples_per_epoch",
            "batch_size", ("num_workers", "workers"),
            "lr", "boundary_probability", "prior_shift_probability",
            "prior_shift_max_mm", "seed", "save_every",
        )),
    ]
    if bool(ocre.get("amp", False)):
        command.append("--amp")
    _run(config, command, dry_run)


def _infer_ocre(
    config: dict[str, Any], images: Path, gsp_predictions: Path, output: Path, dry_run: bool
) -> None:
    ocre = config.get("ocre", {})
    command = [
        _python(config), "-m", "pipeline.refinement",
        "--model_type", "ocre",
        "--images", str(images),
        "--coarse_masks", str(gsp_predictions),
        "--checkpoint", str(_path(config, "ocre.checkpoint")),
        "--output", str(output),
        "--patch_size", *map(str, require(config, "ocre.patch_size")),
        "--max_patches", str(require(config, "ocre.max_patches")),
        "--inference_batch_size", str(ocre.get("inference_batch_size", 4)),
        "--threshold", str(ocre.get("threshold", 0.5)),
    ]
    if bool(ocre.get("amp", False)):
        command.append("--amp")
    _run(config, command, dry_run)


def _evaluate_stage(
    config: dict[str, Any], predictions: Path, output: Path, dry_run: bool
) -> None:
    _run(config, [
        _python(config), "-m", "evaluation.dice",
        "--predictions", str(predictions),
        "--labels", str(_path(config, "dataset.test_labels")),
        "--references", str(_path(config, "dataset.test_images")),
        "--per_case_csv", str(output / "dice_per_case.csv"),
        "--summary_csv", str(output / "dice_summary.csv"),
    ], dry_run)
    _run(config, [
        _python(config), "-m", "evaluation.surface",
        "--images", str(_path(config, "dataset.test_images")),
        "--labels", str(_path(config, "dataset.test_labels")),
        "--predictions", str(predictions),
        "--per_case_csv", str(output / "surface_per_case.csv"),
        "--summary_csv", str(output / "surface_summary.csv"),
        "--case_ids", *map(str, require(config, "dataset.test_case_ids")),
    ], dry_run)


def _prepare_root(root: Path, overwrite: bool, dry_run: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {root}")
        if not dry_run:
            shutil.rmtree(root)
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)


def train_pipeline(
    config: dict[str, Any], *, dry_run: bool = False, overwrite: bool = False
) -> dict[str, Any]:
    _verify_split_and_data(config)
    start_stage = str(config.get("project", {}).get("start_stage", "gsp_dino"))
    stop_stage = str(config.get("project", {}).get("stop_after_stage", "ocre"))
    if start_stage not in MODEL_STAGES or stop_stage not in MODEL_STAGES:
        raise ValueError(f"Only GSP and OCRE are available: start={start_stage}, stop={stop_stage}")
    if MODEL_STAGES.index(stop_stage) < MODEL_STAGES.index(start_stage):
        raise ValueError(f"stop_after_stage precedes start_stage: {stop_stage} < {start_stage}")
    root = _path(config, "results.root")
    if start_stage == "gsp_dino":
        _prepare_root(root, overwrite, dry_run)
        _train_gsp(config, dry_run)
        gsp_sha = "DRY_RUN" if dry_run else sha256_file(_path(config, "gsp_dino.checkpoint"))
        _infer_gsp(config, _path(config, "dataset.train_images"), _path(config, "gsp_dino.train_predictions"), gsp_sha, dry_run)
        _infer_gsp(config, _path(config, "dataset.test_images"), _path(config, "gsp_dino.test_predictions"), gsp_sha, dry_run)
    else:
        gsp_sha = _verify_stage(config, "gsp_dino")
    if stop_stage == "gsp_dino":
        return {"status": "dry_run" if dry_run else "complete", "stage": "gsp_dino", "checkpoint_sha256": gsp_sha}

    _train_ocre(config, dry_run)
    _infer_ocre(config, _path(config, "dataset.train_images"), _path(config, "gsp_dino.train_predictions"), _path(config, "ocre.train_predictions"), dry_run)
    _infer_ocre(config, _path(config, "dataset.test_images"), _path(config, "gsp_dino.test_predictions"), _path(config, "ocre.test_predictions"), dry_run)
    _evaluate_stage(
        config,
        _path(config, "gsp_dino.test_predictions"),
        root / "gsp_dino" / "reports" / "test",
        dry_run,
    )
    _evaluate_stage(
        config,
        _path(config, "ocre.test_predictions"),
        root / "ocre" / "reports" / "test",
        dry_run,
    )
    ocre_sha = "DRY_RUN" if dry_run else sha256_file(_path(config, "ocre.checkpoint"))
    record = {
        "status": "dry_run" if dry_run else "complete",
        "dataset": str(require(config, "dataset.name")),
        "stages": list(MODEL_STAGES),
        "checkpoints": {"gsp_dino": gsp_sha, "ocre": ocre_sha},
    }
    if not dry_run:
        (root / "TRAINING_COMPLETE.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return record


def reproduce(
    config: dict[str, Any], *, dry_run: bool = False, overwrite: bool = False
) -> dict[str, Any]:
    checkpoints = verify_final_assets(config)
    root = _path(config, "results.reproduction_root")
    _prepare_root(root, overwrite, dry_run)
    gsp_output = root / "gsp_dino" / "predictions" / "test"
    ocre_output = root / "ocre" / "predictions" / "test"
    _infer_gsp(config, _path(config, "dataset.test_images"), gsp_output, checkpoints["gsp_dino"], dry_run)
    _infer_ocre(config, _path(config, "dataset.test_images"), gsp_output, ocre_output, dry_run)
    _evaluate_stage(
        config, gsp_output, root / "gsp_dino" / "reports" / "test", dry_run
    )
    _evaluate_stage(
        config, ocre_output, root / "ocre" / "reports" / "test", dry_run
    )
    record = {
        "status": "dry_run" if dry_run else "complete",
        "checkpoints": checkpoints,
        "fixed_test_previously_viewed": True,
    }
    if not dry_run:
        (root / "REPRODUCTION_COMPLETE.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return record
