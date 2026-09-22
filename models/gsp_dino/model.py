import os
import math
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from dinov3.hub.backbones import dinov3_vitl16
from transformers import CLIPTextModel

from .semantics import get_anatomy_relation_matrix


# =============================================================================
# DINOv3 HF safetensors -> Meta dinov3.hub key conversion
# =============================================================================

def _strip_known_prefix(key: str) -> str:
    """Remove common wrapper prefixes without hiding unknown key patterns."""
    prefixes = [
        "module.",
        "model.",
        "backbone.",
        "vision_model.",
        "encoder.",
    ]
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def _fix_token_shape(value: torch.Tensor, target_ndim: int = 3) -> torch.Tensor:
    """Meta DINOv3 expects cls/storage tokens as [1, N, C]."""
    if target_ndim == 3:
        if value.ndim == 2:
            return value.unsqueeze(0)
        if value.ndim == 1:
            return value.view(1, 1, -1)
    return value


def convert_hf_to_meta_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert the observed HuggingFace-style DINOv3 ViT-L/16 keys to Meta DINOv3 keys.

    Your checkpoint uses patterns like:
        layer.0.attention.q_proj.weight
        layer.0.attention.k_proj.weight
        layer.0.attention.v_proj.weight
        layer.0.attention.o_proj.weight
        layer.0.mlp.up_proj.weight
        layer.0.mlp.down_proj.weight

    Meta dinov3.hub.backbones.dinov3_vitl16 expects:
        blocks.0.attn.qkv.weight
        blocks.0.attn.proj.weight
        blocks.0.mlp.fc1.weight
        blocks.0.mlp.fc2.weight

    This function merges q/k/v into qkv and fixes token tensor shapes.
    Unknown keys are kept so the loading report can expose them as unexpected.
    """
    converted: Dict[str, torch.Tensor] = {}
    qkv_parts: Dict[Tuple[int, str, str], torch.Tensor] = {}

    for raw_k, raw_v in state_dict.items():
        k = _strip_known_prefix(raw_k)
        v = raw_v
        nk = None

        # ------------------------------------------------------------------
        # Embedding / token keys
        # ------------------------------------------------------------------
        if k in ["embeddings.cls_token", "cls_token"]:
            nk = "cls_token"
            v = _fix_token_shape(v, target_ndim=3)

        elif k in ["embeddings.mask_token", "mask_token"]:
            nk = "mask_token"
            # Most Meta DINOv3 builds expect [C] for mask_token.
            if v.ndim == 2 and v.shape[0] == 1:
                v = v.squeeze(0)
            elif v.ndim == 3 and v.shape[0] == 1 and v.shape[1] == 1:
                v = v.squeeze(0).squeeze(0)

        elif k in ["embeddings.register_tokens", "register_tokens", "storage_tokens"]:
            nk = "storage_tokens"
            v = _fix_token_shape(v, target_ndim=3)

        elif k in ["embeddings.patch_embeddings.weight", "patch_embed.proj.weight"]:
            nk = "patch_embed.proj.weight"
        elif k in ["embeddings.patch_embeddings.bias", "patch_embed.proj.bias"]:
            nk = "patch_embed.proj.bias"

        # Some HF checkpoints include abs position embeddings; your Meta DINOv3
        # backbone uses rope_embed, so this key may remain unexpected if there is
        # no matching pos_embed in the model.
        elif k in ["embeddings.position_embeddings", "pos_embed"]:
            nk = "pos_embed"

        # ------------------------------------------------------------------
        # Already Meta-style keys
        # ------------------------------------------------------------------
        elif k.startswith("blocks.") or k.startswith("patch_embed.") or k.startswith("norm.") or k.startswith("rope_embed."):
            nk = k
            if nk in ["cls_token", "storage_tokens"]:
                v = _fix_token_shape(v, target_ndim=3)

        # ------------------------------------------------------------------
        # Observed HF-style layer keys: layer.{i}.attention.q_proj/k_proj/...
        # ------------------------------------------------------------------
        elif k.startswith("layer."):
            parts = k.split(".")
            if len(parts) >= 4 and parts[1].isdigit():
                i = int(parts[1])
                sub = ".".join(parts[2:])

                # Norms
                if sub == "norm1.weight":
                    nk = f"blocks.{i}.norm1.weight"
                elif sub == "norm1.bias":
                    nk = f"blocks.{i}.norm1.bias"
                elif sub == "norm2.weight":
                    nk = f"blocks.{i}.norm2.weight"
                elif sub == "norm2.bias":
                    nk = f"blocks.{i}.norm2.bias"

                # LayerScale names observed in various HF exports
                elif sub in ["layer_scale1.gamma", "layer_scale1.lambda1", "ls1.gamma"]:
                    nk = f"blocks.{i}.ls1.gamma"
                elif sub in ["layer_scale2.gamma", "layer_scale2.lambda1", "ls2.gamma"]:
                    nk = f"blocks.{i}.ls2.gamma"

                # Attention q/k/v are collected and concatenated later.
                elif sub == "attention.q_proj.weight":
                    qkv_parts[(i, "weight", "q")] = v
                    continue
                elif sub == "attention.k_proj.weight":
                    qkv_parts[(i, "weight", "k")] = v
                    continue
                elif sub == "attention.v_proj.weight":
                    qkv_parts[(i, "weight", "v")] = v
                    continue
                elif sub == "attention.q_proj.bias":
                    qkv_parts[(i, "bias", "q")] = v
                    continue
                elif sub == "attention.k_proj.bias":
                    qkv_parts[(i, "bias", "k")] = v
                    continue
                elif sub == "attention.v_proj.bias":
                    qkv_parts[(i, "bias", "v")] = v
                    continue

                # Other common attention naming variants
                elif sub == "attention.attention.query.weight":
                    qkv_parts[(i, "weight", "q")] = v
                    continue
                elif sub == "attention.attention.key.weight":
                    qkv_parts[(i, "weight", "k")] = v
                    continue
                elif sub == "attention.attention.value.weight":
                    qkv_parts[(i, "weight", "v")] = v
                    continue
                elif sub == "attention.attention.query.bias":
                    qkv_parts[(i, "bias", "q")] = v
                    continue
                elif sub == "attention.attention.key.bias":
                    qkv_parts[(i, "bias", "k")] = v
                    continue
                elif sub == "attention.attention.value.bias":
                    qkv_parts[(i, "bias", "v")] = v
                    continue

                # Output projection. Your checkpoint uses o_proj.
                elif sub in ["attention.o_proj.weight", "attention.output.dense.weight"]:
                    nk = f"blocks.{i}.attn.proj.weight"
                elif sub in ["attention.o_proj.bias", "attention.output.dense.bias"]:
                    nk = f"blocks.{i}.attn.proj.bias"

                # MLP. Your checkpoint uses up_proj/down_proj.
                elif sub in ["mlp.up_proj.weight", "intermediate.dense.weight", "mlp.fc1.weight"]:
                    nk = f"blocks.{i}.mlp.fc1.weight"
                elif sub in ["mlp.up_proj.bias", "intermediate.dense.bias", "mlp.fc1.bias"]:
                    nk = f"blocks.{i}.mlp.fc1.bias"
                elif sub in ["mlp.down_proj.weight", "output.dense.weight", "mlp.fc2.weight"]:
                    nk = f"blocks.{i}.mlp.fc2.weight"
                elif sub in ["mlp.down_proj.bias", "output.dense.bias", "mlp.fc2.bias"]:
                    nk = f"blocks.{i}.mlp.fc2.bias"

        if nk is not None:
            converted[nk] = v
        else:
            # Keep unknown keys visible in the loading report.
            converted[k] = v

    # Merge q/k/v into qkv for every layer and for weight/bias separately.
    layer_ids = sorted({i for (i, _kind, _qkv) in qkv_parts.keys()})
    for i in layer_ids:
        for kind in ["weight", "bias"]:
            q = qkv_parts.get((i, kind, "q"))
            k = qkv_parts.get((i, kind, "k"))
            v = qkv_parts.get((i, kind, "v"))
            if q is not None and k is not None and v is not None:
                converted[f"blocks.{i}.attn.qkv.{kind}"] = torch.cat([q, k, v], dim=0)

    return converted


def _is_tolerated_missing_key(key: str) -> bool:
    """Buffers that are present in Meta implementation but often absent from HF safetensors."""
    if key == "rope_embed.periods":
        return True
    if key.endswith("attn.qkv.bias_mask"):
        return True
    return False


def _load_raw_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return load_file(path)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")
    return obj


def _coerce_tensor_to_model_shape(value: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Return tensor aligned to target shape/dtype when this is mathematically safe.

    This fixes harmless exported-token shape differences such as:
        checkpoint mask_token: [1024]
        Meta DINOv3 mask_token: [1, 1024]

    It does not interpolate or invent pretrained values.  It only reshapes tensors
    with the same number of elements.
    """
    out = value.detach().cpu()
    if tuple(out.shape) == tuple(target.shape):
        return out.to(dtype=target.dtype), False
    if out.numel() == target.numel():
        return out.reshape(tuple(target.shape)).to(dtype=target.dtype), True
    return out, False


def _is_zero_fillable_required_key(key: str) -> bool:
    """Keys that are absent because the released DINO checkpoint is no-bias.

    The Meta DINOv3 module instantiated by dinov3_vitl16 contains qkv.bias
    parameters, while some official/HF-exported DINOv3 ViT checkpoints contain
    only q/k/v weights and no q/k/v bias tensors.  In that case the correct
    no-bias equivalent is a zero bias, not a random PyTorch initialization.
    """
    return key.startswith("blocks.") and key.endswith("attn.qkv.bias")


def load_dinov3_backbone_weights(
    backbone: nn.Module,
    pretrained_weights: str,
    min_load_ratio: float = 1.0,
    allow_partial: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load DINOv3 weights with safe completion of export-format gaps.

    Compared with a strict exact-shape loader, this version does three extra
    safety-preserving things:
      1. reshapes same-numel token tensors, e.g. mask_token [C] -> [1, C];
      2. keeps tolerated implementation buffers initialized, e.g. rope/bias masks;
      3. initializes missing qkv.bias tensors to zero when the checkpoint is a
         no-bias export.  This is deterministic and equivalent to no-bias qkv.

    Therefore the report separates:
      - pretrained tensors loaded exactly,
      - pretrained tensors loaded after shape fix,
      - deterministic zero-completed tensors,
      - tolerated buffers.
    """
    raw_state = _load_raw_checkpoint(pretrained_weights)
    converted = convert_hf_to_meta_keys(raw_state)
    model_state = backbone.state_dict()

    exact_matched: Dict[str, torch.Tensor] = {}
    shape_fixed: Dict[str, torch.Tensor] = {}
    unresolved_shape_mismatch = []
    unexpected = []

    # 1) Load every checkpoint tensor that maps to this DINOv3 backbone.
    for k, v in converted.items():
        if k not in model_state:
            unexpected.append(k)
            continue
        target = model_state[k]
        aligned, was_shape_fixed = _coerce_tensor_to_model_shape(v, target)
        if tuple(aligned.shape) == tuple(target.shape):
            if was_shape_fixed:
                shape_fixed[k] = aligned
            else:
                exact_matched[k] = aligned
        else:
            unresolved_shape_mismatch.append((k, tuple(v.shape), tuple(target.shape)))

    completed: Dict[str, torch.Tensor] = {}
    completed.update(exact_matched)
    completed.update(shape_fixed)

    # 2) Classify missing keys after exact + safe shape-fixed loading.
    tolerated_missing = []
    zero_initialized = []
    missing_required = []

    for k, target in model_state.items():
        if k in completed:
            continue
        if _is_tolerated_missing_key(k):
            tolerated_missing.append(k)
        elif _is_zero_fillable_required_key(k):
            completed[k] = torch.zeros_like(target)
            zero_initialized.append(k)
        else:
            missing_required.append(k)

    required_total = len(model_state) - len(tolerated_missing)
    completed_required = len(exact_matched) + len(shape_fixed) + len(zero_initialized)
    load_ratio = completed_required / max(required_total, 1)

    # 3) Load completed tensors.  Tolerated buffers stay initialized by dinov3_vitl16.
    new_state = dict(model_state)
    new_state.update(completed)
    backbone.load_state_dict(new_state, strict=True)

    if verbose:
        print("=" * 90)
        print("[DINOv3 Weight Loading Report - COMPLETE]")
        print(f"Pretrained path: {pretrained_weights}")
        print(f"Raw checkpoint keys: {len(raw_state)}")
        print(f"Converted keys: {len(converted)}")
        print(f"Backbone keys: {len(model_state)}")
        print(f"Exact pretrained tensors loaded: {len(exact_matched)}")
        print(f"Shape-fixed pretrained tensors loaded: {len(shape_fixed)}")
        print(f"Zero-initialized no-bias qkv tensors: {len(zero_initialized)}")
        print(f"Tolerated missing buffers: {len(tolerated_missing)}")
        print(f"Missing required pretrained keys: {len(missing_required)}")
        print(f"Unexpected converted keys: {len(unexpected)}")
        print(f"Unresolved shape mismatches: {len(unresolved_shape_mismatch)}")
        print(f"Effective completed ratio: {load_ratio:.2%}")
        print("-" * 90)
        if shape_fixed:
            print("Shape-fixed pretrained keys:")
            for k in list(shape_fixed.keys())[:40]:
                print("  ", k, f"-> model shape {tuple(model_state[k].shape)}")
        else:
            print("No shape-fixed pretrained keys.")
        print("-" * 90)
        if zero_initialized:
            print("Zero-initialized no-bias qkv keys (first 40):")
            for k in zero_initialized[:40]:
                print("  ", k)
        else:
            print("No zero-initialized qkv bias keys.")
        print("-" * 90)
        if missing_required:
            print("First 40 missing required keys:")
            for k in missing_required[:40]:
                print("  ", k)
        else:
            print("No missing required keys.")
        print("-" * 90)
        if unexpected:
            print("First 40 unexpected converted keys:")
            for k in unexpected[:40]:
                print("  ", k)
        else:
            print("No unexpected converted keys.")
        print("-" * 90)
        if unresolved_shape_mismatch:
            print("First 20 unresolved shape mismatches:")
            for k, s_ckpt, s_model in unresolved_shape_mismatch[:20]:
                print(f"  {k}: checkpoint={s_ckpt}, model={s_model}")
        else:
            print("No unresolved shape mismatches.")
        print("=" * 90)

    if load_ratio < min_load_ratio and not allow_partial:
        raise RuntimeError(
            f"DINOv3 weights were not fully completed: completed ratio {load_ratio:.2%} < {min_load_ratio:.2%}. "
            "Check checkpoint architecture or conversion."
        )
    if missing_required and not allow_partial:
        raise RuntimeError(
            "DINOv3 has missing required pretrained keys after completion. "
            f"First missing keys: {missing_required[:10]}"
        )
    if unresolved_shape_mismatch and not allow_partial:
        raise RuntimeError(
            "DINOv3 has unresolved shape mismatches after safe same-numel reshaping. "
            f"First mismatches: {unresolved_shape_mismatch[:5]}"
        )

    return {
        "raw_keys": len(raw_state),
        "converted_keys": len(converted),
        "backbone_keys": len(model_state),
        "exact_matched_keys": len(exact_matched),
        "shape_fixed_keys": list(shape_fixed.keys()),
        "zero_initialized_keys": zero_initialized,
        "tolerated_missing": tolerated_missing,
        "missing_required": missing_required,
        "unexpected": unexpected,
        "unresolved_shape_mismatch": unresolved_shape_mismatch,
        "load_ratio": load_ratio,
    }


# =============================================================================
# Segmentation model
# =============================================================================

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        hidden = max(in_planes // ratio, 1)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x * self.ca(x)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out = torch.max(out, dim=1, keepdim=True)[0]
        out = out * self.sa(torch.cat([avg_out, max_out], dim=1))
        return x + out


class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim=256):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.proj_back = nn.Linear(hidden_dim, visual_dim)

    def forward(self, visual_tokens, text_tokens):
        q = self.visual_proj(visual_tokens)
        kv = self.text_proj(text_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv)
        return visual_tokens + self.proj_back(attn_out)


class DepthContextAdapter(nn.Module):
    """Learn residual fusion of the lower, center, and upper CT slices.

    The final projection is zero-initialized, so enabling the adapter preserves
    the original three-channel input exactly at initialization. This keeps the
    adapter close to the locked GSP model while allowing training to
    learn asymmetric through-plane evidence for weak or discontinuous organs.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        center = images[:, 1:2]
        context = torch.cat(
            (center, images[:, 0:1] - center, images[:, 2:3] - center),
            dim=1,
        )
        return images + self.encoder(context)


class AnatomySemanticGraph(nn.Module):
    """Fixed anatomical relation propagation over prompt-level text embeddings."""

    def __init__(self, num_classes: int = 14, steps: int = 1, residual: float = 0.5):
        super().__init__()
        relation = torch.tensor(get_anatomy_relation_matrix(num_classes), dtype=torch.float32)
        self.register_buffer("relation", relation, persistent=False)
        self.steps = int(steps)
        self.residual = float(residual)

    def forward(self, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        if prompt_embeddings.ndim != 2 or prompt_embeddings.shape[0] != self.relation.shape[0]:
            return prompt_embeddings
        relation = self.relation.to(device=prompt_embeddings.device, dtype=prompt_embeddings.dtype)
        out = prompt_embeddings
        for _ in range(max(self.steps, 0)):
            neighbor_context = relation @ out
            out = F.layer_norm(out + self.residual * neighbor_context, (out.shape[-1],))
        return out


class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        layers = []
        num_blocks = int(math.log2(scale_factor))
        if 2 ** num_blocks != scale_factor:
            raise ValueError(f"scale_factor must be power of 2, got {scale_factor}")
        for i in range(num_blocks):
            out = out_channels if i == num_blocks - 1 else max(in_channels // 2, out_channels)
            layers.extend([
                nn.ConvTranspose2d(in_channels, out, 2, 2),
                nn.GroupNorm(8, out),
                nn.ReLU(inplace=True),
            ])
            in_channels = out
        self.up = nn.Sequential(*layers)

    def forward(self, x):
        return self.up(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNETRDecoder(nn.Module):
    """Standard shared UNETR decoder for all foreground classes."""

    def __init__(self, embed_dim=1024, num_classes=14, deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.up4 = UpSampleLayer(embed_dim, 512, 2)
        self.skip3 = UpSampleLayer(embed_dim, 512, 2)
        self.conv3 = DoubleConv(1024, 512)
        self.up3 = UpSampleLayer(512, 256, 2)
        self.skip2 = UpSampleLayer(embed_dim, 256, 4)
        self.conv2 = DoubleConv(512, 256)
        self.up2 = UpSampleLayer(256, 128, 2)
        self.skip1 = UpSampleLayer(embed_dim, 128, 8)
        self.conv1 = DoubleConv(256, 128)
        self.final = nn.Sequential(
            UpSampleLayer(128, 64, 2),
            nn.Conv2d(64, num_classes, 1),
        )
        if self.deep_supervision:
            self.aux_x1 = nn.Conv2d(128, num_classes, kernel_size=1)
            self.aux_x2 = nn.Conv2d(256, num_classes, kernel_size=1)
            nn.init.zeros_(self.aux_x1.weight)
            nn.init.zeros_(self.aux_x1.bias)
            nn.init.zeros_(self.aux_x2.weight)
            nn.init.zeros_(self.aux_x2.bias)

    def forward(self, f1, f2, f3, bottleneck, return_aux: bool = False):
        x3 = self.conv3(torch.cat([self.up4(bottleneck), self.skip3(f3)], dim=1))
        x2 = self.conv2(torch.cat([self.up3(x3), self.skip2(f2)], dim=1))
        x1 = self.conv1(torch.cat([self.up2(x2), self.skip1(f1)], dim=1))
        logits = self.final(x1)
        if not return_aux:
            return logits
        if not self.deep_supervision:
            return logits, []
        return logits, [self.aux_x1(x1), self.aux_x2(x2)]


class CTSegModel(nn.Module):
    """Final GSP-DINO model: DINOv3 + fixed anatomy prompts + one UNETR decoder."""

    def __init__(
        self,
        pretrained_weights=None,
        num_classes=14,
        min_load_ratio=None,
        allow_partial_backbone=False,
        use_depth_context_adapter: bool = False,
        use_presence_head: bool = False,
        use_spatial_view_context: bool = False,
        use_deep_supervision: bool = False,
        trainable_block_start: int = 6,
    ):
        super().__init__()
        self.backbone = dinov3_vitl16(pretrained=False)
        if pretrained_weights:
            if min_load_ratio is None:
                min_load_ratio = float(os.environ.get("DINO_MIN_LOAD_RATIO", "0.85"))
            load_dinov3_backbone_weights(
                self.backbone,
                pretrained_weights,
                min_load_ratio=float(min_load_ratio),
                allow_partial=bool(allow_partial_backbone),
                verbose=True,
            )

        for p in self.backbone.parameters():
            p.requires_grad = False

        if not 0 <= trainable_block_start < len(self.backbone.blocks):
            raise ValueError(
                f"trainable_block_start must be in [0, {len(self.backbone.blocks) - 1}], "
                f"got {trainable_block_start}"
            )

        # Adapt the middle and late DINOv3 blocks while retaining the generic stem.
        for name, param in self.backbone.named_parameters():
            if (
                any(
                    f"blocks.{i}." in name
                    for i in range(trainable_block_start, len(self.backbone.blocks))
                )
                or "norm" in name
            ):
                param.requires_grad = True

        unfrozen_layers = [name for name, p in self.backbone.named_parameters() if p.requires_grad]
        print(
            f"--- DINOv3 trainable blocks: {trainable_block_start}-"
            f"{len(self.backbone.blocks) - 1}; unfrozen tensors: {len(unfrozen_layers)} ---"
        )

        self.depth_context_adapter = (
            DepthContextAdapter() if use_depth_context_adapter else None
        )
        self.cbam = CBAMBlock(1024)
        self.text_encoder = CLIPTextModel.from_pretrained(
            os.environ.get("CLIP_MODEL_DIR", "/pd/heyang/weights/clip-vit-base-patch32")
        )
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.spatial_view_context = bool(use_spatial_view_context)
        context_dim = 4 if self.spatial_view_context else 1
        self.z_encoder = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
        )
        self.semantic_graph = AnatomySemanticGraph(
            num_classes=num_classes,
            steps=1,
            residual=0.3,
        )
        self.fusion = CrossAttentionFusion(1024, 512)
        self.semantic_gate = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.semantic_gate[2].weight)
        nn.init.constant_(self.semantic_gate[2].bias, -1.3862944)
        self.decoder = UNETRDecoder(
            1024,
            num_classes,
            deep_supervision=use_deep_supervision,
        )
        self.presence_head = (
            nn.Linear(1024, num_classes - 1) if use_presence_head else None
        )
        if self.presence_head is not None:
            nn.init.zeros_(self.presence_head.weight)
            nn.init.zeros_(self.presence_head.bias)
        self._cached_anatomy_semantics = None

    def _compute_anatomy_semantics(self, input_ids, attention_mask):
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        mask = attention_mask.to(dtype=text_out.dtype).unsqueeze(-1)
        prompt_emb = (text_out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.semantic_graph(prompt_emb)

    @torch.no_grad()
    def cache_anatomy_semantics(self, input_ids, attention_mask):
        """Cache fixed prompt features after the model has been moved to its device."""
        self.text_encoder.eval()
        self._cached_anatomy_semantics = self._compute_anatomy_semantics(
            input_ids, attention_mask
        ).detach()
        return self._cached_anatomy_semantics

    def encode_anatomy_semantics(self, input_ids, attention_mask, batch_size: int):
        if self._cached_anatomy_semantics is not None:
            prompt_emb = self._cached_anatomy_semantics
        else:
            if input_ids is None or attention_mask is None:
                raise ValueError("Text inputs are required until anatomy semantics are cached.")
            prompt_emb = self._compute_anatomy_semantics(input_ids, attention_mask)
        return prompt_emb.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        images,
        input_ids,
        attention_mask,
        z_depths,
        return_aux: bool = False,
        return_deep_supervision: bool = False,
    ):
        expected_context = 4 if self.spatial_view_context else 1
        if z_depths.ndim != 2 or z_depths.shape[1] != expected_context:
            raise ValueError(
                f"Expected context shape [B, {expected_context}], got {tuple(z_depths.shape)}"
            )
        if self.depth_context_adapter is not None:
            images = self.depth_context_adapter(images)
        f1, f2, f3, f4 = self.backbone.get_intermediate_layers(images, n=[5, 11, 17, 23], reshape=True)
        text_emb = self.encode_anatomy_semantics(input_ids, attention_mask, images.shape[0])
        f4 = self.cbam(f4)
        f4_flat = f4.flatten(2).transpose(1, 2) + self.z_encoder(z_depths).unsqueeze(1)
        fused = self.fusion(f4_flat, text_emb)
        gate = self.semantic_gate(f4_flat.mean(dim=1)).unsqueeze(1)
        fused = f4_flat + gate * (fused - f4_flat)
        bottleneck = fused.transpose(1, 2).reshape(f4.shape)
        decoder_output = self.decoder(
            f1,
            f2,
            f3,
            bottleneck,
            return_aux=return_deep_supervision,
        )
        if return_deep_supervision:
            logits, deep_logits = decoder_output
        else:
            logits = decoder_output
            deep_logits = []
        logits = F.interpolate(
            logits,
            size=images.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        if not return_aux and not return_deep_supervision:
            return logits
        presence_logits = (
            self.presence_head(fused.mean(dim=1))
            if self.presence_head is not None
            else None
        )
        if return_deep_supervision:
            return logits, presence_logits, deep_logits
        return logits, presence_logits
