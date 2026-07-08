#!/usr/bin/env python3
"""Stage 1 multi-reference teacher-forcing inference for overfit samples.

This is not the final Stage 2 planner inference path. It uses precomputed
target-video GT SigLIP visual tokens as teacher visual conditions, so it can
check whether a Stage 1 checkpoint learned to consume:

- vlm_conditions
- gt_siglip_tokens
- multi_reference_latents

CFG, negative prompts, ValidationRunner and online Gemma/VLM are intentionally
not used here.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import typer
import yaml
from einops import rearrange
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from rich.console import Console
from rich.progress import track
from safetensors.torch import load_file
from torch import Tensor

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.multicond.rope_mask_builder import build_multiref_sequence
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_core.types import VideoLatentShape
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.model_loader import load_embeddings_processor, load_transformer, load_video_vae_decoder
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.training_strategies.multi_reference_video import MultiReferenceVideoStrategy
from ltx_trainer.utils import open_image_as_srgb
from ltx_trainer.video_utils import save_video

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Run Stage 1 multi-reference teacher-forcing inference on one overfit sample.",
)

_DEFAULT_TILING = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=192, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=48, tile_overlap_in_frames=24),
)


class _TupleSafeLoader(yaml.SafeLoader):
    pass


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


_TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)


def _velocity_to_denoised(latent: Tensor, velocity: Tensor, timesteps: Tensor) -> Tensor:
    if latent.shape != velocity.shape:
        raise ValueError(f"Velocity shape {list(velocity.shape)} does not match latent shape {list(latent.shape)}")
    if timesteps.shape != latent.shape[:2]:
        raise ValueError(
            f"Timesteps shape {list(timesteps.shape)} must match packed token shape {list(latent.shape[:2])}"
        )
    denoise_timesteps = timesteps.unsqueeze(-1).to(device=latent.device, dtype=torch.float32)
    return (latent.to(torch.float32) - velocity.to(torch.float32) * denoise_timesteps).to(latent.dtype)


def _read_manifest_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return list(data.values())
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON manifest must contain a list or dict of objects: {path}")
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError(f"Unsupported manifest suffix: {path.suffix}")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for pattern in ("*.json", "*.jsonl", "*.csv"):
            for file in sorted(path.glob(pattern)):
                rows.extend(_read_manifest_file(file))
        if not rows:
            raise FileNotFoundError(f"No manifest shards found in {path}")
        return rows
    if path.is_file():
        return _read_manifest_file(path)
    raise FileNotFoundError(f"Manifest does not exist: {path}")


def _resolve_path(value: str, root_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root_dir / path


def _output_relative(path: Path, data_root: Path) -> Path:
    try:
        return path.relative_to(data_root)
    except ValueError:
        return Path(*path.parts[1:]) if path.is_absolute() else path


def _normalize_precomputed_root(path: Path) -> Path:
    if (path / ".precomputed").is_dir():
        return path / ".precomputed"
    return path


def _parse_reference_images(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("reference_images JSON must decode to a list")
            return [str(item) for item in parsed]
        for sep in ("|", ";", ","):
            if sep in text:
                return [part.strip() for part in text.split(sep) if part.strip()]
        return [text]
    raise ValueError(f"Unsupported reference_images value type: {type(value).__name__}")


def _load_pt_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing precomputed file: {path}")
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def _unsqueeze_sample_dim(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Tensor):
            out[key] = value.unsqueeze(0)
        else:
            out[key] = torch.tensor([value]) if isinstance(value, (int, float, bool)) else value
    return out


def _build_single_sample_batch(
    *,
    latents: dict[str, Any],
    conditions: dict[str, Any],
    multi_reference_latents: dict[str, Any],
    gt_visual_tokens: dict[str, Any],
) -> dict[str, Any]:
    latents = PrecomputedDataset._normalize_video_latents(latents)
    return {
        "latents": _unsqueeze_sample_dim(latents),
        "conditions": _unsqueeze_sample_dim(conditions),
        "multi_ref_latents": _unsqueeze_sample_dim(multi_reference_latents),
        "gt_visual_tokens": _unsqueeze_sample_dim(gt_visual_tokens),
    }


def _move_nested_to_device(data: dict[str, Any], *, device: torch.device, dtype: torch.dtype | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Tensor):
            if dtype is not None and value.is_floating_point():
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def _shape(value: Tensor) -> list[int]:
    return list(value.shape)


def _condition_feature_key(conditions: dict[str, Tensor]) -> str:
    return "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"


def _load_config(config_path: Path) -> LtxTrainerConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    config_data = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_TupleSafeLoader)
    cfg = LtxTrainerConfig(**config_data)
    if cfg.training_strategy.name != "multi_reference_video":
        raise ValueError(
            f"Stage 1 teacher inference requires training_strategy.name='multi_reference_video', "
            f"got {cfg.training_strategy.name!r}."
        )
    if cfg.training_strategy.cfg_dropout_enabled:
        console.print("[yellow]Stage1 teacher inference runs without CFG; overriding cfg_dropout_enabled=false.[/yellow]")
        cfg.training_strategy.cfg_dropout_enabled = False
    return cfg


def _setup_lora(transformer: torch.nn.Module, cfg: LtxTrainerConfig) -> torch.nn.Module:
    if cfg.lora is None:
        raise ValueError("model.training_mode='lora' requires a lora config")
    lora_config = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.dropout,
        init_lora_weights=True,
    )
    return get_peft_model(transformer, lora_config)


def _load_checkpoint_weights(
    *,
    checkpoint_path: Path,
    cfg: LtxTrainerConfig,
    transformer: torch.nn.Module,
    embeddings_processor: torch.nn.Module,
    strategy: MultiReferenceVideoStrategy,
) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    state_dict = load_file(checkpoint_path)

    strategy.load_extra_checkpoint_state_dict(state_dict)

    processor_state = {
        key.removeprefix("embeddings_processor."): value
        for key, value in state_dict.items()
        if key.startswith("embeddings_processor.")
    }
    if processor_state:
        embeddings_processor.load_state_dict(processor_state, strict=False)
        console.print("Loaded embeddings_processor auxiliary checkpoint state")

    if cfg.model.training_mode == "full":
        transformer_state = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("training_strategy.")
            and not key.startswith("embeddings_processor.")
            and not key.startswith("text_encoder.")
        }
        if transformer_state:
            transformer.load_state_dict(transformer_state, strict=True)
        return

    lora_state = {
        key.replace("diffusion_model.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("diffusion_model.")
    }
    if not lora_state:
        console.print("[yellow]No diffusion_model.* LoRA weights found; loaded auxiliary state only.[/yellow]")
        return
    base_model = transformer.get_base_model()
    set_peft_model_state_dict(base_model, lora_state)
    console.print(f"Loaded LoRA checkpoint: {checkpoint_path}")


def _copy_sample_media(
    *,
    sample: dict[str, Any],
    sample_dir: Path,
    manifest_root: Path,
    video_column: str,
    reference_column: str,
) -> tuple[Path, list[Path]]:
    gt_path = _resolve_path(str(sample[video_column]), manifest_root)
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT video does not exist: {gt_path}")
    gt_dst = sample_dir / "gt.mp4"
    shutil.copy2(gt_path, gt_dst)

    ref_outputs: list[Path] = []
    for idx, ref_value in enumerate(_parse_reference_images(sample.get(reference_column))):
        ref_path = _resolve_path(ref_value, manifest_root)
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference image does not exist: {ref_path}")
        dst = sample_dir / f"ref_{idx}.jpg"
        try:
            image = open_image_as_srgb(ref_path)
            image.save(dst, quality=95)
        except Exception:
            dst = dst.with_suffix(ref_path.suffix or ".jpg")
            shutil.copy2(ref_path, dst)
        ref_outputs.append(dst)
    return gt_dst, ref_outputs


def _load_sample_precomputed(
    *,
    row: dict[str, Any],
    manifest_root: Path,
    precomputed_root: Path,
    video_column: str,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    video_path = _resolve_path(str(row[video_column]), manifest_root)
    rel_path = _output_relative(video_path, manifest_root).with_suffix(".pt")
    files = {
        "latents": precomputed_root / "latents" / rel_path,
        "conditions": precomputed_root / "vlm_conditions" / rel_path,
        "multi_reference_latents": precomputed_root / "multi_reference_latents" / rel_path,
        "gt_visual_tokens": precomputed_root / "gt_siglip_tokens" / rel_path,
    }
    return rel_path, {key: _load_pt_file(path) for key, path in files.items()}


def _prepare_condition_context(
    *,
    strategy: MultiReferenceVideoStrategy,
    embeddings_processor: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Tensor], dict[str, list[int]]]:
    batch["conditions"] = _move_nested_to_device(batch["conditions"], device=device, dtype=dtype)
    batch["gt_visual_tokens"] = _move_nested_to_device(batch["gt_visual_tokens"], device=device, dtype=dtype)

    conditions = batch["conditions"]
    key = _condition_feature_key(conditions)
    original_shape = _shape(conditions[key])
    raw_visual = batch["gt_visual_tokens"][strategy.config.visual_token_key]
    raw_visual_shape = _shape(raw_visual)

    source_dim = strategy.config.visual_token_source_dim
    if source_dim is not None and raw_visual.shape[-1] != source_dim:
        raise ValueError(f"GT visual token raw dim {raw_visual.shape[-1]} does not match config source dim {source_dim}")

    conditions = strategy.prepare_conditions(batch, conditions)
    pre_connector_key = _condition_feature_key(conditions)
    pre_connector_shape = _shape(conditions[pre_connector_key])
    target_dim = strategy.config.visual_token_target_dim
    if target_dim is not None and conditions[pre_connector_key].shape[-1] != target_dim:
        raise ValueError(
            f"Final pre-connector condition dim {conditions[pre_connector_key].shape[-1]} does not match "
            f"config target dim {target_dim}"
        )
    appended_tokens = conditions[pre_connector_key].shape[1] - original_shape[1]
    if appended_tokens < raw_visual.shape[1]:
        raise ValueError(
            "Final condition sequence did not append the expected GT visual tokens: "
            f"original {original_shape}, raw GT {raw_visual_shape}, final {pre_connector_shape}"
        )

    video_features = conditions[pre_connector_key]
    audio_features = conditions.get("audio_prompt_embeds")
    additive_mask = convert_to_additive_mask(conditions["prompt_attention_mask"], video_features.dtype)
    video_embeds, audio_embeds, attention_mask = embeddings_processor.create_embeddings(
        video_features,
        audio_features,
        additive_mask,
    )
    conditions["video_prompt_embeds"] = video_embeds
    if audio_embeds is not None:
        conditions["audio_prompt_embeds"] = audio_embeds
    conditions["prompt_attention_mask"] = attention_mask

    shapes = {
        "original_condition_shape": original_shape,
        "raw_gt_visual_shape": raw_visual_shape,
        "pre_connector_condition_shape": pre_connector_shape,
        "transformer_condition_shape": _shape(video_embeds),
    }
    return conditions, shapes


def _denoise_stage1(
    *,
    transformer: torch.nn.Module,
    strategy: MultiReferenceVideoStrategy,
    batch: dict[str, Any],
    conditions: dict[str, Tensor],
    num_inference_steps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be >= 1")

    latents_data = _move_nested_to_device(batch["latents"], device=device, dtype=dtype)
    ref_data = _move_nested_to_device(batch["multi_ref_latents"], device=device, dtype=dtype)
    target_shape_tensor = latents_data["latents"]
    batch_size = target_shape_tensor.shape[0]
    if batch_size != 1:
        raise ValueError(f"Stage1 overfit inference expects batch size 1, got {batch_size}")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    target_latents = torch.randn(target_shape_tensor.shape, generator=generator, device=device, dtype=dtype)
    target_tokens = strategy._video_patchifier.patchify(target_latents)
    target_seq_len = target_tokens.shape[1]

    ref_latents = strategy._normalize_reference_latents(ref_data["latents"])
    ref_valid_mask = strategy._get_reference_valid_mask(ref_data, ref_latents)
    if strategy.config.max_ref_images_per_sample is not None:
        max_refs = strategy.config.max_ref_images_per_sample
        ref_latents = ref_latents[:, :max_refs]
        ref_valid_mask = ref_valid_mask[:, :max_refs]
    _, num_refs, _channels, ref_frames, ref_height, ref_width = ref_latents.shape

    num_frames = int(latents_data["num_frames"].flatten()[0].item())
    height = int(latents_data["height"].flatten()[0].item())
    width = int(latents_data["width"].flatten()[0].item())
    fps_value = strategy._first_scalar(latents_data.get("fps"), default=24.0)

    ref_tokens = strategy._video_patchifier.patchify(ref_latents.reshape(num_refs, *ref_latents.shape[2:]))
    ref_tokens = ref_tokens.reshape(1, num_refs, ref_tokens.shape[1], ref_tokens.shape[2])
    target_positions = strategy._get_video_positions(
        num_frames=num_frames,
        height=height,
        width=width,
        batch_size=1,
        fps=fps_value,
        device=device,
    )
    ref_positions = strategy._get_video_positions(
        num_frames=ref_frames,
        height=ref_height,
        width=ref_width,
        batch_size=num_refs,
        fps=strategy._first_scalar(ref_data.get("fps"), default=1.0),
        device=device,
    )
    ref_positions = ref_positions.reshape(1, num_refs, *ref_positions.shape[1:])
    ref_positions = strategy._scale_reference_positions(ref_positions, height, width, ref_height, ref_width)

    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(device=device).float()
    stepper = EulerDiffusionStep()
    context_key = _condition_feature_key(conditions)
    context = conditions[context_key]
    context_mask = conditions["prompt_attention_mask"]

    transformer.eval()
    with torch.inference_mode():
        for step_idx, sigma in enumerate(track(sigmas[:-1], description="Stage 1 denoising")):
            sigma_batch = sigma.unsqueeze(0)
            target_timesteps = torch.full(
                (1, target_seq_len),
                float(sigma.item()),
                dtype=torch.float32,
                device=device,
            )
            target_loss_mask = torch.ones((1, target_seq_len), dtype=torch.bool, device=device)
            packed = build_multiref_sequence(
                ref_tokens=ref_tokens,
                ref_positions=ref_positions,
                ref_valid_mask=ref_valid_mask,
                target_tokens=target_tokens,
                target_positions=target_positions,
                target_timesteps=target_timesteps,
                target_loss_mask=target_loss_mask,
                reference_time_stride=strategy.config.reference_time_stride,
            )
            video = Modality(
                enabled=True,
                latent=packed.latents,
                sigma=sigma_batch,
                timesteps=packed.timesteps,
                positions=packed.positions,
                context=context,
                context_mask=context_mask,
                attention_mask=packed.attention_mask,
            )
            velocity_video, _ = transformer(video=video, audio=None, perturbations=None)
            if velocity_video is None:
                raise RuntimeError("Transformer returned no video velocity during Stage 1 inference")
            denoised_video = _velocity_to_denoised(video.latent, velocity_video, packed.timesteps)
            next_packed = stepper.step(video.latent, denoised_video, sigmas, step_idx)
            target_tokens = next_packed[:, -target_seq_len:, :]

    return strategy._video_patchifier.unpatchify(
        target_tokens,
        output_shape=VideoLatentShape.from_torch_shape(target_shape_tensor.shape),
    )


def _decode_video_latents(
    *,
    vae_decoder: torch.nn.Module,
    latents: Tensor,
    device: torch.device,
    decode_tile: bool,
) -> Tensor:
    vae_decoder.to(device)
    latents = latents.to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        if decode_tile:
            chunks = list(vae_decoder.tiled_decode(latents, tiling_config=_DEFAULT_TILING))
            video = torch.cat(chunks, dim=2)
        else:
            video = vae_decoder(latents)
    video = ((video + 1.0) / 2.0).clamp(0.0, 1.0)
    return rearrange(video, "1 c f h w -> f c h w").float().cpu()


def _load_models_and_strategy(
    *,
    cfg: LtxTrainerConfig,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, MultiReferenceVideoStrategy]:
    transformer = load_transformer(cfg.model.model_path, device=device, dtype=dtype)
    embeddings_processor = load_embeddings_processor(cfg.model.model_path, device=device, dtype=dtype)
    embeddings_processor.feature_extractor = None
    embeddings_processor.requires_grad_(False)

    strategy = get_training_strategy(cfg.training_strategy)
    if not isinstance(strategy, MultiReferenceVideoStrategy):
        raise TypeError(f"Expected MultiReferenceVideoStrategy, got {type(strategy).__name__}")
    strategy.attach_models(transformer=transformer, embeddings_processor=embeddings_processor, text_encoder=None)

    projection = strategy.get_trainable_modules().get("visual_token_projection")
    if projection is None:
        raise ValueError(
            "visual_token_projection is not enabled. Set training_strategy.visual_token_source_dim=3840 and "
            "visual_token_target_dim=4096 in the config."
        )
    console.print(
        "visual_token_projection enabled: "
        f"source_dim={cfg.training_strategy.visual_token_source_dim}, "
        f"target_dim={cfg.training_strategy.visual_token_target_dim}"
    )

    if cfg.model.training_mode == "lora":
        transformer = _setup_lora(transformer, cfg)
    _load_checkpoint_weights(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        transformer=transformer,
        embeddings_processor=embeddings_processor,
        strategy=strategy,
    )
    transformer.eval()
    embeddings_processor.eval()
    for module in strategy.get_trainable_modules().values():
        module.eval()

    vae_decoder = load_video_vae_decoder(cfg.model.model_path, device=device, dtype=dtype)
    vae_decoder.eval()
    return transformer, embeddings_processor, vae_decoder, strategy


@app.command()
def main(  # noqa: PLR0913
    config: str = typer.Option(..., help="Stage 1 overfit/training YAML config."),
    checkpoint: str = typer.Option(..., help="Stage 1 checkpoint safetensors file."),
    manifest: str = typer.Option(..., help="Overfit manifest JSON/JSONL/CSV."),
    precomputed_root: str = typer.Option(..., help="Root containing latents/, vlm_conditions/, etc."),
    sample_index: int = typer.Option(0, help="Sample index inside manifest."),
    output_dir: str = typer.Option(..., help="Directory for sample_<index>/ outputs."),
    device: str = typer.Option("cuda", help="Torch device, e.g. cuda or cuda:0."),
    seed: int = typer.Option(42, help="Random seed for target latent initialization."),
    num_inference_steps: int = typer.Option(50, help="Number of denoising steps."),
    video_column: str = typer.Option("video", help="Manifest video column."),
    caption_column: str = typer.Option("caption", help="Manifest caption column."),
    reference_column: str = typer.Option("reference_images", help="Manifest reference image column."),
    root_dir: str | None = typer.Option(None, help="Root for relative manifest media paths. Defaults to manifest parent."),
    fps: float | None = typer.Option(None, help="Override output fps. Defaults to latent metadata fps."),
    decode_tile: bool = typer.Option(True, "--decode-tile/--no-decode-tile", help="Use tiled VAE decode."),
) -> None:
    if sample_index < 0:
        raise typer.BadParameter("--sample-index must be >= 0")
    if num_inference_steps < 1:
        raise typer.BadParameter("--num-inference-steps must be >= 1")

    cfg = _load_config(Path(config))
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    torch_device = torch.device(device)
    dtype = torch.bfloat16
    manifest_path = Path(manifest)
    manifest_root = Path(root_dir) if root_dir is not None else manifest_path.parent
    rows = _read_manifest(manifest_path)
    if sample_index >= len(rows):
        raise typer.BadParameter(f"--sample-index must be in [0, {len(rows) - 1}]")
    sample = rows[sample_index]
    if video_column not in sample:
        raise ValueError(f"Selected sample has no video column '{video_column}'")

    sample_dir = Path(output_dir) / f"sample_{sample_index}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    gt_copy, ref_copies = _copy_sample_media(
        sample=sample,
        sample_dir=sample_dir,
        manifest_root=manifest_root,
        video_column=video_column,
        reference_column=reference_column,
    )

    rel_path, precomputed = _load_sample_precomputed(
        row=sample,
        manifest_root=manifest_root,
        precomputed_root=_normalize_precomputed_root(Path(precomputed_root)),
        video_column=video_column,
    )
    batch = _build_single_sample_batch(
        latents=precomputed["latents"],
        conditions=precomputed["conditions"],
        multi_reference_latents=precomputed["multi_reference_latents"],
        gt_visual_tokens=precomputed["gt_visual_tokens"],
    )

    transformer, embeddings_processor, vae_decoder, strategy = _load_models_and_strategy(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        device=torch_device,
        dtype=dtype,
    )

    conditions, condition_shapes = _prepare_condition_context(
        strategy=strategy,
        embeddings_processor=embeddings_processor,
        batch=batch,
        device=torch_device,
        dtype=dtype,
    )
    console.print(f"original condition shape: {condition_shapes['original_condition_shape']}")
    console.print(f"raw GT visual token shape: {condition_shapes['raw_gt_visual_shape']}")
    console.print(f"pre-connector final condition shape: {condition_shapes['pre_connector_condition_shape']}")
    console.print(f"transformer condition shape: {condition_shapes['transformer_condition_shape']}")

    generated_latents = _denoise_stage1(
        transformer=transformer,
        strategy=strategy,
        batch=batch,
        conditions=conditions,
        num_inference_steps=num_inference_steps,
        seed=seed,
        device=torch_device,
        dtype=dtype,
    )
    decoded = _decode_video_latents(
        vae_decoder=vae_decoder,
        latents=generated_latents,
        device=torch_device,
        decode_tile=decode_tile,
    )

    latent_fps = strategy._first_scalar(batch["latents"].get("fps"), default=24.0)
    output_fps = float(fps) if fps is not None else float(latent_fps)
    generated_path = sample_dir / "generated.mp4"
    save_video(video_tensor=decoded, output_path=generated_path, fps=output_fps, video_format="FCHW")

    metadata = {
        "sample_index": sample_index,
        "checkpoint": str(checkpoint_path),
        "config": str(Path(config)),
        "prompt": str(sample.get(caption_column, "")),
        "gt_path": str(_resolve_path(str(sample[video_column]), manifest_root)),
        "gt_copy": str(gt_copy),
        "reference_paths": [str(_resolve_path(value, manifest_root)) for value in _parse_reference_images(sample.get(reference_column))],
        "reference_copies": [str(path) for path in ref_copies],
        "precomputed_relative_path": str(rel_path),
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": 1.0,
        "condition_mode": "stage1_teacher_gt_siglip",
        "raw_gt_visual_shape": condition_shapes["raw_gt_visual_shape"],
        "pre_connector_condition_shape": condition_shapes["pre_connector_condition_shape"],
        "final_condition_shape": condition_shapes["transformer_condition_shape"],
        "reference_latent_shape": _shape(batch["multi_ref_latents"]["latents"]),
        "target_latent_shape": _shape(batch["latents"]["latents"]),
        "fps": output_fps,
        "decode_tile": decode_tile,
        "generated": str(generated_path),
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    console.print(f"[green]Saved generated video:[/green] {generated_path}")


if __name__ == "__main__":
    app()
