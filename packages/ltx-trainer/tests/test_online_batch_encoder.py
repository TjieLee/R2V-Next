from __future__ import annotations

import traceback
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import typer
from torch import Tensor, nn

from ltx_core.multicond.visual_tokens import extract_projected_visual_tokens
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_trainer.online_data.constants import IMAGE_TASK
from ltx_trainer.online_data.online_batch_encoder import OnlineBatchEncoder, OnlineSampleEncodeError
from ltx_trainer.trainer import LtxvTrainer
from scripts import check_multitask_online_real_encode as real_encode_script


class _RecordingVisionTower(nn.Module):
    def __init__(self, *, compute_dtype: torch.dtype, output_dtype: torch.dtype) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), dtype=compute_dtype))
        self.output_dtype = output_dtype
        self.input_dtypes: list[torch.dtype] = []

    def forward(self, *, pixel_values: Tensor) -> SimpleNamespace:
        self.input_dtypes.append(pixel_values.dtype)
        hidden = torch.ones(
            pixel_values.shape[0],
            256,
            4,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ) * self.scale
        return SimpleNamespace(last_hidden_state=hidden.to(dtype=self.output_dtype))


class _RecordingProjector(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False).to(dtype=dtype)
        self.input_dtypes: list[torch.dtype] = []
        self.output_dtypes: list[torch.dtype] = []

    def forward(self, hidden: Tensor) -> Tensor:
        self.input_dtypes.append(hidden.dtype)
        output = self.projection(hidden)
        self.output_dtypes.append(output.dtype)
        return output


class _WrappedModule(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


def _fake_gemma(vision_tower: nn.Module, projector: nn.Module) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
        )
    )


def test_extract_projected_visual_tokens_aligns_projector_dtype() -> None:
    vision_tower = _RecordingVisionTower(compute_dtype=torch.float32, output_dtype=torch.float32)
    projector = _RecordingProjector(torch.bfloat16)
    wrapped_projector = _WrappedModule(projector)

    result = extract_projected_visual_tokens(
        _fake_gemma(vision_tower, wrapped_projector),
        torch.randn(1, 3, 2, 2, dtype=torch.float32),
        dtype_diagnostics=(diagnostics := {}),
    )

    assert vision_tower.input_dtypes == [torch.float32]
    assert projector.input_dtypes == [torch.bfloat16]
    assert projector.output_dtypes == [torch.bfloat16]
    assert result.tokens.dtype == torch.bfloat16
    assert diagnostics == {
        "vision_module": "_RecordingVisionTower",
        "vision_input_dtype": "torch.float32",
        "vision_input_device": "cpu",
        "vision_output_dtype": "torch.float32",
        "vision_output_device": "cpu",
        "projector_module": "_RecordingProjector",
        "projector_input_dtype": "torch.bfloat16",
        "projector_input_device": "cpu",
        "projector_output_dtype": "torch.bfloat16",
        "projector_output_device": "cpu",
    }


def test_extract_projected_visual_tokens_aligns_vision_input_dtype() -> None:
    vision_tower = _RecordingVisionTower(compute_dtype=torch.bfloat16, output_dtype=torch.float32)
    projector = _RecordingProjector(torch.bfloat16)

    result = extract_projected_visual_tokens(
        _fake_gemma(_WrappedModule(vision_tower), projector),
        torch.randn(1, 3, 2, 2, dtype=torch.float32),
    )

    assert vision_tower.input_dtypes == [torch.bfloat16]
    assert projector.input_dtypes == [torch.bfloat16]
    assert result.tokens.dtype == torch.bfloat16


class _FakeVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), dtype=torch.bfloat16))
        self.input_dtypes: list[torch.dtype] = []

    def forward(self, pixels: Tensor) -> Tensor:
        self.input_dtypes.append(pixels.dtype)
        return pixels[:, :2] * self.scale


class _FakeEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((), dtype=torch.bfloat16))

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.weight * torch.ones(*input_ids.shape, 4, device=input_ids.device, dtype=self.weight.dtype)


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.ones((), dtype=torch.bfloat16))
        self.embedding = _FakeEmbedding()
        self.input_dtypes: list[torch.dtype] = []
        self.attention_mask_dtypes: list[torch.dtype] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert output_hidden_states and return_dict
        self.input_dtypes.append(inputs_embeds.dtype)
        self.attention_mask_dtypes.append(attention_mask.dtype)
        hidden = inputs_embeds.to(dtype=torch.float32)
        return SimpleNamespace(hidden_states=(hidden, hidden + 1.0))


class _FakeFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 6, bias=False).to(dtype=torch.bfloat16)
        self.input_dtypes: list[tuple[torch.dtype, ...]] = []
        self.attention_mask_dtypes: list[torch.dtype] = []

    def forward(
        self,
        hidden_states: tuple[Tensor, ...],
        attention_mask: Tensor,
        _padding_side: str,
    ) -> tuple[Tensor, None]:
        self.input_dtypes.append(tuple(hidden.dtype for hidden in hidden_states))
        self.attention_mask_dtypes.append(attention_mask.dtype)
        return self.projection(hidden_states[-1]), None


class _FakeCausalLm(nn.Module):
    def __init__(
        self,
        vision_tower: nn.Module,
        projector: nn.Module,
        language_model: nn.Module,
    ) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.vision_tower = vision_tower
        self.model.multi_modal_projector = projector
        self.model.language_model = language_model


class _FakeTextEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


class _FakeTokenizer:
    pad_token_id = 0

    @staticmethod
    def apply_chat_template(*_args: Any, **_kwargs: Any) -> str:
        return "serialized prompt"


class _FakeConditionProcessor:
    def __call__(self, *, images: list[Any] | None, **_kwargs: Any) -> dict[str, Tensor]:
        if images:
            input_ids = torch.tensor(
                [[5] + [GEMMA3_CONFIG_FOR_LTX.image_token_index] * 256],
                dtype=torch.long,
            )
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "pixel_values": torch.randn(len(images), 3, 2, 2, dtype=torch.float32),
            }
        return {
            "input_ids": torch.tensor([[5]], dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        }


class _FakeImageProcessor:
    def __call__(self, *, images: list[Any], **_kwargs: Any) -> dict[str, Tensor]:
        return {"pixel_values": torch.randn(len(images), 3, 2, 2, dtype=torch.float32)}


def _minimal_online_encoder() -> tuple[
    OnlineBatchEncoder,
    _FakeVae,
    _RecordingVisionTower,
    _RecordingProjector,
    _FakeLanguageModel,
    _FakeFeatureExtractor,
]:
    vae = _FakeVae()
    vision_tower = _RecordingVisionTower(compute_dtype=torch.bfloat16, output_dtype=torch.float32)
    projector = _RecordingProjector(torch.bfloat16)
    language_model = _FakeLanguageModel()
    feature_extractor = _FakeFeatureExtractor()
    causal_lm = _FakeCausalLm(vision_tower, projector, language_model)

    encoder = object.__new__(OnlineBatchEncoder)
    encoder.config = SimpleNamespace(
        encoder_dtype="bfloat16",
        encoder_device_policy="resident_cuda",
        max_ref_images=1,
        planner_max_length=2300,
        raw_visual_dim=4,
        vlm_reference_preprocess="original",
        video_decoder="pyav",
    )
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.bfloat16
    encoder.last_dtype_diagnostics = {}
    encoder.vae_encoder = vae
    encoder.text_encoder = _FakeTextEncoder(causal_lm)
    encoder.embeddings_processor = SimpleNamespace(feature_extractor=feature_extractor)
    encoder.tokenizer = _FakeTokenizer()
    encoder.processor = _FakeConditionProcessor()
    encoder.image_processor = _FakeImageProcessor()
    encoder.system_prompts = {IMAGE_TASK: "image edit system prompt"}
    return encoder, vae, vision_tower, projector, language_model, feature_extractor


def test_online_encoder_bf16_outside_accelerator_autocast() -> None:
    encoder, vae, vision_tower, projector, language_model, feature_extractor = _minimal_online_encoder()
    target = torch.randint(0, 256, (1, 1, 2, 2, 3), dtype=torch.uint8)
    reference = torch.randint(0, 256, (2, 2, 3), dtype=torch.uint8)
    raw_batch = {
        "target_pixels": target,
        "target_fps": torch.tensor([1.0]),
        "reference_pixels_vae": [[reference]],
        "reference_images_vlm": [[reference]],
        "sample_key": ["sample"],
        "sample_plan_sha256": ["plan"],
        "task": [IMAGE_TASK],
        "target_modality": ["image"],
        "vlm_reference_preprocess": ["original"],
        "manifest_index": torch.tensor([0]),
        "target_source_frame_indices": [[0]],
        "vlm_target_frame_indices": [[0]],
        "vlm_source_frame_indices": [[0]],
        "caption": ["edit the image"],
        "data_decode_ms": torch.tensor([1.0]),
    }
    strategy = SimpleNamespace(config=SimpleNamespace(name="multi_reference_video"))

    result = encoder.encode_for_strategy(raw_batch, strategy=strategy, training_phase="stage3")

    assert vae.input_dtypes == [torch.bfloat16, torch.bfloat16]
    assert vision_tower.input_dtypes == [torch.bfloat16, torch.bfloat16]
    assert projector.input_dtypes == [torch.bfloat16, torch.bfloat16]
    assert projector.output_dtypes == [torch.bfloat16, torch.bfloat16]
    assert language_model.input_dtypes == [torch.bfloat16, torch.bfloat16]
    assert all(dtype == torch.long for dtype in language_model.attention_mask_dtypes)
    assert feature_extractor.input_dtypes == [
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
    ]
    assert all(dtype == torch.long for dtype in feature_extractor.attention_mask_dtypes)
    assert feature_extractor.projection.weight.dtype == torch.bfloat16
    assert result["gt_visual_tokens"]["visual_tokens"].dtype == torch.bfloat16
    assert result["conditions"]["video_prompt_embeds"].dtype == torch.bfloat16
    assert encoder.last_dtype_diagnostics["vision_output_dtype"] == "torch.float32"
    assert encoder.last_dtype_diagnostics["projector_input_dtype"] == "torch.bfloat16"
    assert encoder.last_dtype_diagnostics["projector_output_dtype"] == "torch.bfloat16"
    assert encoder.last_dtype_diagnostics["feature_extractor_input_dtype"] == "torch.bfloat16"
    assert encoder.last_dtype_diagnostics["feature_extractor_weight_dtype"] == "torch.bfloat16"
    assert encoder.last_dtype_diagnostics["feature_extractor_attention_mask_dtype"] == "torch.int64"


def test_frozen_encoder_uses_explicit_cuda_autocast(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = object.__new__(OnlineBatchEncoder)
    encoder.device = torch.device("cuda")
    encoder.dtype = torch.bfloat16
    calls: list[tuple[str, torch.dtype]] = []

    def fake_autocast(*, device_type: str, dtype: torch.dtype) -> Any:
        calls.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    with encoder._frozen_encode_autocast():
        pass
    assert calls == [("cuda", torch.bfloat16)]


def test_real_encode_entry_allows_i2i_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_check(
        _config: str,
        *,
        num_image_samples: int,
        num_video_samples: int,
    ) -> None:
        calls.append((num_image_samples, num_video_samples))

    monkeypatch.setattr(real_encode_script, "run_real_encode_check", fake_check)
    real_encode_script.main("config.yaml", num_image_samples=1, num_video_samples=0)
    assert calls == [(1, 0)]

    with pytest.raises(typer.BadParameter, match="At least one image or video"):
        real_encode_script.main("config.yaml", num_image_samples=0, num_video_samples=0)


class _FakeAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.reduced_values: list[int] = []

    def reduce(self, flag: Tensor, *, reduction: str) -> Tensor:
        assert reduction == "max"
        self.reduced_values.append(int(flag.item()))
        return flag


class _FakeSampler:
    @staticmethod
    def was_consumed_in_current_step(_index: int) -> bool:
        return False


class _FailingOnlineEncoder:
    def __init__(self, errors: list[Exception]) -> None:
        self.errors = errors
        self.calls = 0

    def encode_for_strategy(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"encoded": True}


def _retry_trainer(encoder: _FailingOnlineEncoder, *, retries: int = 8) -> LtxvTrainer:
    trainer = object.__new__(LtxvTrainer)
    trainer._online_batch_encoder = encoder
    trainer._online_sampler = _FakeSampler()
    trainer._config = SimpleNamespace(
        data=SimpleNamespace(online_encoding=SimpleNamespace(runtime_max_retries=retries))
    )
    trainer._training_strategy = SimpleNamespace(config=SimpleNamespace(training_phase="stage3"))
    trainer._accelerator = _FakeAccelerator()
    trainer._load_online_retry_batch = lambda _attempt, _collate: {  # type: ignore[method-assign]
        "manifest_index": torch.tensor([0])
    }
    trainer._log_online_reject = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    return trainer


def test_programming_error_is_not_retried() -> None:
    encoder = _FailingOnlineEncoder([RuntimeError("mat1 and mat2 must have the same dtype")])
    trainer = _retry_trainer(encoder, retries=8)

    with pytest.raises(RuntimeError, match="mat1 and mat2") as exc_info:
        trainer._prepare_online_batch_with_retry({"manifest_index": torch.tensor([0])})

    assert encoder.calls == 1
    assert "encode_for_strategy" in "".join(traceback.format_tb(exc_info.tb))


def test_data_error_still_retries_synchronously() -> None:
    encoder = _FailingOnlineEncoder(
        [
            OnlineSampleEncodeError("bad sample one"),
            OnlineSampleEncodeError("bad sample two"),
        ]
    )
    trainer = _retry_trainer(encoder, retries=8)

    result = trainer._prepare_online_batch_with_retry({"manifest_index": torch.tensor([0])})

    assert result == {"encoded": True}
    assert encoder.calls == 3
    assert trainer._accelerator.reduced_values.count(1) == 2
