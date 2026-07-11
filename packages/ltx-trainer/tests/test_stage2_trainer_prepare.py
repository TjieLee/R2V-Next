from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model
from torch import nn

from ltx_core.model.transformer.model import LTXModel
from ltx_trainer.trainer import LtxvTrainer, TrainingStepOutput
from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Config,
    MultiReferencePlannerStage2Strategy,
)


class _TrackingLinear(nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4)
        self.to_devices = []
        self._enable_gradient_checkpointing = False

    def set_gradient_checkpointing(self, enable: bool) -> None:
        self._enable_gradient_checkpointing = enable

    def to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        self.to_devices.append(torch.device(device) if device is not None else None)
        return super().to(*args, **kwargs)


class _FakeAccelerator:
    def __init__(self, distributed_type=DistributedType.NO) -> None:
        self.distributed_type = distributed_type
        self.device = torch.device("cpu")
        self.prepared = []
        self.autocast_active = False
        self.autocast_entered = False
        self.autocast_exited = False

    @contextmanager
    def autocast(self):
        self.autocast_entered = True
        self.autocast_active = True
        try:
            yield
        finally:
            self.autocast_active = False
            self.autocast_exited = True

    def prepare(self, *modules):
        self.prepared.extend(modules)
        return modules[0] if len(modules) == 1 else modules

    @staticmethod
    def unwrap_model(module, keep_torch_compile=False):
        del keep_torch_compile
        return getattr(module, "module", module)

    @staticmethod
    def get_state_dict(module):
        return getattr(module, "module", module).state_dict()


class _FakeStrategy:
    def __init__(self, module: nn.Module) -> None:
        self.module = module

    def get_trainable_modules(self):
        return {"planner": self.module}

    def set_trainable_modules(self, modules):
        self.module = modules["planner"]


class _FakeEmbeddingsProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_connector = nn.Linear(4, 4)


def _trainer_for_prepare(distributed_type=DistributedType.NO):
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._accelerator = _FakeAccelerator(distributed_type)
    trainer._transformer = _TrackingLinear()
    trainer._embeddings_processor = _FakeEmbeddingsProcessor()
    trainer._text_encoder = None
    trainer._training_strategy = _FakeStrategy(nn.Linear(4, 4))
    trainer._train_transformer = False
    trainer._train_embeddings_processor = False
    trainer._train_text_encoder = False
    trainer._config = SimpleNamespace(
        model=SimpleNamespace(training_mode="lora"),
        optimization=SimpleNamespace(enable_gradient_checkpointing=True),
    )
    return trainer


def test_frozen_transformer_is_not_prepared_by_ddp() -> None:
    trainer = _trainer_for_prepare()
    transformer = trainer._transformer

    trainer._prepare_models_for_training()

    assert transformer not in trainer._accelerator.prepared
    assert transformer not in trainer._accumulation_models
    assert transformer.to_devices == [torch.device("cpu")]
    assert transformer.training is False
    assert all(not parameter.requires_grad for parameter in transformer.parameters())
    assert trainer._accumulation_models == [trainer._training_strategy.module]


def test_frozen_transformer_enables_gradient_checkpointing() -> None:
    trainer = _trainer_for_prepare()
    transformer = trainer._transformer

    trainer._prepare_models_for_training()

    assert transformer not in trainer._accelerator.prepared
    assert transformer.training is False
    assert all(not parameter.requires_grad for parameter in transformer.parameters())
    assert transformer._enable_gradient_checkpointing is True


def test_only_trainable_video_connector_is_prepared() -> None:
    trainer = _trainer_for_prepare()
    processor = trainer._embeddings_processor
    connector = processor.video_connector
    trainer._train_embeddings_processor = True

    trainer._prepare_models_for_training()

    assert processor not in trainer._accelerator.prepared
    assert connector in trainer._accelerator.prepared
    assert processor.video_connector in trainer._accumulation_models


def test_frozen_transformer_fsdp_fails_clearly() -> None:
    trainer = _trainer_for_prepare(DistributedType.FSDP)

    try:
        trainer._prepare_models_for_training()
    except RuntimeError as exc:
        assert "frozen-Transformer FSDP is not implemented" in str(exc)
    else:
        raise AssertionError("Expected frozen Transformer FSDP to fail")


def test_frozen_transformer_allows_input_gradient() -> None:
    upstream = nn.Linear(4, 4)
    frozen_renderer = nn.Linear(4, 4)
    frozen_renderer.requires_grad_(False)
    output = frozen_renderer(upstream(torch.randn(2, 4)))

    output.square().mean().backward()

    assert all(parameter.grad is None for parameter in frozen_renderer.parameters())
    assert all(parameter.grad is not None and torch.any(parameter.grad != 0) for parameter in upstream.parameters())


class _IdentityBlockInputProcessor:
    def __call__(self, modality, perturbations, block_idx, **kwargs):
        del perturbations, block_idx, kwargs
        return modality


class _TinyTransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, video=None, audio=None):
        return self.projection(video) if video is not None else None, audio


def _tiny_ltx_model() -> LTXModel:
    model = LTXModel.__new__(LTXModel)
    nn.Module.__init__(model)
    model._enable_gradient_checkpointing = True
    model.transformer_blocks = nn.ModuleList([_TinyTransformerBlock()])
    model.block_input_processor = _IdentityBlockInputProcessor()
    model.eval()
    return model


def test_eval_model_uses_checkpoint_when_grad_enabled(monkeypatch) -> None:
    model = _tiny_ltx_model()
    calls = []

    def fake_checkpoint(function, *args, **kwargs):
        calls.append((function, kwargs))
        return function(*args)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", fake_checkpoint)
    video, _ = model._process_transformer_blocks(
        torch.randn(2, 4, requires_grad=True),
        None,
        object(),
    )

    assert video is not None
    assert len(calls) == 1
    assert calls[0][1] == {"use_reentrant": False}


def test_eval_model_skips_checkpoint_under_no_grad(monkeypatch) -> None:
    model = _tiny_ltx_model()
    calls = []

    def fake_checkpoint(function, *args, **kwargs):
        calls.append((function, kwargs))
        return function(*args)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", fake_checkpoint)
    with torch.no_grad():
        video, _ = model._process_transformer_blocks(torch.randn(2, 4), None, object())

    assert video is not None
    assert calls == []


def test_frozen_checkpointed_renderer_propagates_input_gradient() -> None:
    upstream = nn.Linear(4, 4)
    renderer = _tiny_ltx_model()
    renderer.requires_grad_(False)

    output, _ = renderer._process_transformer_blocks(
        upstream(torch.randn(2, 4)),
        None,
        object(),
    )
    output.square().mean().backward()

    assert all(parameter.grad is None for parameter in renderer.parameters())
    assert all(parameter.grad is not None and torch.any(parameter.grad != 0) for parameter in upstream.parameters())


def test_training_step_enters_accelerator_autocast(monkeypatch) -> None:
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._accelerator = _FakeAccelerator()
    expected = TrainingStepOutput(loss=torch.tensor([1.0]), sigma=torch.tensor([0.5]))

    def fake_training_step_autocast(batch):
        assert batch == {}
        assert trainer._accelerator.autocast_active is True
        return expected

    monkeypatch.setattr(trainer, "_training_step_autocast", fake_training_step_autocast)

    actual = trainer._training_step({})

    assert actual is expected
    assert trainer._accelerator.autocast_entered is True
    assert trainer._accelerator.autocast_exited is True
    assert trainer._accelerator.autocast_active is False


def test_frozen_bf16_renderer_accepts_float32_input_under_autocast() -> None:
    upstream = nn.Linear(4, 4)
    renderer = nn.Linear(4, 4).to(dtype=torch.bfloat16)
    renderer.requires_grad_(False)
    renderer.eval()
    renderer_input = upstream(torch.randn(2, 4))

    assert renderer_input.dtype == torch.float32
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = renderer(renderer_input)
            loss = output.float().square().mean()
    except RuntimeError as exc:
        pytest.skip(f"CPU BF16 autocast is unavailable: {exc}")
    loss.backward()

    assert torch.isfinite(output).all()
    assert all(parameter.grad is None for parameter in renderer.parameters())
    assert all(parameter.grad is not None and torch.any(parameter.grad != 0) for parameter in upstream.parameters())


class _TrainingStepEmbeddingsProcessor:
    @staticmethod
    def create_embeddings(video_features, audio_features, additive_mask):
        return video_features, audio_features, additive_mask


class _TrainingStepStrategy:
    @staticmethod
    def prepare_conditions(batch, conditions):
        del batch
        return conditions

    @staticmethod
    def postprocess_conditions_after_connector(batch, conditions):
        del batch
        return conditions

    @staticmethod
    def prepare_training_inputs(batch, timestep_sampler):
        del batch, timestep_sampler
        video = SimpleNamespace(enabled=True, sigma=torch.tensor([0.5]))
        return SimpleNamespace(video=video, audio=None)

    @staticmethod
    def compute_loss(video_pred, audio_pred, model_inputs):
        del audio_pred, model_inputs
        return video_pred


class _AutocastRecordingTransformer(nn.Module):
    def __init__(self, accelerator: _FakeAccelerator) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.forward_saw_autocast = False

    def forward(self, *, video, audio, perturbations):
        del video, audio, perturbations
        self.forward_saw_autocast = self.accelerator.autocast_active
        return torch.ones(1), None


def test_training_step_autocast_covers_transformer_forward() -> None:
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._accelerator = _FakeAccelerator()
    trainer._training_strategy = _TrainingStepStrategy()
    trainer._embeddings_processor = _TrainingStepEmbeddingsProcessor()
    trainer._transformer = _AutocastRecordingTransformer(trainer._accelerator)
    trainer._timestep_sampler = object()
    batch = {
        "conditions": {
            "video_prompt_embeds": torch.randn(1, 2, 4),
            "audio_prompt_embeds": None,
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
    }

    output = trainer._training_step(batch)

    assert trainer._transformer.forward_saw_autocast is True
    assert output.loss.shape == (1,)
    assert output.sigma.shape == (1,)


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden):
        return self.o_proj(self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden))


class _TinyGemmaTextEncoder(nn.Module):
    def __init__(self, language_model: nn.Module) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.model = nn.Module()
        self.model.model.language_model = language_model


def _tiny_lora_text_encoder(seed: int) -> _TinyGemmaTextEncoder:
    torch.manual_seed(seed)
    language_model = get_peft_model(
        _TinyLanguageModel(),
        LoraConfig(
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            init_lora_weights=True,
        ),
    )
    return _TinyGemmaTextEncoder(language_model)


def test_gemma_lora_checkpoint_roundtrip() -> None:
    first_encoder = _tiny_lora_text_encoder(seed=123)
    first_strategy = MultiReferencePlannerStage2Strategy(
        MultiReferencePlannerStage2Config(use_online_vlm=False, cfg_dropout_enabled=False)
    )
    first_strategy.text_encoder = first_encoder
    for name, parameter in first_encoder.named_parameters():
        if "lora_" in name:
            parameter.data.fill_(0.125)
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(False)
    first_base = {
        name: parameter.detach().clone()
        for name, parameter in first_encoder.named_parameters()
        if "lora_" not in name
    }

    saved = first_strategy.get_text_encoder_checkpoint_state_dict(_FakeAccelerator())

    assert saved
    assert all(key.startswith("model.model.language_model.") for key in saved)
    assert all("lora_" in key for key in saved)
    second_encoder = _tiny_lora_text_encoder(seed=123)
    second_base_before = {
        name: parameter.detach().clone()
        for name, parameter in second_encoder.named_parameters()
        if "lora_" not in name
    }
    missing, unexpected = second_encoder.load_state_dict(saved, strict=False)
    assert not unexpected
    assert all("lora_" not in name for name in missing)

    for name, parameter in second_encoder.named_parameters():
        if "lora_" in name:
            assert torch.equal(parameter, dict(first_encoder.named_parameters())[name])
        else:
            assert torch.equal(parameter, first_base[name])
            assert torch.equal(parameter, second_base_before[name])
    hidden = torch.randn(2, 4)
    first_output = first_encoder.model.model.language_model(hidden)
    second_output = second_encoder.model.model.language_model(hidden)
    assert torch.allclose(first_output, second_output)
