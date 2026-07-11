from types import SimpleNamespace

import torch
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model
from torch import nn

from ltx_trainer.trainer import LtxvTrainer
from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Config,
    MultiReferencePlannerStage2Strategy,
)


class _TrackingLinear(nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4)
        self.to_devices = []

    def to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        self.to_devices.append(torch.device(device) if device is not None else None)
        return super().to(*args, **kwargs)


class _FakeAccelerator:
    def __init__(self, distributed_type=DistributedType.NO) -> None:
        self.distributed_type = distributed_type
        self.device = torch.device("cpu")
        self.prepared = []

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
