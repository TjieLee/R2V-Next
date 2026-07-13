from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from torch import nn

from ltx_trainer.trainer import LtxvTrainer
from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Config,
    MultiReferencePlannerStage2Strategy,
    configure_stage3_transformer_trainability,
)


def _stage3_config(**overrides) -> MultiReferencePlannerStage2Config:
    values = {
        "training_phase": "stage3",
        "train_stage1_dit_lora": True,
        "use_online_vlm": False,
        "cfg_dropout_enabled": False,
    }
    values.update(overrides)
    return MultiReferencePlannerStage2Config(**values)


def _valid_stage3_checkpoint() -> dict[str, torch.Tensor]:
    return {
        "diffusion_model.block.lora_A.default.weight": torch.ones(1),
        "diffusion_model.block.lora_B.default.weight": torch.ones(1),
        "text_encoder.model.model.language_model.block.lora_A.default.weight": torch.ones(1),
        "text_encoder.model.model.language_model.block.lora_B.default.weight": torch.ones(1),
        "training_strategy.planner_tokens.query_tokens": torch.ones(1),
        "training_strategy.visual_token_projection.weight": torch.ones(1),
        "training_strategy.visual_full_encoder.input_norm.weight": torch.ones(1),
        "embeddings_processor.video_connector.weight": torch.ones(1),
    }


def test_stage3_phase_validation_preserves_stage2_defaults() -> None:
    stage2 = MultiReferencePlannerStage2Config(use_online_vlm=False, cfg_dropout_enabled=False)
    assert stage2.training_phase == "stage2"
    assert stage2.train_stage1_dit_lora is False
    assert stage2.freeze_transformer is True

    with pytest.raises(ValueError, match="Stage 2 requires train_stage1_dit_lora=false"):
        MultiReferencePlannerStage2Config(
            training_phase="stage2",
            train_stage1_dit_lora=True,
            use_online_vlm=False,
            cfg_dropout_enabled=False,
        )
    with pytest.raises(ValueError, match="Stage 3 requires train_stage1_dit_lora=true"):
        MultiReferencePlannerStage2Config(
            training_phase="stage3",
            train_stage1_dit_lora=False,
            use_online_vlm=False,
            cfg_dropout_enabled=False,
        )


class _TinyDitWithLora(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.base(hidden) + self.lora_B(self.lora_A(hidden))


def test_stage3_only_dit_lora_is_trainable() -> None:
    transformer = _TinyDitWithLora()

    trainable_names = configure_stage3_transformer_trainability(transformer)

    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    assert all(
        parameter.requires_grad == ("lora_" in name)
        for name, parameter in transformer.named_parameters()
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "diffusion_model.",
        "text_encoder.model.model.language_model.",
        "training_strategy.planner_tokens.",
        "training_strategy.visual_token_projection.",
        "training_strategy.visual_full_encoder.",
        "embeddings_processor.video_connector.",
    ],
)
def test_stage3_initial_checkpoint_requires_every_component(prefix: str) -> None:
    strategy = MultiReferencePlannerStage2Strategy(_stage3_config())
    checkpoint = {
        key: value
        for key, value in _valid_stage3_checkpoint().items()
        if not key.startswith(prefix)
    }

    with pytest.raises(RuntimeError, match="missing required components"):
        strategy.validate_initial_checkpoint_state_dict(checkpoint)


def test_stage3_checkpoint_rejects_base_and_non_finite_weights() -> None:
    strategy = MultiReferencePlannerStage2Strategy(_stage3_config())
    checkpoint = _valid_stage3_checkpoint()
    strategy.validate_initial_checkpoint_state_dict(checkpoint)

    base_weight = dict(checkpoint)
    base_weight["diffusion_model.block.weight"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="frozen base weights"):
        strategy.validate_initial_checkpoint_state_dict(base_weight)

    non_finite = dict(checkpoint)
    non_finite["training_strategy.planner_tokens.query_tokens"] = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="non-finite"):
        strategy.validate_initial_checkpoint_state_dict(non_finite)


def test_stage2_initial_checkpoint_does_not_require_stage3_components() -> None:
    strategy = MultiReferencePlannerStage2Strategy(
        MultiReferencePlannerStage2Config(use_online_vlm=False, cfg_dropout_enabled=False)
    )
    strategy.validate_initial_checkpoint_state_dict(
        {"diffusion_model.block.lora_A.default.weight": torch.ones(1)}
    )


def test_stage3_metadata_and_loss_terms_are_unchanged() -> None:
    strategy = MultiReferencePlannerStage2Strategy(_stage3_config())
    metadata = strategy.get_checkpoint_metadata()
    total = strategy._combine_losses(
        torch.tensor([2.0]),
        torch.tensor([3.0]),
        torch.tensor([5.0]),
    )

    assert metadata["stage"] == 3
    assert metadata["training_phase"] == "stage3"
    assert metadata["train_stage1_dit_lora"] is True
    assert metadata["losses"] == ["ntp", "flow_matching", "siglip_mse"]
    assert torch.equal(total, torch.tensor([5.5]))


class _FakeAccelerator:
    def __init__(self) -> None:
        self.distributed_type = DistributedType.MULTI_GPU
        self.device = torch.device("cpu")
        self.prepared: list[nn.Module] = []

    def prepare(self, *modules):
        self.prepared.extend(modules)
        return modules[0] if len(modules) == 1 else modules


class _CheckpointableTransformer(_TinyDitWithLora):
    def __init__(self) -> None:
        super().__init__()
        self.checkpointing = False

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.checkpointing = enabled


class _FakeStrategy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(training_phase="stage3")
        self.planner = nn.Linear(4, 4)

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        return {"planner": self.planner}

    def set_trainable_modules(self, modules: dict[str, nn.Module]) -> None:
        self.planner = modules["planner"]


def test_stage3_transformer_participates_in_ddp_prepare_and_accumulate() -> None:
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._accelerator = _FakeAccelerator()
    trainer._transformer = _CheckpointableTransformer()
    trainer._embeddings_processor = SimpleNamespace(video_connector=nn.Linear(4, 4))
    trainer._text_encoder = None
    trainer._training_strategy = _FakeStrategy()
    trainer._train_transformer = True
    trainer._train_embeddings_processor = False
    trainer._train_text_encoder = False
    trainer._config = SimpleNamespace(
        model=SimpleNamespace(training_mode="lora"),
        optimization=SimpleNamespace(enable_gradient_checkpointing=True),
    )

    transformer = trainer._transformer
    trainer._prepare_models_for_training()

    assert transformer in trainer._accelerator.prepared
    assert trainer._transformer in trainer._accumulation_models
    assert trainer._transformer.training is True
    assert transformer.checkpointing is True


def test_optimizer_parameter_filter_is_unique_and_excludes_frozen() -> None:
    first = nn.Parameter(torch.ones(1), requires_grad=True)
    second = nn.Parameter(torch.ones(1), requires_grad=True)
    frozen = nn.Parameter(torch.ones(1), requires_grad=False)

    unique = LtxvTrainer._deduplicate_parameters([first, first, frozen, second, second])

    assert len(unique) == 2
    assert unique[0] is first
    assert unique[1] is second
    assert len({id(parameter) for parameter in unique}) == len(unique)
    assert all(parameter.requires_grad for parameter in unique)


class _TinyGemmaLora(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)
        self.base.requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.base(hidden) + self.lora_B(self.lora_A(hidden))


def _has_nonzero_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def test_stage3_joint_backward_reaches_all_six_trainable_groups() -> None:
    dit = _TinyDitWithLora()
    configure_stage3_transformer_trainability(dit)
    gemma = _TinyGemmaLora()
    planner = nn.Linear(4, 4)
    projection = nn.Linear(4, 4)
    visual_encoder = nn.Linear(4, 4)
    connector = nn.Linear(4, 4)
    vision_tower = nn.Linear(4, 4).requires_grad_(False).eval()
    projector = nn.Linear(4, 4).requires_grad_(False).eval()

    source = torch.randn(3, 4)
    frozen_visual = projector(vision_tower(source))
    gemma_hidden = gemma(source + frozen_visual)
    planned = planner(gemma_hidden)
    visual = connector(visual_encoder(projection(planned)))
    rendered = dit(visual)
    flow_loss = rendered.square().mean()
    siglip_loss = (planned - torch.randn_like(planned)).square().mean()
    ntp_loss = gemma_hidden.square().mean()
    (flow_loss + siglip_loss + 0.1 * ntp_loss).backward()

    for module in (dit, gemma, planner, projection, visual_encoder, connector):
        assert _has_nonzero_gradient(module)
    assert all(parameter.grad is None for parameter in dit.base.parameters())
    assert all(parameter.grad is None for parameter in gemma.base.parameters())
    assert all(parameter.grad is None for parameter in vision_tower.parameters())
    assert all(parameter.grad is None for parameter in projector.parameters())
