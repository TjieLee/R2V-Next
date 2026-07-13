from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import nn

from ltx_trainer.trainer import LtxvTrainer, normalize_peft_adapter_key
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


class _TinyPeftBackbone(nn.Module):
    def __init__(self, target_names: tuple[str, ...] = ("q_proj",)) -> None:
        super().__init__()
        for name in target_names:
            setattr(self, name, nn.Linear(4, 4, bias=False))
        self.target_names = target_names

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for name in self.target_names:
            hidden = getattr(self, name)(hidden)
        return hidden


def _tiny_peft_model(target_names: tuple[str, ...] = ("q_proj",)) -> nn.Module:
    return get_peft_model(
        _TinyPeftBackbone(target_names),
        LoraConfig(
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            target_modules=list(target_names),
        ),
    )


def _with_default_adapter_name(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    named: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = key.replace(".lora_A.", ".lora_A.default.")
        key = key.replace(".lora_B.", ".lora_B.default.")
        named[key] = value
    return named


@pytest.mark.parametrize("adapter_kind", ["lora_A", "lora_B"])
def test_stage3_strict_dit_adapter_rejects_missing_matrix(adapter_kind: str) -> None:
    source = _tiny_peft_model()
    checkpoint = get_peft_model_state_dict(source)
    removed_key = next(key for key in checkpoint if adapter_kind in key)
    checkpoint.pop(removed_key)

    with pytest.raises(RuntimeError, match="Incomplete Stage 1 DiT LoRA checkpoint"):
        LtxvTrainer._strict_load_peft_adapter_state(
            _tiny_peft_model(),
            checkpoint,
            label="Stage 1 DiT LoRA",
        )


def test_stage3_strict_dit_adapter_rejects_unknown_key_and_shape() -> None:
    checkpoint = get_peft_model_state_dict(_tiny_peft_model())
    first_key = next(iter(checkpoint))
    with_unknown = dict(checkpoint)
    with_unknown[f"unknown.{first_key}"] = checkpoint[first_key].clone()
    with pytest.raises(RuntimeError, match="unexpected"):
        LtxvTrainer._strict_load_peft_adapter_state(
            _tiny_peft_model(),
            with_unknown,
            label="Stage 1 DiT LoRA",
        )

    wrong_shape = dict(checkpoint)
    wrong_shape[first_key] = torch.zeros(checkpoint[first_key].numel() + 1)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        LtxvTrainer._strict_load_peft_adapter_state(
            _tiny_peft_model(),
            wrong_shape,
            label="Stage 1 DiT LoRA",
        )


def test_stage3_strict_gemma_adapter_rejects_missing_q_proj_and_roundtrips() -> None:
    targets = ("q_proj", "k_proj", "v_proj", "o_proj")
    source = _tiny_peft_model(targets)
    checkpoint = _with_default_adapter_name(get_peft_model_state_dict(source))
    missing_q_proj = {
        key: value
        for key, value in checkpoint.items()
        if not ("q_proj" in key and "lora_A" in key)
    }
    with pytest.raises(RuntimeError, match="Incomplete Stage 2 Gemma LoRA checkpoint"):
        LtxvTrainer._strict_load_peft_adapter_state(
            _tiny_peft_model(targets),
            missing_q_proj,
            label="Stage 2 Gemma LoRA",
        )

    q_proj_key = next(key for key in checkpoint if "q_proj" in key)
    wrong_shape = dict(checkpoint)
    wrong_shape[q_proj_key] = torch.zeros(checkpoint[q_proj_key].numel() + 1)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        LtxvTrainer._strict_load_peft_adapter_state(
            _tiny_peft_model(targets),
            wrong_shape,
            label="Stage 2 Gemma LoRA",
        )

    target = _tiny_peft_model(targets)
    count = LtxvTrainer._strict_load_peft_adapter_state(
        target,
        checkpoint,
        label="Stage 2 Gemma LoRA",
    )
    loaded = {
        normalize_peft_adapter_key(key): value
        for key, value in get_peft_model_state_dict(target).items()
    }
    expected = {
        normalize_peft_adapter_key(key): value
        for key, value in checkpoint.items()
    }
    assert count == len(expected)
    assert loaded.keys() == expected.keys()
    assert all(torch.equal(loaded[key], expected[key]) for key in expected)


def test_stage3_strict_connector_rejects_missing_real_state_key() -> None:
    source = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    checkpoint = source.state_dict()
    checkpoint.pop(next(iter(checkpoint)))

    with pytest.raises(RuntimeError, match="Incomplete video connector checkpoint"):
        LtxvTrainer._strict_load_module_state(
            nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)),
            checkpoint,
            label="video connector",
        )


def test_stage3_six_group_checkpoint_roundtrip_is_tensor_exact(tmp_path: Path) -> None:
    dit_source = _tiny_peft_model()
    gemma_source = _tiny_peft_model(("q_proj", "k_proj", "v_proj", "o_proj"))
    source_modules = {
        "planner": nn.Linear(4, 4),
        "projection": nn.Linear(4, 4),
        "visual_encoder": nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
        "connector": nn.Linear(4, 4),
    }
    checkpoint: dict[str, torch.Tensor] = {
        **{
            f"diffusion_model.{key}": value
            for key, value in get_peft_model_state_dict(dit_source).items()
        },
        **{
            f"text_encoder.model.model.language_model.{key}": value
            for key, value in _with_default_adapter_name(
                get_peft_model_state_dict(gemma_source)
            ).items()
        },
    }
    prefixes = {
        "planner": "training_strategy.planner_tokens.",
        "projection": "training_strategy.visual_token_projection.",
        "visual_encoder": "training_strategy.visual_full_encoder.",
        "connector": "embeddings_processor.video_connector.",
    }
    for name, module in source_modules.items():
        checkpoint.update({f"{prefixes[name]}{key}": value for key, value in module.state_dict().items()})
    checkpoint_path = tmp_path / "stage3_roundtrip.safetensors"
    save_file(checkpoint, checkpoint_path, metadata={"training_phase": "stage3"})
    checkpoint = load_file(checkpoint_path)

    dit_target = _tiny_peft_model()
    gemma_target = _tiny_peft_model(("q_proj", "k_proj", "v_proj", "o_proj"))
    assert LtxvTrainer._strict_load_peft_adapter_state(
        dit_target,
        {
            key.removeprefix("diffusion_model."): value
            for key, value in checkpoint.items()
            if key.startswith("diffusion_model.")
        },
        label="Stage 1 DiT LoRA",
    )
    assert LtxvTrainer._strict_load_peft_adapter_state(
        gemma_target,
        {
            key.removeprefix("text_encoder.model.model.language_model."): value
            for key, value in checkpoint.items()
            if key.startswith("text_encoder.model.model.language_model.")
        },
        label="Stage 2 Gemma LoRA",
    )

    target_modules = {
        "planner": nn.Linear(4, 4),
        "projection": nn.Linear(4, 4),
        "visual_encoder": nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
        "connector": nn.Linear(4, 4),
    }
    for name, module in target_modules.items():
        module_state = {
            key.removeprefix(prefixes[name]): value
            for key, value in checkpoint.items()
            if key.startswith(prefixes[name])
        }
        assert LtxvTrainer._strict_load_module_state(module, module_state, label=name)


class _CheckpointAccelerator:
    distributed_type = DistributedType.NO

    def __init__(self) -> None:
        self.state_dict_calls = 0

    def wait_for_everyone(self) -> None:
        return None

    def get_state_dict(self, module: nn.Module) -> dict[str, torch.Tensor]:
        self.state_dict_calls += 1
        return module.state_dict()

    def unwrap_model(self, module: nn.Module, keep_torch_compile: bool = False) -> nn.Module:
        del keep_torch_compile
        return module


def _checkpoint_test_trainer(tmp_path: Path) -> LtxvTrainer:
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(
        output_dir=str(tmp_path),
        model=SimpleNamespace(training_mode="lora"),
        checkpoints=SimpleNamespace(precision="float32", keep_last_n=1),
    )
    trainer._accelerator = _CheckpointAccelerator()
    trainer._transformer = _tiny_peft_model()
    trainer._training_strategy = SimpleNamespace(validate_checkpoint_state_dict=lambda state: None)
    trainer._global_step = 1
    trainer._checkpoint_paths = []
    trainer._last_saved_step = None
    trainer._last_saved_weights_path = None
    trainer._collect_auxiliary_checkpoint_state = lambda save_dtype: {}
    trainer._build_checkpoint_metadata = lambda: {"training_phase": "stage3"}
    return trainer


def test_final_save_is_idempotent_when_interval_matches_last_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _checkpoint_test_trainer(tmp_path)
    write_count = 0
    state_count = 0

    def fake_save_file(
        state: dict[str, torch.Tensor],
        path: Path,
        metadata: dict[str, str],
    ) -> None:
        nonlocal write_count
        assert state
        assert metadata["training_phase"] == "stage3"
        write_count += 1
        path.write_bytes(b"checkpoint")

    def fake_save_training_state(save_dir: Path) -> None:
        nonlocal state_count
        state_count += 1
        (save_dir / "training_state_step_00001.pt").write_bytes(b"state")

    monkeypatch.setattr("ltx_trainer.trainer.save_file", fake_save_file)
    trainer._save_training_state = fake_save_training_state

    interval_path = trainer._save_checkpoint()
    final_path = trainer._save_checkpoint()

    assert write_count == 1
    assert state_count == 1
    assert interval_path == final_path
    assert final_path is not None and final_path.is_file()
    assert len(trainer._checkpoint_paths) == 1
    assert (tmp_path / "checkpoints/training_state_step_00001.pt").is_file()
    assert trainer._accelerator.state_dict_calls == 1


def test_checkpoint_cleanup_keeps_four_unique_interval_steps(tmp_path: Path) -> None:
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(checkpoints=SimpleNamespace(keep_last_n=4))
    checkpoints = []
    for step in (500, 1000, 1500, 2000):
        path = tmp_path / f"lora_weights_step_{step:05d}.safetensors"
        path.write_bytes(b"checkpoint")
        checkpoints.append(path)
    trainer._checkpoint_paths = [*checkpoints, checkpoints[-1]]

    trainer._cleanup_checkpoints()

    assert trainer._checkpoint_paths == checkpoints
    assert all(path.is_file() for path in checkpoints)


def test_stage3_resume_rejects_stage2_metadata_and_missing_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "lora_weights_step_00500.safetensors"
    save_file({"weight": torch.ones(1)}, checkpoint, metadata={"training_phase": "stage2"})
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._loaded_checkpoint_path = checkpoint
    trainer._training_strategy = SimpleNamespace(config=SimpleNamespace(training_phase="stage3"))
    trainer._config = SimpleNamespace(checkpoints=SimpleNamespace(no_resume=False))

    with pytest.raises(RuntimeError, match="Stage 3 resume requires a Stage 3 checkpoint"):
        trainer._resolve_resume_state()

    save_file({"weight": torch.ones(1)}, checkpoint, metadata={"training_phase": "stage3"})
    with pytest.raises(RuntimeError, match="matching training state"):
        trainer._resolve_resume_state()
