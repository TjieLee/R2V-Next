from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from accelerate.scheduler import AcceleratedScheduler
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.trainer import LtxvTrainer, normalize_peft_adapter_key
from ltx_trainer.training_state import ConfigFingerprint, RngStates, TrainingState
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


def _training_state(
    step: int,
    *,
    optimizer_state: dict | None,
    scheduler_state: dict | None = None,
) -> TrainingState:
    return TrainingState(
        global_step=step,
        config_fingerprint=ConfigFingerprint(
            optimizer_type="adamw",
            scheduler_type="linear",
            training_mode="lora",
            lora_rank=128,
        ),
        rng_states=RngStates(torch_state=torch.random.get_rng_state()),
        lr_scheduler_state_dict=scheduler_state
        or {"last_epoch": step, "_last_lr": [1.0e-5]},
        optimizer_state_dict=optimizer_state,
    )


def _resume_test_trainer(
    checkpoint_path: Path,
    *,
    warm: bool = False,
) -> LtxvTrainer:
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._loaded_checkpoint_path = checkpoint_path
    trainer._training_strategy = SimpleNamespace(config=SimpleNamespace(training_phase="stage3"))
    trainer._config = SimpleNamespace(
        checkpoints=SimpleNamespace(
            no_resume=False,
            save_training_state="minimal" if warm else "full",
            allow_warm_resume_without_optimizer=warm,
        ),
        optimization=SimpleNamespace(
            optimizer_type="adamw",
            scheduler_type="linear",
            steps=2000,
        ),
        model=SimpleNamespace(training_mode="lora"),
        lora=SimpleNamespace(rank=128),
    )
    return trainer


def _write_resume_pair(
    directory: Path,
    *,
    filename_step: int,
    metadata_step: int | None,
    state: TrainingState,
) -> Path:
    checkpoint = directory / f"lora_weights_step_{filename_step:05d}.safetensors"
    metadata = {"training_phase": "stage3"}
    if metadata_step is not None:
        metadata["global_step"] = str(metadata_step)
    save_file({"weight": torch.ones(1)}, checkpoint, metadata=metadata)
    torch.save(
        state.to_save_dict(),
        directory / f"training_state_step_{filename_step:05d}.pt",
    )
    return checkpoint


def test_stage3_exact_resume_rejects_minimal_state_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "multiref_stage3_joint_full_tokens_planner_2048_resume.yaml"
    )
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(LtxTrainerConfig, "_validate_data_dirs_exist", lambda _self: None)
    exact_config = LtxTrainerConfig.model_validate(config_data)
    assert exact_config.checkpoints.save_training_state == "full"
    assert exact_config.checkpoints.allow_warm_resume_without_optimizer is False

    config_data["checkpoints"]["save_training_state"] = "minimal"
    config_data["checkpoints"]["allow_warm_resume_without_optimizer"] = False

    with pytest.raises(ValueError, match="Exact Stage 3 resume requires save_training_state='full'"):
        LtxTrainerConfig.model_validate(config_data)

    config_data["checkpoints"]["allow_warm_resume_without_optimizer"] = True
    config = LtxTrainerConfig.model_validate(config_data)
    assert config.checkpoints.save_training_state == "minimal"
    assert config.checkpoints.allow_warm_resume_without_optimizer is True


def test_stage3_warm_resume_logs_optimizer_reset_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint = _write_resume_pair(
        tmp_path,
        filename_step=500,
        metadata_step=500,
        state=_training_state(500, optimizer_state=None),
    )
    trainer = _resume_test_trainer(checkpoint, warm=True)

    step, state = trainer._resolve_resume_state()

    assert step == 500
    assert state is not None
    assert "Warm Stage 3 resume: optimizer moments are reset." in caplog.text


def test_stage3_no_resume_warmstart_ignores_training_state_and_starts_at_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _write_resume_pair(
        tmp_path,
        filename_step=500,
        metadata_step=500,
        state=_training_state(500, optimizer_state={"state": {}}),
    )
    trainer = _resume_test_trainer(checkpoint)
    trainer._config.checkpoints.no_resume = True

    def _forbid_training_state_load(_path: Path) -> None:
        raise AssertionError("training state must not load")

    monkeypatch.setattr(trainer, "_load_training_state", _forbid_training_state_load)

    assert trainer._resolve_resume_state() == (0, None)


@pytest.mark.parametrize(
    ("metadata_step", "state_step", "expected"),
    [
        (500, 1000, "filename=500, metadata=500, training_state=1000"),
        (1000, 500, "filename=500, metadata=1000, training_state=500"),
    ],
)
def test_stage3_resume_rejects_step_mismatch(
    tmp_path: Path,
    metadata_step: int,
    state_step: int,
    expected: str,
) -> None:
    checkpoint = _write_resume_pair(
        tmp_path,
        filename_step=500,
        metadata_step=metadata_step,
        state=_training_state(state_step, optimizer_state={}),
    )
    trainer = _resume_test_trainer(checkpoint)

    with pytest.raises(RuntimeError, match=expected):
        trainer._resolve_resume_state()


def test_stage3_resume_accepts_matching_filename_metadata_and_state(tmp_path: Path) -> None:
    checkpoint = _write_resume_pair(
        tmp_path,
        filename_step=500,
        metadata_step=500,
        state=_training_state(500, optimizer_state={}),
    )
    trainer = _resume_test_trainer(checkpoint)
    trainer._accelerator = SimpleNamespace(num_processes=8, split_batches=False)

    step, state = trainer._resolve_resume_state()

    assert step == 500
    assert state is not None and state.global_step == 500


def test_stage3_resume_rejects_scheduler_epoch_mismatch(tmp_path: Path) -> None:
    checkpoint = _write_resume_pair(
        tmp_path,
        filename_step=500,
        metadata_step=500,
        state=_training_state(
            500,
            optimizer_state={},
            scheduler_state={"last_epoch": 499, "_last_lr": [1.0e-5]},
        ),
    )
    trainer = _resume_test_trainer(checkpoint)

    with pytest.raises(RuntimeError, match="scheduler/global_step mismatch"):
        trainer._resolve_resume_state()


def test_stage3_resume_rejects_world_size_scaled_scheduler_epoch(tmp_path: Path) -> None:
    checkpoint = _write_resume_pair(
        tmp_path,
        filename_step=500,
        metadata_step=500,
        state=_training_state(
            500,
            optimizer_state={},
            scheduler_state={"last_epoch": 4000, "_last_lr": [1.0e-5]},
        ),
    )
    trainer = _resume_test_trainer(checkpoint)
    trainer._accelerator = SimpleNamespace(num_processes=8, split_batches=False)

    with pytest.raises(RuntimeError, match="expected_last_epoch=500"):
        trainer._resolve_resume_state()


def test_stage3_resume_rejects_legacy_metadata_without_global_step(tmp_path: Path) -> None:
    checkpoint = _write_resume_pair(
        tmp_path,
        filename_step=500,
        metadata_step=None,
        state=_training_state(500, optimizer_state={}),
    )
    trainer = _resume_test_trainer(checkpoint)

    with pytest.raises(RuntimeError, match="metadata is missing global_step"):
        trainer._resolve_resume_state()


def test_stage3_resume_rejects_stage2_metadata_and_missing_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "lora_weights_step_00500.safetensors"
    save_file(
        {"weight": torch.ones(1)},
        checkpoint,
        metadata={"training_phase": "stage2", "global_step": "500"},
    )
    trainer = _resume_test_trainer(checkpoint)

    with pytest.raises(RuntimeError, match="Stage 3 resume requires a Stage 3 checkpoint"):
        trainer._resolve_resume_state()

    save_file(
        {"weight": torch.ones(1)},
        checkpoint,
        metadata={"training_phase": "stage3", "global_step": "500"},
    )
    with pytest.raises(RuntimeError, match="matching training state"):
        trainer._resolve_resume_state()


def test_exact_resume_restores_adam_state_and_lr_continuity() -> None:
    source_parameter = nn.Parameter(torch.tensor([1.0]))
    source_optimizer = AdamW([source_parameter], lr=1.0e-3)
    source_scheduler = LinearLR(
        source_optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=1000,
    )
    for _ in range(500):
        source_parameter.grad = torch.ones_like(source_parameter)
        source_optimizer.step()
        source_optimizer.zero_grad(set_to_none=True)
        source_scheduler.step()

    resumed_parameter = nn.Parameter(source_parameter.detach().clone())
    resumed_optimizer = AdamW([resumed_parameter], lr=1.0e-3)
    resumed_scheduler = LinearLR(
        resumed_optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=1000,
    )
    state = _training_state(
        500,
        optimizer_state=source_optimizer.state_dict(),
        scheduler_state=source_scheduler.state_dict(),
    )
    trainer = _resume_test_trainer(Path("lora_weights_step_00500.safetensors"))
    trainer._optimizer = resumed_optimizer
    trainer._lr_scheduler = resumed_scheduler
    trainer._accelerator = SimpleNamespace(num_processes=1)

    saved_rng_state = state.rng_states.torch_state.clone()
    torch.random.set_rng_state(saved_rng_state)
    expected_random_values = torch.rand(4)
    torch.manual_seed(12345)

    assert trainer._restore_training_state(state)
    source_adam_state = source_optimizer.state[source_parameter]
    resumed_adam_state = resumed_optimizer.state[resumed_parameter]
    assert torch.equal(resumed_adam_state["exp_avg"], source_adam_state["exp_avg"])
    assert torch.equal(resumed_adam_state["exp_avg_sq"], source_adam_state["exp_avg_sq"])
    assert resumed_optimizer.param_groups[0]["lr"] == source_optimizer.param_groups[0]["lr"]
    assert resumed_optimizer.param_groups[0]["lr"] == resumed_scheduler.get_last_lr()[0]
    assert torch.equal(torch.rand(4), expected_random_values)

    source_parameter.grad = torch.ones_like(source_parameter)
    resumed_parameter.grad = torch.ones_like(resumed_parameter)
    source_optimizer.step()
    source_scheduler.step()
    resumed_optimizer.step()
    resumed_scheduler.step()

    assert torch.allclose(resumed_parameter, source_parameter)
    assert resumed_optimizer.param_groups[0]["lr"] == source_optimizer.param_groups[0]["lr"]


def test_warm_resume_restores_scheduler_lr_without_adam_state() -> None:
    source_parameter = nn.Parameter(torch.tensor([1.0]))
    source_optimizer = AdamW([source_parameter], lr=1.0e-3)
    source_scheduler = LinearLR(source_optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
    for _ in range(20):
        source_parameter.grad = torch.ones_like(source_parameter)
        source_optimizer.step()
        source_optimizer.zero_grad(set_to_none=True)
        source_scheduler.step()

    resumed_parameter = nn.Parameter(torch.tensor([1.0]))
    resumed_optimizer = AdamW([resumed_parameter], lr=1.0e-3)
    resumed_scheduler = LinearLR(resumed_optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
    trainer = _resume_test_trainer(Path("lora_weights_step_00020.safetensors"), warm=True)
    trainer._optimizer = resumed_optimizer
    trainer._lr_scheduler = resumed_scheduler
    trainer._accelerator = SimpleNamespace(num_processes=1)
    state = _training_state(
        20,
        optimizer_state=None,
        scheduler_state=source_scheduler.state_dict(),
    )

    assert trainer._restore_training_state(state)
    assert resumed_optimizer.state == {}
    assert resumed_optimizer.param_groups[0]["lr"] == source_optimizer.param_groups[0]["lr"]
    assert resumed_optimizer.param_groups[0]["lr"] == resumed_scheduler.get_last_lr()[0]


def test_setup_accelerator_disables_automatic_scheduler_stepping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_accelerator(**kwargs: object) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            num_processes=1,
            state=SimpleNamespace(dynamo_plugin=SimpleNamespace(backend="NO")),
        )

    monkeypatch.setattr("ltx_trainer.trainer.Accelerator", fake_accelerator)
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(
        acceleration=SimpleNamespace(mixed_precision_mode="bf16"),
        optimization=SimpleNamespace(batch_size=1, gradient_accumulation_steps=4),
    )

    trainer._setup_accelerator()

    assert captured_kwargs["step_scheduler_with_optimizer"] is False


@pytest.mark.parametrize(
    ("world_size", "gradient_accumulation_steps"),
    [(1, 1), (8, 1), (8, 4)],
)
def test_scheduler_steps_once_per_global_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    gradient_accumulation_steps: int,
) -> None:
    accelerator_state_calls = 0

    def fake_accelerator_state() -> SimpleNamespace:
        nonlocal accelerator_state_calls
        accelerator_state_calls += 1
        return SimpleNamespace(num_processes=world_size)

    monkeypatch.setattr("accelerate.scheduler.AcceleratorState", fake_accelerator_state)
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamW([parameter], lr=1.0e-3)
    base_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=2000)
    scheduler = AcceleratedScheduler(
        base_scheduler,
        optimizer,
        step_with_optimizer=False,
        split_batches=False,
    )

    for micro_step in range(500 * gradient_accumulation_steps):
        sync_gradients = (micro_step + 1) % gradient_accumulation_steps == 0
        if sync_gradients:
            optimizer.step()
        LtxvTrainer._step_lr_scheduler(scheduler, sync_gradients=sync_gradients)

    assert base_scheduler.last_epoch == 500
    assert scheduler.get_last_lr() == base_scheduler.get_last_lr()
    assert accelerator_state_calls == 0


def test_linear_scheduler_finishes_at_global_step_2000() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamW([parameter], lr=1.0e-3)
    base_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=2000)
    scheduler = AcceleratedScheduler(
        base_scheduler,
        optimizer,
        step_with_optimizer=False,
        split_batches=False,
    )

    for _ in range(2000):
        optimizer.step()
        LtxvTrainer._step_lr_scheduler(scheduler, sync_gradients=True)

    assert base_scheduler.last_epoch == 2000
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)


def _write_checkpoint_pair(checkpoints_dir: Path, step: int) -> tuple[Path, Path]:
    checkpoint = checkpoints_dir / f"lora_weights_step_{step:05d}.safetensors"
    state = checkpoints_dir / f"training_state_step_{step:05d}.pt"
    checkpoint.write_bytes(b"checkpoint")
    state.write_bytes(b"state")
    return checkpoint, state


def test_resume_cleanup_scans_disk_and_removes_checkpoint_state_pairs(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    checkpoint_500, state_500 = _write_checkpoint_pair(checkpoints_dir, 500)
    checkpoint_1000, _ = _write_checkpoint_pair(checkpoints_dir, 1000)
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(
        output_dir=str(tmp_path),
        checkpoints=SimpleNamespace(keep_last_n=4),
    )
    trainer._loaded_checkpoint_path = checkpoint_1000
    trainer._checkpoint_paths = []
    trainer._training_state_paths = []

    for step in (1500, 2000, 2500):
        _write_checkpoint_pair(checkpoints_dir, step)
        trainer._cleanup_checkpoints()

    expected_steps = [1000, 1500, 2000, 2500]
    assert [trainer._checkpoint_step(path) for path in trainer._checkpoint_paths] == expected_steps
    assert [trainer._checkpoint_step(path) for path in trainer._training_state_paths] == expected_steps
    assert checkpoint_1000.is_file()
    assert not checkpoint_500.exists()
    assert not state_500.exists()
    assert len(list(checkpoints_dir.glob("lora_weights_step_*.safetensors"))) == 4
    assert len(list(checkpoints_dir.glob("training_state_step_*.pt"))) == 4


def test_stage3_checkpoint_metadata_contains_global_step() -> None:
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._global_step = 500
    trainer._training_strategy = SimpleNamespace(get_checkpoint_metadata=lambda: {"training_phase": "stage3"})
    trainer._config = SimpleNamespace(text_encoder_lora=SimpleNamespace(enabled=False))

    metadata = trainer._build_checkpoint_metadata()

    assert metadata["training_phase"] == "stage3"
    assert metadata["global_step"] == "500"


def test_effective_global_batch_and_throughput_include_gradient_accumulation() -> None:
    global_batch = LtxvTrainer._effective_global_batch_size(
        batch_size=1,
        num_processes=8,
        gradient_accumulation_steps=4,
    )

    assert global_batch == 32
    assert 2.5 * global_batch == 80.0
