from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import runpy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from safetensors.torch import save_file
from torch import nn

from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2
from ltx_trainer.online_data.constants import IMAGE_TASK
from ltx_trainer.online_data.online_batch_encoder import (
    OnlineBatchEncoder,
    phase2_condition_axes,
)
from ltx_trainer.trainer import LtxvTrainer
from ltx_trainer.training_state import ConfigFingerprint, RngStates, TrainingState
from ltx_trainer.training_strategies.semantic_flow import (
    DEFAULT_PHASE2_CONDITION_PROBABILITIES,
    PHASE2_CONDITION_MODES,
    SemanticFlowConfig,
    SemanticFlowStrategy,
)
from ltx_trainer.training_strategies.semantic_flow_bridge import (
    configure_phase2_bridge_trainability,
    expected_phase2_bridge_state,
    phase2_bridge_audit_rows,
    phase2_bridge_parameters,
    validate_and_load_phase2_bridge_state,
)


class _TinyConnector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.learnable_registers = nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16))
        self.projection = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)


class _FakePhase2LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 2)
        self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(4)])
        self.config = SimpleNamespace(sliding_window=4)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: dict[str, torch.Tensor],
        position_ids: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        assert output_hidden_states
        assert return_dict
        assert not use_cache
        return SimpleNamespace(hidden_states=tuple(layer(inputs_embeds) for layer in self.layers))


def _processor() -> EmbeddingsProcessor:
    return EmbeddingsProcessor(
        feature_extractor=FeatureExtractorV2(
            video_aggregate_embed=nn.Linear(8, 4),
            embedding_dim=2,
            audio_aggregate_embed=nn.Linear(8, 3),
        ),
        video_connector=_TinyConnector(),
        audio_connector=nn.Linear(3, 3),
    )


def _strategy_modules() -> dict[str, nn.Module]:
    return {
        "semantic_query": nn.Linear(2, 2, bias=False),
        "semantic_encoder": nn.Linear(2, 3, bias=False),
        "semantic_reconstruction_decoder": nn.Linear(3, 2, bias=False),
        "semantic_alignment_head": nn.Linear(3, 4, bias=False),
    }


def _runtime_audit_trainer() -> LtxvTrainer:
    trainer = object.__new__(LtxvTrainer)
    trainer._training_strategy = SemanticFlowStrategy(SemanticFlowConfig(training_phase="phase2"))
    trainer._embeddings_processor = _processor()
    configure_phase2_bridge_trainability(trainer._embeddings_processor)
    trainer._text_encoder = nn.Linear(4, 4).requires_grad_(False)
    trainer._online_vae_encoder = nn.Linear(4, 4).requires_grad_(False)
    trainer._phase2_runtime_trainability_checked = False
    return trainer


def _training_state(step: int) -> TrainingState:
    return TrainingState(
        global_step=step,
        config_fingerprint=ConfigFingerprint(
            optimizer_type="adamw",
            scheduler_type="linear",
            training_mode="full",
        ),
        rng_states=RngStates(torch_state=torch.random.get_rng_state()),
        lr_scheduler_state_dict={"last_epoch": step},
        data_state={
            "task_schedule_cursor": step,
            "microstep_in_optimizer_step": 0,
        },
    )


def _write_phase2_bundle(checkpoint_dir: Path, step: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"model_weights_step_{step:05d}.safetensors"
    save_file(
        {"weight": torch.ones(1)},
        checkpoint,
        metadata={
            "architecture": "semantic_flow_v2",
            "training_phase": "phase2",
            "global_step": str(step),
        },
    )
    training_state = checkpoint_dir / f"training_state_step_{step:05d}.pt"
    torch.save(_training_state(step).to_save_dict(), training_state)
    accelerator_state = checkpoint_dir / f"accelerator_state_step_{step:05d}"
    accelerator_state.mkdir()
    (accelerator_state / "state.bin").write_bytes(b"state")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    marker = checkpoint_dir / f"checkpoint_step_{step:05d}.ready.json"
    marker.write_text(
        json.dumps(
            {
                "global_step": step,
                "checkpoint_path": str(checkpoint.resolve()),
                "training_state_path": str(training_state.resolve()),
                "accelerator_state_path": str(accelerator_state.resolve()),
                "checkpoint_sha256": digest,
                "metadata_global_step": str(step),
                "training_phase": "phase2",
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def _gradient_audit_trainer(tmp_path: Path) -> LtxvTrainer:
    trainer = _runtime_audit_trainer()
    bridge = phase2_bridge_parameters(trainer._embeddings_processor)
    for item in bridge:
        item.parameter.grad = torch.ones_like(item.parameter)
    dit_parameter = nn.Parameter(torch.ones(1))
    trainer._optimizer = SimpleNamespace(
        param_groups=[
            {
                "name": "dit_semantic",
                "lr": 5.0e-6,
                "params": [dit_parameter],
            },
            {
                "name": "conditioning_bridge",
                "lr": 3.0e-6,
                "params": [item.parameter for item in bridge],
            },
        ]
    )
    trainer._config = SimpleNamespace(
        output_dir=str(tmp_path),
        optimization=SimpleNamespace(
            learning_rate=5.0e-6,
            bridge_learning_rate=3.0e-6,
        ),
    )

    class _Accelerator:
        device = torch.device("cpu")
        num_processes = 1
        is_main_process = True

        @staticmethod
        def reduce(value: torch.Tensor, *, reduction: str) -> torch.Tensor:
            assert reduction == "sum"
            return value

        @staticmethod
        def wait_for_everyone() -> None:
            return None

    trainer._accelerator = _Accelerator()
    trainer._global_step = 0
    trainer._phase2_smoke_audit_enabled = True
    trainer._phase2_gradient_audit_completed = False
    return trainer


def test_phase1_defaults_and_sampling_sequence_remain_unchanged() -> None:
    config = SemanticFlowConfig()
    assert config.training_phase == "phase1"
    assert config.phase2_condition_probabilities == DEFAULT_PHASE2_CONDITION_PROBABILITIES
    modes = [
        OnlineBatchEncoder._sample_condition_mode(
            object(),
            sample_key=key,
            optimizer_step=7,
            microstep=2,
            global_seed=42,
            strategy_config=config,
        )
        for key in "abcdefgh"
    ]
    assert modes == [
        "full",
        "drop_text",
        "full",
        "drop_text",
        "full",
        "full",
        "full",
        "full",
    ]
    assert SemanticFlowStrategy(config).train_embeddings_processor() is False
    assert "training_phase" not in SemanticFlowStrategy(config).get_checkpoint_metadata()


def test_phase2_probability_schema_is_exact_and_normalized() -> None:
    config = SemanticFlowConfig(training_phase="phase2")
    assert tuple(config.phase2_condition_probabilities) == PHASE2_CONDITION_MODES
    with pytest.raises(ValueError, match="exactly"):
        SemanticFlowConfig(
            training_phase="phase2",
            phase2_condition_probabilities={"til_111": 1.0},
        )
    probabilities = dict(DEFAULT_PHASE2_CONDITION_PROBABILITIES)
    probabilities["til_111"] = 0.51
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        SemanticFlowConfig(
            training_phase="phase2",
            phase2_condition_probabilities=probabilities,
        )


@pytest.mark.parametrize(
    ("mode", "axes"),
    [
        ("til_111", (True, True, True)),
        ("til_110", (True, True, False)),
        ("til_101", (True, False, True)),
        ("til_011", (False, True, True)),
        ("til_100", (True, False, False)),
        ("til_010", (False, True, False)),
        ("til_001", (False, False, True)),
        ("til_000", (False, False, False)),
    ],
)
def test_phase2_condition_axes(mode: str, axes: tuple[bool, bool, bool]) -> None:
    assert phase2_condition_axes(mode) == axes


def test_phase2_categorical_sampling_is_deterministic_and_reaches_every_mode() -> None:
    config = SemanticFlowConfig(training_phase="phase2")
    first = [
        OnlineBatchEncoder._sample_condition_mode(
            object(),
            sample_key=f"sample-{index}",
            optimizer_step=1,
            microstep=0,
            global_seed=42,
            strategy_config=config,
        )
        for index in range(200)
    ]
    second = [
        OnlineBatchEncoder._sample_condition_mode(
            object(),
            sample_key=f"sample-{index}",
            optimizer_step=1,
            microstep=0,
            global_seed=42,
            strategy_config=config,
        )
        for index in range(200)
    ]
    assert first == second
    assert set(first) == set(PHASE2_CONDITION_MODES)


def test_phase2_til_000_does_not_reuse_phase1_zero_helper() -> None:
    source = Path(__file__).parents[1] / "src" / "ltx_trainer" / "online_data" / "online_batch_encoder.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_zero_condition_tensors"
    ]
    assert len(calls) == 2
    assert all(
        not (isinstance(parent, ast.If) and "til_000" in ast.unparse(parent.test))
        for call in calls
        for parent in ast.walk(module)
        if call in ast.walk(parent)
    )


def test_phase2_bridge_allowlist_freezes_audio_and_uses_fp32_master_weights() -> None:
    processor = _processor()
    parameters = configure_phase2_bridge_trainability(processor)
    names = {item.name for item in parameters}
    assert names == {
        "feature_extractor.video_aggregate_embed.weight",
        "feature_extractor.video_aggregate_embed.bias",
        "video_connector.learnable_registers",
        "video_connector.projection.weight",
    }
    assert all(item.parameter.requires_grad for item in parameters)
    assert all(item.parameter.dtype == torch.float32 for item in parameters)
    assert all(
        not parameter.requires_grad for parameter in processor.feature_extractor.audio_aggregate_embed.parameters()
    )
    assert all(not parameter.requires_grad for parameter in processor.audio_connector.parameters())
    rows = phase2_bridge_audit_rows(processor)
    assert sum(row["numel"] for row in rows if row["owner_group"] == "bridge") == sum(
        item.parameter.numel() for item in parameters
    )


def test_phase2_feature_extractor_runs_in_training_graph_from_detached_hidden_states() -> None:
    processor = _processor()
    configure_phase2_bridge_trainability(processor)
    strategy = SemanticFlowStrategy(SemanticFlowConfig(training_phase="phase2"))
    strategy._embeddings_processor = processor
    hidden_states = tuple(torch.randn(1, 3, 2).detach() for _ in range(4))
    conditions = strategy.prepare_conditions(
        {},
        {
            "frozen_vlm_hidden_states": hidden_states,
            "prompt_attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
    )
    assert "frozen_vlm_hidden_states" not in conditions
    assert conditions["video_prompt_embeds"].shape == (1, 3, 4)
    conditions["video_prompt_embeds"].square().mean().backward()
    video_projection = processor.feature_extractor.video_aggregate_embed
    assert video_projection.weight.grad is not None
    assert torch.isfinite(video_projection.weight.grad).all()
    assert torch.count_nonzero(video_projection.weight.grad)
    assert processor.feature_extractor.audio_aggregate_embed.weight.grad is None
    assert all(value.grad_fn is None and not value.requires_grad for value in hidden_states)


def test_phase2_prefix_encoding_preserves_bridge_trainability_and_projection_gradient() -> None:
    processor = _processor()
    configure_phase2_bridge_trainability(processor)
    language_model = _FakePhase2LanguageModel().requires_grad_(False)
    encoder = OnlineBatchEncoder.__new__(OnlineBatchEncoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.embeddings_processor = processor
    encoder.last_dtype_diagnostics = {}
    encoder.last_hidden_state_diagnostics = {}
    encoder._phase2_hidden_state_shape_logged = False
    encoder._frozen_encode_autocast = nullcontext  # type: ignore[method-assign]
    encoder._get_language_model = lambda: language_model  # type: ignore[method-assign]
    encoder._process_multimodal_prefix = (  # type: ignore[method-assign]
        lambda **_kwargs: (
            {
                "input_ids": torch.tensor([[1, 2, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
            },
            torch.zeros(3, dtype=torch.bool),
            torch.zeros(3, dtype=torch.bool),
        )
    )

    conditions, teacher_prefix = encoder._encode_prefix(
        caption="edit the image",
        reference_images=[],
        task=IMAGE_TASK,
        sample_key="phase2-prefix",
        defer_feature_extractor=True,
    )

    video_projection = processor.feature_extractor.video_aggregate_embed
    assert all(parameter.requires_grad for parameter in video_projection.parameters())
    assert all(parameter.requires_grad for parameter in processor.video_connector.parameters())
    assert all(not parameter.requires_grad for parameter in processor.audio_connector.parameters())
    assert all(not value.requires_grad for value in conditions["frozen_vlm_hidden_states"])
    assert all(not torch.is_inference(value) for value in conditions["frozen_vlm_hidden_states"])
    assert teacher_prefix["prefix_inputs_embeds"].requires_grad is False

    strategy = SemanticFlowStrategy(SemanticFlowConfig(training_phase="phase2"))
    strategy._embeddings_processor = processor
    prepared = strategy.prepare_conditions({}, conditions)
    prepared["video_prompt_embeds"].square().mean().backward()
    assert video_projection.weight.grad is not None
    assert torch.isfinite(video_projection.weight.grad).all()
    assert torch.count_nonzero(video_projection.weight.grad)


def test_phase2_frozen_processor_modules_remain_in_eval_mode() -> None:
    processor = _processor()
    configure_phase2_bridge_trainability(processor)
    strategy = SemanticFlowStrategy(SemanticFlowConfig(training_phase="phase2"))
    strategy._embeddings_processor = processor
    processor.train()
    strategy.enforce_frozen_module_eval()
    assert processor.feature_extractor.training is False
    assert processor.feature_extractor.video_aggregate_embed.training is True
    assert processor.video_connector.training is True
    assert processor.audio_connector.training is False


def test_phase2_hidden_state_selection_enforces_pretrained_layer_contract() -> None:
    feature = _processor().feature_extractor
    hidden_states = tuple(torch.randn(1, 3, 2) for _ in range(4))
    selected, indices = OnlineBatchEncoder._select_feature_hidden_states(
        hidden_states,
        feature,
    )
    assert selected == hidden_states
    assert indices == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="hidden-state count"):
        OnlineBatchEncoder._select_feature_hidden_states(hidden_states[:-1], feature)


def test_phase2_bridge_checkpoint_roundtrip_is_strict() -> None:
    source = _processor()
    target = _processor()
    configure_phase2_bridge_trainability(source)
    configure_phase2_bridge_trainability(target)
    state = {key: parameter.detach().clone() + 0.25 for key, parameter in expected_phase2_bridge_state(source).items()}
    assert validate_and_load_phase2_bridge_state(target, state) == len(state)
    for key, parameter in expected_phase2_bridge_state(target).items():
        torch.testing.assert_close(parameter, state[key])
    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_and_load_phase2_bridge_state(target, missing)


def test_phase2_optimizer_has_named_disjoint_groups_and_independent_lrs() -> None:
    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(
        optimization=SimpleNamespace(
            learning_rate=5.0e-6,
            bridge_learning_rate=3.0e-6,
            optimizer_type="adamw",
            scheduler_type="linear",
            scheduler_params={},
            steps=10,
        )
    )
    trainer._transformer = nn.Linear(4, 4)
    strategy = SemanticFlowStrategy(SemanticFlowConfig(training_phase="phase2"))
    strategy.set_trainable_modules(_strategy_modules())
    trainer._training_strategy = strategy
    processor = _processor()
    configure_phase2_bridge_trainability(processor)
    trainer._embeddings_processor_trainable_modules = {
        "feature_extractor.video_aggregate_embed": (processor.feature_extractor.video_aggregate_embed),
        "video_connector": processor.video_connector,
    }
    trainer._trainable_params = trainer._deduplicate_parameters(
        [
            *trainer._transformer.parameters(),
            *[parameter for module in strategy.get_trainable_modules().values() for parameter in module.parameters()],
            *[
                parameter
                for module in trainer._embeddings_processor_trainable_modules.values()
                for parameter in module.parameters()
                if parameter.requires_grad
            ],
        ]
    )

    class _Accelerator:
        @staticmethod
        def prepare(*values: object) -> tuple[object, ...]:
            return values

    trainer._accelerator = _Accelerator()
    trainer._init_optimizer()
    assert [group["name"] for group in trainer._optimizer.param_groups] == [
        "dit_semantic",
        "conditioning_bridge",
    ]
    assert [group["lr"] for group in trainer._optimizer.param_groups] == [
        5.0e-6,
        3.0e-6,
    ]
    first_ids = {id(parameter) for parameter in trainer._optimizer.param_groups[0]["params"]}
    second_ids = {id(parameter) for parameter in trainer._optimizer.param_groups[1]["params"]}
    assert not first_ids & second_ids


def test_phase2_trainability_rejects_frozen_dit_or_bridge_allowlist_parameter() -> None:
    trainer = object.__new__(LtxvTrainer)
    trainer._training_strategy = SemanticFlowStrategy(SemanticFlowConfig(training_phase="phase2"))
    trainer._transformer = nn.Linear(4, 4)
    trainer._text_encoder = nn.Linear(4, 4)
    trainer._text_encoder.requires_grad_(False)
    trainer._online_vae_encoder = nn.Linear(4, 4)
    trainer._online_vae_encoder.requires_grad_(False)
    trainer._embeddings_processor = _processor()
    configure_phase2_bridge_trainability(trainer._embeddings_processor)
    trainer._embeddings_processor_trainable_modules = {
        "feature_extractor.video_aggregate_embed": (
            trainer._embeddings_processor.feature_extractor.video_aggregate_embed
        ),
        "video_connector": trainer._embeddings_processor.video_connector,
    }
    strategy_modules = _strategy_modules()

    trainer._transformer.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="full DiT"):
        trainer._validate_phase2_trainability(strategy_modules)

    trainer._transformer.weight.requires_grad_(True)
    trainer._embeddings_processor.video_connector.learnable_registers.requires_grad_(False)
    with pytest.raises(RuntimeError, match="allowlist contains frozen"):
        trainer._validate_phase2_trainability(strategy_modules)


def test_phase2_post_encoding_trainability_audit_runs_once_and_fails_closed() -> None:
    trainer = _runtime_audit_trainer()

    trainer._validate_phase2_runtime_trainability_once()
    assert trainer._phase2_runtime_trainability_checked is True
    trainer._validate_phase2_runtime_trainability_once()

    failing = _runtime_audit_trainer()
    failing._embeddings_processor.feature_extractor.video_aggregate_embed.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="frozen after online encoding"):
        failing._validate_phase2_runtime_trainability_once()
    assert failing._phase2_runtime_trainability_checked is False


@pytest.mark.parametrize(
    ("component", "message"),
    [
        ("text_encoder", "Gemma/SigLIP/projector"),
        ("vae_encoder", "VAE encoder"),
        ("audio_connector", "audio connector"),
    ],
)
def test_phase2_post_encoding_audit_rejects_newly_trainable_frozen_modules(
    component: str,
    message: str,
) -> None:
    trainer = _runtime_audit_trainer()
    if component == "text_encoder":
        trainer._text_encoder.weight.requires_grad_(True)
    elif component == "vae_encoder":
        trainer._online_vae_encoder.weight.requires_grad_(True)
    else:
        trainer._embeddings_processor.audio_connector.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match=message):
        trainer._validate_phase2_runtime_trainability_once()


def test_phase2_runtime_audit_occurs_after_online_encoding_before_forward() -> None:
    source = inspect.getsource(LtxvTrainer.train)
    encoded = source.index("batch = self._prepare_online_batch_with_retry(batch)")
    audited = source.index("self._validate_phase2_runtime_trainability_once()")
    forwarded = source.index("output = self._training_step(batch)")
    assert encoded < audited < forwarded


def test_phase2_smoke_gradient_audit_records_every_bridge_parameter(
    tmp_path: Path,
) -> None:
    trainer = _gradient_audit_trainer(tmp_path)
    trainer._run_phase2_gradient_audit_once()
    report = json.loads(
        (tmp_path / "phase2_gradient_audit.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert set(report["parameters"]) == {
        item.name for item in phase2_bridge_parameters(trainer._embeddings_processor)
    }
    for record in report["parameters"].values():
        assert record["requires_grad"] is True
        assert record["grad_exists"] is True
        assert record["grad_finite"] is True
        assert record["grad_nonzero"] is True
        assert record["grad_norm"] > 0
        assert record["parameter_norm"] >= 0
    assert report["frozen_module_gradients"] == {
        "audio_connector": False,
        "text_encoder": False,
        "vae_encoder": False,
    }
    assert [group["name"] for group in report["optimizer_groups"]] == [
        "dit_semantic",
        "conditioning_bridge",
    ]
    assert trainer._phase2_gradient_audit_completed is True


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing_projection", "missing, frozen, or non-finite"),
        ("zero_register", "learnable_registers has zero gradient"),
        ("nan_projection", "missing, frozen, or non-finite"),
        ("inf_projection", "missing, frozen, or non-finite"),
        ("frozen_text_gradient", "frozen modules received gradients"),
    ],
)
def test_phase2_smoke_gradient_audit_fails_closed(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    trainer = _gradient_audit_trainer(tmp_path)
    projection = trainer._embeddings_processor.feature_extractor.video_aggregate_embed
    registers = trainer._embeddings_processor.video_connector.learnable_registers
    if failure == "missing_projection":
        projection.weight.grad = None
    elif failure == "zero_register":
        registers.grad = torch.zeros_like(registers)
    elif failure == "nan_projection":
        projection.weight.grad.fill_(float("nan"))
    elif failure == "inf_projection":
        projection.weight.grad.fill_(float("inf"))
    else:
        trainer._text_encoder.weight.grad = torch.ones_like(
            trainer._text_encoder.weight
        )
    with pytest.raises(RuntimeError, match=message):
        trainer._run_phase2_gradient_audit_once()
    report = json.loads(
        (tmp_path / "phase2_gradient_audit.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is False
    assert trainer._phase2_gradient_audit_completed is False


def test_phase2_smoke_gradient_audit_runs_between_backward_and_optimizer_step() -> None:
    source = inspect.getsource(LtxvTrainer.train)
    backward = source.index("self._accelerator.backward(output.loss.mean())")
    audit = source.index("self._run_phase2_gradient_audit_once()")
    optimizer_step = source.index("self._optimizer.step()")
    zero_grad = source.index("self._optimizer.zero_grad()")
    assert backward < audit < optimizer_step < zero_grad


def test_phase2_checkpoint_metadata_is_additive_only_for_phase2() -> None:
    phase1 = SemanticFlowStrategy(SemanticFlowConfig()).get_checkpoint_metadata()
    assert "training_phase" not in phase1
    phase2_strategy = SemanticFlowStrategy(SemanticFlowConfig(training_phase="phase2"))
    phase2_strategy.phase2_parent_checkpoint_sha256 = "abc123"
    phase2 = phase2_strategy.get_checkpoint_metadata()
    assert phase2["training_phase"] == "phase2"
    assert phase2["parent_checkpoint_step"] == 14000
    assert phase2["parent_checkpoint_sha256"] == "abc123"
    assert phase2["condition_factorization"] == "T_I_L_8way_v1"


def test_phase2_exact_resume_uses_matching_distributed_accelerate_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_weights_step_00007.safetensors"
    checkpoint.touch()
    state_dir = tmp_path / "accelerator_state_step_00007"
    state_dir.mkdir()
    loaded: list[str] = []

    class _Accelerator:
        @staticmethod
        def load_state(path: str) -> None:
            loaded.append(path)

    trainer = object.__new__(LtxvTrainer)
    trainer._loaded_checkpoint_path = checkpoint
    trainer._accelerator = _Accelerator()
    trainer._lr_scheduler = SimpleNamespace(last_epoch=7)
    trainer._optimizer = SimpleNamespace(
        param_groups=[
            {"name": "dit_semantic"},
            {"name": "conditioning_bridge"},
        ]
    )
    training_state = TrainingState(
        global_step=7,
        config_fingerprint=ConfigFingerprint(
            optimizer_type="adamw",
            scheduler_type="linear",
            training_mode="full",
        ),
        rng_states=RngStates(torch_state=torch.random.get_rng_state()),
    )
    trainer._restore_phase2_accelerator_state(training_state)
    assert loaded == [str(state_dir)]
    assert trainer._phase2_accelerator_state_restored is True


def test_phase2_required_training_state_fails_closed_and_preserves_cause(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_weights_step_00001.safetensors"
    checkpoint.touch()
    with pytest.raises(RuntimeError, match="training state is missing"):
        LtxvTrainer._load_training_state(checkpoint, required=True)
    assert LtxvTrainer._load_training_state(checkpoint) is None

    state_path = tmp_path / "training_state_step_00001.pt"
    state_path.write_bytes(b"not a torch state")
    with pytest.raises(RuntimeError, match="Failed to load required") as exc_info:
        LtxvTrainer._load_training_state(checkpoint, required=True)
    assert exc_info.value.__cause__ is not None
    assert LtxvTrainer._load_training_state(checkpoint) is None


@pytest.mark.parametrize("corrupt", [False, True])
def test_phase2_resolve_resume_never_falls_back_to_step_zero(
    tmp_path: Path,
    corrupt: bool,
) -> None:
    checkpoint = tmp_path / "model_weights_step_00001.safetensors"
    save_file(
        {"weight": torch.ones(1)},
        checkpoint,
        metadata={"training_phase": "phase2", "global_step": "1"},
    )
    if corrupt:
        (tmp_path / "training_state_step_00001.pt").write_bytes(b"corrupt")
    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(
        checkpoints=SimpleNamespace(no_resume=False),
        optimization=SimpleNamespace(
            optimizer_type="adamw",
            scheduler_type="linear",
            steps=2,
        ),
        model=SimpleNamespace(training_mode="full"),
        lora=None,
    )
    trainer._training_strategy = SemanticFlowStrategy(
        SemanticFlowConfig(training_phase="phase2")
    )
    trainer._loaded_checkpoint_path = checkpoint
    with pytest.raises(RuntimeError, match="training state"):
        trainer._resolve_resume_state()


def test_phase1_resolve_resume_keeps_missing_state_compatibility(
    tmp_path: Path,
) -> None:
    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(
        checkpoints=SimpleNamespace(no_resume=False),
    )
    trainer._training_strategy = SemanticFlowStrategy(SemanticFlowConfig())
    trainer._loaded_checkpoint_path = (
        tmp_path / "model_weights_step_00001.safetensors"
    )
    assert trainer._resolve_resume_state() == (0, None)


def test_phase2_resume_state_validates_step_scheduler_sampler_and_accelerate(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_weights_step_00007.safetensors"
    checkpoint.touch()
    (tmp_path / "accelerator_state_step_00007").mkdir()
    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(
        optimization=SimpleNamespace(steps=10),
    )
    metadata = {"training_phase": "phase2", "global_step": "7"}
    trainer._validate_phase2_resume_state(
        checkpoint_path=checkpoint,
        metadata=metadata,
        state=_training_state(7),
    )

    with pytest.raises(RuntimeError, match="step mismatch"):
        trainer._validate_phase2_resume_state(
            checkpoint_path=checkpoint,
            metadata={**metadata, "global_step": "6"},
            state=_training_state(7),
        )
    bad_scheduler = _training_state(7)
    bad_scheduler.lr_scheduler_state_dict = {"last_epoch": 6}
    with pytest.raises(RuntimeError, match="scheduler/global_step"):
        trainer._validate_phase2_resume_state(
            checkpoint_path=checkpoint,
            metadata=metadata,
            state=bad_scheduler,
        )
    missing_sampler = _training_state(7)
    missing_sampler.data_state = None
    with pytest.raises(RuntimeError, match="sampler state"):
        trainer._validate_phase2_resume_state(
            checkpoint_path=checkpoint,
            metadata=metadata,
            state=missing_sampler,
        )
    (tmp_path / "accelerator_state_step_00007").rmdir()
    with pytest.raises(RuntimeError, match="Accelerate state"):
        trainer._validate_phase2_resume_state(
            checkpoint_path=checkpoint,
            metadata=metadata,
            state=_training_state(7),
        )


def test_phase2_resume_bundle_is_complete_and_latest_selection_fails_on_partial(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "check_semantic_flow_phase2_runtime.py"
    runtime = runpy.run_path(str(script))
    validate = runtime["validate_phase2_resume_bundle"]
    latest = runtime["find_latest_phase2_resume_checkpoint"]
    checkpoint_dir = tmp_path / "checkpoints"
    first = _write_phase2_bundle(checkpoint_dir, 1)
    assert validate(first)["global_step"] == 1
    assert latest(checkpoint_dir) == first.resolve()

    partial = checkpoint_dir / "model_weights_step_00002.safetensors"
    save_file(
        {"weight": torch.ones(1)},
        partial,
        metadata={"training_phase": "phase2", "global_step": "2"},
    )
    with pytest.raises(RuntimeError, match="ready marker is missing"):
        latest(checkpoint_dir)


@pytest.mark.parametrize(
    "broken_artifact",
    ["marker", "checkpoint", "sha", "training_state", "accelerator_state"],
)
def test_phase2_resume_bundle_rejects_each_missing_or_corrupt_component(
    tmp_path: Path,
    broken_artifact: str,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "check_semantic_flow_phase2_runtime.py"
    validate = runpy.run_path(str(script))["validate_phase2_resume_bundle"]
    checkpoint = _write_phase2_bundle(tmp_path, 1)
    marker = tmp_path / "checkpoint_step_00001.ready.json"
    if broken_artifact == "marker":
        marker.unlink()
    elif broken_artifact == "checkpoint":
        checkpoint.unlink()
    elif broken_artifact == "sha":
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["checkpoint_sha256"] = "0" * 64
        marker.write_text(json.dumps(payload), encoding="utf-8")
    elif broken_artifact == "training_state":
        (tmp_path / "training_state_step_00001.pt").unlink()
    else:
        (tmp_path / "accelerator_state_step_00001" / "state.bin").unlink()
        (tmp_path / "accelerator_state_step_00001").rmdir()
    with pytest.raises(RuntimeError):
        validate(checkpoint)


@pytest.mark.parametrize(
    "mismatch",
    ["marker_step", "metadata_step", "training_state_step"],
)
def test_phase2_resume_bundle_rejects_step_mismatches(
    tmp_path: Path,
    mismatch: str,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "check_semantic_flow_phase2_runtime.py"
    validate = runpy.run_path(str(script))["validate_phase2_resume_bundle"]
    checkpoint = _write_phase2_bundle(tmp_path, 1)
    marker = tmp_path / "checkpoint_step_00001.ready.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if mismatch == "marker_step":
        payload["global_step"] = 2
    elif mismatch == "metadata_step":
        save_file(
            {"weight": torch.ones(1)},
            checkpoint,
            metadata={"training_phase": "phase2", "global_step": "2"},
        )
        payload["checkpoint_sha256"] = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
    else:
        torch.save(
            _training_state(2).to_save_dict(),
            tmp_path / "training_state_step_00001.pt",
        )
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="step mismatch"):
        validate(checkpoint)


def test_phase2_smoke_result_requires_gradient_resume_and_inference_evidence(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "check_semantic_flow_phase2_runtime.py"
    validate = runpy.run_path(str(script))["validate_phase2_smoke_output"]
    first = _write_phase2_bundle(tmp_path / "checkpoints", 1)
    final = _write_phase2_bundle(tmp_path / "checkpoints", 2)
    del first
    (tmp_path / "phase2_gradient_audit.json").write_text(
        json.dumps(
            {
                "passed": True,
                "world_size": 8,
                "parameters": {
                    "feature_extractor.video_aggregate_embed.weight": {
                        "requires_grad": True,
                        "grad_exists": True,
                        "grad_finite": True,
                        "grad_nonzero": True,
                    },
                    "video_connector.learnable_registers": {
                        "requires_grad": True,
                        "grad_exists": True,
                        "grad_finite": True,
                        "grad_nonzero": True,
                    },
                },
                "frozen_module_gradients": {
                    "text_encoder": False,
                    "vae_encoder": False,
                    "audio_connector": False,
                },
                "optimizer_groups": [
                    {"name": "dit_semantic", "learning_rate": 5.0e-6},
                    {"name": "conditioning_bridge", "learning_rate": 3.0e-6},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "phase2_resume_runtime_audit.json").write_text(
        json.dumps(
            {
                "initial_step": 1,
                "scheduler_last_epoch": 1,
                "sampler_task_schedule_cursor": 1,
                "sampler_microstep_in_optimizer_step": 0,
                "accelerator_state_restored": True,
                "scheduler_restored": True,
                "sampler_restored": True,
                "optimizer_groups_restored": True,
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    for mode in ("positive_ref", "latent_ref"):
        summary = tmp_path / "inference" / mode / "run_summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(
            json.dumps(
                {
                    "guidance_mode": mode,
                    "success_count": 1,
                    "failure_count": 0,
                    "checkpoint": str(final.resolve()),
                }
            ),
            encoding="utf-8",
        )
    report = validate(
        tmp_path,
        expected_processes=8,
        expected_final_step=2,
        require_exact_resume=True,
        require_inference=True,
    )
    assert report["passed"] is True
    assert report["first_bundle"]["global_step"] == 1
    assert report["final_bundle"]["global_step"] == 2
    assert set(report["inference"]) == {"positive_ref", "latent_ref"}


def test_phase2_distributed_state_save_is_additive_to_minimal_state(
    tmp_path: Path,
) -> None:
    saved: list[tuple[str, bool]] = []

    class _Accelerator:
        @staticmethod
        def save_state(*, output_dir: str, safe_serialization: bool) -> None:
            Path(output_dir).mkdir(parents=True)
            saved.append((output_dir, safe_serialization))

        @staticmethod
        def wait_for_everyone() -> None:
            return None

    trainer = object.__new__(LtxvTrainer)
    trainer._training_strategy = SemanticFlowStrategy(SemanticFlowConfig(training_phase="phase2"))
    trainer._config = SimpleNamespace(checkpoints=SimpleNamespace(save_training_state="minimal"))
    trainer._accelerator = _Accelerator()
    trainer._global_step = 7
    trainer._last_phase2_accelerator_state_path = None
    state_path = trainer._save_phase2_accelerator_state(tmp_path)
    assert state_path == tmp_path / "accelerator_state_step_00007"
    assert saved == [(str(state_path), True)]


def test_phase2_production_config_and_launcher_are_isolated_from_phase1() -> None:
    trainer_root = Path(__file__).parents[1]
    phase1_path = trainer_root / "configs" / "semantic_flow_multitask_480p121.yaml"
    phase2_path = trainer_root / "configs" / "semantic_flow_multitask_480p121_phase2.yaml"
    phase1 = yaml.safe_load(phase1_path.read_text(encoding="utf-8"))
    phase2 = yaml.safe_load(phase2_path.read_text(encoding="utf-8"))
    assert "training_phase" not in phase1["training_strategy"]
    assert phase2["training_strategy"]["training_phase"] == "phase2"
    assert phase2["optimization"]["learning_rate"] == 5.0e-6
    assert phase2["optimization"]["bridge_learning_rate"] == 3.0e-6
    assert phase2["output_dir"] != phase1["output_dir"]
    assert phase2["checkpoints"] == {
        "interval": 1000,
        "keep_last_n": 3,
        "precision": "bfloat16",
        "no_resume": True,
        "save_training_state": "minimal",
    }
    phase1_launcher = trainer_root / "scripts" / "run_semantic_flow_opens2v.sh"
    phase2_launcher = trainer_root / "scripts" / "run_semantic_flow_phase2_8gpu.sh"
    assert phase1_launcher.is_file()
    launcher_text = phase2_launcher.read_text(encoding="utf-8")
    assert "set -euo pipefail" not in launcher_text
    assert "0,1,2,3,4,5,6,7" in launcher_text
    assert "--guidance-mode positive_ref" in launcher_text
    assert "--ref-guidance-scale 0" in launcher_text
    assert "--guidance-mode latent_ref" in launcher_text
    assert "--ref-guidance-scale 1" in launcher_text
    assert 'PYTHON="${PYTHON:-/mnt/workspace/litengjie/R2V-Next/.venv/bin/python}"' in launcher_text
    assert 'ACCELERATE="${ACCELERATE:-/mnt/workspace/litengjie/R2V-Next/.venv/bin/accelerate}"' in launcher_text
    assert '"$PYTHON" "$CHECKER" --find-latest-resume-checkpoint' in launcher_text
    assert '"$ACCELERATE" launch' in launcher_text
    assert "--phase2-smoke-audit" in launcher_text
    assert "semantic_flow_phase2_smoke_8gpu_stage_a.yaml" in launcher_text
    assert "semantic_flow_phase2_smoke_8gpu_stage_b.yaml" in launcher_text
    assert "model_weights_step_00001.safetensors" in launcher_text
    assert "model_weights_step_00002.safetensors" in launcher_text
    assert (
        launcher_text.count(
            'launch_train "$STAGE_A_CONFIG" "$TRAIN_ACCELERATE_CONFIG" true'
        )
        == 1
    )
    assert (
        launcher_text.count(
            'launch_train "$STAGE_B_CONFIG" "$TRAIN_ACCELERATE_CONFIG" false'
        )
        == 1
    )
    assert '"$STAGE_A_CHECKPOINT" \\\n      false \\\n      2 \\' in launcher_text
    assert "--require-exact-resume" in launcher_text
    assert "--require-inference" in launcher_text
    assert "phase2_smoke_8gpu_result.json" in launcher_text
    for command in ("start", "resume", "smoke-2gpu", "smoke-8gpu", "preflight"):
        assert command in launcher_text

    train_cli = (trainer_root / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "--phase2-smoke-audit" in train_cli
    assert "trainer.enable_phase2_smoke_audit()" in train_cli


def test_phase2_runtime_contract_accepts_production_shape_and_rejects_wrong_world_size() -> None:
    script = Path(__file__).parents[1] / "scripts" / "check_semantic_flow_phase2_runtime.py"
    validate = runpy.run_path(str(script))["validate_phase2_config_contract"]
    config = {
        "model": {"training_mode": "full"},
        "text_encoder_lora": {"enabled": False},
        "training_strategy": {
            "name": "semantic_flow",
            "training_phase": "phase2",
            "phase2_condition_probabilities": copy.deepcopy(DEFAULT_PHASE2_CONDITION_PROBABILITIES),
        },
        "optimization": {
            "learning_rate": 5.0e-6,
            "bridge_learning_rate": 3.0e-6,
        },
        "checkpoints": {"no_resume": True},
    }
    accelerate = {
        "num_processes": 8,
        "distributed_type": "FSDP",
        "mixed_precision": "bf16",
        "fsdp_config": {
            "fsdp_version": 1,
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
        },
    }
    contract = validate(config, accelerate, mode="start")
    assert contract["num_processes"] == 8
    assert contract["dit_semantic_learning_rate"] == 5.0e-6
    assert contract["conditioning_bridge_learning_rate"] == 3.0e-6

    unified = copy.deepcopy(config)
    unified["optimization"]["bridge_learning_rate"] = 5.0e-6
    assert (
        validate(
            unified,
            accelerate,
            mode="start",
        )["conditioning_bridge_learning_rate"]
        == 5.0e-6
    )

    implicit = copy.deepcopy(config)
    implicit["optimization"]["bridge_learning_rate"] = None
    assert (
        validate(
            implicit,
            accelerate,
            mode="start",
        )["conditioning_bridge_learning_rate"]
        == 5.0e-6
    )

    invalid_bridge = copy.deepcopy(config)
    invalid_bridge["optimization"]["bridge_learning_rate"] = 1.0e-5
    with pytest.raises(RuntimeError, match="3e-6 or 5e-6"):
        validate(invalid_bridge, accelerate, mode="start")

    invalid_dit = copy.deepcopy(config)
    invalid_dit["optimization"]["learning_rate"] = 1.0e-5
    with pytest.raises(RuntimeError, match="DiT/semantic learning rate"):
        validate(invalid_dit, accelerate, mode="start")

    with pytest.raises(RuntimeError, match="8 processes"):
        validate(
            config,
            {**accelerate, "num_processes": 7},
            mode="start",
        )


@pytest.mark.parametrize(
    ("artifact_name", "is_directory"),
    [
        ("checkpoint_step_00001.ready.json", False),
        ("model_weights_step_00001.safetensors", False),
        ("training_state_step_00001.pt", False),
        ("accelerator_state_step_00001", True),
    ],
)
def test_phase2_start_refuses_existing_training_artifacts_but_resume_allows_them(
    tmp_path: Path,
    artifact_name: str,
    is_directory: bool,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "check_semantic_flow_phase2_runtime.py"
    guard = runpy.run_path(str(script))["assert_phase2_start_output_is_empty"]
    output_dir = tmp_path / "phase2"
    guard(output_dir, mode="start")

    artifact = output_dir / "checkpoints" / artifact_name
    if is_directory:
        artifact.mkdir(parents=True)
    else:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()

    with pytest.raises(RuntimeError, match="Use .* resume"):
        guard(output_dir, mode="start")
    guard(output_dir, mode="resume")
