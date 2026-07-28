from __future__ import annotations

import ast
import copy
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2
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
    validate_and_load_phase2_bridge_state,
)


class _TinyConnector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.learnable_registers = nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16))
        self.projection = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)


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
    for command in ("start", "resume", "smoke-2gpu", "smoke-8gpu", "preflight"):
        assert command in launcher_text


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
    assert validate(config, accelerate, mode="start")["num_processes"] == 8
    with pytest.raises(RuntimeError, match="8 processes"):
        validate(
            config,
            {**accelerate, "num_processes": 7},
            mode="start",
        )
