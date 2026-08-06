import logging
from dataclasses import replace
from enum import Enum

import torch
import torch.utils.checkpoint

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.model_protocol import LTXModelProtocol
from ltx_core.model.transformer.adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from ltx_core.model.transformer.attention import attention_label
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import (
    DEFAULT_TRANSFORMER_OPS,
    BasicAVTransformerBlock,
    TransformerConfig,
    TransformerOpsConfig,
)
from ltx_core.model.transformer.transformer_args import (
    BlockPerturbationsProcessor,
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ltx_core.utils import to_denoised

logger = logging.getLogger(__name__)


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        # Log the attention backends this transformer is built with. Reading the resolved
        # ``label`` off the ops reports whatever was selected -- AUTOMATIC, an explicit pin
        # (PYTORCH/XFORMERS/FA3/FA4/SDPA_*), or a directly supplied callable -- so this is the
        # single source of truth for which kernel a build uses. Fires once per build.
        logger.info(
            "Building transformer with attention backends -- self: %s, masked: %s",
            attention_label(ops.attention_ops.attention_function),
            attention_label(ops.attention_ops.masked_attention_function),
        )
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            ops=ops,
            apply_gated_attention=apply_gated_attention,
        )
        # Hook for per-block input prep. Compile transforms in `compiling.py`
        # wrap (not replace) this with a processor that also marks the seq dim
        # dynamic, so any caller customisation here is preserved as the inner.
        self.block_input_processor = BlockPerturbationsProcessor()
        self.semantic_token_type_embedding: torch.nn.Embedding | None = None
        self.semantic_entity_embedding: torch.nn.Embedding | None = None
        self.semantic_position_adapter: torch.nn.Module | None = None
        self.semantic_norm_out: torch.nn.RMSNorm | None = None
        self.semantic_proj_out: torch.nn.Linear | None = None
        self.semantic_token_type_id: int | None = None
        self.reference_token_type_id: int | None = None
        self.semantic_repae_reference_type_embedding: torch.nn.Embedding | None = None
        self.semantic_repae_reference_slot_embedding: torch.nn.Embedding | None = None
        self.semantic_repae_semantic_type_embedding: torch.nn.Embedding | None = None
        self._semantic_repae_capture_request: tuple[int, int, int] | None = None
        self._semantic_repae_captured_hidden: torch.Tensor | None = None

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        ops: TransformerOpsConfig,
        apply_gated_attention: bool,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    ops=ops,
                )
                for _ in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def enable_semantic_flow_conditioning(
        self,
        *,
        semantic_dim: int,
        num_token_types: int = 3,
        num_entities: int = 5,
        semantic_token_type_id: int = 1,
        reference_token_type_id: int = 0,
    ) -> None:
        """Install trainable token metadata adapters and the semantic flow head."""
        if semantic_dim != self.patchify_proj.in_features:
            raise ValueError(
                f"semantic_dim={semantic_dim} must match video input token dim={self.patchify_proj.in_features}"
            )
        if self.semantic_proj_out is not None:
            if self.semantic_proj_out.out_features != semantic_dim:
                raise ValueError("semantic flow conditioning was already initialized with a different dimension")
            return
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        self.semantic_token_type_embedding = torch.nn.Embedding(num_token_types, self.inner_dim).to(
            device=device, dtype=dtype
        )
        self.semantic_entity_embedding = torch.nn.Embedding(num_entities, self.inner_dim).to(device=device, dtype=dtype)
        self.semantic_position_adapter = torch.nn.Sequential(
            torch.nn.Linear(6, self.inner_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.inner_dim, self.inner_dim),
        ).to(device=device, dtype=dtype)
        self.semantic_norm_out = torch.nn.RMSNorm(self.inner_dim, elementwise_affine=True).to(
            device=device, dtype=dtype
        )
        self.semantic_proj_out = torch.nn.Linear(self.inner_dim, semantic_dim).to(device=device, dtype=dtype)
        torch.nn.init.zeros_(self.semantic_proj_out.weight)
        torch.nn.init.zeros_(self.semantic_proj_out.bias)
        self.semantic_token_type_id = int(semantic_token_type_id)
        torch.nn.init.zeros_(self.semantic_token_type_embedding.weight)
        self.reference_token_type_id = int(reference_token_type_id)
        torch.nn.init.zeros_(self.semantic_entity_embedding.weight)
        torch.nn.init.normal_(self.semantic_position_adapter[0].weight, std=1.0e-4)
        torch.nn.init.zeros_(self.semantic_position_adapter[0].bias)
        torch.nn.init.zeros_(self.semantic_position_adapter[2].weight)
        torch.nn.init.zeros_(self.semantic_position_adapter[2].bias)

    def enable_semantic_repae_conditioning(
        self,
        *,
        semantic_dim: int,
        num_reference_slots: int = 4,
        semantic_token_type_id: int = 1,
        reference_token_type_id: int = 0,
    ) -> None:
        """Install REPA-E metadata adapters without changing target-token hidden states."""
        if semantic_dim != self.patchify_proj.in_features:
            raise ValueError(
                f"semantic_dim={semantic_dim} must match video input token dim={self.patchify_proj.in_features}"
            )
        if self.semantic_repae_semantic_type_embedding is not None:
            if self.semantic_proj_out is None or self.semantic_proj_out.out_features != semantic_dim:
                raise ValueError("semantic REPA-E conditioning was initialized with a different dimension")
            return
        if self.semantic_token_type_embedding is not None:
            raise RuntimeError("semantic flow and semantic REPA-E conditioning cannot be enabled together")
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        self.semantic_repae_reference_type_embedding = torch.nn.Embedding(1, self.inner_dim).to(
            device=device, dtype=dtype
        )
        self.semantic_repae_reference_slot_embedding = torch.nn.Embedding(
            num_reference_slots, self.inner_dim
        ).to(device=device, dtype=dtype)
        self.semantic_repae_semantic_type_embedding = torch.nn.Embedding(1, self.inner_dim).to(
            device=device, dtype=dtype
        )
        self.semantic_norm_out = torch.nn.RMSNorm(self.inner_dim, elementwise_affine=True).to(
            device=device, dtype=dtype
        )
        self.semantic_proj_out = torch.nn.Linear(self.inner_dim, semantic_dim).to(device=device, dtype=dtype)
        torch.nn.init.zeros_(self.semantic_repae_reference_type_embedding.weight)
        torch.nn.init.zeros_(self.semantic_repae_reference_slot_embedding.weight)
        torch.nn.init.zeros_(self.semantic_repae_semantic_type_embedding.weight)
        torch.nn.init.zeros_(self.semantic_proj_out.weight)
        torch.nn.init.zeros_(self.semantic_proj_out.bias)
        self.semantic_token_type_id = int(semantic_token_type_id)
        self.reference_token_type_id = int(reference_token_type_id)

    def configure_semantic_repae_capture(
        self,
        *,
        block_index: int,
        semantic_start: int,
        semantic_end: int,
    ) -> None:
        """Capture one semantic span after one transformer block on the next forward."""
        if not 0 <= block_index < len(self.transformer_blocks):
            raise ValueError(f"semantic REPA-E capture block index is out of range: {block_index}")
        if not 0 <= semantic_start < semantic_end:
            raise ValueError(
                f"invalid semantic REPA-E capture span [{semantic_start},{semantic_end})"
            )
        self._semantic_repae_capture_request = (block_index, semantic_start, semantic_end)
        self._semantic_repae_captured_hidden = None

    def consume_semantic_repae_capture(self) -> torch.Tensor:
        """Return and clear the semantic-only intermediate captured by the last forward."""
        captured = self._semantic_repae_captured_hidden
        self._semantic_repae_captured_hidden = None
        self._semantic_repae_capture_request = None
        if captured is None:
            raise RuntimeError("semantic REPA-E intermediate capture is missing")
        return captured

    def apply_video_token_metadata(self, video_args: TransformerArgs, video: Modality) -> TransformerArgs:
        """Apply enabled token metadata adapters through a stable public interface."""
        if self.semantic_repae_semantic_type_embedding is None:
            return self._apply_semantic_flow_video_token_metadata(video_args, video)
        if video.token_type_ids is None:
            raise ValueError("semantic REPA-E conditioning requires token_type_ids")
        if video.entity_ids is None:
            raise ValueError("semantic REPA-E conditioning requires entity_ids")
        if video.token_type_ids.shape != video_args.x.shape[:2]:
            raise ValueError("token_type_ids must match the video token shape")
        if video.entity_ids.shape != video_args.x.shape[:2]:
            raise ValueError("entity_ids must match the video token shape")

        x = video_args.x
        reference_mask = video.token_type_ids == self.reference_token_type_id
        semantic_mask = video.token_type_ids == self.semantic_token_type_id
        reference_type = self.semantic_repae_reference_type_embedding.weight[0].to(dtype=x.dtype)
        semantic_type = self.semantic_repae_semantic_type_embedding.weight[0].to(dtype=x.dtype)
        slot_count = self.semantic_repae_reference_slot_embedding.num_embeddings
        slot_ids = (video.entity_ids - 1).clamp(min=0, max=slot_count - 1)
        slot = self.semantic_repae_reference_slot_embedding(slot_ids).to(dtype=x.dtype)
        metadata = (
            reference_mask.unsqueeze(-1).to(dtype=x.dtype) * (reference_type + slot)
            + semantic_mask.unsqueeze(-1).to(dtype=x.dtype) * semantic_type
        )
        return replace(video_args, x=x + metadata)

    def _apply_video_token_metadata(self, video_args: TransformerArgs, video: Modality) -> TransformerArgs:
        return self.apply_video_token_metadata(video_args, video)

    def _apply_semantic_flow_video_token_metadata(
        self,
        video_args: TransformerArgs,
        video: Modality,
    ) -> TransformerArgs:
        x = video_args.x
        if video.token_type_ids is not None:
            if self.semantic_token_type_embedding is None:
                raise RuntimeError("token_type_ids require semantic flow conditioning to be initialized")
            x = x + self.semantic_token_type_embedding(video.token_type_ids).to(dtype=x.dtype)
        if video.entity_ids is not None:
            if self.semantic_entity_embedding is None:
                raise RuntimeError("entity_ids require semantic flow conditioning to be initialized")
            x = x + self.semantic_entity_embedding(video.entity_ids).to(dtype=x.dtype)
        if video.semantic_position_bounds is not None:
            if self.semantic_position_adapter is None or self.semantic_token_type_id is None:
                raise RuntimeError("semantic_position_bounds require semantic flow conditioning to be initialized")
            if video.semantic_position_bounds.shape != (*x.shape[:2], 6):
                raise ValueError(
                    "semantic_position_bounds must be [B,T,6], got "
                    f"{tuple(video.semantic_position_bounds.shape)} for hidden {tuple(x.shape)}"
                )
            semantic_position = self.semantic_position_adapter(
                video.semantic_position_bounds.to(device=x.device, dtype=x.dtype)
            )
            if video.token_type_ids is None:
                raise ValueError("semantic_position_bounds require token_type_ids")
            semantic_mask = (video.token_type_ids == self.semantic_token_type_id).unsqueeze(-1)
            x = x + semantic_position * semantic_mask.to(dtype=x.dtype)
        return replace(video_args, x=x)

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Process transformer blocks for LTXAV.
        Per-block perturbation masks are precomputed here and attached to each
        modality's ``TransformerArgs`` so the block forward has no per-block
        identity to specialise on — all blocks share a single Dynamo cache slot.
        """
        if perturbations is None:
            batch_size = (video or audio).x.shape[0]
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        for block_idx, block in enumerate(self.transformer_blocks):
            if video is not None:
                video = self.block_input_processor(
                    video,
                    perturbations,
                    block_idx,
                    self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                    cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
                )
            if audio is not None:
                audio = self.block_input_processor(
                    audio,
                    perturbations,
                    block_idx,
                    self_attn_type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                    cross_attn_type=PerturbationType.SKIP_V2A_CROSS_ATTN,
                )

            if self._enable_gradient_checkpointing and torch.is_grad_enabled():
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    use_reentrant=False,
                )
            else:
                video, audio = block(video=video, audio=audio)

            request = self._semantic_repae_capture_request
            if request is not None and block_idx == request[0]:
                if video is None:
                    raise RuntimeError("semantic REPA-E capture requires an enabled video modality")
                semantic_start, semantic_end = request[1:]
                if semantic_end > video.x.shape[1]:
                    raise RuntimeError(
                        "semantic REPA-E capture span exceeds the video sequence: "
                        f"end={semantic_end}, length={video.x.shape[1]}"
                    )
                self._semantic_repae_captured_hidden = video.x[:, semantic_start:semantic_end].clone()

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        self._semantic_repae_captured_hidden = None
        if video_args is not None and video is not None:
            video_args = self._apply_video_token_metadata(video_args, video)
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

        # Process output
        vx = None
        if video_out is not None:
            vx = self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if (
                video is not None
                and video.token_type_ids is not None
                and self.semantic_norm_out is not None
                and self.semantic_proj_out is not None
                and self.semantic_token_type_id is not None
            ):
                semantic_velocity = self.semantic_proj_out(self.semantic_norm_out(video_out.x))
                semantic_mask = (video.token_type_ids == self.semantic_token_type_id).unsqueeze(-1)
                vx = torch.where(semantic_mask, semantic_velocity, vx)
                if self.reference_token_type_id is not None:
                    reference_mask = (video.token_type_ids == self.reference_token_type_id).unsqueeze(-1)
                    vx = torch.where(reference_mask, torch.zeros_like(vx), vx)
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModelProtocol):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModelProtocol):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
