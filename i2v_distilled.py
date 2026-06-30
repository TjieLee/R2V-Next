#!/usr/bin/env python3
"""
LTX-2 图生视频 (Image-to-Video) 脚本
基于 DistilledPipeline，复用已有的本地蒸馏模型路径。

用法:
    python i2v_distilled.py --image-path input.jpg --prompt "描述文本"
    
    # 或使用完整参数
    python i2v_distilled.py \
        --image-path input.jpg \
        --prompt "A small robot walks through a neon-lit street at night." \
        --image-strength 1.0 \
        --seed 42 \
        --height 512 \
        --width 768 \
        --num-frames 97 \
        --frame-rate 24 \
        --output-path outputs/i2v_output.mp4
"""

import argparse
import logging
from collections.abc import Iterator

import torch

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LTX-2 图生视频 (Image-to-Video) 脚本")

    # ============ 模型路径（复用你已有的本地路径） ============
    parser.add_argument(
        "--distilled-checkpoint-path",
        type=str,
        default="models/ltx-2.3-22b-distilled-1.1.safetensors",
        help="蒸馏模型 checkpoint 路径",
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        type=str,
        default="models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        help="空间上采样器模型路径",
    )
    parser.add_argument(
        "--gemma-root",
        type=str,
        default="models/gemma-3-12b-it-qat-q4_0-unquantized",
        help="Gemma 文本编码器根目录",
    )

    # ============ 图生视频条件控制参数 ============
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="输入图片路径（作为视频首帧或参考帧的条件控制）",
    )
    parser.add_argument(
        "--image-strength",
        type=float,
        default=1.0,
        help=(
            "图片条件强度，控制输入图片对生成视频的影响力。"
            "1.0 = 完全遵循输入图片，0.0 = 无影响。"
            "推荐范围: 0.7~1.0，较低值允许更多创意自由度 (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--image-frame-idx",
        type=int,
        default=0,
        help=(
            "输入图片对应的目标帧索引。"
            "0 = 首帧（最常用），也可设为其他帧索引实现关键帧控制 (default: 0)"
        ),
    )
    parser.add_argument(
        "--image-crf",
        type=int,
        default=33,
        help=(
            "图片 H.264 压缩质量参数 (CRF)。"
            "0 = 无损，值越大压缩越强。较低值保留更多细节 (default: 33)"
        ),
    )

    # ============ 生成参数 ============
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="文本提示词，描述期望生成的视频内容",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于可复现生成 (default: 42)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="输出视频高度（像素），须为 64 的倍数 (default: 512)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="输出视频宽度（像素），须为 64 的倍数 (default: 768)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=97,
        help="输出视频帧数，须满足 (8*k)+1 格式，如 97, 121, 161 (default: 97)",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=24,
        help="输出视频帧率 (fps) (default: 24)",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="使用 Gemma 增强提示词（基于输入图片自动补充描述）",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/i2v_output.mp4",
        help="输出视频文件路径 (default: outputs/i2v_output.mp4)",
    )

    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # 构建图片条件输入
    image_condition = ImageConditioningInput(
        path=args.image_path,
        frame_idx=args.image_frame_idx,
        strength=args.image_strength,
        crf=args.image_crf,
    )

    # 初始化 DistilledPipeline（复用你的本地模型路径）
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=(),
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    # 执行图生视频推理
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=[image_condition],
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )

    # 编码输出视频
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )
    print(f"视频已保存至: {args.output_path}")


if __name__ == "__main__":
    main()