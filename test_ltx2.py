import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained(
    "Lightricks/LTX-2",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

out = pipe(
    prompt="A small robot walking through a neon-lit street at night, cinematic, smooth motion.",
    width=768,
    height=512,
    num_frames=97,
)

export_to_video(out.frames[0], "ltx2_test.mp4", fps=24)
print("Saved: ltx2_test.mp4")
