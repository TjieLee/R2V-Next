#!/usr/bin/env python3
"""Create a simple HTML page for qualitative overfit generation checks."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Build an HTML side-by-side report for overfit generated samples.",
)
console = Console()

try:
    import cv2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency at runtime
    cv2 = None


def _extract_first_frame(video_path: Path, frame_path: Path) -> Path | None:
    if not video_path.is_file() or cv2 is None:
        return None
    if frame_path.is_file():
        return frame_path
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
        if not ok:
            return None
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_path), frame)
        return frame_path
    finally:
        cap.release()


def _rel(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _load_metadata(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _sample_dirs(root: Path) -> list[Path]:
    dirs = sorted(path for path in root.glob("sample_*") if path.is_dir())
    if not dirs and (root / "metadata.json").is_file():
        dirs = [root]
    return dirs


def _img_tag(path: Path | None, base: Path, label: str) -> str:
    if path is None or not path.is_file():
        return f'<div class="missing">Missing {html.escape(label)}</div>'
    src = html.escape(_rel(path, base) or "")
    return f'<img src="{src}" alt="{html.escape(label)}">'


def _video_tag(path: Path | None, base: Path) -> str:
    if path is None or not path.is_file():
        return ""
    src = html.escape(_rel(path, base) or "")
    return f'<video src="{src}" controls muted loop preload="metadata"></video>'


@app.command()
def main(
    generated_dir: str = typer.Argument(..., help="Directory containing sample_x folders."),
    output_html: str | None = typer.Option(None, help="Output HTML path. Defaults to generated_dir/index.html."),
    max_samples: int | None = typer.Option(None, help="Optional cap on samples included in the report."),
) -> None:
    root = Path(generated_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Generated directory does not exist: {root}")
    if max_samples is not None and max_samples < 1:
        raise typer.BadParameter("--max-samples must be >= 1")

    html_path = Path(output_html) if output_html is not None else root / "index.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    base = html_path.parent

    samples = _sample_dirs(root)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No sample_* directories found under {root}")

    cards: list[str] = []
    for sample_dir in samples:
        metadata = _load_metadata(sample_dir)
        prompt = html.escape(str(metadata.get("prompt", "")))
        gt_video = sample_dir / "gt.mp4"
        generated_video = sample_dir / "generated.mp4"
        gt_frame = _extract_first_frame(gt_video, sample_dir / "gt_first.jpg")
        generated_frame = _extract_first_frame(generated_video, sample_dir / "generated_first.jpg")
        refs = sorted(sample_dir.glob("ref_*"))
        ref_html = "".join(_img_tag(ref, base, ref.name) for ref in refs) or '<div class="missing">No references</div>'
        cards.append(
            f"""
<section class="sample">
  <h2>{html.escape(sample_dir.name)}</h2>
  <p class="prompt">{prompt}</p>
  <div class="grid">
    <div><h3>References</h3><div class="refs">{ref_html}</div></div>
    <div><h3>GT First Frame</h3>{_img_tag(gt_frame, base, 'GT first frame')}{_video_tag(gt_video, base)}</div>
    <div><h3>Generated First Frame</h3>{_img_tag(generated_frame, base, 'Generated first frame')}{_video_tag(generated_video, base)}</div>
  </div>
  <div class="metrics">Metrics TODO: identity similarity, motion/content similarity, visual-token ablation.</div>
</section>
"""
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Multi-Reference Overfit Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    h2 {{ font-size: 20px; margin: 0 0 8px; }}
    h3 {{ font-size: 14px; margin: 0 0 8px; color: #475569; }}
    .sample {{ border-top: 1px solid #d8dee9; padding: 20px 0 28px; }}
    .prompt {{ max-width: 1100px; color: #475569; }}
    .grid {{ display: grid; grid-template-columns: minmax(220px, 1fr) minmax(220px, 1fr) minmax(220px, 1fr); gap: 16px; align-items: start; }}
    .refs {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); gap: 8px; }}
    img, video {{ width: 100%; max-height: 360px; object-fit: contain; background: #f8fafc; border: 1px solid #e2e8f0; }}
    video {{ margin-top: 8px; }}
    .missing {{ min-height: 120px; display: grid; place-items: center; background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }}
    .metrics {{ margin-top: 12px; color: #64748b; font-size: 13px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Multi-Reference Overfit Report</h1>
  <p>{len(samples)} sample(s). Check whether generated identity matches references and motion/content memorizes GT.</p>
  {''.join(cards)}
</body>
</html>
"""
    html_path.write_text(page, encoding="utf-8")
    console.print(f"Overfit comparison report written to: {html_path}")


if __name__ == "__main__":
    app()
