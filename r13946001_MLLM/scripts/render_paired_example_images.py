from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render selected same-question VQA examples with their images.")
    parser.add_argument("--selected", default="presentation/demo_examples/paired_base_vs_fl_examples_selected.csv")
    parser.add_argument("--out", default="presentation/demo_final_assets/final_paired_image_examples.png")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_path = Path(args.selected)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not selected_path.exists():
        write_empty(out, "No selected paired examples found")
        return

    selected = pd.read_csv(selected_path)
    selected = selected[selected["selection_reason"].ne("representative_caa_correct")].head(args.limit).copy()
    if selected.empty:
        write_empty(out, "No useful paired examples selected")
        return

    datasets = {}
    for dataset in selected["dataset"].unique():
        datasets[str(dataset)] = load_limited_raw_testset(str(dataset), args.seed, args.max_test_samples)

    rows = []
    for _, row in selected.iterrows():
        example = datasets[str(row["dataset"])][int(row["sample_index"])]
        image = example["image"]
        rows.append((row, image))

    render_pil_grid(rows, out)
    print(f"wrote {out} examples={len(rows)}")


def shorten(text: str, width: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def render_pil_grid(rows: list[tuple[pd.Series, Image.Image]], out: Path) -> None:
    canvas_w, canvas_h = 2200, 1240
    margin = 46
    title_h = 92
    gutter = 30
    panel_w = (canvas_w - 2 * margin - gutter) // 2
    panel_h = (canvas_h - title_h - 2 * margin - gutter) // 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(44, bold=True)
    header_font = load_font(24, bold=True)
    body_font = load_font(25, bold=False)
    small_font = load_font(23, bold=False)

    title = "Same-question VQA DEMO examples with images"
    tw = draw.textbbox((0, 0), title, font=title_font)[2]
    draw.text(((canvas_w - tw) // 2, 22), title, fill=(15, 23, 42), font=title_font)

    for idx, (row, image) in enumerate(rows[:4]):
        col = idx % 2
        r = idx // 2
        x0 = margin + col * (panel_w + gutter)
        y0 = title_h + margin + r * (panel_h + gutter)
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=(248, 250, 252), outline=(203, 213, 225), width=3)

        img_box = (x0 + 22, y0 + 58, x0 + 390, y1 - 22)
        fitted = fit_image(image, img_box[2] - img_box[0], img_box[3] - img_box[1])
        ix = img_box[0] + (img_box[2] - img_box[0] - fitted.width) // 2
        iy = img_box[1] + (img_box[3] - img_box[1] - fitted.height) // 2
        canvas.paste(fitted, (ix, iy))

        header = f"{display_dataset_name(str(row['dataset']))} #{int(row['sample_index'])}"
        draw.text((x0 + 22, y0 + 16), header, fill=(15, 23, 42), font=header_font)

        tx = x0 + 420
        ty = y0 + 66
        question = "Q: " + str(row["question"])
        for line in wrap_text(draw, question, body_font, panel_w - 445):
            draw.text((tx, ty), line, fill=(15, 23, 42), font=body_font)
            ty += 33
        ty += 10
        draw.text((tx, ty), "Options:", fill=(51, 65, 85), font=small_font)
        ty += 30
        for option_line in format_choices(str(row.get("choices", ""))):
            for line in wrap_text(draw, option_line, small_font, panel_w - 445):
                draw.text((tx, ty), line, fill=(71, 85, 105), font=small_font)
                ty += 30
        ty += 8

        facts = [
            f"Gold: {row['gold_answer']}",
            f"Base: {row['parsed_answer_Base']}",
            f"Naive: {row['parsed_answer_Naive_Async']}",
            f"Sync: {row['parsed_answer_Sync']}",
            f"CAA-v2: {row['parsed_answer_CAA-v2']}",
        ]
        for fact in facts:
            color = (5, 150, 105) if fact.startswith("CAA-v2") else (51, 65, 85)
            draw.text((tx, ty), fact, fill=color, font=small_font)
            ty += 30
    canvas.save(out)


def load_font(size: int, *, bold: bool) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def fit_image(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return image


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:4]


def load_limited_raw_testset(dataset: str, seed: int, max_samples: int) -> list[dict[str, object]]:
    from datasets import load_dataset

    if dataset == "vqa_rad_closed":
        raw = load_dataset("abhay2812/vqa-rad", split="test", cache_dir="data/hf")
    elif dataset == "path_vqa_closed":
        raw = load_dataset("flaviagiammarino/path-vqa", split="test", cache_dir="data/hf")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    rng = np.random.default_rng(seed + 1)
    indices = rng.permutation(len(raw))[:max_samples].tolist()
    rows: list[dict[str, object]] = []
    for index in indices:
        item = raw[int(index)]
        image = item.get("image") or item.get("Image")
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        rows.append(
            {
                "question": str(item.get("question") or item.get("Question") or ""),
                "image": image.convert("RGB"),
            }
        )
    return rows


def display_dataset_name(dataset: str) -> str:
    names = {
        "vqa_rad_closed": "VQA-RAD",
        "path_vqa_closed": "PathVQA",
    }
    return names.get(dataset, dataset)


def format_choices(choices: str) -> list[str]:
    values = [part.strip() for part in choices.split("|")]
    labels = ["A", "B", "C", "D"]
    lines = []
    for label, value in zip(labels, values):
        if value:
            lines.append(f"({label}) {value}")
    return lines if lines else ["not available"]


def write_empty(path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
