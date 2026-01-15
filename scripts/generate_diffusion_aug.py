#!/usr/bin/env python
"""Generate diffusion-augmented images offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--style_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--strength", type=float, default=0.4)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()

    from diffusers import StableDiffusionImg2ImgPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe = pipe.to(device)

    out_root = Path(args.output_dir) / args.style_name / "images"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, str] = {}
    images = sorted([p for p in Path(args.input_dir).glob("**/*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if args.max_images > 0:
        images = images[: args.max_images]

    generator = torch.Generator(device=device).manual_seed(args.seed)
    for img_path in tqdm(images, desc="diffusion-aug"):
        image = Image.open(img_path).convert("RGB")
        result = pipe(
            prompt=args.prompt,
            image=image,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            generator=generator,
        ).images[0]
        out_path = out_root / img_path.name
        result.save(out_path)
        manifest[str(img_path)] = str(out_path)

    manifest_path = Path(args.output_dir) / args.style_name / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
