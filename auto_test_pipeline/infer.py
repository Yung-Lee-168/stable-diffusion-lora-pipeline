import sys
import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

def load_prompts():
    with open("city_features.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def main(mode):
    prompts = load_prompts()
    if mode == "baseline":
        model_path = "base_model.safetensors"  # 請放入你的 base model
    else:
        model_path = os.path.join("lora_output", "lora.safetensors")  # 請根據實際 LoRA 輸出命名
    pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16).to("cuda")
    os.makedirs("images", exist_ok=True)
    for i, prompt in enumerate(prompts):
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        image.save(f"images/{mode}_{i+1}.png")
    print(f"✅ {mode} 產圖完成")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "lora"], required=True)
    args = parser.parse_args()
    main(args.mode)
