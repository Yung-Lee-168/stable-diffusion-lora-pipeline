# analyze.py
"""
分析腳本：自動化比較 baseline 與 LoRA 產生的圖像，支援簡單統計、CLIP score、圖像相似度等擴充。
建議：先以檔案命名規則比對、簡單像素差異、檔案大小等為主，進階可加 CLIP score。
"""
import os
from PIL import Image, ImageChops
import numpy as np

def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def mse(img1, img2):
    arr1 = np.asarray(img1).astype(np.float32)
    arr2 = np.asarray(img2).astype(np.float32)
    return np.mean((arr1 - arr2) ** 2)

def analyze(baseline_dir, lora_dir, output_path):
    baseline_imgs = list_images(baseline_dir)
    lora_imgs = list_images(lora_dir)
    results = []
    for base, lora in zip(baseline_imgs, lora_imgs):
        base_img = Image.open(os.path.join(baseline_dir, base)).convert('RGB')
        lora_img = Image.open(os.path.join(lora_dir, lora)).convert('RGB')
        diff = mse(base_img, lora_img)
        results.append(f"{base}\t{lora}\tMSE: {diff:.2f}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("baseline\tlora\tMSE\n")
        for line in results:
            f.write(line + "\n")
    print(f"分析完成，結果已儲存至 {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析 baseline 與 LoRA 產圖差異")
    parser.add_argument('--baseline_dir', type=str, default='../auto_test_pipeline/images/baseline', help='baseline 圖片資料夾')
    parser.add_argument('--lora_dir', type=str, default='../auto_test_pipeline/images/lora', help='lora 圖片資料夾')
    parser.add_argument('--output', type=str, default='../auto_test_pipeline/analyze_result.txt', help='分析結果輸出檔')
    args = parser.parse_args()
    analyze(args.baseline_dir, args.lora_dir, args.output)
