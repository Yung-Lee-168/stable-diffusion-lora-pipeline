# analyze_fashionclip.py
"""
進階分析腳本：利用 FashionCLIP 與自訂特徵分類，針對 baseline 與 LoRA 產圖進行語意特徵比對。
- 會自動載入 baseline/lora 圖片，並用 FashionCLIP 對每張圖進行特徵分類（依據特徵值.py定義）。
- 輸出每張圖的分類結果與 baseline/lora 差異。
"""
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json

# 直接複製特徵值.py的分類定義
CATEGORIES = {
    "Gender": ["male", "female"],
    "Age": ["child", "teenager", "young adult", "adult", "senior"],
    "Season": ["spring", "summer", "autumn", "winter"],
    "Occasion": ["casual", "formal", "business", "sport", "party", "beach", "wedding", "date", "travel", "home"],
    "Upper Body": ["t-shirt", "shirt", "jacket", "coat", "sweater", "blazer", "hoodie", "tank top", "blouse", "dress"],
    "Lower Body": ["jeans", "trousers", "shorts", "skirt", "leggings", "cargo pants", "sweatpants", "culottes", "capris", "dress"]
}

DETAILED_FEATURES = {
    "Dress Style": ["A-line dress", "sheath dress", "wrap dress", "maxi dress", "midi dress", "mini dress", "bodycon dress", "shift dress", "empire waist dress", "fit and flare dress", "slip dress", "shirt dress", "sweater dress"],
    "Shirt Features": ["button-down shirt", "polo shirt", "henley shirt", "flannel shirt", "dress shirt", "peasant blouse", "crop top", "off-shoulder top", "turtleneck", "v-neck shirt", "crew neck", "collared shirt"],
    "Jacket Types": ["denim jacket", "leather jacket", "bomber jacket", "trench coat", "peacoat", "blazer jacket", "cardigan", "windbreaker", "puffer jacket", "motorcycle jacket", "varsity jacket"],
    "Pants Details": ["skinny jeans", "straight leg jeans", "bootcut jeans", "wide leg pants", "high-waisted pants", "low-rise pants", "cropped pants", "palazzo pants", "joggers", "dress pants", "cargo pants with pockets"],
    "Skirt Varieties": ["pencil skirt", "A-line skirt", "pleated skirt", "wrap skirt", "mini skirt", "maxi skirt", "denim skirt", "leather skirt", "tulle skirt", "asymmetrical skirt"],
    "Fabric Texture": ["cotton fabric", "silk material", "denim texture", "leather finish", "wool texture", "linen fabric", "chiffon material", "velvet texture", "knit fabric", "lace material", "satin finish", "corduroy texture"],
    "Pattern Details": ["solid color", "striped pattern", "floral print", "polka dots", "geometric pattern", "animal print", "plaid pattern", "paisley design", "abstract print", "tie-dye pattern", "checkered pattern"],
    "Color Scheme": ["monochrome outfit", "pastel colors", "bright colors", "earth tones", "neutral colors", "bold colors", "metallic accents", "neon colors", "vintage colors", "gradient colors"],
    "Fit Description": ["loose fit", "tight fit", "oversized", "fitted", "relaxed fit", "tailored fit", "slim fit", "regular fit", "cropped length", "flowing silhouette", "structured shape"],
    "Style Details": ["minimalist style", "vintage style", "bohemian style", "gothic style", "preppy style", "streetwear style", "romantic style", "edgy style", "classic style", "trendy style", "elegant style"]
}

ALL_FEATURES = {**CATEGORIES, **DETAILED_FEATURES}


def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def extract_features(image_path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    result = {}
    for cat, labels in ALL_FEATURES.items():
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        best_idx = probs.argmax().item()
        result[cat] = labels[best_idx]
    return result

def analyze_fashionclip(baseline_dir, lora_dir, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    baseline_imgs = list_images(baseline_dir)
    lora_imgs = list_images(lora_dir)
    results = []
    for base, lora in zip(baseline_imgs, lora_imgs):
        base_feat = extract_features(os.path.join(baseline_dir, base), model, processor, device)
        lora_feat = extract_features(os.path.join(lora_dir, lora), model, processor, device)
        diff = {k: (base_feat[k], lora_feat[k]) for k in ALL_FEATURES if base_feat[k] != lora_feat[k]}
        results.append({"baseline": base, "lora": lora, "diff": diff, "base_feat": base_feat, "lora_feat": lora_feat})
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"分析完成，結果已儲存至 {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="使用 FashionCLIP 進行語意特徵分析")
    parser.add_argument('--baseline_dir', type=str, default='../auto_test_pipeline/images/baseline', help='baseline 圖片資料夾')
    parser.add_argument('--lora_dir', type=str, default='../auto_test_pipeline/images/lora', help='lora 圖片資料夾')
    parser.add_argument('--output', type=str, default='../auto_test_pipeline/analyze_fashionclip.json', help='分析結果輸出檔')
    args = parser.parse_args()
    analyze_fashionclip(args.baseline_dir, args.lora_dir, args.output)
