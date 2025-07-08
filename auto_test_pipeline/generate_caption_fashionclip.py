from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
import torch
import traceback
import shutil

print("🚩 開始自動產生圖片描述（特徵標籤）...")

# 獲取腳本所在目錄
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"📁 腳本目錄: {script_dir}")

# 智能檢測目錄結構，支援兩種運行模式
is_in_auto_test_pipeline = script_dir.endswith("auto_test_pipeline")
is_in_project_root = "stable-diffusion-webui" in script_dir and not script_dir.endswith("auto_test_pipeline")

if is_in_auto_test_pipeline:
    print("✅ 檢測到在 auto_test_pipeline 目錄中執行")
    base_dir = script_dir
    feature_file = os.path.join(script_dir, "特徵值.py")
    source_dir = os.path.join(script_dir, "source_image")
    target_dir = os.path.join(script_dir, "lora_train_set", "10_test")
elif is_in_project_root:
    print("✅ 檢測到在項目根目錄中執行")
    base_dir = script_dir
    # 在項目根目錄時，特徵值.py 可能在根目錄或 auto_test_pipeline 目錄
    feature_file = os.path.join(script_dir, "特徵值.py")
    if not os.path.exists(feature_file):
        feature_file = os.path.join(script_dir, "auto_test_pipeline", "特徵值.py")
    source_dir = os.path.join(script_dir, "auto_test_pipeline", "source_image")
    target_dir = os.path.join(script_dir, "auto_test_pipeline", "lora_train_set", "10_test")
else:
    print("❌ 錯誤：此腳本只能在以下目錄中執行：")
    print(f"   1. E:\\Yung_Folder\\Project\\stable-diffusion-webui\\auto_test_pipeline")
    print(f"   2. E:\\Yung_Folder\\Project\\stable-diffusion-webui")
    print(f"💡 當前目錄: {script_dir}")
    exit(1)

print(f"📁 基礎目錄: {base_dir}")
print(f"📁 特徵值文件: {feature_file}")
print(f"📁 來源目錄: {source_dir}")
print(f"📁 目標目錄: {target_dir}")

# 檢查特徵值.py 是否存在
if not os.path.exists(feature_file):
    print("❌ 錯誤：找不到 特徵值.py 文件")
    print(f"💡 特徵值.py 必須位於: {feature_file}")
    if is_in_project_root:
        print(f"💡 或者位於: {os.path.join(script_dir, '特徵值.py')}")
    exit(1)

# 添加特徵值.py 所在目錄到 Python 路徑
feature_dir = os.path.dirname(feature_file)
import sys
if feature_dir not in sys.path:
    sys.path.insert(0, feature_dir)

try:
    import 特徵值
    print("✅ 成功匯入 特徵值.py")
except Exception as e:
    print(f"❌ 匯入特徵值.py 失敗：{e}")
    print(f"💡 確認 特徵值.py 是否存在於: {feature_file}")
    traceback.print_exc()
    exit(1)

# 合併所有特徵詞典的值為 labels_dict，保留類別資訊
labels_dict = {}
for k, v in 特徵值.__dict__.items():
    if isinstance(v, (list, tuple)):
        labels_dict[k] = list(v)
    elif isinstance(v, dict):
        for kk, vv in v.items():
            if isinstance(vv, (list, tuple)):
                labels_dict[kk] = list(vv)

# 展平成 labels（所有特徵詞，去重）
labels = [str(x).strip() for v in labels_dict.values() for x in v]
labels = [x for x in set(labels) if x]
print(f"✅ 已載入 {len(labels)} 個特徵標籤，前10個：{labels[:10]}")

try:
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    print("✅ FashionCLIP (transformers) 載入成功")
except Exception as e:
    print(f"❌ FashionCLIP 載入失敗：{e}")
    traceback.print_exc()
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 檢查來源目錄是否存在
if not os.path.exists(source_dir):
    print(f"❌ 來源目錄不存在: {source_dir}")
    print(f"💡 請確保圖片放在此目錄中")
    
    # 創建來源目錄
    os.makedirs(source_dir, exist_ok=True)
    print(f"✅ 已創建來源目錄: {source_dir}")
    print(f"🔧 請將要處理的圖片放入此目錄中")
    exit(1)

# 檢查圖片數量
img_files = [f for f in os.listdir(source_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
if not img_files:
    print(f"❌ 來源目錄中沒有圖片: {source_dir}")
    print(f"💡 支援的格式: .png, .jpg, .jpeg, .bmp, .gif")
    exit(1)

print(f"✅ 在來源目錄中發現 {len(img_files)} 張圖片")

# 確保目標資料夾存在
os.makedirs(target_dir, exist_ok=True)
print(f"📁 來源目錄: {source_dir}")
print(f"📁 目標目錄: {target_dir}")

def resize_image_keep_aspect(image, max_size=512):
    """調整圖片尺寸，保持寬高比，確保不超過 max_size"""
    width, height = image.size
    
    # 如果圖片已經符合要求，直接返回
    if width <= max_size and height <= max_size:
        return image, False
    
    # 計算縮放比例
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 調整圖片大小
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image, True

# 處理來源圖片
print(f"🔄 開始處理 {len(img_files)} 張圖片...")

# 第一階段：處理圖片尺寸並複製到目標資料夾
processed_files = []
for i, fname in enumerate(img_files, 1):
    source_path = os.path.join(source_dir, fname)
    print(f"🔄 [{i}/{len(img_files)}] 處理: {fname}")
    
    try:
        image = Image.open(source_path).convert("RGB")
        width, height = image.size
        
        # 檢查圖片尺寸
        if width <= 512 and height <= 512:
            # 尺寸符合要求，直接複製
            target_path = os.path.join(target_dir, fname)
            shutil.copy2(source_path, target_path)
            processed_files.append(fname)
            print(f"   ✅ {fname} ({width}x{height}) 尺寸符合要求，已複製")
        else:
            # 需要縮放
            resized_image, was_resized = resize_image_keep_aspect(image, 512)
            if was_resized:
                # 產生新檔名
                name, ext = os.path.splitext(fname)
                new_fname = f"{name}_scale{ext}"
                target_path = os.path.join(target_dir, new_fname)
                resized_image.save(target_path, quality=95)
                processed_files.append(new_fname)
                new_width, new_height = resized_image.size
                print(f"   ✅ {fname} ({width}x{height}) 已縮放至 {new_fname} ({new_width}x{new_height})")
            
    except Exception as e:
        print(f"   ❌ 處理 {fname} 時發生錯誤：{e}")
        traceback.print_exc()

print(f"\n✅ 第一階段完成，共處理 {len(processed_files)} 張圖片")
print("🔄 開始第二階段：生成 FashionCLIP 描述...")

# 第二階段：對目標資料夾中的圖片生成描述
target_img_files = [f for f in os.listdir(target_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
print(f"✅ 目標資料夾中共有 {len(target_img_files)} 張圖片需要生成描述")

for fname in target_img_files:
    img_path = os.path.join(target_dir, fname)
    try:
        image = Image.open(img_path).convert("RGB")
        desc_list = []
        # 對每個特徵類別都找分數最高的詞
        for cat, cat_labels in labels_dict.items():
            if not cat_labels:
                continue
            inputs = processor(text=cat_labels, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            best_idx = probs[0].argmax().item()
            best_label = cat_labels[best_idx]
            desc_list.append(str(best_label))
            print(f"🏷️  {fname} [{cat}] best_label: {best_label} (index: {best_idx})")
        # 組成完整描述
        full_desc = ", ".join(desc_list)
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_desc)
        print(f"✅ {fname} → {full_desc}")
    except Exception as e:
        print(f"❌ {fname} 發生錯誤：{e}")
        traceback.print_exc()

print("🏁 全部完成！")