from PIL import Image
import os

# 檢查來源圖片尺寸
source_dir = "source_image"
files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print("📏 來源圖片尺寸：")
for f in files[:5]:
    img_path = os.path.join(source_dir, f)
    size = Image.open(img_path).size
    print(f"{f}: {size}")

# 檢查目標圖片尺寸
target_dir = "lora_train_set/10_test"
target_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print("\n📏 目標圖片尺寸：")
for f in target_files[:5]:
    img_path = os.path.join(target_dir, f)
    size = Image.open(img_path).size
    print(f"{f}: {size}")
