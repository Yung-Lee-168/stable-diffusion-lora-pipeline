import os
import json

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_local_base_model.json')

REQUIRED_SUBDIRS = ["vae", "unet", "tokenizer", "text_encoder"]
REQUIRED_FILES = ["diffusion_pytorch_model.bin", "model.ckpt", "model.safetensors"]

def check_base_model():
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ 找不到 {CONFIG_PATH}")
        return False
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    base_model_path = config.get("base_model_path")
    if not base_model_path:
        print("❌ config_local_base_model.json 缺少 base_model_path")
        return False
    if not os.path.exists(base_model_path):
        print(f"❌ 找不到 base_model_path: {base_model_path}")
        return False
    # 檢查主權重
    has_weight = any(os.path.exists(os.path.join(base_model_path, f)) for f in REQUIRED_FILES)
    if not has_weight:
        print(f"❌ {base_model_path} 缺少主權重檔 (diffusion_pytorch_model.bin / model.ckpt / model.safetensors)")
        return False
    # 檢查子資料夾
    for subdir in REQUIRED_SUBDIRS:
        subdir_path = os.path.join(base_model_path, subdir)
        if not os.path.isdir(subdir_path):
            print(f"⚠️ 缺少子資料夾: {subdir_path}")
        else:
            files = os.listdir(subdir_path)
            if not files:
                print(f"⚠️ {subdir_path} 內沒有檔案")
    print("✅ 本地模型結構檢查完成！")
    return True

if __name__ == "__main__":
    check_base_model()
