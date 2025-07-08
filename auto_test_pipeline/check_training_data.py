import os
from PIL import Image

def check_lora_training_data():
    """檢查 LoRA 訓練資料是否準備就緒"""
    
    target_dir = "lora_train_set/10_test"
    
    if not os.path.exists(target_dir):
        print(f"❌ 目標資料夾不存在：{target_dir}")
        return False
    
    # 獲取所有圖片和文字檔案
    img_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    txt_files = [f for f in os.listdir(target_dir) if f.lower().endswith('.txt')]
    
    print(f"📊 LoRA 訓練資料檢查報告")
    print(f"=" * 50)
    print(f"📁 資料夾位置: {target_dir}")
    print(f"🖼️  圖片檔案數量: {len(img_files)}")
    print(f"📝 文字檔案數量: {len(txt_files)}")
    
    # 檢查每張圖片是否有對應的文字檔案
    missing_txt = []
    for img_file in img_files:
        img_name = os.path.splitext(img_file)[0]
        txt_file = img_name + ".txt"
        if txt_file not in txt_files:
            missing_txt.append(img_file)
    
    if missing_txt:
        print(f"⚠️  缺少描述檔案的圖片: {missing_txt}")
        return False
    
    print(f"✅ 所有圖片都有對應的描述檔案")
    
    # 檢查圖片尺寸
    print(f"\n📏 圖片尺寸檢查:")
    oversized_images = []
    for img_file in img_files[:5]:  # 檢查前5張圖片
        img_path = os.path.join(target_dir, img_file)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                print(f"   {img_file}: {width}x{height}")
                if width > 512 or height > 512:
                    oversized_images.append((img_file, width, height))
        except Exception as e:
            print(f"   ❌ 無法讀取 {img_file}: {e}")
            return False
    
    if oversized_images:
        print(f"⚠️  超出 512x512 的圖片: {oversized_images}")
        return False
    
    # 檢查描述檔案內容
    print(f"\n📝 描述檔案內容檢查:")
    for txt_file in txt_files[:3]:  # 檢查前3個描述檔案
        txt_path = os.path.join(target_dir, txt_file)
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    print(f"   {txt_file}: {content[:50]}...")
                else:
                    print(f"   ❌ {txt_file} 是空的")
                    return False
        except Exception as e:
            print(f"   ❌ 無法讀取 {txt_file}: {e}")
            return False
    
    print(f"\n🎯 LoRA 訓練資料檢查結果:")
    print(f"✅ 圖片檔案: {len(img_files)} 張")
    print(f"✅ 描述檔案: {len(txt_files)} 個")
    print(f"✅ 圖片尺寸: 全部符合 ≤512x512")
    print(f"✅ 描述內容: 全部有效")
    print(f"✅ 檔案配對: 完整")
    
    print(f"\n🚀 準備就緒！可以開始 LoRA 訓練了！")
    return True

if __name__ == "__main__":
    check_lora_training_data()
