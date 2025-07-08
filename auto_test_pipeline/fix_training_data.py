#!/usr/bin/env python3
"""
檢查和修正訓練資料的配對問題
確保每個 .jpg 文件都有對應的 .txt 文件，反之亦然
"""

import os
import sys

def check_and_fix_training_data(data_folder):
    """檢查並修正訓練資料配對問題"""
    print(f"🔍 檢查訓練資料配對: {data_folder}")
    
    if not os.path.exists(data_folder):
        print(f"❌ 資料夾不存在: {data_folder}")
        return False
    
    # 獲取所有文件
    all_files = os.listdir(data_folder)
    jpg_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))]
    txt_files = [f for f in all_files if f.lower().endswith('.txt')]
    
    print(f"📊 文件統計:")
    print(f"   JPG 文件: {len(jpg_files)}")
    print(f"   TXT 文件: {len(txt_files)}")
    
    # 檢查配對
    jpg_basenames = {os.path.splitext(f)[0] for f in jpg_files}
    txt_basenames = {os.path.splitext(f)[0] for f in txt_files}
    
    # 找出沒有配對的文件
    jpg_without_txt = jpg_basenames - txt_basenames
    txt_without_jpg = txt_basenames - jpg_basenames
    
    print(f"\n🔄 配對檢查結果:")
    if jpg_without_txt:
        print(f"   ❌ 沒有 TXT 文件的 JPG: {len(jpg_without_txt)} 個")
        for basename in sorted(jpg_without_txt):
            print(f"      - {basename}")
    
    if txt_without_jpg:
        print(f"   ❌ 沒有 JPG 文件的 TXT: {len(txt_without_jpg)} 個")
        for basename in sorted(txt_without_jpg):
            print(f"      - {basename}")
    
    # 修正策略
    fixed_count = 0
    
    # 為沒有 TXT 的 JPG 創建空白 TXT 文件
    if jpg_without_txt:
        print(f"\n🔧 為 {len(jpg_without_txt)} 個 JPG 文件創建空白 TXT 文件...")
        for basename in jpg_without_txt:
            # 找到對應的 JPG 文件（可能是 .jpg 或 .jpeg）
            jpg_file = None
            for ext in ['.jpg', '.jpeg']:
                if os.path.exists(os.path.join(data_folder, basename + ext)):
                    jpg_file = basename + ext
                    break
            
            if jpg_file:
                txt_path = os.path.join(data_folder, basename + '.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write('test')  # 使用 'test' 作為默認 caption
                print(f"   ✅ 創建: {basename}.txt")
                fixed_count += 1
    
    # 刪除沒有 JPG 的 TXT 文件
    if txt_without_jpg:
        print(f"\n🗑️ 刪除 {len(txt_without_jpg)} 個沒有對應 JPG 的 TXT 文件...")
        for basename in txt_without_jpg:
            txt_path = os.path.join(data_folder, basename + '.txt')
            if os.path.exists(txt_path):
                os.remove(txt_path)
                print(f"   🗑️ 刪除: {basename}.txt")
                fixed_count += 1
    
    # 最終檢查
    all_files = os.listdir(data_folder)
    final_jpg = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))]
    final_txt = [f for f in all_files if f.lower().endswith('.txt')]
    
    print(f"\n✅ 修正完成:")
    print(f"   最終 JPG 文件: {len(final_jpg)}")
    print(f"   最終 TXT 文件: {len(final_txt)}")
    print(f"   修正操作數: {fixed_count}")
    
    if len(final_jpg) == len(final_txt):
        print(f"🎯 成功! JPG 和 TXT 文件數量一致")
        return True
    else:
        print(f"⚠️ 警告: JPG 和 TXT 文件數量仍不一致")
        return False

def main():
    """主函數"""
    data_folder = "lora_train_set/10_test"
    
    print("🛠️ 訓練資料配對修正工具")
    print("=" * 50)
    
    success = check_and_fix_training_data(data_folder)
    
    if success:
        print("\n🎉 資料修正完成! 現在可以重新嘗試訓練")
    else:
        print("\n❌ 資料修正過程中出現問題")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
