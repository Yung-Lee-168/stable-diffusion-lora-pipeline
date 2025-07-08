#!/usr/bin/env python3
"""
測試 generate_caption_fashionclip.py 的目錄設定
"""

import os
import sys

def test_directory_setup():
    """測試目錄設定"""
    print("🧪 測試 generate_caption_fashionclip.py 目錄設定")
    print("=" * 60)
    
    # 檢查當前目錄
    current_dir = os.getcwd()
    print(f"📁 當前工作目錄: {current_dir}")
    
    # 檢查是否在 auto_test_pipeline 目錄
    if current_dir.endswith("auto_test_pipeline"):
        print("✅ 正確：在 auto_test_pipeline 目錄中")
    else:
        print("❌ 錯誤：不在 auto_test_pipeline 目錄中")
        return False
    
    # 檢查必要的檔案和目錄
    required_items = {
        "特徵值.py": "檔案",
        "generate_caption_fashionclip.py": "檔案", 
        "source_image": "目錄",
        "lora_train_set": "目錄"
    }
    
    print(f"\n📋 檢查必要的檔案和目錄:")
    all_exists = True
    
    for item, item_type in required_items.items():
        exists = os.path.exists(item)
        status = "✅" if exists else "❌"
        print(f"  {status} {item} ({item_type})")
        
        if exists:
            abs_path = os.path.abspath(item)
            print(f"     → {abs_path}")
            
            # 如果是 source_image 目錄，檢查內容
            if item == "source_image" and os.path.isdir(item):
                images = [f for f in os.listdir(item) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"     → 包含 {len(images)} 張圖片")
        else:
            all_exists = False
    
    # 檢查目標目錄
    target_dir = "lora_train_set/10_test"
    if os.path.exists(target_dir):
        print(f"\n✅ 目標目錄存在: {os.path.abspath(target_dir)}")
    else:
        print(f"\n💡 目標目錄將自動創建: {os.path.abspath(target_dir)}")
    
    return all_exists

def test_feature_import():
    """測試特徵值.py 匯入"""
    print(f"\n🧪 測試特徵值.py 匯入:")
    
    try:
        import 特徵值
        print("✅ 成功匯入 特徵值.py")
        
        # 檢查特徵值內容
        feature_count = 0
        for k, v in 特徵值.__dict__.items():
            if isinstance(v, (list, tuple)):
                feature_count += len(v)
                print(f"  📊 {k}: {len(v)} 個特徵")
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (list, tuple)):
                        feature_count += len(vv)
                        print(f"  📊 {k}.{kk}: {len(vv)} 個特徵")
        
        print(f"  🎯 總共 {feature_count} 個特徵值")
        return True
        
    except Exception as e:
        print(f"❌ 匯入失敗: {e}")
        return False

def main():
    """主函數"""
    print("🎯 generate_caption_fashionclip.py 環境測試")
    print("基準：所有檔案都在 auto_test_pipeline 目錄")
    print("=" * 60)
    
    # 測試目錄設定
    dir_ok = test_directory_setup()
    
    # 測試特徵值匯入
    import_ok = test_feature_import()
    
    # 總結
    print(f"\n📊 測試結果:")
    print(f"  目錄設定: {'✅ 通過' if dir_ok else '❌ 失敗'}")
    print(f"  特徵值匯入: {'✅ 通過' if import_ok else '❌ 失敗'}")
    
    if dir_ok and import_ok:
        print(f"\n🎉 環境設定完成！")
        print(f"💡 現在可以執行: python generate_caption_fashionclip.py")
    else:
        print(f"\n⚠️  環境設定需要修復")
    
    return dir_ok and import_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
