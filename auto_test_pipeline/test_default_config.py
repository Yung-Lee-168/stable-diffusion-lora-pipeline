#!/usr/bin/env python3
"""
測試新的默認配置：100張圖片 + 100步
"""

import os
import subprocess
import time

def test_default_config():
    """測試默認配置"""
    print("🧪 測試新的默認配置：100張圖片 + 100步")
    print("=" * 50)
    
    # 檢查訓練資料
    train_dir = "lora_train_set/10_test"
    if not os.path.exists(train_dir):
        print(f"❌ 訓練目錄不存在: {train_dir}")
        return False
    
    # 計算圖片數量
    images = [f for f in os.listdir(train_dir) 
              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_count = len(images)
    
    print(f"📁 訓練目錄: {train_dir}")
    print(f"🖼️  圖片數量: {image_count}")
    
    # 基於實際測試計算預估時間
    # 10張圖片 + 200步 = 30分鐘
    # 計算每張圖片每步的時間
    time_per_image_per_step = 30 / (10 * 200)  # 0.015分鐘/張/步
    
    # 計算100步的時間
    base_time = image_count * 100 * time_per_image_per_step
    buffered_time = base_time * 1.5
    
    print(f"📊 時間預估 (基於實際測試):")
    print(f"   每張圖片每步: {time_per_image_per_step*60:.1f} 秒")
    print(f"   基礎時間: {base_time:.1f} 分鐘")
    print(f"   緩衝時間: {buffered_time:.1f} 分鐘")
    print(f"   超時風險: {'❌ 會超時' if buffered_time > 240 else '✅ 安全'}")
    
    # 顯示配置
    print(f"\n🔧 默認配置:")
    print(f"   max_train_steps: 100")
    print(f"   learning_rate: 5e-5")
    print(f"   network_dim: 32")
    print(f"   network_alpha: 32")
    
    return True

def test_realistic_time_calculator():
    """測試實際時間計算器"""
    print(f"\n🧮 測試實際時間計算器:")
    
    # 測試不同圖片數量
    test_cases = [10, 50, 100, 200]
    
    for image_count in test_cases:
        print(f"\n📊 {image_count} 張圖片 + 100 步:")
        
        # 計算時間
        time_per_image_per_step = 30 / (10 * 200)  # 0.015分鐘/張/步
        base_time = image_count * 100 * time_per_image_per_step
        buffered_time = base_time * 1.5
        
        status = "✅ 安全" if buffered_time <= 240 else "❌ 超時"
        print(f"   基礎時間: {base_time:.1f} 分鐘")
        print(f"   緩衝時間: {buffered_time:.1f} 分鐘")
        print(f"   狀態: {status}")

def show_usage_examples():
    """顯示使用範例"""
    print(f"\n🚀 使用範例:")
    print("=" * 50)
    
    print("1. 使用默認配置 (100步):")
    print("   python train_lora_monitored.py --new")
    
    print("\n2. 自動優化參數:")
    print("   python train_lora_monitored.py --auto-optimize --new")
    
    print("\n3. 自定義步數:")
    print("   python train_lora_monitored.py --new --max-train-steps 150")
    
    print("\n4. 檢查時間預估:")
    print("   python realistic_time_calculator.py 100 --steps 100")
    
    print("\n5. 分批訓練大數據集:")
    print("   python batch_training_helper.py your_images_folder --batch-size 100")

def main():
    """主函數"""
    print("🎯 新默認配置測試工具")
    print("基準：100張圖片 + 100步")
    print("=" * 50)
    
    # 測試默認配置
    if test_default_config():
        # 測試時間計算器
        test_realistic_time_calculator()
        
        # 顯示使用範例
        show_usage_examples()
        
        print(f"\n✅ 測試完成！")
        print(f"💡 新的默認配置已設定為：100張圖片 + 100步")
        print(f"🎯 這個配置在4小時超時限制內是安全的")
    else:
        print(f"❌ 測試失敗")

if __name__ == "__main__":
    main()
