#!/usr/bin/env python3
"""
測試 train_lora_monitored.py 的圖片尺寸檢查功能
"""

from train_lora_monitored import LoRATrainer

def test_image_size_check():
    """測試圖片尺寸檢查功能"""
    print("🧪 測試 train_lora_monitored.py 圖片尺寸檢查功能")
    print("=" * 60)
    
    trainer = LoRATrainer()
    
    # 測試圖片尺寸檢查
    result = trainer.check_image_sizes('lora_train_set/10_test')
    
    print(f"\n🎯 測試結果：{'✅ 通過' if result else '❌ 失敗'}")
    
    # 測試完整的訓練需求檢查
    print("\n🔄 測試完整的訓練需求檢查...")
    requirements_ok = trainer.check_training_requirements()
    
    print(f"🎯 訓練需求檢查結果：{'✅ 通過' if requirements_ok else '❌ 失敗'}")
    
    return result and requirements_ok

if __name__ == "__main__":
    success = test_image_size_check()
    print(f"\n🏁 總體測試結果：{'✅ 成功' if success else '❌ 失敗'}")
