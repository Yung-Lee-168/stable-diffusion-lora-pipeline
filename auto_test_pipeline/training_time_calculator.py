#!/usr/bin/env python3
"""
LoRA 訓練時間預估工具
根據圖片數量計算訓練時間和超時設定
"""

def calculate_training_timeout(image_count: int) -> dict:
    """計算訓練參數"""
    # 根據經驗：10張圖片 = 30分鐘
    base_time_per_image = 3 * 60  # 3分鐘/張
    buffer_multiplier = 1.5  # 50%緩衝
    
    estimated_time = int(image_count * base_time_per_image * buffer_multiplier)
    
    # 設定最小和最大超時時間
    min_timeout = 1800  # 最少30分鐘
    max_timeout = 14400  # 最多4小時
    
    timeout = max(min_timeout, min(estimated_time, max_timeout))
    
    return {
        "image_count": image_count,
        "estimated_minutes": estimated_time / 60,
        "timeout_minutes": timeout / 60,
        "timeout_seconds": timeout,
        "will_timeout": estimated_time > max_timeout
    }

def show_timeout_table():
    """顯示不同圖片數量的超時設定表"""
    print("📊 LoRA 訓練時間預估表")
    print("=" * 80)
    print(f"{'圖片數量':<10} {'預估時間':<12} {'超時設定':<12} {'狀態':<15} {'建議'}")
    print("-" * 80)
    
    test_counts = [10, 20, 50, 100, 150, 200, 300, 500]
    
    for count in test_counts:
        result = calculate_training_timeout(count)
        
        if result["will_timeout"]:
            status = "⚠️  會超時"
            suggestion = "需要分批訓練"
        elif result["estimated_minutes"] > 180:  # 超過3小時
            status = "⏱️ 時間較長"
            suggestion = "建議減少圖片"
        else:
            status = "✅ 正常"
            suggestion = "可以直接訓練"
        
        print(f"{count:<10} {result['estimated_minutes']:<12.1f} {result['timeout_minutes']:<12.1f} {status:<15} {suggestion}")
    
    print("\n💡 說明：")
    print("• 預估時間：基於 10張圖片=30分鐘 的經驗值")
    print("• 超時設定：預估時間 × 1.5 (50%緩衝)")
    print("• 最大超時：4小時 (14400秒)")
    print("• 最小超時：30分鐘 (1800秒)")
    
    print("\n🚀 優化建議：")
    print("• 100張以下：可以直接訓練")
    print("• 100-200張：考慮分批或調整參數")
    print("• 200張以上：建議分批訓練或減少 max_train_steps")

if __name__ == "__main__":
    show_timeout_table()
