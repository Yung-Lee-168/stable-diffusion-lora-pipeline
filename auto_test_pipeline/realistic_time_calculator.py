#!/usr/bin/env python3
"""
基於實際測試結果的 LoRA 訓練時間計算器
實際測試基準：10張圖片 + 200步 = 30分鐘
"""

import argparse
import math

def calculate_realistic_training_time(image_count: int, train_steps: int) -> dict:
    """
    基於實際測試數據計算訓練時間
    基準：10張圖片 + 200步 = 30分鐘
    """
    # 基準數據
    base_images = 10
    base_steps = 200
    base_time_minutes = 30
    
    # 計算每張圖片每步的時間
    time_per_image_per_step = base_time_minutes / (base_images * base_steps)
    
    # 計算總時間
    total_time_minutes = image_count * train_steps * time_per_image_per_step
    
    # 加上50%緩衝時間
    buffered_time_minutes = total_time_minutes * 1.5
    
    # 超時限制（4小時 = 240分鐘）
    timeout_limit_minutes = 240
    will_timeout = buffered_time_minutes > timeout_limit_minutes
    
    # 建議的批次大小（如果會超時）
    max_images_per_batch = None
    if will_timeout:
        # 計算在超時限制內最多能處理多少張圖片
        max_time_per_batch = timeout_limit_minutes / 1.5  # 去掉緩衝時間
        max_images_per_batch = int(max_time_per_batch / (train_steps * time_per_image_per_step))
    
    return {
        "image_count": image_count,
        "train_steps": train_steps,
        "base_time_minutes": total_time_minutes,
        "buffered_time_minutes": buffered_time_minutes,
        "timeout_limit_minutes": timeout_limit_minutes,
        "will_timeout": will_timeout,
        "max_images_per_batch": max_images_per_batch,
        "time_per_image_per_step": time_per_image_per_step
    }

def print_analysis(result: dict):
    """打印分析結果"""
    print(f"📊 訓練時間分析 (基於實際測試: 10張圖片+200步=30分鐘)")
    print("=" * 60)
    print(f"🖼️  圖片數量: {result['image_count']} 張")
    print(f"🔢 訓練步數: {result['train_steps']} 步")
    print(f"⏱️  基礎時間: {result['base_time_minutes']:.1f} 分鐘")
    print(f"🛡️  緩衝時間: {result['buffered_time_minutes']:.1f} 分鐘 (含50%緩衝)")
    print(f"⏰ 超時限制: {result['timeout_limit_minutes']} 分鐘")
    
    if result['will_timeout']:
        print(f"⚠️  結果: 會超時 (超出 {result['buffered_time_minutes'] - result['timeout_limit_minutes']:.1f} 分鐘)")
        print(f"💡 建議: 分批訓練，每批最多 {result['max_images_per_batch']} 張圖片")
    else:
        print(f"✅ 結果: 不會超時")
    
    print()

def main():
    parser = argparse.ArgumentParser(description="基於實際測試的 LoRA 訓練時間計算器")
    parser.add_argument("image_count", type=int, help="圖片數量")
    parser.add_argument("--steps", type=int, nargs='+', default=[200, 150, 100], 
                       help="訓練步數列表 (默認: 200 150 100)")
    
    args = parser.parse_args()
    
    print(f"🔍 分析 {args.image_count} 張圖片的訓練時間")
    print(f"📋 測試步數: {args.steps}")
    print()
    
    for steps in args.steps:
        result = calculate_realistic_training_time(args.image_count, steps)
        print_analysis(result)
    
    # 特別分析：找出最佳步數
    print("🎯 最佳配置建議:")
    print("=" * 60)
    
    best_config = None
    for steps in args.steps:
        result = calculate_realistic_training_time(args.image_count, steps)
        if not result['will_timeout']:
            if best_config is None or steps > best_config['train_steps']:
                best_config = result
    
    if best_config:
        print(f"✅ 建議配置: {best_config['train_steps']} 步")
        print(f"   預估時間: {best_config['buffered_time_minutes']:.1f} 分鐘")
        print(f"   安全餘量: {best_config['timeout_limit_minutes'] - best_config['buffered_time_minutes']:.1f} 分鐘")
    else:
        print("⚠️  所有配置都會超時，建議分批訓練")
        # 找出最接近的配置
        min_steps = min(args.steps)
        result = calculate_realistic_training_time(args.image_count, min_steps)
        print(f"   最小步數 {min_steps} 步仍需 {result['buffered_time_minutes']:.1f} 分鐘")
        print(f"   建議每批最多 {result['max_images_per_batch']} 張圖片")

if __name__ == "__main__":
    main()
