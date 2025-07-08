#!/usr/bin/env python3
"""
比較兩個 LoRA 訓練腳本的參數設定
"""

import re

# 從 train_lora.py 提取的參數
train_lora_params = {
    "pretrained_model_name_or_path": "../models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
    "train_data_dir": "lora_train_set",
    "output_dir": "lora_output",
    "resolution": "512,512",
    "network_module": "networks.lora",
    "network_dim": "8",
    "train_batch_size": "1",
    "max_train_steps": "200",
    "mixed_precision": "fp16",
    "cache_latents": True,
    "learning_rate": "1e-4",
    "sample_every_n_steps": None,
    "sample_sampler": None
}

# 從 train_lora_monitored.py 提取的參數
train_lora_monitored_params = {
    "pretrained_model_name_or_path": "../models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
    "train_data_dir": "lora_train_set",
    "output_dir": "lora_output",
    "resolution": "512,512",
    "network_module": "networks.lora",
    "network_dim": "8",
    "train_batch_size": "1",
    "max_train_steps": "200",
    "mixed_precision": "fp16",
    "cache_latents": True,
    "learning_rate": "1e-4",
    "sample_every_n_steps": "100",
    "sample_sampler": "euler_a"
}

def compare_params():
    """比較兩個腳本的參數"""
    print("🔍 LoRA 訓練腳本參數比較")
    print("=" * 80)
    
    all_keys = set(train_lora_params.keys()) | set(train_lora_monitored_params.keys())
    
    print(f"{'參數名稱':<30} {'train_lora.py':<25} {'train_lora_monitored.py':<25} {'狀態':<10}")
    print("-" * 90)
    
    differences = []
    
    for key in sorted(all_keys):
        val1 = train_lora_params.get(key)
        val2 = train_lora_monitored_params.get(key)
        
        if val1 == val2:
            status = "✅ 相同"
        elif val1 is None and val2 is not None:
            status = "➕ 新增"
            differences.append(f"{key}: monitored 版本新增 {val2}")
        elif val1 is not None and val2 is None:
            status = "➖ 移除"
            differences.append(f"{key}: monitored 版本移除")
        else:
            status = "❌ 不同"
            differences.append(f"{key}: {val1} -> {val2}")
        
        print(f"{key:<30} {str(val1):<25} {str(val2):<25} {status}")
    
    print("\n" + "=" * 80)
    print("📋 差異總結:")
    
    if not differences:
        print("🎉 兩個腳本的核心參數完全一致！")
    else:
        print(f"發現 {len(differences)} 個差異:")
        for diff in differences:
            print(f"  - {diff}")
    
    # 檢查額外功能
    print("\n📊 額外功能比較:")
    print("train_lora.py:")
    print("  - 基本訓練功能")
    print("  - 圖片尺寸檢查")
    print("  - 簡單的訓練指令執行")
    
    print("\ntrain_lora_monitored.py:")
    print("  - 基本訓練功能")
    print("  - 圖片尺寸檢查")
    print("  - 訓練進度監控")
    print("  - 訓練結果評估")
    print("  - 詳細日誌記錄")
    print("  - 訓練報告生成")
    print("  - 自動決策是否繼續推理")
    print("  - 採樣預覽功能 (sample_every_n_steps)")

if __name__ == "__main__":
    compare_params()
