#!/usr/bin/env python3
"""
LoRA 訓練參數優化器 - 根據圖片數量自動調整訓練參數
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any

class LoRATrainingOptimizer:
    def __init__(self):
        self.base_params = {
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "output_dir": "lora_output",
            "train_data_dir": "lora_train_set",
            "resolution": 512,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "dataloader_num_workers": 0,
            "num_train_epochs": 1,
            "max_train_steps": 200,
            "learning_rate": 1e-4,
            "scale_lr": False,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 0,
            "snr_gamma": 5.0,
            "use_8bit_adam": True,
            "mixed_precision": "fp16",
            "save_precision": "fp16",
            "enable_xformers_memory_efficient_attention": True,
            "cache_latents": True,
            "save_model_as": "safetensors",
            "network_module": "networks.lora",
            "network_dim": 32,
            "network_alpha": 32,
            "network_train_unet_only": True,
            "network_train_text_encoder_only": False,
            "save_every_n_epochs": 1
        }
    
    def calculate_optimal_params(self, image_count: int, target_time_minutes: int = 120) -> Dict[str, Any]:
        """根據圖片數量和目標時間計算最優參數"""
        params = self.base_params.copy()
        
        # 根據圖片數量調整參數
        if image_count <= 20:
            # 小數據集：增加訓練步數
            params["max_train_steps"] = 300
            params["learning_rate"] = 1e-4
            params["network_dim"] = 32
            recommendation = "小數據集，增加訓練步數以充分學習"
            
        elif image_count <= 50:
            # 中等數據集：標準參數
            params["max_train_steps"] = 200
            params["learning_rate"] = 1e-4
            params["network_dim"] = 32
            recommendation = "中等數據集，使用標準參數"
            
        elif image_count <= 100:
            # 大數據集：減少訓練步數
            params["max_train_steps"] = 150
            params["learning_rate"] = 5e-5
            params["network_dim"] = 32
            recommendation = "大數據集，減少訓練步數避免過擬合"
            
        elif image_count <= 200:
            # 超大數據集：進一步減少訓練步數
            params["max_train_steps"] = 100
            params["learning_rate"] = 5e-5
            params["network_dim"] = 32
            recommendation = "超大數據集，大幅減少訓練步數"
            
        else:
            # 海量數據集：建議分批訓練
            params["max_train_steps"] = 50
            params["learning_rate"] = 2e-5
            params["network_dim"] = 32
            recommendation = "海量數據集，強烈建議分批訓練"
        
        # 根據目標時間進一步調整
        if target_time_minutes <= 60:
            # 快速訓練模式
            params["max_train_steps"] = min(params["max_train_steps"], 100)
            params["network_dim"] = 16  # 減少參數量
            recommendation += " + 快速訓練模式"
            
        elif target_time_minutes <= 120:
            # 標準訓練模式
            pass  # 保持當前參數
            
        else:
            # 深度訓練模式
            params["max_train_steps"] = min(params["max_train_steps"] + 50, 400)
            params["network_dim"] = 64  # 增加參數量
            recommendation += " + 深度訓練模式"
        
        # 計算預估訓練時間
        estimated_time = self.estimate_training_time(image_count, params["max_train_steps"])
        
        return {
            "params": params,
            "image_count": image_count,
            "estimated_time_minutes": estimated_time,
            "recommendation": recommendation,
            "fits_in_timeout": estimated_time <= 240  # 4小時限制
        }
    
    def estimate_training_time(self, image_count: int, max_train_steps: int) -> int:
        """估算訓練時間（分鐘）"""
        # 基礎時間：每個step約需要 0.5-2 秒，取決於圖片數量
        if image_count <= 20:
            seconds_per_step = 0.5
        elif image_count <= 50:
            seconds_per_step = 1.0
        elif image_count <= 100:
            seconds_per_step = 1.5
        else:
            seconds_per_step = 2.0
        
        # 計算總時間
        total_seconds = max_train_steps * seconds_per_step
        
        # 加上初始化和保存時間
        overhead_seconds = 60 + (image_count * 2)  # 初始化 + 每張圖片2秒處理時間
        
        total_time_minutes = (total_seconds + overhead_seconds) / 60
        
        # 加上50%緩衝時間
        return int(total_time_minutes * 1.5)
    
    def generate_training_config(self, image_count: int, target_time_minutes: int = 120) -> str:
        """生成訓練配置文件"""
        config = self.calculate_optimal_params(image_count, target_time_minutes)
        
        # 生成配置文件內容
        config_content = f"""# LoRA 訓練優化配置
# 圖片數量: {config['image_count']}
# 預估時間: {config['estimated_time_minutes']} 分鐘
# 推薦策略: {config['recommendation']}
# 適合超時限制: {'✅ 是' if config['fits_in_timeout'] else '❌ 否，建議分批訓練'}

import json

TRAINING_CONFIG = {json.dumps(config['params'], indent=4)}

def get_training_args():
    return TRAINING_CONFIG
"""
        
        return config_content
    
    def save_config(self, image_count: int, target_time_minutes: int = 120, 
                   output_file: str = "lora_training_config.py") -> str:
        """保存配置到文件"""
        config_content = self.generate_training_config(image_count, target_time_minutes)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        return output_file

def main():
    parser = argparse.ArgumentParser(description="LoRA 訓練參數優化器")
    parser.add_argument("image_count", type=int, help="訓練圖片數量")
    parser.add_argument("--target-time", type=int, default=120, help="目標訓練時間（分鐘）")
    parser.add_argument("--output", default="lora_training_config.py", help="輸出配置文件名")
    parser.add_argument("--analyze-only", action="store_true", help="只分析不生成配置文件")
    
    args = parser.parse_args()
    
    optimizer = LoRATrainingOptimizer()
    config = optimizer.calculate_optimal_params(args.image_count, args.target_time)
    
    print("🔧 LoRA 訓練參數優化分析")
    print("=" * 50)
    print(f"📊 圖片數量: {config['image_count']}")
    print(f"⏱️  目標時間: {args.target_time} 分鐘")
    print(f"📈 預估時間: {config['estimated_time_minutes']} 分鐘")
    print(f"💡 推薦策略: {config['recommendation']}")
    print(f"✅ 適合超時: {'是' if config['fits_in_timeout'] else '否'}")
    
    print("\n🎯 優化後的關鍵參數:")
    key_params = ['max_train_steps', 'learning_rate', 'network_dim', 'network_alpha']
    for param in key_params:
        if param in config['params']:
            print(f"  {param}: {config['params'][param]}")
    
    if not config['fits_in_timeout']:
        print("\n⚠️  警告：預估時間超過4小時限制")
        print("📋 建議方案：")
        print("  1. 減少圖片數量")
        print("  2. 使用分批訓練")
        print("  3. 進一步減少 max_train_steps")
    
    if not args.analyze_only:
        config_file = optimizer.save_config(args.image_count, args.target_time, args.output)
        print(f"\n📄 配置文件已保存: {config_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())
