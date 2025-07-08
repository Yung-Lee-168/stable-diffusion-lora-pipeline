#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: Fine-tuning Configuration Manager
微調配置管理器 - 支持不同的訓練策略和參數組合
"""

import json
import os
from datetime import datetime

class FineTuningConfig:
    """微調配置類"""
    
    def __init__(self):
        self.configs = {
            # 快速測試配置 - 適合初始驗證
            "quick_test": {
                "learning_rate": 5e-4,
                "batch_size": 1,
                "num_epochs": 5,
                "save_steps": 2,
                "validation_steps": 2,
                "max_grad_norm": 1.0,
                "use_lora": True,
                "lora_rank": 4,
                "lora_alpha": 32,
                "image_size": 512,
                "mixed_precision": True,
                "description": "快速測試配置，適合驗證訓練流程"
            },
            
            # 標準訓練配置 - 平衡效果和速度
            "standard": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "num_epochs": 50,
                "save_steps": 10,
                "validation_steps": 5,
                "max_grad_norm": 1.0,
                "use_lora": True,
                "lora_rank": 8,
                "lora_alpha": 32,
                "image_size": 512,
                "mixed_precision": True,
                "description": "標準訓練配置，平衡效果和訓練時間"
            },
            
            # 高品質配置 - 追求最佳效果
            "high_quality": {
                "learning_rate": 5e-5,
                "batch_size": 1,
                "num_epochs": 100,
                "save_steps": 20,
                "validation_steps": 10,
                "max_grad_norm": 1.0,
                "use_lora": True,
                "lora_rank": 16,
                "lora_alpha": 64,
                "image_size": 768,
                "mixed_precision": True,
                "description": "高品質配置，追求最佳生成效果"
            },
            
            # 全量微調配置 - 完整模型訓練
            "full_finetuning": {
                "learning_rate": 1e-5,
                "batch_size": 1,
                "num_epochs": 30,
                "save_steps": 5,
                "validation_steps": 3,
                "max_grad_norm": 1.0,
                "use_lora": False,  # 不使用 LoRA，訓練全部參數
                "lora_rank": None,
                "lora_alpha": None,
                "image_size": 512,
                "mixed_precision": True,
                "description": "全量微調，訓練完整 UNet 參數（需要大量 GPU 記憶體）"
            },
            
            # DreamBooth 風格配置
            "dreambooth": {
                "learning_rate": 2e-6,
                "batch_size": 1,
                "num_epochs": 200,
                "save_steps": 50,
                "validation_steps": 25,
                "max_grad_norm": 1.0,
                "use_lora": True,
                "lora_rank": 4,
                "lora_alpha": 16,
                "image_size": 512,
                "mixed_precision": True,
                "prior_preservation": True,  # DreamBooth 特有
                "class_prompt": "fashion outfit",
                "description": "DreamBooth 風格配置，適合個性化風格學習"
            }
        }
    
    def get_config(self, config_name="standard"):
        """獲取指定配置"""
        if config_name not in self.configs:
            print(f"⚠️  配置 '{config_name}' 不存在，使用默認 'standard' 配置")
            config_name = "standard"
        
        config = self.configs[config_name].copy()
        config["config_name"] = config_name
        return config
    
    def list_configs(self):
        """列出所有可用配置"""
        print("📋 可用的訓練配置:")
        print("=" * 50)
        
        for name, config in self.configs.items():
            print(f"🔧 {name}")
            print(f"   描述: {config['description']}")
            print(f"   學習率: {config['learning_rate']}")
            print(f"   Epochs: {config['num_epochs']}")
            print(f"   LoRA: {'是' if config['use_lora'] else '否'}")
            print()
    
    def save_config(self, config, save_path):
        """保存配置到文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"💾 配置已保存: {save_path}")
    
    def create_custom_config(self, base_config="standard", **kwargs):
        """創建自定義配置"""
        config = self.get_config(base_config)
        config.update(kwargs)
        config["config_name"] = "custom"
        config["description"] = f"基於 {base_config} 的自定義配置"
        return config

class TrainingValidator:
    """訓練參數驗證器"""
    
    @staticmethod
    def validate_config(config):
        """驗證配置參數"""
        errors = []
        warnings = []
        
        # 檢查必需參數
        required_params = [
            "learning_rate", "batch_size", "num_epochs", 
            "save_steps", "validation_steps", "image_size"
        ]
        
        for param in required_params:
            if param not in config:
                errors.append(f"缺少必需參數: {param}")
        
        # 檢查參數合理性
        if config.get("learning_rate", 0) <= 0:
            errors.append("學習率必須大於 0")
        
        if config.get("learning_rate", 0) > 1e-2:
            warnings.append("學習率可能過高，建議 < 1e-2")
        
        if config.get("batch_size", 0) <= 0:
            errors.append("批次大小必須大於 0")
        
        if config.get("num_epochs", 0) <= 0:
            errors.append("訓練 epochs 必須大於 0")
        
        if config.get("image_size", 0) not in [512, 768, 1024]:
            warnings.append("圖片尺寸建議使用 512, 768 或 1024")
        
        # LoRA 參數檢查
        if config.get("use_lora", False):
            if config.get("lora_rank", 0) <= 0:
                errors.append("LoRA rank 必須大於 0")
            
            if config.get("lora_rank", 0) > 64:
                warnings.append("LoRA rank 過高可能影響訓練效果")
        
        # GPU 記憶體估算
        memory_estimate = TrainingValidator.estimate_gpu_memory(config)
        if memory_estimate > 24:  # GB
            warnings.append(f"預估 GPU 記憶體需求: {memory_estimate:.1f} GB，可能超出限制")
        
        return errors, warnings
    
    @staticmethod
    def estimate_gpu_memory(config):
        """估算 GPU 記憶體需求"""
        base_memory = 4.0  # SD v1.5 基礎記憶體 (GB)
        
        # 圖片尺寸影響
        size_factor = (config.get("image_size", 512) / 512) ** 2
        image_memory = 2.0 * size_factor
        
        # 批次大小影響
        batch_memory = config.get("batch_size", 1) * 1.5
        
        # LoRA vs 全量微調
        if config.get("use_lora", True):
            training_memory = 1.0  # LoRA 記憶體較少
        else:
            training_memory = 6.0  # 全量微調記憶體較多
        
        # 混合精度
        if not config.get("mixed_precision", True):
            training_memory *= 1.5
        
        total_memory = base_memory + image_memory + batch_memory + training_memory
        return total_memory

def main():
    """主函數 - 配置管理演示"""
    config_manager = FineTuningConfig()
    validator = TrainingValidator()
    
    print("🚀 Fine-tuning 配置管理器")
    print("=" * 50)
    
    # 列出所有配置
    config_manager.list_configs()
    
    # 測試配置驗證
    print("🧪 配置驗證測試:")
    for config_name in ["quick_test", "standard", "high_quality"]:
        config = config_manager.get_config(config_name)
        errors, warnings = validator.validate_config(config)
        memory = validator.estimate_gpu_memory(config)
        
        print(f"\n📋 {config_name} 配置:")
        print(f"   GPU 記憶體估算: {memory:.1f} GB")
        
        if errors:
            print(f"   ❌ 錯誤: {', '.join(errors)}")
        if warnings:
            print(f"   ⚠️  警告: {', '.join(warnings)}")
        if not errors and not warnings:
            print("   ✅ 配置有效")
    
    # 創建自定義配置示例
    print(f"\n🔧 自定義配置示例:")
    custom_config = config_manager.create_custom_config(
        base_config="standard",
        learning_rate=2e-4,
        num_epochs=30,
        lora_rank=12
    )
    
    errors, warnings = validator.validate_config(custom_config)
    memory = validator.estimate_gpu_memory(custom_config)
    
    print(f"   學習率: {custom_config['learning_rate']}")
    print(f"   Epochs: {custom_config['num_epochs']}")
    print(f"   LoRA Rank: {custom_config['lora_rank']}")
    print(f"   GPU 記憶體估算: {memory:.1f} GB")
    
    if errors:
        print(f"   ❌ 錯誤: {', '.join(errors)}")
    if warnings:
        print(f"   ⚠️  警告: {', '.join(warnings)}")

if __name__ == "__main__":
    main()
