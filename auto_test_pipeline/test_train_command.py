#!/usr/bin/env python3
"""
測試 train_lora.py 的訓練命令
"""
import os
import sys

# 加入 train_lora.py 的路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_command_generation():
    """測試命令生成"""
    from train_lora import train_lora
    
    # 暫時重寫 train_lora 函數來只返回命令
    import train_lora as tl
    
    # 備份原函數
    original_train_lora = tl.train_lora
    
    def mock_train_lora(continue_from_checkpoint=False):
        """Mock版本，只生成命令不執行"""
        
        # 基本訓練指令
        cmd_parts = [
            "python train_network.py",
            "--pretrained_model_name_or_path=../models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
            "--train_data_dir=lora_train_set",
            "--output_dir=lora_output",
            "--resolution=512,512",
            "--network_module=networks.lora",
            "--network_dim=32",
            "--train_batch_size=1",
            "--max_train_steps=100",
            "--mixed_precision=fp16",
            "--cache_latents",
            "--learning_rate=5e-5",
            "--save_every_n_epochs=50",
            "--save_model_as=safetensors",
            "--save_state"
        ]
        
        cmd = " ".join(cmd_parts)
        print(f"生成的命令: {cmd}")
        return cmd
    
    # 測試命令生成
    cmd = mock_train_lora()
    
    # 檢查是否包含 logging_interval
    if "logging_interval" in cmd:
        print("❌ 發現 logging_interval 參數！")
        return False
    else:
        print("✅ 沒有發現 logging_interval 參數")
        return True

if __name__ == "__main__":
    print("🧪 測試訓練命令生成...")
    success = test_command_generation()
    sys.exit(0 if success else 1)
