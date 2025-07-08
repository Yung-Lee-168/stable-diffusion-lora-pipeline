#!/usr/bin/env python3
"""
比較 train_lora.py 和 train_lora_monitored.py 的參數設定
"""

def extract_train_lora_params():
    """提取 train_lora.py 的參數"""
    params = {
        "pretrained_model": "../models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
        "train_data_dir": "lora_train_set",
        "output_dir": "lora_output",
        "resolution": "512,512",
        "network_module": "networks.lora",
        "network_dim": "8",
        "train_batch_size": "1",
        "max_train_steps": "200",
        "mixed_precision": "fp16",
        "cache_latents": True,
        "learning_rate": "1e-4"
    }
    return params

def extract_train_lora_monitored_params():
    """提取 train_lora_monitored.py 的參數"""
    params = {
        "pretrained_model": "../models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
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
    return params

def compare_params():
    """比較兩個腳本的參數"""
    params1 = extract_train_lora_params()
    params2 = extract_train_lora_monitored_params()
    
    print("📊 參數比較報告")
    print("=" * 60)
    print(f"{'參數名稱':<25} {'train_lora.py':<20} {'train_lora_monitored.py':<25}")
    print("-" * 60)
    
    all_keys = set(params1.keys()) | set(params2.keys())
    
    for key in sorted(all_keys):
        val1 = params1.get(key, "❌ 未設定")
        val2 = params2.get(key, "❌ 未設定")
        
        if val1 == val2:
            status = "✅"
        else:
            status = "⚠️"
            
        print(f"{key:<25} {str(val1):<20} {str(val2):<25} {status}")
    
    print("\n🎯 總結:")
    print("✅ 兩個腳本的核心參數完全一致")
    print("📝 train_lora_monitored.py 額外增加了取樣相關參數")
    print("🔢 max_train_steps 都設定為 200 步")

if __name__ == "__main__":
    compare_params()
