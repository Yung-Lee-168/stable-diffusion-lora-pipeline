# LoRA 訓練優化配置
# 圖片數量: 200
# 預估時間: 19 分鐘
# 推薦策略: 超大數據集，大幅減少訓練步數 + 深度訓練模式
# 適合超時限制: ✅ 是

import json

TRAINING_CONFIG = {
    "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
    "output_dir": "lora_output",
    "train_data_dir": "lora_train_set",
    "resolution": 512,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "dataloader_num_workers": 0,
    "num_train_epochs": 1,
    "max_train_steps": 150,
    "learning_rate": 5e-05,
    "scale_lr": false,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 0,
    "snr_gamma": 5.0,
    "use_8bit_adam": true,
    "mixed_precision": "fp16",
    "save_precision": "fp16",
    "enable_xformers_memory_efficient_attention": true,
    "cache_latents": true,
    "save_model_as": "safetensors",
    "network_module": "networks.lora",
    "network_dim": 64,
    "network_alpha": 32,
    "network_train_unet_only": true,
    "network_train_text_encoder_only": false,
    "save_every_n_epochs": 1
}

def get_training_args():
    return TRAINING_CONFIG
