#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: Fashion AI Training - Google Colab 版本
專為 Google Colab 環境優化的 SD v1.5 真正微調系統

🎯 特色:
- 針對 Colab 的 T4/V100 GPU 優化
- 自動記憶體管理和清理
- LoRA 高效微調 (節省記憶體)
- 自動上傳/下載 Google Drive
- 漸進式訓練和檢查點保存
- 實時訓練監控

🔧 記憶體需求:
- T4 (16GB): 支援 LoRA + 混合精度
- V100 (16GB): 支援完整訓練
- 自動偵測和調整配置
"""

import os
import json
import torch
import gc
import zipfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Colab 專用導入
try:
    from google.colab import drive, files
    IN_COLAB = True
    print("🌐 在 Google Colab 環境中運行")
except ImportError:
    IN_COLAB = False
    print("💻 在本地環境中運行")

# 深度學習框架
from diffusers import (
    StableDiffusionPipeline, 
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType

class ColabEnvironmentSetup:
    """Google Colab 環境設置和優化"""
    
    def __init__(self):
        self.gpu_info = self.check_gpu()
        self.drive_mounted = False
        
    def check_gpu(self):
        """檢查 GPU 狀態和記憶體"""
        if not torch.cuda.is_available():
            print("❌ 沒有可用的 GPU")
            return None
            
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"🔧 GPU: {gpu_name}")
        print(f"💾 VRAM: {gpu_memory:.1f} GB")
        
        return {
            "name": gpu_name,
            "memory_gb": gpu_memory,
            "is_t4": "T4" in gpu_name,
            "is_v100": "V100" in gpu_name,
            "is_a100": "A100" in gpu_name
        }
    
    def mount_drive(self):
        """掛載 Google Drive"""
        if IN_COLAB and not self.drive_mounted:
            try:
                drive.mount('/content/drive')
                self.drive_mounted = True
                print("✅ Google Drive 已掛載")
                
                # 創建工作目錄
                work_dir = "/content/drive/MyDrive/fashion_ai_training"
                os.makedirs(work_dir, exist_ok=True)
                os.chdir(work_dir)
                print(f"📁 工作目錄: {work_dir}")
                
            except Exception as e:
                print(f"❌ Drive 掛載失敗: {e}")
        
    def setup_environment(self):
        """設置 Colab 環境"""
        if IN_COLAB:
            # 安裝必要套件
            os.system("pip install -q diffusers[torch] transformers accelerate peft")
            os.system("pip install -q xformers --index-url https://download.pytorch.org/whl/cu118")
            
            # 設置環境變數
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            
        # 清理記憶體
        self.cleanup_memory()
        
    def cleanup_memory(self):
        """清理 GPU 記憶體"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 記憶體清理完成，可用: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class FashionDataset(Dataset):
    """時尚圖片數據集"""
    
    def __init__(self, image_paths: List[str], captions: List[str], 
                 tokenizer, image_size: int = 512):
        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 載入和預處理圖片
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()
        
        # 處理文字描述
        caption = self.captions[idx]
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "image": image,
            "input_ids": tokens.input_ids.squeeze(),
            "attention_mask": tokens.attention_mask.squeeze()
        }

class FashionSDFineTuner:
    """SD v1.5 時尚微調器 - Colab 優化版"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.env_setup = ColabEnvironmentSetup()
        self.config = self._get_colab_optimized_config(config)
        self.accelerator = None
        self.models = {}
        self.fashion_clip_model = None
        
        # 設置環境
        self.env_setup.setup_environment()
        self.env_setup.mount_drive()
        
        # 初始化模型
        self._init_models()
        
    def _get_colab_optimized_config(self, custom_config: Optional[Dict] = None):
        """根據 Colab GPU 自動優化配置"""
        gpu_info = self.env_setup.gpu_info
        
        if gpu_info is None:
            raise RuntimeError("需要 GPU 支援")
        
        # 基礎配置
        base_config = {
            "model_id": "runwayml/stable-diffusion-v1-5",
            "output_dir": "/content/drive/MyDrive/fashion_ai_training/models",
            "cache_dir": "/content/cache",
            "image_size": 512,
            "train_batch_size": 1,
            "num_epochs": 20,
            "learning_rate": 1e-4,
            "use_lora": True,
            "lora_rank": 4,
            "mixed_precision": "fp16",
            "gradient_accumulation_steps": 4,
            "save_steps": 100,
            "validation_steps": 50,
            "max_grad_norm": 1.0,
            "warmup_steps": 100
        }
        
        # 根據 GPU 調整配置
        if gpu_info["is_t4"]:
            # T4 優化 (16GB)
            base_config.update({
                "train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "lora_rank": 4,
                "image_size": 512,
                "use_xformers": True
            })
            print("🎯 T4 GPU 優化配置")
            
        elif gpu_info["is_v100"]:
            # V100 優化 (16GB)
            base_config.update({
                "train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "lora_rank": 8,
                "image_size": 512,
                "use_xformers": True
            })
            print("🎯 V100 GPU 優化配置")
            
        elif gpu_info["is_a100"]:
            # A100 優化 (40GB)
            base_config.update({
                "train_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "lora_rank": 16,
                "image_size": 768,
                "use_xformers": True
            })
            print("🎯 A100 GPU 優化配置")
        
        # 合併自定義配置
        if custom_config:
            base_config.update(custom_config)
            
        return base_config
    
    def _init_models(self):
        """初始化模型"""
        print("📦 載入模型...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 載入 Stable Diffusion v1.5
        try:
            self.models["tokenizer"] = CLIPTokenizer.from_pretrained(
                self.config["model_id"], 
                subfolder="tokenizer",
                cache_dir=self.config["cache_dir"]
            )
            
            self.models["text_encoder"] = CLIPTextModel.from_pretrained(
                self.config["model_id"], 
                subfolder="text_encoder",
                cache_dir=self.config["cache_dir"],
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            self.models["vae"] = AutoencoderKL.from_pretrained(
                self.config["model_id"], 
                subfolder="vae",
                cache_dir=self.config["cache_dir"],
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            self.models["unet"] = UNet2DConditionModel.from_pretrained(
                self.config["model_id"], 
                subfolder="unet",
                cache_dir=self.config["cache_dir"],
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            print("✅ Stable Diffusion v1.5 載入成功")
            
        except Exception as e:
            print(f"❌ SD 模型載入失敗: {e}")
            raise
        
        # 載入 FashionCLIP (用於特徵提取和評估)
        try:
            self.fashion_clip_model = CLIPModel.from_pretrained(
                "patrickjohncyh/fashion-clip",
                cache_dir=self.config["cache_dir"],
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            self.fashion_clip_processor = CLIPProcessor.from_pretrained(
                "patrickjohncyh/fashion-clip",
                cache_dir=self.config["cache_dir"]
            )
            
            print("✅ FashionCLIP 載入成功")
            
        except Exception as e:
            print(f"⚠️ FashionCLIP 載入失敗: {e}")
            self.fashion_clip_model = None
        
        # 設置 LoRA
        if self.config["use_lora"]:
            self._setup_lora()
        
        # 初始化 accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.config["mixed_precision"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"]
        )
        
        print(f"🚀 使用設備: {device}")
        print(f"💾 混合精度: {self.config['mixed_precision']}")
    
    def _setup_lora(self):
        """設置 LoRA 微調"""
        print("🔧 設置 LoRA 微調...")
        
        # UNet LoRA 配置
        unet_lora_config = LoraConfig(
            r=self.config["lora_rank"],
            lora_alpha=self.config["lora_rank"],
            target_modules=[
                "to_k", "to_q", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2"
            ],
            lora_dropout=0.1,
        )
        
        # 應用 LoRA 到 UNet
        self.models["unet"] = get_peft_model(self.models["unet"], unet_lora_config)
        
        # 凍結其他模型參數
        self.models["vae"].requires_grad_(False)
        self.models["text_encoder"].requires_grad_(False)
        
        # 只訓練 LoRA 參數
        trainable_params = sum(p.numel() for p in self.models["unet"].parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.models["unet"].parameters())
        
        print(f"📊 可訓練參數: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def upload_training_images(self):
        """上傳訓練圖片 (Colab 專用)"""
        if not IN_COLAB:
            print("💻 本地環境，跳過上傳")
            return []
            
        print("📤 請上傳訓練圖片...")
        uploaded = files.upload()
        
        image_paths = []
        upload_dir = "/content/uploaded_images"
        os.makedirs(upload_dir, exist_ok=True)
        
        for filename, content in uploaded.items():
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(upload_dir, filename)
                with open(file_path, 'wb') as f:
                    f.write(content)
                image_paths.append(file_path)
                
        print(f"✅ 上傳了 {len(image_paths)} 張圖片")
        return image_paths
    
    def extract_features_from_images(self, image_paths: List[str]) -> List[str]:
        """使用 FashionCLIP 從圖片提取特徵並生成描述"""
        print("🔍 使用 FashionCLIP 提取特徵...")
        
        if not self.fashion_clip_model:
            # 如果沒有 FashionCLIP，使用簡單描述
            print("⚠️ 使用簡單描述")
            return [f"fashion photo {i+1}" for i in range(len(image_paths))]
        
        captions = []
        device = next(self.fashion_clip_model.parameters()).device
        
        # 特徵類別 (簡化版)
        categories = {
            "clothing": ["dress", "shirt", "jacket", "pants", "skirt", "blouse"],
            "style": ["casual", "formal", "elegant", "sporty", "vintage", "modern"],
            "color": ["black", "white", "blue", "red", "green", "pink", "brown"],
            "pattern": ["solid", "striped", "floral", "geometric", "plain"]
        }
        
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                
                caption_parts = []
                
                # 分析每個類別
                for category, labels in categories.items():
                    inputs = self.fashion_clip_processor(
                        text=labels, 
                        images=image, 
                        return_tensors="pt", 
                        padding=True
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.fashion_clip_model(**inputs)
                        probs = outputs.logits_per_image.softmax(dim=1)
                        
                    # 取最高分的標籤
                    best_idx = probs.argmax().item()
                    best_label = labels[best_idx]
                    caption_parts.append(best_label)
                
                # 組合描述
                caption = f"a photo of a person wearing {caption_parts[0]} in {caption_parts[1]} style"
                captions.append(caption)
                
                print(f"   {os.path.basename(image_path)}: {caption}")
                
            except Exception as e:
                print(f"⚠️ 處理 {image_path} 時出錯: {e}")
                captions.append("fashion photo")
        
        return captions
    
    def prepare_dataset(self, image_paths: List[str], captions: List[str]) -> DataLoader:
        """準備訓練數據集"""
        print("📊 準備數據集...")
        
        dataset = FashionDataset(
            image_paths=image_paths,
            captions=captions,
            tokenizer=self.models["tokenizer"],
            image_size=self.config["image_size"]
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["train_batch_size"],
            shuffle=True,
            num_workers=0,  # Colab 建議設為 0
            pin_memory=True
        )
        
        print(f"✅ 數據集準備完成: {len(dataset)} 張圖片")
        return dataloader
    
    def train(self, dataloader: DataLoader):
        """執行訓練"""
        print("🚀 開始訓練...")
        
        # 優化器
        optimizer = torch.optim.AdamW(
            self.models["unet"].parameters(),
            lr=self.config["learning_rate"],
            weight_decay=0.01
        )
        
        # 學習率調度器
        num_training_steps = len(dataloader) * self.config["num_epochs"]
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps
        )
        
        # Accelerator 準備
        unet, optimizer, dataloader, lr_scheduler = self.accelerator.prepare(
            self.models["unet"], optimizer, dataloader, lr_scheduler
        )
        
        # 噪聲調度器
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.config["model_id"], 
            subfolder="scheduler"
        )
        
        # 訓練循環
        global_step = 0
        training_losses = []
        
        for epoch in range(self.config["num_epochs"]):
            unet.train()
            epoch_losses = []
            
            for step, batch in enumerate(dataloader):
                with self.accelerator.accumulate(unet):
                    # 編碼圖片
                    images = batch["image"].to(self.accelerator.device)
                    latents = self.models["vae"].encode(images).latent_dist.sample()
                    latents = latents * 0.18215
                    
                    # 添加噪聲
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, 
                        (latents.shape[0],), device=latents.device
                    ).long()
                    
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # 編碼文字
                    encoder_hidden_states = self.models["text_encoder"](
                        batch["input_ids"].to(self.accelerator.device)
                    )[0]
                    
                    # 預測噪聲
                    noise_pred = unet(
                        noisy_latents, 
                        timesteps, 
                        encoder_hidden_states
                    ).sample
                    
                    # 計算損失
                    loss = F.mse_loss(noise_pred, noise, reduction="mean")
                    
                    # 反向傳播
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(unet.parameters(), self.config["max_grad_norm"])
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # 記錄損失
                epoch_losses.append(loss.item())
                training_losses.append(loss.item())
                
                if global_step % 10 == 0:
                    avg_loss = np.mean(epoch_losses[-10:])
                    print(f"Epoch {epoch+1}/{self.config['num_epochs']}, "
                          f"Step {step+1}/{len(dataloader)}, "
                          f"Loss: {avg_loss:.4f}")
                
                # 保存檢查點
                if global_step % self.config["save_steps"] == 0:
                    self._save_checkpoint(global_step, optimizer, lr_scheduler)
                
                # 驗證
                if global_step % self.config["validation_steps"] == 0:
                    self._validate(global_step)
                
                global_step += 1
                
                # 記憶體清理
                if global_step % 50 == 0:
                    self.env_setup.cleanup_memory()
            
            # Epoch 結束
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"✅ Epoch {epoch+1} 完成，平均損失: {avg_epoch_loss:.4f}")
        
        # 保存最終模型
        self._save_final_model()
        
        # 生成訓練圖表
        self._plot_training_progress(training_losses)
        
        print("🎉 訓練完成！")
    
    def _save_checkpoint(self, step: int, optimizer, lr_scheduler):
        """保存訓練檢查點"""
        checkpoint_dir = os.path.join(self.config["output_dir"], f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存 LoRA 權重
        if self.config["use_lora"]:
            self.models["unet"].save_pretrained(checkpoint_dir)
        
        # 保存訓練狀態
        torch.save({
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        print(f"💾 檢查點已保存: {checkpoint_dir}")
    
    def _validate(self, step: int):
        """驗證生成效果"""
        print(f"🔍 第 {step} 步驗證...")
        
        try:
            # 創建臨時管道
            pipeline = StableDiffusionPipeline(
                vae=self.models["vae"],
                text_encoder=self.models["text_encoder"],
                tokenizer=self.models["tokenizer"],
                unet=self.accelerator.unwrap_model(self.models["unet"]),
                scheduler=DDPMScheduler.from_pretrained(
                    self.config["model_id"], subfolder="scheduler"
                ),
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # 測試提示詞
            test_prompts = [
                "a woman wearing a elegant dress",
                "a man in casual shirt and jeans",
                "person in formal business attire"
            ]
            
            validation_dir = os.path.join(self.config["output_dir"], "validation")
            os.makedirs(validation_dir, exist_ok=True)
            
            for i, prompt in enumerate(test_prompts):
                image = pipeline(
                    prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    width=512,
                    height=512
                ).images[0]
                
                image.save(os.path.join(validation_dir, f"step_{step}_prompt_{i+1}.png"))
            
            print(f"✅ 驗證圖片已保存到 {validation_dir}")
            
        except Exception as e:
            print(f"⚠️ 驗證失敗: {e}")
    
    def _save_final_model(self):
        """保存最終模型"""
        final_dir = os.path.join(self.config["output_dir"], "final_model")
        os.makedirs(final_dir, exist_ok=True)
        
        if self.config["use_lora"]:
            # 保存 LoRA 權重
            self.models["unet"].save_pretrained(final_dir)
            print(f"💾 LoRA 模型已保存: {final_dir}")
        else:
            # 保存完整模型
            pipeline = StableDiffusionPipeline(
                vae=self.models["vae"],
                text_encoder=self.models["text_encoder"],
                tokenizer=self.models["tokenizer"],
                unet=self.accelerator.unwrap_model(self.models["unet"]),
                scheduler=DDPMScheduler.from_pretrained(
                    self.config["model_id"], subfolder="scheduler"
                ),
                safety_checker=None,
                requires_safety_checker=False
            )
            pipeline.save_pretrained(final_dir)
            print(f"💾 完整模型已保存: {final_dir}")
    
    def _plot_training_progress(self, losses: List[float]):
        """繪製訓練進度圖"""
        plt.figure(figsize=(12, 6))
        
        # 損失曲線
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        
        # 平滑損失曲線
        plt.subplot(1, 2, 2)
        window_size = min(50, len(losses) // 10)
        if window_size > 1:
            smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_losses)
            plt.title(f"Smoothed Training Loss (window={window_size})")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True)
        
        plt.tight_layout()
        
        # 保存圖表
        plot_path = os.path.join(self.config["output_dir"], "training_progress.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 訓練圖表已保存: {plot_path}")
    
    def create_download_package(self):
        """創建下載包 (Colab 專用)"""
        if not IN_COLAB:
            print("💻 本地環境，跳過打包")
            return
            
        print("📦 創建下載包...")
        
        # 打包模型和結果
        package_name = f"fashion_ai_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加模型文件
            model_dir = os.path.join(self.config["output_dir"], "final_model")
            if os.path.exists(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.config["output_dir"])
                        zipf.write(file_path, arcname)
            
            # 添加訓練圖表
            plot_path = os.path.join(self.config["output_dir"], "training_progress.png")
            if os.path.exists(plot_path):
                zipf.write(plot_path, "training_progress.png")
            
            # 添加驗證圖片
            validation_dir = os.path.join(self.config["output_dir"], "validation")
            if os.path.exists(validation_dir):
                for file in os.listdir(validation_dir):
                    file_path = os.path.join(validation_dir, file)
                    zipf.write(file_path, f"validation/{file}")
        
        print(f"📦 打包完成: {package_name}")
        
        # 提供下載
        files.download(package_name)

def main():
    """主函數 - Colab 互動式執行"""
    print("🎨 Day 3: Fashion AI Training - Google Colab 版本")
    print("=" * 60)
    
    # 初始化訓練器
    trainer = FashionSDFineTuner()
    
    # 上傳圖片
    print("\n📤 步驟 1: 上傳訓練圖片")
    image_paths = trainer.upload_training_images()
    
    if not image_paths:
        print("❌ 沒有有效的圖片，請重新上傳")
        return
    
    # 提取特徵
    print("\n🔍 步驟 2: 提取圖片特徵")
    captions = trainer.extract_features_from_images(image_paths)
    
    # 準備數據集
    print("\n📊 步驟 3: 準備訓練數據")
    dataloader = trainer.prepare_dataset(image_paths, captions)
    
    # 開始訓練
    print("\n🚀 步驟 4: 開始微調訓練")
    trainer.train(dataloader)
    
    # 創建下載包
    print("\n📦 步驟 5: 準備下載")
    trainer.create_download_package()
    
    print("\n🎉 所有步驟完成！")

if __name__ == "__main__":
    main()
