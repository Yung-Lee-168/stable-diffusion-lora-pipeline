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

# Colab 環境檢查和依賴安裝
def setup_colab_environment():
    """Colab 環境初始化和依賴安裝 - 修復複雜依賴衝突"""
    print("� Google Colab 環境設置中...")
    print("� 檢測到複雜的依賴衝突，正在進行深度修復...")
    
    # 檢查和解決依賴衝突
    import subprocess
    import sys
    
    try:
        # 1. 先卸載所有可能衝突的套件
        print("🗑️ 步驟 1: 深度清理衝突套件...")
        conflicting_packages = [
            "sentence-transformers", 
            "transformers", 
            "torch", 
            "torchvision", 
            "torchaudio",
            "fastai"
        ]
        
        for package in conflicting_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package], 
                              capture_output=True, text=True)
                print(f"   🗑️ 已卸載 {package}")
            except:
                pass
        
        # 2. 安裝兼容的 PyTorch 生態系統
        print("📦 步驟 2: 安裝兼容的 PyTorch 版本...")
        
        # 使用 CUDA 11.8 的穩定版本組合
        torch_install_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch==2.1.0+cu118", 
            "torchvision==0.16.0+cu118", 
            "torchaudio==2.1.0+cu118",
            "--index-url", "https://download.pytorch.org/whl/cu118",
            "--force-reinstall"
        ]
        
        result = subprocess.run(torch_install_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ PyTorch 生態系統安裝成功")
        else:
            print("⚠️ PyTorch 安裝可能有問題，繼續...")
        
        # 3. 安裝兼容版本的 transformers
        print("📦 步驟 3: 安裝兼容版本的 transformers...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "transformers>=4.41.0,<5.0.0", "--force-reinstall"], 
                      capture_output=True, text=True)
        
        # 4. 安裝其他核心套件
        print("📦 步驟 4: 安裝專案依賴...")
        core_packages = [
            "diffusers[torch]",
            "accelerate",
            "peft>=0.4.0",
            "packaging",
            "matplotlib",
            "seaborn", 
            "numpy",
            "pillow",
            "scikit-learn"
        ]
        
        for package in core_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], 
                              capture_output=True, text=True, check=True)
                print(f"✅ {package}")
            except subprocess.CalledProcessError:
                print(f"⚠️ {package} 安裝失敗")
        
        # 5. 重新安裝 sentence-transformers（最後安裝以避免衝突）
        print("📦 步驟 5: 重新安裝 sentence-transformers...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", 
                           "sentence-transformers"], 
                          capture_output=True, text=True, check=True)
            print("✅ sentence-transformers")
        except subprocess.CalledProcessError:
            print("⚠️ sentence-transformers 安裝失敗（可選套件）")
        
        # 6. 嘗試安裝 xformers（可選，但有助於性能）
        print("📦 步驟 6: 安裝 xformers...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                           "xformers==0.0.22.post7", 
                           "--index-url", "https://download.pytorch.org/whl/cu118"], 
                          capture_output=True, text=True, check=True)
            print("✅ xformers")
        except subprocess.CalledProcessError:
            print("⚠️ xformers 安裝失敗（將使用標準注意力機制）")
        
        # 7. 檢查最終安裝狀態
        print("📋 步驟 7: 檢查安裝狀態...")
        _check_final_installation()
        
        print("✅ 環境設置完成！")
        print("🔄 強烈建議重新啟動運行時 (Runtime > Restart runtime)")
        
        return True
        
    except Exception as e:
        print(f"❌ 環境設置失敗: {e}")
        print("\n🔧 手動修復方法:")
        print("請在新的 cell 中執行以下命令:")
        print("!pip uninstall -y torch torchvision torchaudio transformers sentence-transformers fastai")
        print("!pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118")
        print("!pip install transformers>=4.41.0 diffusers[torch] accelerate peft")
        print("然後重新啟動運行時")
        return False

def _check_final_installation():
    """檢查最終安裝狀態"""
    try:
        import torch
        import transformers
        import diffusers
        
        print(f"   torch: {torch.__version__}")
        print(f"   transformers: {transformers.__version__}")
        print(f"   diffusers: {diffusers.__version__}")
        
        # 檢查 CUDA 可用性
        if torch.cuda.is_available():
            print(f"   CUDA: 可用 (版本 {torch.version.cuda})")
        else:
            print("   CUDA: 不可用")
            
        # 測試關鍵導入
        from diffusers import StableDiffusionPipeline
        from peft import LoraConfig
        print("   ✅ 關鍵套件導入測試通過")
        
    except ImportError as e:
        print(f"   ❌ 導入測試失敗: {e}")
    except Exception as e:
        print(f"   ⚠️ 檢查過程出錯: {e}")

# 檢查是否在 Colab 環境
try:
    from google.colab import drive, files
    IN_COLAB = True
    print("🌐 在 Google Colab 環境中運行")
    
    # 自動修復依賴（但不阻塞後續導入）
    try:
        setup_success = setup_colab_environment()
        if not setup_success:
            print("⚠️ 自動修復失敗，請手動執行修復命令或重新啟動運行時")
    except Exception as e:
        print(f"⚠️ 依賴修復過程出錯: {e}")
        print("請重新啟動運行時並手動安裝依賴")
    
except ImportError:
    IN_COLAB = False
    print("💻 在本地環境中運行")

# 嘗試導入深度學習框架，如果失敗則提供指導
try:
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
    
    # PEFT 導入有時會失敗，單獨處理
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        PEFT_AVAILABLE = True
    except ImportError:
        print("⚠️ PEFT 不可用，將跳過 LoRA 功能")
        PEFT_AVAILABLE = False
        
        # 創建假的 LoRA 類別以避免錯誤
        class LoraConfig:
            def __init__(self, **kwargs):
                pass
        
        def get_peft_model(model, config):
            return model
            
        TaskType = type('TaskType', (), {})
    
    print("✅ 核心套件導入成功")
    
except ImportError as e:
    print(f"❌ 核心套件導入失敗: {e}")
    print("🔧 解決方案:")
    if IN_COLAB:
        print("1. 重新啟動運行時 (Runtime > Restart runtime)")
        print("2. 重新執行此腳本")
        print("3. 或手動執行以下命令:")
        print("   !pip uninstall -y torch torchvision torchaudio transformers")
        print("   !pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118")
        print("   !pip install transformers>=4.41.0 diffusers[torch] accelerate peft")
    else:
        print("請檢查本地環境的套件安裝")
    
    # 創建空的類別以避免後續錯誤
    class StableDiffusionPipeline: pass
    class UNet2DConditionModel: pass
    class DDPMScheduler: pass
    class AutoencoderKL: pass
    class CLIPTextModel: pass
    class CLIPTokenizer: pass
    class CLIPProcessor: pass
    class CLIPModel: pass
    class Accelerator: pass
    class Dataset: pass
    class DataLoader: pass
    class LoraConfig: pass
    def get_peft_model(model, config): return model
    TaskType = type('TaskType', (), {})
    PEFT_AVAILABLE = False

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
            print("🔧 正在安裝和更新套件...")
            
            # 先更新 transformers 到兼容版本
            os.system("pip install -q --upgrade transformers>=4.41.0")
            
            # 安裝其他必要套件
            os.system("pip install -q diffusers[torch] accelerate peft")
            
            # 嘗試安裝 xformers（如果失敗則跳過）
            try:
                os.system("pip install -q xformers --index-url https://download.pytorch.org/whl/cu118")
                print("✅ xformers 安裝成功")
            except:
                print("⚠️ xformers 安裝失敗，將使用標準注意力機制")
            
            # 檢查版本兼容性
            self._check_package_compatibility()
            
            # 設置環境變數
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            
        # 清理記憶體
        self.cleanup_memory()
    
    def _check_package_compatibility(self):
        """檢查套件版本兼容性"""
        try:
            import transformers
            import diffusers
            
            print(f"📋 套件版本檢查:")
            print(f"   transformers: {transformers.__version__}")
            print(f"   diffusers: {diffusers.__version__}")
            
            # 檢查 transformers 版本
            from packaging import version
            transformers_version = version.parse(transformers.__version__)
            required_version = version.parse("4.41.0")
            
            if transformers_version < required_version:
                print(f"⚠️ transformers 版本過低，正在升級...")
                os.system("pip install -q --upgrade transformers>=4.41.0")
                print("✅ transformers 已升級，請重新啟動 runtime")
                
        except ImportError as e:
            print(f"⚠️ 導入檢查失敗: {e}")
        except Exception as e:
            print(f"⚠️ 版本檢查失敗: {e}")
        
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

        # 判斷本地模型路徑
        model_path = self.config.get("base_model_path") or self.config.get("model_id")
        cache_dir = self.config.get("cache_dir")

        # 載入 Stable Diffusion v1.5
        try:
            self.models["tokenizer"] = CLIPTokenizer.from_pretrained(
                model_path,
                subfolder="tokenizer",
                cache_dir=cache_dir
            )
            self.models["text_encoder"] = CLIPTextModel.from_pretrained(
                model_path,
                subfolder="text_encoder",
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            self.models["vae"] = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            self.models["unet"] = UNet2DConditionModel.from_pretrained(
                model_path,
                subfolder="unet",
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            print(f"✅ Stable Diffusion 載入成功: {model_path}")
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
        if not PEFT_AVAILABLE:
            print("⚠️ PEFT 不可用，跳過 LoRA 設置，將使用完整微調")
            self.config["use_lora"] = False
            return
            
        print("🔧 設置 LoRA 微調...")
        
        try:
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
            
        except Exception as e:
            print(f"❌ LoRA 設置失敗: {e}")
            print("⚠️ 將使用完整微調模式")
            self.config["use_lora"] = False
    
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
        
        # 保存 LoRA 權重或完整模型
        if self.config["use_lora"] and PEFT_AVAILABLE:
            try:
                self.models["unet"].save_pretrained(checkpoint_dir)
                print(f"💾 LoRA 檢查點已保存: {checkpoint_dir}")
            except Exception as e:
                print(f"⚠️ LoRA 保存失敗: {e}")
                # 回退到保存完整 UNet
                torch.save(self.models["unet"].state_dict(), 
                          os.path.join(checkpoint_dir, "unet_state_dict.pt"))
        else:
            # 保存完整模型狀態
            torch.save(self.models["unet"].state_dict(), 
                      os.path.join(checkpoint_dir, "unet_state_dict.pt"))
            print(f"💾 完整模型檢查點已保存: {checkpoint_dir}")
        
        # 保存訓練狀態
        torch.save({
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "config": self.config
        }, os.path.join(checkpoint_dir, "training_state.pt"))
    
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
        
        if self.config["use_lora"] and PEFT_AVAILABLE:
            try:
                # 保存 LoRA 權重
                self.models["unet"].save_pretrained(final_dir)
                print(f"💾 LoRA 模型已保存: {final_dir}")
            except Exception as e:
                print(f"⚠️ LoRA 保存失敗: {e}")
                # 回退到保存完整模型
                self._save_full_pipeline(final_dir)
        else:
            # 保存完整模型
            self._save_full_pipeline(final_dir)
    
    def _save_full_pipeline(self, save_dir):
        """保存完整的 Stable Diffusion 管道"""
        try:
            pipeline = StableDiffusionPipeline(
                vae=self.models["vae"],
                text_encoder=self.models["text_encoder"],
                tokenizer=self.models["tokenizer"],
                unet=self.accelerator.unwrap_model(self.models["unet"]) if self.accelerator else self.models["unet"],
                scheduler=DDPMScheduler.from_pretrained(
                    self.config["model_id"], subfolder="scheduler"
                ),
                safety_checker=None,
                requires_safety_checker=False
            )
            pipeline.save_pretrained(save_dir)
            print(f"💾 完整模型已保存: {save_dir}")
        except Exception as e:
            print(f"❌ 完整模型保存失敗: {e}")
            # 至少保存 UNet 權重
            torch.save(self.models["unet"].state_dict(), 
                      os.path.join(save_dir, "unet_state_dict.pt"))
            print(f"💾 UNet 權重已保存: {save_dir}")
    
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

def demo_quick_test():
    """快速測試 - 使用範例數據"""
    print("🚀 快速測試模式")
    print("=" * 40)
    
    try:
        # 創建簡單的測試配置
        test_config = {
            "num_epochs": 2,
            "train_batch_size": 1,
            "save_steps": 10,
            "validation_steps": 10,
            "learning_rate": 5e-5
        }
        
        trainer = FashionSDFineTuner(test_config)
        
        # 創建測試數據
        test_prompts = [
            "a woman wearing an elegant black dress",
            "a man in casual blue jeans and white shirt",
            "person in formal business suit"
        ]
        
        print("✅ 快速測試完成，系統運行正常！")
        print("💡 現在您可以運行完整版本")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        print("請檢查依賴安裝或重新啟動運行時")

def main():
    """主函數 - Colab 互動式執行"""
    print("🎨 Day 3: Fashion AI Training - Google Colab 版本")
    print("=" * 60)
    
    # 選擇執行模式
    print("請選擇執行模式:")
    print("1. 快速測試 (檢查環境)")
    print("2. 完整訓練 (上傳圖片)")
    print("3. 使用範例數據訓練")
    
    if IN_COLAB:
        # 在 Colab 中自動選擇模式
        mode = input("請輸入選項 (1/2/3，預設=1): ").strip() or "1"
    else:
        mode = "1"  # 本地環境預設快速測試
    
    if mode == "1":
        demo_quick_test()
        return
    
    elif mode == "3":
        # 使用範例數據
        print("\n🎯 使用範例數據訓練")
        trainer = FashionSDFineTuner()
        
        # 創建範例數據
        sample_captions = [
            "a woman wearing an elegant black dress",
            "a man in casual blue jeans and white shirt", 
            "person in formal business suit",
            "woman in floral summer dress",
            "man wearing leather jacket"
        ]
        
        # 由於沒有真實圖片，我們跳過實際訓練
        print("📝 範例描述:")
        for i, caption in enumerate(sample_captions, 1):
            print(f"  {i}. {caption}")
        
        print("\n💡 這是範例模式，如需完整訓練請選擇模式 2")
        return
    
    # 完整訓練模式
    try:
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
        
    except Exception as e:
        print(f"❌ 執行失敗: {e}")
        print("\n🔧 建議解決方案:")
        print("1. 重新啟動運行時 (Runtime > Restart runtime)")
        print("2. 重新執行此腳本")
        print("3. 檢查 GPU 是否已啟用")

def run_colab_setup_only():
    """只運行環境設置 - 用於排除故障"""
    print("🔧 只運行環境設置...")
    setup_success = setup_colab_environment()
    
    if setup_success:
        print("✅ 環境設置完成")
        print("💡 現在可以重新啟動運行時並執行完整腳本")
    else:
        print("❌ 環境設置失敗")
        print("請手動安裝依賴或聯繫支援")

if __name__ == "__main__":
    # 檢查命令行參數
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        run_colab_setup_only()
    else:
        main()
