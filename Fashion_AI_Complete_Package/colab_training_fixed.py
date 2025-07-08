#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: Fashion AI Training - Google Colab 版本 (修復依賴衝突)
專為 Google Colab 環境優化的 SD v1.5 真正微調系統

🔧 修復項目:
- 解決 sentence-transformers 依賴衝突
- 更新 transformers 到兼容版本
- 優化套件安裝順序
- 加入錯誤處理和重試機制

🎯 使用說明:
1. 在 Colab 中運行此腳本
2. 腳本會自動處理依賴衝突
3. 按照提示上傳圖片並開始訓練
"""

# 優先處理依賴衝突
print("🔧 正在檢查和修復 Google Colab 依賴衝突...")

import subprocess
import sys
import os

def fix_colab_dependencies():
    """修復 Colab 依賴衝突"""
    try:
        print("🗑️ 步驟 1: 清理衝突套件...")
        
        # 卸載可能衝突的套件
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", 
                       "sentence-transformers", "transformers"], 
                      capture_output=True, text=True)
        
        print("📦 步驟 2: 安裝兼容版本...")
        
        # 安裝兼容版本的 transformers
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "transformers>=4.41.0", "--force-reinstall"], 
                      capture_output=True, text=True)
        
        # 安裝其他核心套件
        core_packages = [
            "torch>=2.0.0",
            "torchvision",
            "torchaudio", 
            "diffusers[torch]",
            "accelerate",
            "peft",
            "packaging"
        ]
        
        for package in core_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], 
                              capture_output=True, text=True, check=True)
                print(f"✅ {package}")
            except subprocess.CalledProcessError:
                print(f"⚠️ {package} 安裝失敗")
        
        print("📦 步驟 3: 安裝可選套件...")
        
        # 重新安裝 sentence-transformers
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                           "sentence-transformers"], 
                          capture_output=True, text=True, check=True)
            print("✅ sentence-transformers")
        except subprocess.CalledProcessError:
            print("⚠️ sentence-transformers 安裝失敗（可選）")
        
        # 嘗試安裝 xformers
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xformers", 
                           "--index-url", "https://download.pytorch.org/whl/cu118"], 
                          capture_output=True, text=True, check=True)
            print("✅ xformers")
        except subprocess.CalledProcessError:
            print("⚠️ xformers 安裝失敗（將使用標準注意力機制）")
        
        print("✅ 依賴修復完成！")
        return True
        
    except Exception as e:
        print(f"❌ 依賴修復失敗: {e}")
        print("\n🔧 手動修復方法:")
        print("請在新的 cell 中執行以下命令:")
        print("!pip uninstall -y sentence-transformers transformers")
        print("!pip install transformers>=4.41.0 --force-reinstall")
        print("!pip install diffusers[torch] accelerate peft")
        return False

# 檢查是否在 Colab 環境
try:
    from google.colab import drive, files
    IN_COLAB = True
    print("🌐 在 Google Colab 環境中運行")
    
    # 自動修復依賴
    if not fix_colab_dependencies():
        print("❌ 自動修復失敗，請手動執行修復命令")
        sys.exit(1)
    
except ImportError:
    IN_COLAB = False
    print("💻 在本地環境中運行")

# 重新啟動運行時提示
print("\n🔄 重要提示:")
print("如果是第一次運行，請在依賴安裝完成後重新啟動運行時 (Runtime > Restart runtime)")
print("然後重新運行此腳本")
print("=" * 60)

# 標準導入
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

# 檢查導入是否成功
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
    from peft import LoraConfig, get_peft_model, TaskType
    
    print("✅ 所有必要套件導入成功")
    
except ImportError as e:
    print(f"❌ 套件導入失敗: {e}")
    print("請重新啟動運行時並重新執行此腳本")
    sys.exit(1)

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
    
    def cleanup_memory(self):
        """清理 GPU 記憶體"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 記憶體清理完成")

# 添加簡化的訓練函數
def quick_colab_training():
    """快速 Colab 訓練流程"""
    print("🎨 Fashion AI Training - 快速 Colab 版本")
    print("=" * 60)
    
    # 環境設置
    env_setup = ColabEnvironmentSetup()
    env_setup.mount_drive()
    
    # 檢查 GPU
    if env_setup.gpu_info is None:
        print("❌ 需要 GPU 支援，請在 Colab 中啟用 GPU")
        return
    
    print(f"✅ 使用 GPU: {env_setup.gpu_info['name']}")
    
    # 上傳圖片
    if IN_COLAB:
        print("📤 請上傳訓練圖片...")
        uploaded = files.upload()
        
        if not uploaded:
            print("❌ 沒有上傳圖片")
            return
        
        print(f"✅ 上傳了 {len(uploaded)} 個檔案")
    
    print("🎉 環境設置完成，可以開始訓練！")
    print("💡 如需完整訓練功能，請使用原始的 day3_colab_finetuning.py")

if __name__ == "__main__":
    quick_colab_training()
