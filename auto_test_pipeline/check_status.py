#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速狀態檢查 - 檢查訓練環境和現有檔案
"""
import os
import sys
from pathlib import Path
import glob

def check_training_status():
    """檢查訓練狀態"""
    print("🔍 檢查 LoRA 訓練狀態...")
    print("=" * 50)
    
    current_dir = Path(__file__).parent
    
    # 檢查訓練資料
    train_data_dir = current_dir / "lora_train_set" / "10_test"
    if train_data_dir.exists():
        images = list(train_data_dir.glob("*.jpg")) + list(train_data_dir.glob("*.jpeg"))
        texts = list(train_data_dir.glob("*.txt"))
        print(f"📁 訓練資料目錄: {train_data_dir}")
        print(f"🖼️ 圖片數量: {len(images)}")
        print(f"📝 標籤數量: {len(texts)}")
        
        if len(images) != len(texts):
            print("⚠️ 圖片和標籤數量不一致！")
        else:
            print("✅ 圖片和標籤數量一致")
    else:
        print("❌ 訓練資料目錄不存在")
    
    print()
    
    # 檢查輸出目錄
    output_dir = current_dir / "lora_output"
    if output_dir.exists():
        print(f"📁 輸出目錄: {output_dir}")
        
        # 檢查 LoRA 檔案
        lora_files = list(output_dir.glob("*.safetensors"))
        backup_files = list(output_dir.glob("backup_*.safetensors"))
        
        print(f"🎯 LoRA 檔案: {len(lora_files)}")
        for f in lora_files:
            if not f.name.startswith("backup_"):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   - {f.name} ({size_mb:.1f} MB)")
        
        print(f"💾 備份檔案: {len(backup_files)}")
        for f in backup_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.1f} MB)")
        
        # 檢查狀態目錄
        state_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("state_")]
        print(f"🔄 狀態目錄: {len(state_dirs)}")
        for d in state_dirs:
            print(f"   - {d.name}")
        
        # 檢查日誌
        log_files = list(output_dir.glob("*.log"))
        print(f"📜 日誌檔案: {len(log_files)}")
        for f in log_files:
            print(f"   - {f.name}")
        
    else:
        print("📁 輸出目錄不存在 (將在首次訓練時創建)")
    
    print()
    
    # 檢查腳本
    scripts = [
        "train_lora.py",
        "train_lora_monitor.py", 
        "train_lora_monitored_new.py",
        "infer_lora_direct.py"
    ]
    
    print("🔧 腳本檢查:")
    for script in scripts:
        script_path = current_dir / script
        if script_path.exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script} (不存在)")
    
    print()
    
    # 檢查訓練環境
    print("🌐 環境檢查:")
    
    # 檢查 Python 版本
    print(f"🐍 Python 版本: {sys.version.split()[0]}")
    
    # 檢查 SD WebUI 路徑
    webui_path = current_dir.parent
    if (webui_path / "webui.py").exists():
        print(f"✅ SD WebUI 路徑: {webui_path}")
    else:
        print(f"❌ SD WebUI 路徑無效: {webui_path}")
    
    # 檢查基礎模型
    models_path = webui_path / "models" / "Stable-diffusion"
    if models_path.exists():
        model_files = list(models_path.glob("*.safetensors")) + list(models_path.glob("*.ckpt"))
        print(f"🎭 基礎模型數量: {len(model_files)}")
        if model_files:
            print(f"   - 最新: {model_files[-1].name}")
    
    print()
    print("🎉 狀態檢查完成！")

if __name__ == "__main__":
    check_training_status()
