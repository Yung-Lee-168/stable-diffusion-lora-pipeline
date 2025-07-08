#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試訓練流程 - 驗證不同情況下的繼續訓練行為
"""
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

# 添加路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_training_scenarios():
    """測試不同的訓練情況"""
    
    print("🧪 開始測試訓練流程...")
    
    # 測試目錄
    lora_output = current_dir / "lora_output"
    test_data = current_dir / "lora_train_set" / "10_test"
    
    # 檢查測試資料是否存在
    if not test_data.exists():
        print(f"❌ 測試資料目錄不存在: {test_data}")
        return False
    
    # 清理舊的輸出
    if lora_output.exists():
        print(f"🧹 清理舊的輸出目錄: {lora_output}")
        shutil.rmtree(lora_output)
    
    # 測試 1: 全新訓練 (1 epoch)
    print("\n📍 測試 1: 全新訓練")
    cmd = [
        sys.executable, "train_lora.py",
        "--epochs", "1",
        "--no-continue"
    ]
    
    print(f"執行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=current_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 全新訓練失敗:")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False
    
    print("✅ 全新訓練完成")
    
    # 檢查是否有 LoRA 檔案
    lora_files = list(lora_output.glob("*.safetensors"))
    if not lora_files:
        print("❌ 沒有找到 LoRA 檔案")
        return False
    
    print(f"📁 找到 LoRA 檔案: {[f.name for f in lora_files]}")
    
    # 測試 2: 從現有 LoRA 繼續訓練 (1 epoch)
    print("\n📍 測試 2: 從現有 LoRA 繼續訓練")
    cmd = [
        sys.executable, "train_lora.py",
        "--epochs", "1",
        "--continue"
    ]
    
    print(f"執行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=current_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 繼續訓練失敗:")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False
    
    print("✅ 繼續訓練完成")
    
    # 檢查備份是否存在
    backup_files = list(lora_output.glob("backup_*.safetensors"))
    if backup_files:
        print(f"📁 找到備份檔案: {[f.name for f in backup_files]}")
    
    print("\n🎉 所有測試完成!")
    return True

def test_simple_training():
    """簡單的訓練測試"""
    print("🧪 開始簡單訓練測試...")
    
    # 測試目錄
    current_dir = Path(__file__).parent
    test_data = current_dir / "lora_train_set" / "10_test"
    
    # 檢查測試資料是否存在
    if not test_data.exists():
        print(f"❌ 測試資料目錄不存在: {test_data}")
        return False
    
    # 執行 1 epoch 訓練
    cmd = [
        sys.executable, "train_lora.py",
        "--epochs", "1",
        "--no-continue"
    ]
    
    print(f"執行命令: {' '.join(cmd)}")
    print("訓練開始...")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=current_dir)
    end_time = time.time()
    
    print(f"訓練完成，耗時: {end_time - start_time:.2f} 秒")
    
    if result.returncode == 0:
        print("✅ 訓練成功!")
        return True
    else:
        print("❌ 訓練失敗!")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        test_simple_training()
    else:
        test_training_scenarios()
