#!/usr/bin/env python3
"""
LoRA 訓練腳本 - 完全靜默版本
過濾所有 xFormers 和 Triton 警告
"""
import subprocess
import os
import sys
import warnings
import argparse
import datetime
from contextlib import redirect_stderr
from io import StringIO

# 設定最強力的警告抑制
os.environ['DISABLE_XFORMERS'] = '1'
os.environ['XFORMERS_MORE_DETAILS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['DIFFUSERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 抑制所有可能的警告
warnings.filterwarnings("ignore")

def run_train_lora_silent(*args):
    """靜默執行 train_lora.py"""
    
    # 確保在腳本所在目錄執行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("🚀 開始 LoRA 訓練（靜默模式）...")
    print("📝 警告訊息將被過濾")
    
    # 構建命令
    cmd = [sys.executable, "train_lora.py"] + list(args)
    
    # 設定環境變數
    env = os.environ.copy()
    env['DISABLE_XFORMERS'] = '1'
    env['XFORMERS_MORE_DETAILS'] = '0'
    env['PYTHONWARNINGS'] = 'ignore'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['TRANSFORMERS_VERBOSITY'] = 'error'
    env['DIFFUSERS_VERBOSITY'] = 'error'
    
    try:
        # 執行並過濾輸出
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            bufsize=1,
            universal_newlines=True
        )
        
        # 即時處理輸出，過濾警告
        for line in iter(process.stdout.readline, ''):
            # 過濾掉 xFormers 和 Triton 相關的警告
            if not any(keyword in line.lower() for keyword in [
                'xformers', 'triton', 'warning', 'traceback', 
                'modulenotfounderror', 'c++ extensions'
            ]):
                print(line.rstrip())
        
        # 等待進程完成
        process.wait()
        
        # 檢查返回碼
        if process.returncode == 0:
            print("✅ LoRA 訓練完成")
            return True
        else:
            print("❌ LoRA 訓練失敗")
            return False
            
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        return False

def main():
    """主函數"""
    # 傳遞所有命令行參數給 train_lora.py
    args = sys.argv[1:]
    success = run_train_lora_silent(*args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
