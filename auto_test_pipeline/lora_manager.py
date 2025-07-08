#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 訓練管理器 - 統一管理介面
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description=""):
    """執行命令"""
    if description:
        print(f"🚀 {description}")
    
    print(f"執行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="LoRA 訓練管理器")
    parser.add_argument("action", choices=[
        "status", "train", "continue", "test", "infer", "clean"
    ], help="要執行的動作")
    
    parser.add_argument("--epochs", type=int, default=10, help="訓練輪數")
    parser.add_argument("--monitor", action="store_true", help="使用監控版本")
    parser.add_argument("--silent", action="store_true", help="靜默模式")
    
    args = parser.parse_args()
    
    current_dir = Path(__file__).parent
    
    if args.action == "status":
        # 檢查狀態
        return run_command([sys.executable, "check_status.py"], "檢查訓練狀態")
    
    elif args.action == "train":
        # 全新訓練
        script = "train_lora.py"
        if args.monitor:
            script = "train_lora_monitor.py"
        elif args.silent:
            script = "train_lora_silent.py"
        
        cmd = [sys.executable, script, "--epochs", str(args.epochs), "--no-continue"]
        return run_command(cmd, f"開始全新訓練 ({args.epochs} epochs)")
    
    elif args.action == "continue":
        # 繼續訓練
        script = "train_lora.py"
        if args.monitor:
            script = "train_lora_monitor.py"
        elif args.silent:
            script = "train_lora_silent.py"
        
        cmd = [sys.executable, script, "--epochs", str(args.epochs), "--continue"]
        return run_command(cmd, f"繼續訓練 ({args.epochs} epochs)")
    
    elif args.action == "test":
        # 測試訓練
        return run_command([sys.executable, "test_training_flow.py", "simple"], "執行測試訓練")
    
    elif args.action == "infer":
        # 推理
        return run_command([sys.executable, "infer_lora_direct.py"], "執行推理")
    
    elif args.action == "clean":
        # 清理
        output_dir = current_dir / "lora_output"
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            print("🧹 已清理輸出目錄")
        else:
            print("📁 輸出目錄不存在")
        return True
    
    return False

if __name__ == "__main__":
    try:
        if main():
            print("✅ 操作完成")
        else:
            print("❌ 操作失敗")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 操作被取消")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        sys.exit(1)
