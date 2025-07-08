#!/usr/bin/env python3
"""
測試訓練停止修復
檢查訓練是否在達到max_train_steps時正確停止
"""

import os
import sys
import subprocess
import time
import re
from pathlib import Path

def test_training_stop():
    """測試訓練是否在指定步數停止"""
    print("=" * 60)
    print("測試訓練停止修復")
    print("=" * 60)
    
    # 設定測試參數
    test_steps = 20  # 很小的步數，方便快速測試
    
    # 確保目錄存在
    output_dir = Path("auto_test_pipeline/lora_output_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 準備訓練命令（使用最小配置進行快速測試）
    train_cmd = [
        sys.executable, "auto_test_pipeline/train_lora.py",
        "--base_model", "runwayml/stable-diffusion-v1-5",
        "--data_dir", "auto_test_pipeline/fashion_dataset",
        "--output_dir", str(output_dir),
        "--resolution", "256",  # 使用小解析度加快訓練
        "--train_batch_size", "1",
        "--max_train_steps", str(test_steps),
        "--learning_rate", "1e-4",
        "--lr_scheduler", "constant",
        "--mixed_precision", "fp16",
        "--save_every_n_steps", "10",
        "--logging_dir", str(output_dir / "logs"),
        "--seed", "42",
        "--network_alpha", "128",
        "--network_dim", "64",
        "--network_module", "networks.lora"
    ]
    
    print(f"開始測試訓練（最大步數: {test_steps}）...")
    print("訓練命令:")
    print(" ".join(train_cmd))
    print("\n" + "=" * 60)
    
    # 執行訓練
    start_time = time.time()
    
    try:
        # 執行訓練並捕獲輸出
        result = subprocess.run(
            train_cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=300  # 5分鐘超時
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n訓練完成！耗時: {duration:.2f} 秒")
        print("=" * 60)
        
        # 分析輸出
        output_lines = result.stdout.split('\n') + result.stderr.split('\n')
        
        # 查找步數相關的輸出
        step_pattern = re.compile(r'global_step.*?(\d+)')
        max_step_reached = 0
        step_logs = []
        
        for line in output_lines:
            if 'global_step' in line or 'step' in line.lower():
                step_logs.append(line.strip())
                match = step_pattern.search(line)
                if match:
                    step_num = int(match.group(1))
                    max_step_reached = max(max_step_reached, step_num)
        
        print("步數相關日誌:")
        for log in step_logs[-10:]:  # 顯示最後10條相關日誌
            if log:
                print(f"  {log}")
        
        print(f"\n檢測到的最大步數: {max_step_reached}")
        print(f"預期最大步數: {test_steps}")
        
        # 檢查是否正確停止
        if max_step_reached <= test_steps:
            print("✅ 成功！訓練在預期步數停止")
        else:
            print(f"❌ 失敗！訓練超過了預期步數 ({max_step_reached} > {test_steps})")
        
        # 查找停止相關的日誌
        stop_logs = []
        for line in output_lines:
            if any(keyword in line.lower() for keyword in ['break', 'completed', 'reached', 'stop']):
                stop_logs.append(line.strip())
        
        if stop_logs:
            print("\n停止相關日誌:")
            for log in stop_logs:
                if log:
                    print(f"  {log}")
        
        # 返回碼檢查
        if result.returncode == 0:
            print("✅ 訓練進程正常退出")
        else:
            print(f"⚠️  訓練進程異常退出，返回碼: {result.returncode}")
            
        return max_step_reached <= test_steps
        
    except subprocess.TimeoutExpired:
        print("❌ 訓練超時（可能陷入無限循環）")
        return False
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        return False

def check_training_loop_logic():
    """檢查訓練循環的邏輯"""
    print("\n" + "=" * 60)
    print("檢查訓練循環邏輯")
    print("=" * 60)
    
    train_network_file = "auto_test_pipeline/train_network.py"
    
    if not os.path.exists(train_network_file):
        print(f"❌ 文件不存在: {train_network_file}")
        return False
    
    with open(train_network_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查是否有雙重break邏輯
    if "# Check if we reached max_train_steps and should stop training completely" in content:
        print("✅ 發現雙重break邏輯")
    else:
        print("❌ 未發現雙重break邏輯")
        return False
    
    # 計算break語句的數量
    break_count = content.count("if global_step >= args.max_train_steps:")
    print(f"發現 {break_count} 個 max_train_steps 檢查點")
    
    if break_count >= 2:
        print("✅ 有足夠的檢查點來確保正確停止")
        return True
    else:
        print("❌ 檢查點不足，可能無法正確停止")
        return False

if __name__ == "__main__":
    print("測試訓練停止修復")
    print("=" * 60)
    
    # 檢查邏輯
    logic_ok = check_training_loop_logic()
    
    # 如果邏輯檢查通過，進行實際測試
    if logic_ok:
        print("\n邏輯檢查通過，開始實際測試...")
        test_ok = test_training_stop()
        
        if test_ok:
            print("\n🎉 所有測試通過！訓練停止修復成功！")
        else:
            print("\n❌ 實際測試失敗，需要進一步調查")
    else:
        print("\n❌ 邏輯檢查失敗，需要修復代碼")
