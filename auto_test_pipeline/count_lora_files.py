#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 訓練結果文件計數器
精確計算 LoRA 訓練完成後會產生多少個文件
"""

import os
import sys
from pathlib import Path

def count_lora_output_files(output_dir="lora_output"):
    """統計 LoRA 輸出目錄中的所有文件"""
    
    if not os.path.exists(output_dir):
        print(f"❌ 輸出目錄不存在: {output_dir}")
        return
    
    print(f"📊 統計 LoRA 訓練輸出文件")
    print("=" * 50)
    
    total_files = 0
    total_size = 0
    
    # 1. 主要 LoRA 模型文件 (.safetensors)
    lora_files = []
    for file in os.listdir(output_dir):
        if file.endswith('.safetensors') and os.path.isfile(os.path.join(output_dir, file)):
            lora_files.append(file)
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
    
    print(f"\n🎯 主要 LoRA 模型文件:")
    print(f"   數量: {len(lora_files)} 個")
    for lora_file in lora_files:
        file_path = os.path.join(output_dir, lora_file)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"   📄 {lora_file} ({file_size:.2f} MB)")
    total_files += len(lora_files)
    
    # 2. 訓練狀態目錄中的文件
    state_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and not item.startswith('logs'):
            state_dirs.append(item_path)
    
    state_files_total = 0
    print(f"\n🔄 訓練狀態目錄:")
    print(f"   狀態目錄數量: {len(state_dirs)} 個")
    
    for state_dir in state_dirs:
        if os.path.exists(state_dir):
            state_files = []
            for root, dirs, files in os.walk(state_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    state_files.append(file_path)
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
            
            state_files_total += len(state_files)
            dir_name = os.path.basename(state_dir)
            print(f"   📁 {dir_name}: {len(state_files)} 個文件")
            
            # 顯示主要文件類型
            file_types = {}
            for file_path in state_files:
                ext = os.path.splitext(file_path)[1] or '無副檔名'
                file_types[ext] = file_types.get(ext, 0) + 1
            
            print(f"      文件類型: {dict(file_types)}")
    
    total_files += state_files_total
    
    # 3. TensorBoard 日誌文件
    log_dir = os.path.join(output_dir, "logs")
    log_files_total = 0
    
    print(f"\n📊 TensorBoard 日誌:")
    if os.path.exists(log_dir):
        log_files = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                file_path = os.path.join(root, file)
                log_files.append(file_path)
                file_size = os.path.getsize(file_path)
                total_size += file_size
        
        log_files_total = len(log_files)
        print(f"   日誌文件數量: {log_files_total} 個")
        
        # 按文件類型分類
        tb_events = [f for f in log_files if 'events.out.tfevents' in os.path.basename(f)]
        other_logs = [f for f in log_files if 'events.out.tfevents' not in os.path.basename(f)]
        
        print(f"   📈 TensorBoard 事件文件: {len(tb_events)} 個")
        print(f"   📝 其他日誌文件: {len(other_logs)} 個")
        
        # 顯示一些具體文件
        if tb_events:
            for tb_file in tb_events[:3]:  # 只顯示前3個
                file_size = os.path.getsize(tb_file) / 1024  # KB
                print(f"      📊 {os.path.basename(tb_file)} ({file_size:.1f} KB)")
    else:
        print(f"   ❌ 日誌目錄不存在")
    
    total_files += log_files_total
    
    # 4. 檢查備份文件
    backup_files_total = 0
    backup_patterns = ["lora_backup_", "backup_", "_backup"]
    
    print(f"\n🗄️ 備份文件:")
    backup_files = []
    for file in os.listdir(output_dir):
        if any(pattern in file for pattern in backup_patterns) and file.endswith('.safetensors'):
            backup_files.append(file)
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
    
    backup_files_total = len(backup_files)
    print(f"   備份文件數量: {backup_files_total} 個")
    for backup_file in backup_files:
        file_path = os.path.join(output_dir, backup_file)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"   🗂️ {backup_file} ({file_size:.2f} MB)")
    
    total_files += backup_files_total
    
    # 5. 其他雜項文件
    other_files = []
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            # 排除已統計的文件
            if (not file.endswith('.safetensors') and 
                not any(pattern in file for pattern in backup_patterns)):
                other_files.append(file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
    
    print(f"\n📄 其他文件:")
    print(f"   其他文件數量: {len(other_files)} 個")
    for other_file in other_files:
        file_path = os.path.join(output_dir, other_file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"   📋 {other_file} ({file_size:.1f} KB)")
    
    total_files += len(other_files)
    
    # 總結
    print("\n" + "=" * 50)
    print(f"📈 總計統計:")
    print(f"   🎯 LoRA 模型文件: {len(lora_files)} 個")
    print(f"   🔄 狀態文件: {state_files_total} 個")
    print(f"   📊 日誌文件: {log_files_total} 個")
    print(f"   🗄️ 備份文件: {backup_files_total} 個")
    print(f"   📄 其他文件: {len(other_files)} 個")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   📊 總文件數量: {total_files} 個")
    print(f"   💾 總占用空間: {total_size / (1024*1024):.2f} MB")
    print("=" * 50)

def estimate_typical_file_count():
    """估算典型 LoRA 訓練會產生的文件數量"""
    print(f"\n🎯 典型 LoRA 訓練文件數量估算:")
    print("=" * 40)
    
    print(f"📋 基於訓練參數的典型文件數量:")
    print(f"   🎯 LoRA 模型文件:")
    print(f"      - last.safetensors (最終模型): 1 個")
    print(f"      - 如有 save_every_n_steps: 可能 2-5 個")
    
    print(f"   🔄 訓練狀態目錄 (1個目錄包含):")
    print(f"      - optimizer.pt: 1 個")
    print(f"      - train_state.json: 1 個")
    print(f"      - random_states.pkl: 1 個")
    print(f"      - lr_scheduler.pt: 1 個")
    print(f"      - accelerate相關文件: 2-3 個")
    print(f"      - 其他狀態文件: 1-2 個")
    print(f"      小計: 約 7-10 個文件")
    
    print(f"   📊 TensorBoard 日誌:")
    print(f"      - events.out.tfevents.* : 1 個")
    print(f"      - 其他元數據文件: 0-2 個")
    print(f"      小計: 約 1-3 個文件")
    
    print(f"   🗄️ 備份文件:")
    print(f"      - 訓練前備份: 0-1 個")
    
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   📊 典型總數: 10-20 個文件")
    print(f"   💾 典型大小: 50-200 MB")
    print("=" * 40)
    
    print(f"\n💡 影響文件數量的因素:")
    print(f"   • save_every_n_steps: 設定越小，模型文件越多")
    print(f"   • 訓練時間長短: 日誌文件可能更多")
    print(f"   • 是否繼續訓練: 備份文件數量")
    print(f"   • 訓練參數複雜度: 狀態文件大小")

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA 訓練結果文件計數器")
    parser.add_argument("--output_dir", "-o", default="lora_output",
                       help="LoRA 輸出目錄路徑")
    parser.add_argument("--estimate", "-e", action="store_true",
                       help="只顯示典型估算，不掃描實際文件")
    
    args = parser.parse_args()
    
    if args.estimate:
        estimate_typical_file_count()
    else:
        count_lora_output_files(args.output_dir)
        estimate_typical_file_count()

if __name__ == "__main__":
    main()
