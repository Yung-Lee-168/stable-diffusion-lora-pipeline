#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA訓練日誌檢查器
用於分析TensorBoard日誌，檢查每步loss記錄和訓練進度
"""

import os
import sys
import argparse
from pathlib import Path

def check_tensorboard_logs(log_dir="lora_output/logs"):
    """檢查TensorBoard日誌文件"""
    print(f"🔍 檢查TensorBoard日誌目錄: {log_dir}")
    
    if not os.path.exists(log_dir):
        print(f"❌ 日誌目錄不存在: {log_dir}")
        return False
    
    # 尋找TensorBoard事件文件
    tb_files = []
    for file in os.listdir(log_dir):
        if file.startswith('events.out.tfevents'):
            tb_files.append(file)
            file_path = os.path.join(log_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"📊 找到TensorBoard文件: {file} ({file_size:.1f} KB)")
    
    if not tb_files:
        print(f"❌ 沒有找到TensorBoard日誌文件")
        print(f"💡 請確認訓練時有設定 --logging_dir 參數")
        return False
    
    print(f"✅ 找到 {len(tb_files)} 個TensorBoard日誌文件")
    return True

def extract_loss_data(log_dir="lora_output/logs"):
    """提取loss數據（需要tensorboard庫）"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        print(f"📈 正在提取loss數據...")
        
        # 加載TensorBoard數據
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        # 獲取所有可用的標量標籤
        scalar_tags = ea.Tags()['scalars']
        print(f"📋 可用的數據標籤: {scalar_tags}")
        
        # 尋找loss相關的標籤
        loss_tags = [tag for tag in scalar_tags if 'loss' in tag.lower()]
        
        if not loss_tags:
            print(f"❌ 沒有找到loss相關的數據")
            return
        
        print(f"📊 找到loss標籤: {loss_tags}")
        
        # 提取每個loss標籤的數據
        for tag in loss_tags:
            scalar_events = ea.Scalars(tag)
            print(f"\n📈 {tag}:")
            print(f"   總記錄數: {len(scalar_events)}")
            
            if scalar_events:
                steps = [event.step for event in scalar_events]
                values = [event.value for event in scalar_events]
                
                print(f"   步數範圍: {min(steps)} - {max(steps)}")
                print(f"   Loss範圍: {min(values):.6f} - {max(values):.6f}")
                
                # 顯示最近幾步的數據
                if len(scalar_events) > 0:
                    print(f"   最近5步:")
                    for event in scalar_events[-5:]:
                        print(f"     步數 {event.step}: {event.value:.6f}")
        
        return True
        
    except ImportError:
        print(f"⚠️ 未安裝tensorboard庫，無法解析日誌數據")
        print(f"💡 安裝方法: pip install tensorboard")
        return False
    except Exception as e:
        print(f"❌ 解析TensorBoard數據時出錯: {e}")
        return False

def show_training_summary(log_dir="lora_output/logs", output_dir="lora_output"):
    """顯示完整的訓練摘要"""
    print("\n" + "="*60)
    print("📊 LoRA 訓練日誌分析報告")
    print("="*60)
    
    # 1. 檢查輸出目錄
    print(f"\n🎯 檢查輸出目錄: {output_dir}")
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        lora_files = [f for f in files if f.endswith('.safetensors')]
        state_dirs = [f for f in files if os.path.isdir(os.path.join(output_dir, f)) and not f.startswith('.')]
        
        print(f"   LoRA文件: {len(lora_files)} 個")
        for lora_file in lora_files:
            file_path = os.path.join(output_dir, lora_file)
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"     📄 {lora_file} ({file_size:.2f} MB)")
        
        print(f"   狀態目錄: {len(state_dirs)} 個")
        for state_dir in state_dirs:
            print(f"     📁 {state_dir}")
    else:
        print(f"❌ 輸出目錄不存在")
    
    # 2. 檢查TensorBoard日誌
    print(f"\n📊 檢查訓練日誌:")
    log_exists = check_tensorboard_logs(log_dir)
    
    if log_exists:
        # 3. 提取loss數據
        print(f"\n📈 分析loss數據:")
        extract_loss_data(log_dir)
    
    # 4. 使用說明
    print(f"\n💡 如何查看詳細的訓練曲線:")
    print(f"   1. 打開命令提示符")
    print(f"   2. 切換到日誌目錄: cd {os.path.abspath(log_dir)}")
    print(f"   3. 啟動TensorBoard: tensorboard --logdir .")
    print(f"   4. 在瀏覽器中打開: http://localhost:6006")
    
    print("="*60)

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="LoRA訓練日誌檢查器")
    parser.add_argument("--log_dir", "-l", default="lora_output/logs",
                       help="TensorBoard日誌目錄路徑")
    parser.add_argument("--output_dir", "-o", default="lora_output",
                       help="LoRA輸出目錄路徑")
    parser.add_argument("--extract_only", action="store_true",
                       help="只提取loss數據，不顯示完整報告")
    
    args = parser.parse_args()
    
    if args.extract_only:
        extract_loss_data(args.log_dir)
    else:
        show_training_summary(args.log_dir, args.output_dir)

if __name__ == "__main__":
    main()
