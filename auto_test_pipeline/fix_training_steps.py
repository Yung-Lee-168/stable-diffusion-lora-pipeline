#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速修復 LoRA 訓練步數問題
解決 max_train_steps 錯誤
"""

import os
import json
import shutil

def fix_training_state():
    """修復訓練狀態，允許繼續訓練"""
    
    print("🔧 LoRA 訓練步數問題修復工具")
    print("=" * 50)
    
    state_dir = "lora_output/last-state"
    
    if not os.path.exists(state_dir):
        print("❌ 沒有找到訓練狀態目錄")
        return
    
    # 1. 檢查當前狀態
    state_file = os.path.join(state_dir, "train_state.json")
    if not os.path.exists(state_file):
        print("❌ 沒有找到訓練狀態檔案")
        return
    
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        current_step = state.get('current_step', 0)
        current_epoch = state.get('current_epoch', 0)
        
        print(f"📊 當前訓練狀態:")
        print(f"   當前步數: {current_step}")
        print(f"   當前 epoch: {current_epoch}")
        
        # 2. 提供解決選項
        print(f"\n🎯 解決方案選項:")
        print(f"1. 重置訓練狀態 (從步數 0 開始)")
        print(f"2. 繼續訓練 (自動增加最大步數)")
        print(f"3. 取消操作")
        
        while True:
            choice = input("請選擇 (1/2/3): ").strip()
            
            if choice == "1":
                # 選項 1: 重置狀態
                print(f"\n🔄 重置訓練狀態...")
                
                # 備份當前狀態
                backup_dir = f"lora_output/backup_state_{current_step}steps"
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree(state_dir, backup_dir)
                print(f"📦 已備份當前狀態到: {backup_dir}")
                
                # 刪除狀態目錄
                shutil.rmtree(state_dir)
                print(f"✅ 已重置訓練狀態")
                print(f"💡 現在可以重新開始訓練")
                break
                
            elif choice == "2":
                # 選項 2: 繼續訓練 (已在 train_lora.py 中實現)
                print(f"\n✅ 訓練將繼續進行")
                print(f"💡 train_lora.py 已更新，支持智能步數調整")
                print(f"🎯 下次訓練將從步數 {current_step} 繼續，最大步數會自動增加")
                break
                
            elif choice == "3":
                print(f"\n❌ 操作已取消")
                break
                
            else:
                print("請輸入 1、2 或 3")
    
    except Exception as e:
        print(f"❌ 處理狀態檔案時出錯: {e}")

def main():
    """主函數"""
    fix_training_state()

if __name__ == "__main__":
    main()
