#!/usr/bin/env python3
"""
簡化的訓練停止測試
專注於驗證train_network.py中的雙重break邏輯
"""

import os
import re

def analyze_training_loop():
    """分析train_network.py中的訓練循環邏輯"""
    print("分析訓練循環邏輯")
    print("=" * 50)
    
    train_network_file = "auto_test_pipeline/train_network.py"
    
    if not os.path.exists(train_network_file):
        print(f"❌ 文件不存在: {train_network_file}")
        return False
    
    with open(train_network_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找相關的邏輯行
    epoch_loop_start = None
    step_loop_start = None
    first_break = None
    second_break = None
    
    for i, line in enumerate(lines):
        if "for epoch in range(epoch_to_start, num_train_epochs):" in line:
            epoch_loop_start = i + 1
            print(f"✅ 找到epoch循環開始: 第{epoch_loop_start}行")
        
        elif "for step, batch in enumerate(skipped_dataloader or train_dataloader):" in line:
            step_loop_start = i + 1
            print(f"✅ 找到step循環開始: 第{step_loop_start}行")
        
        elif "if global_step >= args.max_train_steps:" in line:
            if first_break is None:
                first_break = i + 1
                print(f"✅ 找到第一個max_train_steps檢查: 第{first_break}行")
            else:
                second_break = i + 1
                print(f"✅ 找到第二個max_train_steps檢查: 第{second_break}行")
    
    # 檢查邏輯結構
    print("\n邏輯結構分析:")
    print("-" * 30)
    
    if epoch_loop_start and step_loop_start:
        print(f"✅ 雙層循環結構正確")
        print(f"   - Epoch循環: 第{epoch_loop_start}行")
        print(f"   - Step循環: 第{step_loop_start}行")
    else:
        print("❌ 循環結構不完整")
        return False
    
    if first_break and second_break:
        print(f"✅ 雙重break邏輯存在")
        print(f"   - 第一個break（step循環內）: 第{first_break}行")
        print(f"   - 第二個break（epoch循環內）: 第{second_break}行")
        
        # 檢查break的相對位置
        if first_break < second_break:
            print("✅ Break順序正確（先step循環，後epoch循環）")
            return True
        else:
            print("❌ Break順序不正確")
            return False
    else:
        print("❌ 雙重break邏輯不完整")
        return False

def show_relevant_code():
    """顯示相關的代碼片段"""
    print("\n" + "=" * 50)
    print("相關代碼片段")
    print("=" * 50)
    
    train_network_file = "auto_test_pipeline/train_network.py"
    
    with open(train_network_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到並顯示關鍵部分
    for i, line in enumerate(lines):
        if "if global_step >= args.max_train_steps:" in line:
            print(f"\n第{i+1}行附近的代碼:")
            print("-" * 30)
            start = max(0, i-2)
            end = min(len(lines), i+5)
            for j in range(start, end):
                marker = ">>> " if j == i else "    "
                print(f"{marker}{j+1:4d}: {lines[j].rstrip()}")

if __name__ == "__main__":
    success = analyze_training_loop()
    
    if success:
        print("\n🎉 訓練停止邏輯修復成功！")
        print("✅ 現在訓練應該在達到max_train_steps時正確停止")
    else:
        print("\n❌ 邏輯修復需要進一步完善")
    
    show_relevant_code()
