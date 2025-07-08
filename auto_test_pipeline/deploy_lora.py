#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 模型部署到 WebUI
自動將訓練好的 LoRA 模型複製到 WebUI 的模型目錄
"""
import os
import sys
import shutil
from datetime import datetime

def find_latest_lora():
    """找到最新的 LoRA 模型檔案"""
    lora_path = "lora_output"
    if not os.path.exists(lora_path):
        print("❌ 找不到 LoRA 輸出目錄")
        return None
    
    lora_files = [f for f in os.listdir(lora_path) 
                  if f.endswith('.safetensors') and not f.startswith('lora_backup_')]
    if not lora_files:
        print("❌ 沒有找到 LoRA 模型檔案")
        return None
    
    # 找最新的檔案
    latest_lora = max(lora_files, key=lambda x: os.path.getmtime(os.path.join(lora_path, x)))
    lora_full_path = os.path.join(lora_path, latest_lora)
    
    return lora_full_path

def deploy_lora_to_webui():
    """部署 LoRA 到 WebUI"""
    
    print("🚀 開始部署 LoRA 模型到 WebUI...")
    
    # 確保在正確的目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 找到最新的 LoRA 模型
    source_lora = find_latest_lora()
    if not source_lora:
        return False
    
    print(f"📁 找到 LoRA 模型: {os.path.basename(source_lora)}")
    file_size = os.path.getsize(source_lora) / (1024*1024)
    print(f"📊 檔案大小: {file_size:.2f} MB")
    
    # WebUI 的 LoRA 目錄（相對於根目錄）
    webui_lora_dir = "../models/Lora"
    
    # 檢查 WebUI LoRA 目錄是否存在
    if not os.path.exists(webui_lora_dir):
        print(f"❌ WebUI LoRA 目錄不存在: {webui_lora_dir}")
        print("💡 請確認：")
        print("   1. WebUI 已正確安裝")
        print("   2. 此腳本在 auto_test_pipeline 目錄下執行")
        return False
    
    # 生成目標檔名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_name = f"custom_lora_{timestamp}.safetensors"
    target_path = os.path.join(webui_lora_dir, target_name)
    
    try:
        # 複製檔案
        shutil.copy2(source_lora, target_path)
        print(f"✅ 成功複製到: {target_name}")
        
        # 顯示使用說明
        print("\n📋 使用說明：")
        print("1. 重啟 WebUI 或點擊 'Refresh' 按鈕")
        print("2. 在 Additional Networks 或 LoRA 標籤中找到模型")
        print(f"3. 選擇模型: {target_name}")
        print("4. 設定權重: 0.7 - 1.0")
        print("5. 在提示詞中加入觸發詞: test")
        print("\n💡 示例提示詞:")
        print("   test, a beautiful woman, high quality, detailed")
        
        return True
        
    except Exception as e:
        print(f"❌ 複製失敗: {str(e)}")
        return False

def main():
    """主函數"""
    print("=" * 50)
    print("    LoRA 模型部署到 WebUI")
    print("=" * 50)
    
    success = deploy_lora_to_webui()
    
    if success:
        print("\n🎉 部署完成！")
        print("您現在可以在 WebUI 中使用您的自定義 LoRA 模型了。")
    else:
        print("\n❌ 部署失敗！")
        print("請檢查錯誤訊息並重試。")
    
    input("\n按 Enter 鍵退出...")

if __name__ == "__main__":
    main()
