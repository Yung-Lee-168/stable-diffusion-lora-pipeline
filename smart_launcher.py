#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3 Fashion AI Training - 智能啟動器
自動檢測硬體並推薦最適合的訓練方式
"""

import torch
import os
import sys

def check_system_capabilities():
    """檢測系統能力"""
    print("🔍 檢測系統配置...")
    
    capabilities = {
        "has_gpu": False,
        "gpu_name": "",
        "gpu_memory_gb": 0,
        "recommended_mode": "colab"
    }
    
    if torch.cuda.is_available():
        capabilities["has_gpu"] = True
        capabilities["gpu_name"] = torch.cuda.get_device_name()
        capabilities["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✅ GPU: {capabilities['gpu_name']}")
        print(f"💾 VRAM: {capabilities['gpu_memory_gb']:.1f} GB")
        
        # 根據 GPU 記憶體推薦模式
        if capabilities["gpu_memory_gb"] >= 16:
            capabilities["recommended_mode"] = "local_advanced"
        elif capabilities["gpu_memory_gb"] >= 8:
            capabilities["recommended_mode"] = "local_basic"
        elif capabilities["gpu_memory_gb"] >= 6:
            capabilities["recommended_mode"] = "local_minimal"
        else:
            capabilities["recommended_mode"] = "colab"
    else:
        print("❌ 沒有檢測到 GPU")
        capabilities["recommended_mode"] = "colab"
    
    return capabilities

def show_recommendations(capabilities):
    """顯示建議"""
    print("\n🎯 建議的訓練方式:")
    print("=" * 50)
    
    if capabilities["recommended_mode"] == "colab":
        print("🌐 **Google Colab 版本** (強烈推薦)")
        print("   原因: GPU 記憶體不足或沒有 GPU")
        print("   優勢: 免費 16GB+ GPU, 自動配置, 穩定可靠")
        print("   步驟: 上傳 Day3_Fashion_AI_Colab.ipynb 到 Colab")
        
    elif capabilities["recommended_mode"] == "local_minimal":
        print("⚡ **本地最小配置**")
        print("   適用: 您的配置基本滿足要求")
        print("   建議: LoRA rank=4, batch_size=1")
        print("   備選: 如果仍有問題，建議使用 Colab")
        
    elif capabilities["recommended_mode"] == "local_basic":
        print("🔧 **本地標準配置**")
        print("   適用: 您的配置良好")
        print("   建議: LoRA rank=8, batch_size=2")
        
    elif capabilities["recommended_mode"] == "local_advanced":
        print("🚀 **本地高級配置**")
        print("   適用: 您擁有高端 GPU")
        print("   建議: LoRA rank=16, batch_size=4")
        print("   可選: 甚至可以嘗試完整微調")

def show_menu():
    """顯示選單"""
    print("\n📋 選擇執行模式:")
    print("1. 🌐 Google Colab 版本 (推薦)")
    print("2. 💻 本地提示詞優化訓練")
    print("3. 🔧 本地 SD v1.5 微調")
    print("4. 📊 系統狀態檢查")
    print("5. 📚 查看使用指南")
    print("0. 退出")

def launch_colab_guide():
    """顯示 Colab 使用指南"""
    print("\n🌐 Google Colab 使用步驟:")
    print("=" * 40)
    print("1. 開啟 Google Colab (colab.research.google.com)")
    print("2. 上傳 Day3_Fashion_AI_Colab.ipynb")
    print("3. 設置 GPU 運行時 (執行階段 → 變更執行階段類型 → GPU)")
    print("4. 按順序執行所有 Cell")
    print("5. 上傳您的時尚圖片")
    print("6. 等待訓練完成並下載結果")
    
    print("\n📁 需要的檔案:")
    if os.path.exists("Day3_Fashion_AI_Colab.ipynb"):
        print("✅ Day3_Fashion_AI_Colab.ipynb")
    else:
        print("❌ Day3_Fashion_AI_Colab.ipynb (請先創建)")
    
    if os.path.exists("Colab_Deployment_Guide.md"):
        print("✅ Colab_Deployment_Guide.md (詳細指南)")
    else:
        print("❌ Colab_Deployment_Guide.md (請先創建)")

def launch_local_prompt():
    """啟動本地提示詞優化"""
    print("\n💻 啟動本地提示詞優化訓練...")
    
    if os.path.exists("day3_fashion_training.py"):
        print("✅ 找到 day3_fashion_training.py")
        os.system("python day3_fashion_training.py")
    else:
        print("❌ 找不到 day3_fashion_training.py")

def launch_local_finetuning():
    """啟動本地微調"""
    print("\n🔧 啟動本地 SD v1.5 微調...")
    
    capabilities = check_system_capabilities()
    
    if capabilities["gpu_memory_gb"] < 6:
        print("⚠️ 警告: GPU 記憶體可能不足")
        print("🌐 強烈建議使用 Google Colab 版本")
        
        choice = input("是否仍要繼續? (y/N): ").strip().lower()
        if choice not in ['y', 'yes']:
            return
    
    if os.path.exists("day3_real_finetuning.py"):
        print("✅ 找到 day3_real_finetuning.py")
        os.system("python day3_real_finetuning.py")
    else:
        print("❌ 找不到 day3_real_finetuning.py")

def check_system_status():
    """檢查系統狀態"""
    print("\n📊 系統狀態檢查...")
    
    # 檢查檔案
    required_files = [
        "day3_fashion_training.py",
        "Day3_Fashion_AI_Colab.ipynb", 
        "Colab_Deployment_Guide.md"
    ]
    
    print("\n📁 檔案檢查:")
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
    
    # 檢查硬體
    capabilities = check_system_capabilities()
    
    # 檢查來源圖片
    if os.path.exists("day1_results"):
        image_files = [f for f in os.listdir("day1_results") 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\n🖼️ 來源圖片: {len(image_files)} 張")
    else:
        print("\n🖼️ 來源圖片: day1_results 目錄不存在")

def show_guide():
    """顯示使用指南"""
    print("\n📚 Day 3 Fashion AI Training 使用指南")
    print("=" * 50)
    
    print("\n🎯 兩種訓練方式:")
    print("1. 提示詞優化訓練 - 優化生成提示詞，不修改模型權重")
    print("2. 真正的模型微調 - 使用 LoRA 微調 SD v1.5 模型權重")
    
    print("\n🌐 Google Colab vs 本地訓練:")
    print("Google Colab 優勢:")
    print("  ✅ 免費 16GB GPU (T4)")
    print("  ✅ 自動配置和優化")
    print("  ✅ 無需本地環境設置")
    print("  ✅ 穩定且可靠")
    
    print("\n本地訓練優勢:")
    print("  ✅ 完全控制訓練過程")
    print("  ✅ 無網路依賴")
    print("  ✅ 可以長時間訓練")
    print("  ❗ 需要足夠的 GPU 記憶體")
    
    print("\n💡 建議:")
    print("  - GPU VRAM ≤ 4GB: 必須使用 Colab")
    print("  - GPU VRAM 6-8GB: Colab 或本地最小配置") 
    print("  - GPU VRAM ≥ 16GB: 可以選擇任何方式")

def main():
    """主函數"""
    print("🎨 Day 3: Fashion AI Training - 智能啟動器")
    print("=" * 55)
    
    # 檢測系統能力
    capabilities = check_system_capabilities()
    
    # 顯示建議
    show_recommendations(capabilities)
    
    while True:
        try:
            show_menu()
            choice = input("\n請選擇 (0-5): ").strip()
            
            if choice == "0":
                print("👋 再見！")
                break
            elif choice == "1":
                launch_colab_guide()
            elif choice == "2":
                launch_local_prompt()
            elif choice == "3":
                launch_local_finetuning()
            elif choice == "4":
                check_system_status()
            elif choice == "5":
                show_guide()
            else:
                print("❌ 無效選擇，請重新輸入")
                
        except KeyboardInterrupt:
            print("\n👋 使用者中斷，再見！")
            break
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    main()
