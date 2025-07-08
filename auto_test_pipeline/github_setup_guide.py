#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub 版本控制設置指南
幫助您將 Stable Diffusion LoRA Pipeline 上傳到 GitHub
"""

import os
import subprocess
import sys

def print_git_setup_guide():
    """打印 Git 設置指南"""
    
    print("🔧 GitHub 版本控制設置指南")
    print("=" * 60)
    print("📁 當前項目: Stable Diffusion LoRA Training Pipeline")
    print()
    
    # 1. 檢查 Git 狀態
    print("📋 1. 檢查當前 Git 狀態")
    print("-" * 40)
    
    project_dir = r"e:\Yung_Folder\Project\stable-diffusion-webui"
    print(f"📁 項目目錄: {project_dir}")
    
    if os.path.exists(os.path.join(project_dir, ".git")):
        print("✅ Git 倉庫已初始化")
    else:
        print("❌ Git 倉庫未初始化")
        print("💡 需要執行: git init")
    
    # 2. Git 初始化步驟
    print("\n📋 2. Git 初始化和設置")
    print("-" * 40)
    print("🔧 在項目目錄中執行以下命令:")
    print()
    print("# 切換到項目目錄")
    print(f"cd \"{project_dir}\"")
    print()
    print("# 初始化 Git 倉庫 (如果還沒有)")
    print("git init")
    print()
    print("# 設置用戶信息 (替換為您的信息)")
    print("git config user.name \"Your Name\"")
    print("git config user.email \"your.email@example.com\"")
    print()
    
    # 3. 忽略檔案設置
    print("📋 3. 設置 .gitignore 檔案")
    print("-" * 40)
    print("✅ 建議忽略的檔案類型:")
    print("   • 模型檔案 (*.safetensors, *.ckpt) - 太大")
    print("   • 圖片檔案 (*.png, *.jpg) - 數據隱私")
    print("   • 訓練數據 (lora_train_set/, source_image/) - 個人數據")
    print("   • 輸出結果 (test_images/, lora_output/) - 生成內容")
    print("   • Python 快取 (__pycache__/, *.pyc)")
    print("   • IDE 設定 (.vscode/, .idea/)")
    
    # 4. 提交步驟
    print("\n📋 4. 首次提交步驟")
    print("-" * 40)
    print("# 添加所有檔案")
    print("git add .")
    print()
    print("# 檢查狀態")
    print("git status")
    print()
    print("# 首次提交")
    print("git commit -m \"初始提交: LoRA 訓練 Pipeline 完整實現\"")
    print()
    
    # 5. GitHub 設置
    print("📋 5. GitHub 倉庫設置")
    print("-" * 40)
    print("🌐 在 GitHub 上:")
    print("   1. 登入 GitHub.com")
    print("   2. 點擊 '+' → 'New repository'")
    print("   3. 倉庫名稱: stable-diffusion-lora-pipeline")
    print("   4. 描述: Complete LoRA training pipeline for Stable Diffusion")
    print("   5. 選擇 Public 或 Private")
    print("   6. 不要勾選 'Initialize with README' (我們已有檔案)")
    print("   7. 點擊 'Create repository'")
    print()
    
    # 6. 連接遠端倉庫
    print("📋 6. 連接 GitHub 遠端倉庫")
    print("-" * 40)
    print("# 添加遠端倉庫 (替換 YOUR_USERNAME)")
    print("git remote add origin https://github.com/YOUR_USERNAME/stable-diffusion-lora-pipeline.git")
    print()
    print("# 推送到 GitHub")
    print("git branch -M main")
    print("git push -u origin main")
    print()
    
    # 7. 日常使用
    print("📋 7. 日常版本控制命令")
    print("-" * 40)
    print("# 檢查狀態")
    print("git status")
    print()
    print("# 添加修改的檔案")
    print("git add .")
    print("# 或添加特定檔案")
    print("git add auto_test_pipeline/train_lora.py")
    print()
    print("# 提交變更")
    print("git commit -m \"描述您的修改\"")
    print()
    print("# 推送到 GitHub")
    print("git push")
    print()
    print("# 查看提交歷史")
    print("git log --oneline")
    print()
    
    # 8. 回滾操作
    print("📋 8. 回滾操作 (恢復之前版本)")
    print("-" * 40)
    print("# 查看提交歷史")
    print("git log --oneline")
    print()
    print("# 回滾到特定提交 (替換 COMMIT_HASH)")
    print("git reset --hard COMMIT_HASH")
    print()
    print("# 回滾單個檔案")
    print("git checkout HEAD~1 -- auto_test_pipeline/train_lora.py")
    print()
    print("# 創建新分支進行實驗")
    print("git checkout -b experimental-feature")
    print()
    print("# 切換回主分支")
    print("git checkout main")
    print()
    
    # 9. 重要提醒
    print("📋 9. 重要提醒和最佳實踐")
    print("-" * 40)
    print("⚠️  注意事項:")
    print("   • 不要上傳大型模型檔案 (>100MB)")
    print("   • 不要上傳個人訓練數據或圖片")
    print("   • 定期提交代碼變更")
    print("   • 使用有意義的提交訊息")
    print("   • 在重大修改前創建分支")
    print()
    print("✅ 好處:")
    print("   • 代碼歷史追蹤")
    print("   • 多設備同步")
    print("   • 回滾到任何版本")
    print("   • 團隊協作")
    print("   • 自動備份")
    
    # 10. 範例提交訊息
    print("\n📋 10. 範例提交訊息")
    print("-" * 40)
    print("好的提交訊息範例:")
    print("   • '修復 train_lora.py 中的圖片尺寸檢查邏輯'")
    print("   • '統一 SSIM 計算方法在訓練和評估中'")
    print("   • '添加 FashionCLIP 標籤匹配一致性檢查'")
    print("   • '優化 LoRA 訓練參數配置'")
    print("   • '更新 README 文檔和使用指南'")
    
    print("\n🎯 立即開始:")
    print(f"1. 開啟命令提示字元")
    print(f"2. 切換到: {project_dir}")
    print(f"3. 執行上述 Git 命令")
    print(f"4. 在 GitHub 建立倉庫")
    print(f"5. 推送代碼到 GitHub")

def main():
    """主函數"""
    print_git_setup_guide()

if __name__ == "__main__":
    main()
