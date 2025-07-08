#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示詞優化訓練執行狀態檢查
檢查系統準備情況並提供運行建議
"""

import os
import sys
import json
from datetime import datetime

def check_environment():
    """檢查環境準備情況"""
    print("🔧 檢查執行環境...")
    
    checks = {}
    
    # 檢查必要目錄
    checks["source_dir"] = os.path.exists("day1_results")
    checks["output_dir"] = True  # 會自動創建
    
    # 檢查來源圖片
    if checks["source_dir"]:
        image_files = [f for f in os.listdir("day1_results") 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        checks["has_images"] = len(image_files) > 0
        checks["image_count"] = len(image_files)
    else:
        checks["has_images"] = False
        checks["image_count"] = 0
    
    # 檢查必要檔案
    checks["main_script"] = os.path.exists("day3_fashion_training.py")
    checks["demo_script"] = os.path.exists("demo_prompt_optimization.py")
    checks["config_file"] = os.path.exists("prompt_optimization_config.json")
    
    return checks

def check_dependencies():
    """檢查依賴套件"""
    print("📦 檢查依賴套件...")
    
    required_packages = [
        "torch", "transformers", "PIL", "numpy", 
        "opencv-python", "scikit-learn", "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
            elif package == "opencv-python":
                import cv2
            elif package == "scikit-learn":
                import sklearn
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (缺失)")
            missing_packages.append(package)
    
    return missing_packages

def check_sd_webui():
    """檢查 Stable Diffusion WebUI 狀態"""
    print("🎨 檢查 Stable Diffusion WebUI...")
    
    try:
        import requests
        response = requests.get("http://localhost:7860", timeout=5)
        if response.status_code == 200:
            print("   ✅ WebUI 正在運行")
            return True
        else:
            print("   ❌ WebUI 響應異常")
            return False
    except:
        print("   ❌ WebUI 未運行 (localhost:7860)")
        return False

def generate_status_report():
    """生成狀態報告"""
    print("\n📊 生成狀態報告...")
    
    env_checks = check_environment()
    missing_deps = check_dependencies()
    webui_status = check_sd_webui()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "environment": env_checks,
        "missing_dependencies": missing_deps,
        "webui_running": webui_status,
        "ready_to_run": (
            env_checks["main_script"] and 
            env_checks["has_images"] and 
            len(missing_deps) == 0 and
            webui_status
        )
    }
    
    # 保存報告
    with open("system_status_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report

def print_status_summary(report):
    """打印狀態摘要"""
    print("\n🎯 系統狀態摘要")
    print("=" * 50)
    
    # 環境檢查
    env = report["environment"]
    print(f"📁 來源目錄: {'✅' if env['source_dir'] else '❌'}")
    print(f"🖼️ 圖片數量: {env['image_count']}")
    print(f"📄 主要腳本: {'✅' if env['main_script'] else '❌'}")
    print(f"🎮 演示腳本: {'✅' if env['demo_script'] else '❌'}")
    
    # 依賴檢查
    deps = report["missing_dependencies"]
    print(f"📦 依賴套件: {'✅ 完整' if len(deps) == 0 else f'❌ 缺少 {len(deps)} 個'}")
    
    # WebUI 狀態
    print(f"🎨 SD WebUI: {'✅ 運行中' if report['webui_running'] else '❌ 未運行'}")
    
    # 整體準備情況
    ready = report["ready_to_run"]
    print(f"\n🚀 執行準備: {'✅ 就緒' if ready else '❌ 未就緒'}")
    
    if not ready:
        print("\n⚠️ 需要處理的問題:")
        if not env["source_dir"]:
            print("   • 創建 day1_results 目錄並放入圖片")
        if not env["has_images"]:
            print("   • 在 day1_results 中放入圖片檔案")
        if len(deps) > 0:
            print(f"   • 安裝缺失套件: {', '.join(deps)}")
        if not report["webui_running"]:
            print("   • 啟動 Stable Diffusion WebUI (localhost:7860)")

def print_usage_instructions():
    """打印使用說明"""
    print("\n📚 使用說明")
    print("=" * 50)
    print("1. 基礎執行:")
    print("   python day3_fashion_training.py")
    print()
    print("2. 互動式演示:")
    print("   python demo_prompt_optimization.py")
    print()
    print("3. 提示詞配置比較:")
    print("   configs = ['minimal_prompt', 'high_confidence_only']")
    print("   pipeline.compare_prompt_configs(image_path, configs)")
    print()
    print("4. 權重方案比較:")
    print("   schemes = ['default', 'fashion_focused']")
    print("   pipeline.compare_weight_schemes(image_path, schemes)")

def main():
    """主函數"""
    print("🔍 Day 3 提示詞優化訓練 - 系統狀態檢查")
    print("=" * 60)
    
    # 生成並顯示狀態報告
    report = generate_status_report()
    print_status_summary(report)
    
    # 顯示使用說明
    print_usage_instructions()
    
    print(f"\n📄 詳細報告已保存: system_status_report.json")

if __name__ == "__main__":
    main()
