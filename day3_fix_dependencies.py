#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: 依賴問題修復腳本
修復系統測試中發現的依賴問題
"""

import subprocess
import sys
import os

def run_command(command, description):
    """運行命令並顯示結果"""
    print(f"\n🔧 {description}...")
    print(f"執行: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, encoding='utf-8')
        print(f"✅ {description} 成功")
        if result.stdout:
            print(f"輸出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失敗")
        print(f"錯誤: {e.stderr}")
        return False

def check_python_version():
    """檢查 Python 版本"""
    print(f"🐍 Python 版本: {sys.version}")
    return True

def fix_transformers_version():
    """修復 Transformers 版本問題"""
    print("\n🎯 修復 Transformers 版本兼容性問題")
    print("=" * 50)
    
    # 升級 transformers 到支援 SiglipImageProcessor 的版本
    commands = [
        ("pip install --upgrade transformers>=4.37.0", "升級 Transformers"),
        ("pip install --upgrade diffusers>=0.27.0", "升級 Diffusers"),
        ("pip install accelerate", "安裝 Accelerate"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    return success_count == len(commands)

def install_missing_packages():
    """安裝缺少的套件"""
    print("\n📦 安裝缺少的套件")
    print("=" * 50)
    
    packages = [
        ("pip install seaborn", "安裝 Seaborn"),
        ("pip install scipy", "安裝 SciPy"),
        ("pip install plotly", "安裝 Plotly (可選)"),
        ("pip install tqdm", "安裝 TQDM"),
    ]
    
    success_count = 0
    for command, description in packages:
        if run_command(command, description):
            success_count += 1
    
    return success_count >= 2  # 至少成功安裝 seaborn 和 scipy

def create_requirements_file():
    """創建需求文件"""
    print("\n📄 創建 requirements.txt")
    
    requirements = """# Day 3 Fashion AI Training Suite Requirements
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.37.0
diffusers>=0.27.0
accelerate>=0.20.0
pillow>=10.0.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
numpy>=1.24.0
requests>=2.28.0
tqdm>=4.65.0

# 可選套件
wandb>=0.15.0
tensorboard>=2.13.0
xformers>=0.0.20
"""
    
    with open("day3_requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    
    print("✅ requirements.txt 已創建")
    return True

def test_imports():
    """測試關鍵模組導入"""
    print("\n🧪 測試關鍵模組導入")
    print("=" * 50)
    
    test_modules = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"), 
        ("seaborn", "Seaborn"),
        ("scipy", "SciPy"),
    ]
    
    success_count = 0
    for module, name in test_modules:
        try:
            __import__(module)
            print(f"✅ {name} 導入成功")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name} 導入失敗: {e}")
    
    return success_count >= 4

def test_specific_imports():
    """測試特定的有問題的導入"""
    print("\n🔍 測試具體導入問題")
    print("=" * 50)
    
    try:
        from transformers import SiglipImageProcessor
        print("✅ SiglipImageProcessor 導入成功")
        siglip_ok = True
    except ImportError as e:
        print(f"❌ SiglipImageProcessor 導入失敗: {e}")
        siglip_ok = False
    
    try:
        from diffusers import StableDiffusionPipeline
        print("✅ StableDiffusionPipeline 導入成功")
        pipeline_ok = True
    except ImportError as e:
        print(f"❌ StableDiffusionPipeline 導入失敗: {e}")
        pipeline_ok = False
    
    try:
        import seaborn as sns
        print("✅ Seaborn 導入成功")
        seaborn_ok = True
    except ImportError as e:
        print(f"❌ Seaborn 導入失敗: {e}")
        seaborn_ok = False
    
    return siglip_ok and pipeline_ok and seaborn_ok

def main():
    """主修復流程"""
    print("🔧 Day 3: 依賴問題修復工具")
    print("=" * 60)
    
    # 檢查 Python 版本
    check_python_version()
    
    # 修復步驟
    steps = [
        ("修復 Transformers 版本", fix_transformers_version),
        ("安裝缺少套件", install_missing_packages),
        ("創建需求文件", create_requirements_file),
        ("測試模組導入", test_imports),
        ("測試具體導入", test_specific_imports),
    ]
    
    results = {}
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        results[step_name] = step_func()
    
    # 總結報告
    print(f"\n{'='*60}")
    print("📊 修復結果總結")
    print(f"{'='*60}")
    
    success_count = sum(results.values())
    total_steps = len(steps)
    
    for step_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"   {step_name}: {status}")
    
    print(f"\n總成功率: {success_count}/{total_steps} ({success_count/total_steps*100:.1f}%)")
    
    if success_count == total_steps:
        print("\n🎉 所有問題已修復！")
        print("💡 現在可以重新運行系統測試:")
        print("   python day3_system_test.py")
    else:
        print("\n⚠️  部分問題仍未解決")
        print("💡 建議手動檢查失敗的步驟")
        
        if not results.get("修復 Transformers 版本", False):
            print("\n🔧 手動修復 Transformers:")
            print("   pip uninstall transformers diffusers -y")
            print("   pip install transformers>=4.37.0 diffusers>=0.27.0")
        
        if not results.get("安裝缺少套件", False):
            print("\n📦 手動安裝套件:")
            print("   pip install seaborn scipy")

if __name__ == "__main__":
    main()
