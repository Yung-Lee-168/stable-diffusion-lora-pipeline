#!/usr/bin/env python3
"""
Conda環境檢查和LoRA訓練準備腳本
確保在正確的環境中運行訓練
"""

import sys
import os
import subprocess
import platform

def check_environment():
    """全面檢查當前環境"""
    print("=" * 60)
    print("🔍 Conda環境和依賴檢查")
    print("=" * 60)
    
    # 基本Python信息
    print(f"🐍 Python版本: {platform.python_version()}")
    print(f"📁 Python路徑: {sys.executable}")
    print(f"🖥️  操作系統: {platform.system()} {platform.release()}")
    
    # Conda環境信息
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"🐻 當前Conda環境: {conda_env}")
    else:
        print("⚠️  警告: 未檢測到Conda環境")
    
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        print(f"📦 Conda前綴: {conda_prefix}")
    
    print("\n" + "-" * 40)
    print("📚 關鍵依賴檢查")
    print("-" * 40)
    
    # 檢查關鍵依賴
    dependencies = [
        ('torch', 'PyTorch'),
        ('diffusers', 'Diffusers'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm')
    ]
    
    missing_deps = []
    
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', '未知版本')
            print(f"✅ {display_name}: {version}")
            
            # 特別檢查PyTorch的CUDA支持
            if module_name == 'torch':
                cuda_available = module.cuda.is_available()
                print(f"   🎮 CUDA可用: {cuda_available}")
                if cuda_available:
                    print(f"   📱 GPU數量: {module.cuda.device_count()}")
                    for i in range(module.cuda.device_count()):
                        gpu_name = module.cuda.get_device_name(i)
                        print(f"   🔥 GPU {i}: {gpu_name}")
                        
        except ImportError:
            print(f"❌ {display_name}: 未安裝")
            missing_deps.append(display_name)
    
    if missing_deps:
        print(f"\n⚠️  缺少依賴: {', '.join(missing_deps)}")
        return False
    
    print("\n✅ 所有關鍵依賴都已安裝")
    return True

def get_conda_env_list():
    """獲取Conda環境列表"""
    try:
        result = subprocess.run(['conda', 'env', 'list'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("\n📋 可用的Conda環境:")
            for line in result.stdout.split('\n'):
                if line.strip() and not line.startswith('#'):
                    print(f"   {line}")
        else:
            print("⚠️  無法獲取Conda環境列表")
    except FileNotFoundError:
        print("❌ 未找到Conda命令")

def install_missing_dependencies():
    """安裝缺少的依賴"""
    print("\n🔧 自動安裝依賴選項:")
    print("1. 安裝PyTorch (CUDA 11.8)")
    print("2. 安裝Diffusers相關包")
    print("3. 安裝所有依賴")
    print("4. 跳過")
    
    choice = input("請選擇 (1-4): ").strip()
    
    if choice == "1":
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        print(f"執行: {cmd}")
        os.system(cmd)
    elif choice == "2":
        cmd = "pip install diffusers transformers accelerate"
        print(f"執行: {cmd}")
        os.system(cmd)
    elif choice == "3":
        commands = [
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "pip install diffusers transformers accelerate",
            "pip install pillow numpy tqdm"
        ]
        for cmd in commands:
            print(f"執行: {cmd}")
            os.system(cmd)

def run_training():
    """運行LoRA訓練"""
    print("\n🚀 LoRA訓練選項:")
    print("1. 新訓練")
    print("2. 繼續訓練")
    print("3. 返回環境檢查")
    
    choice = input("請選擇 (1-3): ").strip()
    
    if choice == "1":
        print("🆕 啟動新訓練...")
        os.system(f'"{sys.executable}" auto_test_pipeline/train_lora.py --new')
    elif choice == "2":
        print("🔄 啟動繼續訓練...")
        os.system(f'"{sys.executable}" auto_test_pipeline/train_lora.py --continue')
    elif choice == "3":
        return main()
    else:
        print("❌ 無效選擇")

def main():
    """主程序"""
    print("🎯 Conda環境LoRA訓練準備")
    
    # 檢查環境
    env_ok = check_environment()
    
    # 顯示Conda環境
    get_conda_env_list()
    
    if not env_ok:
        print("\n❌ 環境檢查失敗")
        install_missing_dependencies()
        return
    
    print("\n✅ 環境檢查通過")
    
    # 詢問下一步
    print("\n📋 下一步操作:")
    print("1. 開始LoRA訓練")
    print("2. 重新檢查環境")
    print("3. 退出")
    
    choice = input("請選擇 (1-3): ").strip()
    
    if choice == "1":
        run_training()
    elif choice == "2":
        main()
    elif choice == "3":
        print("👋 再見！")
    else:
        print("❌ 無效選擇")

if __name__ == "__main__":
    main()
