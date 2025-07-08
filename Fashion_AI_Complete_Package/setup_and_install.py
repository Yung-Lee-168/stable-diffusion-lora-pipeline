#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fashion AI Complete Package - 自動安裝和配置腳本
自動檢查、安裝依賴和配置系統

功能：
1. 系統環境檢查
2. 自動安裝 Python 依賴
3. 下載必要的模型
4. 配置系統設置
5. 運行系統測試
"""

import os
import sys
import subprocess
import json
import urllib.request
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import platform
import importlib.util

class FashionAIInstaller:
    """Fashion AI 安裝器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_executable = sys.executable
        self.system_info = self.get_system_info()
        self.installation_log = []
        
    def get_system_info(self) -> Dict[str, Any]:
        """獲取系統資訊"""
        return {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'python_executable': self.python_executable
        }
    
    def log(self, message: str, level: str = "INFO"):
        """記錄日誌"""
        log_entry = f"[{level}] {message}"
        print(log_entry)
        self.installation_log.append(log_entry)
    
    def check_python_version(self) -> bool:
        """檢查 Python 版本"""
        self.log("🐍 檢查 Python 版本...")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.log(f"✅ Python {version.major}.{version.minor}.{version.micro} 符合需求")
            return True
        else:
            self.log(f"❌ Python {version.major}.{version.minor}.{version.micro} 不符合需求 (需要 3.8+)")
            return False
    
    def check_gpu(self) -> Dict[str, Any]:
        """檢查 GPU 狀態"""
        self.log("🔧 檢查 GPU 狀態...")
        
        gpu_info = {
            'available': False,
            'name': None,
            'memory_gb': 0
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['name'] = torch.cuda.get_device_name()
                gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                self.log(f"✅ GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
            else:
                self.log("⚠️ 沒有可用的 GPU，將使用 CPU 模式")
        except ImportError:
            self.log("⚠️ PyTorch 未安裝，無法檢查 GPU")
        
        return gpu_info
    
    def install_requirements(self) -> bool:
        """安裝 Python 依賴"""
        self.log("📦 安裝 Python 依賴...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            self.log("❌ requirements.txt 檔案不存在", "ERROR")
            return False
        
        try:
            # 升級 pip
            self.log("⬆️ 升級 pip...")
            subprocess.run([
                self.python_executable, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, capture_output=True)
            
            # 安裝依賴
            self.log("📥 安裝套件依賴...")
            result = subprocess.run([
                self.python_executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("✅ 依賴安裝完成")
                return True
            else:
                self.log(f"❌ 依賴安裝失敗: {result.stderr}", "ERROR")
                return False
                
        except subprocess.CalledProcessError as e:
            self.log(f"❌ 安裝過程發生錯誤: {e}", "ERROR")
            return False
    
    def install_pytorch_gpu(self) -> bool:
        """安裝 PyTorch GPU 版本"""
        self.log("🔥 安裝 PyTorch GPU 版本...")
        
        # 檢查是否已安裝 PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                self.log("✅ PyTorch GPU 版本已安裝")
                return True
        except ImportError:
            pass
        
        try:
            # 安裝 PyTorch GPU 版本
            pytorch_install_cmd = [
                self.python_executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
            
            result = subprocess.run(pytorch_install_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("✅ PyTorch GPU 版本安裝完成")
                return True
            else:
                self.log(f"❌ PyTorch 安裝失敗: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ PyTorch 安裝發生錯誤: {e}", "ERROR")
            return False
    
    def setup_directories(self) -> bool:
        """設置目錄結構"""
        self.log("📁 設置目錄結構...")
        
        directories = [
            "data/input",
            "data/output",
            "data/cache",
            "models/stable_diffusion",
            "models/fashion_clip",
            "models/VAE",
            "logs",
            "web/templates",
            "web/static"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log(f"✅ 創建目錄: {directory}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ 目錄創建失敗: {e}", "ERROR")
            return False
    
    def download_models(self) -> bool:
        """下載必要模型"""
        self.log("📥 檢查和下載模型...")
        
        # 這裡只檢查模型是否存在，實際下載由各個模組處理
        models_to_check = [
            {
                'name': 'FashionCLIP',
                'path': 'models/fashion_clip',
                'required': True
            }
        ]
        
        all_ready = True
        
        for model in models_to_check:
            model_path = self.project_root / model['path']
            
            if model_path.exists() and any(model_path.iterdir()):
                self.log(f"✅ {model['name']} 模型已存在")
            else:
                self.log(f"⚠️ {model['name']} 模型不存在，將在首次使用時自動下載")
                if model['required']:
                    all_ready = False
        
        return all_ready
    
    def create_sample_data(self) -> bool:
        """創建範例數據"""
        self.log("📸 創建範例數據...")
        
        try:
            # 創建範例圖片目錄
            sample_dir = self.project_root / "examples" / "sample_images"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # 創建 README 檔案
            readme_content = """# 範例圖片目錄

請將時尚圖片檔案放在此目錄下，用於測試和演示。

支援的格式：
- JPG/JPEG
- PNG
- BMP
- WEBP
- GIF

建議圖片：
- 時尚服裝照片
- 模特兒展示圖
- 產品圖片
- 街頭時尚照片

檔案命名建議：
- dress_sample.jpg
- top_sample.jpg
- accessories_sample.jpg
"""
            
            readme_path = sample_dir / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            self.log("✅ 範例數據目錄已創建")
            return True
            
        except Exception as e:
            self.log(f"❌ 範例數據創建失敗: {e}", "ERROR")
            return False
    
    def create_web_templates(self) -> bool:
        """創建基本 Web 模板"""
        self.log("🌐 創建 Web 模板...")
        
        try:
            templates_dir = self.project_root / "web" / "templates"
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            # 創建基本的 HTML 模板
            index_html = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fashion AI Complete Package</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .result { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Fashion AI Complete Package</h1>
        <p>智能時尚分析和圖片生成系統</p>
        
        <div class="upload-area">
            <h3>📸 上傳圖片進行分析</h3>
            <input type="file" id="imageInput" accept="image/*">
            <button class="button" onclick="analyzeImage()">分析圖片</button>
        </div>
        
        <div class="upload-area">
            <h3>🎨 文字生成圖片</h3>
            <input type="text" id="promptInput" placeholder="輸入描述..." style="width: 300px;">
            <button class="button" onclick="generateImage()">生成圖片</button>
        </div>
        
        <div id="result" class="result" style="display: none;">
            <h3>📊 結果</h3>
            <div id="resultContent"></div>
        </div>
    </div>
    
    <script>
        function analyzeImage() {
            alert('圖片分析功能需要完整的後端支援');
        }
        
        function generateImage() {
            alert('圖片生成功能需要完整的後端支援');
        }
    </script>
</body>
</html>"""
            
            with open(templates_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(index_html)
            
            self.log("✅ Web 模板已創建")
            return True
            
        except Exception as e:
            self.log(f"❌ Web 模板創建失敗: {e}", "ERROR")
            return False
    
    def test_installation(self) -> bool:
        """測試安裝結果"""
        self.log("🧪 測試安裝結果...")
        
        tests_passed = 0
        total_tests = 4
        
        # 測試 1: 匯入核心模組
        try:
            from core.fashion_analyzer import FashionTrainingPipeline
            self.log("✅ 核心分析模組匯入成功")
            tests_passed += 1
        except Exception as e:
            self.log(f"❌ 核心分析模組匯入失敗: {e}", "ERROR")
        
        # 測試 2: 匯入提示詞生成器
        try:
            from core.prompt_generator import FashionPromptGenerator
            self.log("✅ 提示詞生成器匯入成功")
            tests_passed += 1
        except Exception as e:
            self.log(f"❌ 提示詞生成器匯入失敗: {e}", "ERROR")
        
        # 測試 3: 匯入配置管理器
        try:
            from core.config_manager import FineTuningConfig
            self.log("✅ 配置管理器匯入成功")
            tests_passed += 1
        except Exception as e:
            self.log(f"❌ 配置管理器匯入失敗: {e}", "ERROR")
        
        # 測試 4: 檢查配置檔案
        try:
            import yaml
            config_file = self.project_root / "config" / "default_config.yaml"
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.log("✅ 配置檔案讀取成功")
            tests_passed += 1
        except Exception as e:
            self.log(f"❌ 配置檔案讀取失敗: {e}", "ERROR")
        
        success_rate = tests_passed / total_tests
        self.log(f"📊 測試結果: {tests_passed}/{total_tests} ({success_rate:.1%})")
        
        return success_rate >= 0.75
    
    def save_installation_log(self) -> bool:
        """保存安裝日誌"""
        try:
            log_dir = self.project_root / "logs"
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / "installation.log"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.installation_log))
            
            self.log(f"💾 安裝日誌已保存至: {log_file}")
            return True
            
        except Exception as e:
            self.log(f"❌ 保存安裝日誌失敗: {e}", "ERROR")
            return False
    
    def run_full_installation(self) -> bool:
        """執行完整安裝流程"""
        self.log("🚀 開始 Fashion AI Complete Package 安裝")
        self.log("=" * 60)
        
        steps = [
            ("檢查 Python 版本", self.check_python_version),
            ("設置目錄結構", self.setup_directories),
            ("安裝 PyTorch GPU", self.install_pytorch_gpu),
            ("安裝 Python 依賴", self.install_requirements),
            ("檢查 GPU 狀態", lambda: self.check_gpu() or True),
            ("檢查模型狀態", self.download_models),
            ("創建範例數據", self.create_sample_data),
            ("創建 Web 模板", self.create_web_templates),
            ("測試安裝結果", self.test_installation),
            ("保存安裝日誌", self.save_installation_log)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            self.log(f"\n📍 {step_name}...")
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    self.log(f"⚠️ {step_name} 未完全成功")
            except Exception as e:
                failed_steps.append(step_name)
                self.log(f"❌ {step_name} 發生錯誤: {e}", "ERROR")
        
        # 安裝總結
        self.log("\n" + "=" * 60)
        self.log("📋 安裝總結")
        self.log("=" * 60)
        
        if not failed_steps:
            self.log("🎉 安裝完成！所有步驟都成功執行。")
            self.log("\n📝 下一步：")
            self.log("1. 啟動 Stable Diffusion WebUI（如果要使用圖片生成功能）")
            self.log("2. 運行 'python fashion_ai_main.py' 啟動系統")
            self.log("3. 或運行 'python fashion_web_ui.py' 啟動 Web 介面")
            return True
        else:
            self.log(f"⚠️ 安裝完成，但有 {len(failed_steps)} 個步驟未完全成功：")
            for step in failed_steps:
                self.log(f"   • {step}")
            self.log("\n💡 建議：")
            self.log("1. 檢查安裝日誌以了解詳細錯誤")
            self.log("2. 手動安裝失敗的組件")
            self.log("3. 重新運行安裝腳本")
            return False

def main():
    """主函數"""
    print("🎯 Fashion AI Complete Package - 自動安裝腳本")
    print("=" * 80)
    
    # 確認用戶同意開始安裝
    response = input("是否開始自動安裝？這將安裝 Python 套件和設置系統。(y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("👋 安裝已取消")
        return
    
    # 創建安裝器並執行安裝
    installer = FashionAIInstaller()
    
    try:
        success = installer.run_full_installation()
        
        if success:
            print("\n🎉 恭喜！Fashion AI Complete Package 安裝成功！")
            print("\n🚀 快速開始：")
            print("   python fashion_ai_main.py --mode interactive")
            print("   python fashion_web_ui.py")
        else:
            print("\n⚠️ 安裝過程中遇到一些問題，請檢查日誌檔案。")
            print("您仍然可以嘗試手動啟動系統。")
    
    except KeyboardInterrupt:
        print("\n👋 安裝被用戶中斷")
    except Exception as e:
        print(f"\n❌ 安裝過程發生嚴重錯誤: {e}")
        print("請檢查錯誤訊息並手動安裝必要組件。")

if __name__ == "__main__":
    main()
