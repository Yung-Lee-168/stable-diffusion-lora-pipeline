#!/usr/bin/env python3
"""
CLIP 版本相容性檢查工具
幫助用戶選擇最適合的 CLIP 版本
"""

import sys
import platform
import subprocess
import time

class CLIPCompatibilityChecker:
    def __init__(self):
        self.recommendations = []
        
    def check_system_info(self):
        """檢查系統基本資訊"""
        print("🖥️ 系統資訊檢查")
        print("=" * 50)
        
        # 作業系統
        os_info = platform.system()
        print(f"作業系統: {os_info} {platform.release()}")
        
        # Python 版本
        python_version = sys.version
        print(f"Python 版本: {python_version}")
        
        # 記憶體 (簡單估算)
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"總記憶體: {memory.total // (1024**3)} GB")
            print(f"可用記憶體: {memory.available // (1024**3)} GB")
            
            if memory.total < 8 * (1024**3):  # 少於 8GB
                self.recommendations.append("記憶體較少，建議使用輕量版 CLIP 模型")
        except ImportError:
            print("未安裝 psutil，無法檢查記憶體資訊")
        
        print()
    
    def check_pytorch_gpu(self):
        """檢查 PyTorch 和 GPU 支援"""
        print("🔥 PyTorch 和 GPU 檢查")
        print("=" * 50)
        
        try:
            import torch
            print(f"✅ PyTorch 版本: {torch.__version__}")
            
            # CUDA 檢查
            if torch.cuda.is_available():
                print(f"✅ CUDA 可用: {torch.version.cuda}")
                print(f"✅ GPU 數量: {torch.cuda.device_count()}")
                
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    memory = torch.cuda.get_device_properties(i).total_memory
                    print(f"   GPU {i}: {gpu_name} ({memory // (1024**3)} GB)")
                
                # GPU 記憶體建議
                total_gpu_memory = sum([
                    torch.cuda.get_device_properties(i).total_memory 
                    for i in range(torch.cuda.device_count())
                ])
                
                if total_gpu_memory >= 8 * (1024**3):  # 8GB+
                    self.recommendations.append("GPU 記憶體充足，可使用完整版 CLIP 模型")
                elif total_gpu_memory >= 4 * (1024**3):  # 4-8GB
                    self.recommendations.append("GPU 記憶體中等，建議使用標準 CLIP 模型")
                else:  # <4GB
                    self.recommendations.append("GPU 記憶體較少，建議使用 CPU 或小型模型")
                    
            else:
                print("⚠️ 未偵測到 CUDA GPU，將使用 CPU")
                self.recommendations.append("僅 CPU 可用，建議使用 transformers CLIP (較慢但穩定)")
                
        except ImportError:
            print("❌ PyTorch 未安裝")
            self.recommendations.append("需要先安裝 PyTorch")
        
        print()
    
    def check_installed_packages(self):
        """檢查已安裝的套件"""
        print("📦 已安裝套件檢查")
        print("=" * 50)
        
        packages_to_check = [
            "torch", "transformers", "clip-by-openai", 
            "ftfy", "regex", "tqdm", "pillow"
        ]
        
        installed_packages = {}
        
        for package in packages_to_check:
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "show", package], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    version_line = [line for line in result.stdout.split('\n') if line.startswith('Version:')]
                    if version_line:
                        version = version_line[0].split(':')[1].strip()
                        print(f"✅ {package}: {version}")
                        installed_packages[package] = version
                    else:
                        print(f"✅ {package}: 已安裝")
                        installed_packages[package] = "unknown"
                else:
                    print(f"❌ {package}: 未安裝")
            except Exception:
                print(f"❌ {package}: 檢查失敗")
        
        # 基於已安裝套件給建議
        if "transformers" in installed_packages:
            self.recommendations.append("已安裝 transformers，推薦使用 HuggingFace CLIP")
        
        if "clip-by-openai" in installed_packages:
            self.recommendations.append("已安裝 OpenAI CLIP，可使用官方版本")
        
        print()
        return installed_packages
    
    def test_clip_performance(self):
        """測試不同 CLIP 版本的效能"""
        print("⚡ CLIP 效能測試")
        print("=" * 50)
        
        # 測試 transformers CLIP
        try:
            print("測試 HuggingFace transformers CLIP...")
            start_time = time.time()
            
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image
            import torch
            
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # 創建測試圖片和文字
            test_image = Image.new('RGB', (224, 224), color='red')
            test_texts = ["a red image", "a blue image", "a green image"]
            
            inputs = processor(text=test_texts, images=test_image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            load_time = time.time() - start_time
            print(f"✅ Transformers CLIP 載入時間: {load_time:.2f} 秒")
            
            if load_time < 30:
                self.recommendations.append("Transformers CLIP 載入速度良好")
            else:
                self.recommendations.append("Transformers CLIP 載入較慢，考慮使用更小的模型")
                
        except Exception as e:
            print(f"❌ Transformers CLIP 測試失敗: {e}")
        
        # 測試 OpenAI CLIP (如果可用)
        try:
            print("測試 OpenAI CLIP...")
            import clip
            
            start_time = time.time()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            load_time = time.time() - start_time
            
            print(f"✅ OpenAI CLIP 載入時間: {load_time:.2f} 秒")
            
            if load_time < 20:
                self.recommendations.append("OpenAI CLIP 載入速度優秀")
            
        except ImportError:
            print("⚠️ OpenAI CLIP 未安裝")
        except Exception as e:
            print(f"❌ OpenAI CLIP 測試失敗: {e}")
        
        print()
    
    def generate_recommendations(self):
        """生成最終建議"""
        print("🎯 推薦方案")
        print("=" * 50)
        
        # 基本建議
        if not self.recommendations:
            self.recommendations.append("使用 HuggingFace transformers CLIP (最相容)")
        
        # 根據不同情況給出具體建議
        print("基於您的系統配置，建議：")
        print()
        
        for i, rec in enumerate(self.recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n📋 具體安裝命令：")
        print()
        
        # GPU 使用者
        try:
            import torch
            if torch.cuda.is_available():
                print("🔥 GPU 使用者推薦：")
                print("pip install transformers torch torchvision")
                print("# 或者安裝 OpenAI CLIP：")
                print("pip install git+https://github.com/openai/CLIP.git")
        except:
            pass
        
        # CPU 使用者
        print("\n💻 CPU 使用者推薦：")
        print("pip install transformers torch pillow")
        
        # FashionCLIP
        print("\n👗 時尚專業需求：")
        print("# 嘗試專業 FashionCLIP (需要網路下載)：")
        print("# 模型會自動從 HuggingFace 下載")
        
        print("\n✨ 在當前測試中的建議：")
        print("直接執行 day2_enhanced_test.py，程式會自動選擇最適合的版本！")
    
    def run_full_check(self):
        """執行完整檢查"""
        print("🔍 CLIP 版本相容性完整檢查")
        print("=" * 60)
        print()
        
        self.check_system_info()
        self.check_pytorch_gpu()
        installed = self.check_installed_packages()
        
        # 只有在有基本套件時才做效能測試
        if "transformers" in installed or "torch" in installed:
            self.test_clip_performance()
        
        self.generate_recommendations()

if __name__ == "__main__":
    checker = CLIPCompatibilityChecker()
    checker.run_full_check()
