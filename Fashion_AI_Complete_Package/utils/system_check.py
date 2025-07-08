#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: System Test and Validation
系統測試和驗證腳本 - 檢查所有組件是否正常工作
"""

import os
import sys
import importlib
import torch
from datetime import datetime
import traceback

class SystemTester:
    """系統測試器"""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.total_tests = 0
        
    def run_test(self, test_name, test_func):
        """運行單個測試"""
        print(f"\n🧪 測試: {test_name}")
        print("-" * 40)
        
        self.total_tests += 1
        
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name} - 通過")
                self.test_results[test_name] = "PASS"
                self.passed_tests += 1
            else:
                print(f"❌ {test_name} - 失敗")
                self.test_results[test_name] = "FAIL"
        except Exception as e:
            print(f"❌ {test_name} - 錯誤: {e}")
            self.test_results[test_name] = f"ERROR: {e}"
            # 打印詳細錯誤信息
            if "--verbose" in sys.argv:
                traceback.print_exc()
    
    def test_dependencies(self):
        """測試依賴項"""
        print("檢查 Python 套件...")
        
        required_packages = {
            'torch': '>=1.13.0',
            'transformers': '>=4.21.0',
            'diffusers': '>=0.21.0',
            'PIL': 'any',
            'numpy': 'any',
            'matplotlib': 'any',
            'cv2': 'any',
            'sklearn': 'any'
        }
        
        missing_packages = []
        
        for package, version in required_packages.items():
            try:
                module = importlib.import_module(package)
                if hasattr(module, '__version__'):
                    print(f"   ✅ {package}: {module.__version__}")
                else:
                    print(f"   ✅ {package}: 已安裝")
            except ImportError:
                print(f"   ❌ {package}: 未安裝")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n缺少套件: {', '.join(missing_packages)}")
            print("請運行: pip install torch transformers diffusers pillow opencv-python scikit-learn matplotlib")
            return False
        
        return True
    
    def test_gpu_availability(self):
        """測試 GPU 可用性"""
        print("檢查 GPU 狀態...")
        
        if not torch.cuda.is_available():
            print("   ⚠️  CUDA 不可用，將使用 CPU")
            return True  # CPU 也是可接受的
        
        try:
            device_count = torch.cuda.device_count()
            print(f"   🎮 檢測到 {device_count} 個 GPU")
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"      GPU {i}: {name} ({memory:.1f} GB)")
            
            # 測試 GPU 記憶體分配
            test_tensor = torch.randn(100, 100).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            
            print("   ✅ GPU 測試通過")
            return True
            
        except Exception as e:
            print(f"   ❌ GPU 測試失敗: {e}")
            return False
    
    def test_file_structure(self):
        """測試文件結構"""
        print("檢查文件結構...")
        
        required_files = [
            "day3_fashion_training.py",
            "day3_real_finetuning.py", 
            "day3_finetuning_config.py",
            "day3_training_monitor.py",
            "day3_integrated_launcher.py"
        ]
        
        missing_files = []
        
        for file in required_files:
            if os.path.exists(file):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file}")
                missing_files.append(file)
        
        # 檢查目錄
        required_dirs = ["day1_results"]
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"   ✅ {dir_name}/")
            else:
                print(f"   ⚠️  {dir_name}/ (建議創建)")
        
        return len(missing_files) == 0
    
    def test_module_imports(self):
        """測試模組導入"""
        print("測試模組導入...")
        
        modules_to_test = [
            "day3_fashion_training",
            "day3_real_finetuning", 
            "day3_finetuning_config",
            "day3_training_monitor"
        ]
        
        import_errors = []
        
        for module_name in modules_to_test:
            try:
                if os.path.exists(f"{module_name}.py"):
                    module = importlib.import_module(module_name)
                    print(f"   ✅ {module_name}")
                else:
                    print(f"   ⚠️  {module_name}.py 不存在")
            except Exception as e:
                print(f"   ❌ {module_name}: {e}")
                import_errors.append(module_name)
        
        return len(import_errors) == 0
    
    def test_fashion_clip_access(self):
        """測試 FashionCLIP 模型訪問"""
        print("測試 FashionCLIP 模型訪問...")
        
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            print("   📡 正在載入 FashionCLIP...")
            model = CLIPModel.from_pretrained(
                "patrickjohncyh/fashion-clip",
                torch_dtype=torch.float32
            )
            
            processor = CLIPProcessor.from_pretrained(
                "patrickjohncyh/fashion-clip"
            )
            
            print("   ✅ FashionCLIP 載入成功")
            
            # 釋放記憶體
            del model
            del processor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return True
            
        except Exception as e:
            print(f"   ❌ FashionCLIP 載入失敗: {e}")
            print("   💡 建議: 檢查網路連接或使用 huggingface-cli login")
            return False
    
    def test_stable_diffusion_access(self):
        """測試 Stable Diffusion 模型訪問"""
        print("測試 Stable Diffusion 模型訪問...")
        
        try:
            from diffusers import StableDiffusionPipeline
            
            print("   📡 正在載入 SD v1.5...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            print("   ✅ Stable Diffusion v1.5 載入成功")
            
            # 釋放記憶體
            del pipeline
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return True
            
        except Exception as e:
            print(f"   ❌ Stable Diffusion 載入失敗: {e}")
            print("   💡 建議: 檢查 Hugging Face 訪問權限")
            return False
    
    def test_config_validation(self):
        """測試配置驗證"""
        print("測試配置驗證...")
        
        try:
            if not os.path.exists("day3_finetuning_config.py"):
                print("   ⚠️  配置模組不存在")
                return False
            
            from day3_finetuning_config import FineTuningConfig, TrainingValidator
            
            config_manager = FineTuningConfig()
            validator = TrainingValidator()
            
            # 測試默認配置
            config = config_manager.get_config("quick_test")
            errors, warnings = validator.validate_config(config)
            
            if errors:
                print(f"   ❌ 配置驗證失敗: {errors}")
                return False
            
            print("   ✅ 配置驗證通過")
            return True
            
        except Exception as e:
            print(f"   ❌ 配置測試失敗: {e}")
            return False
    
    def test_basic_image_processing(self):
        """測試基本圖片處理"""
        print("測試基本圖片處理...")
        
        try:
            import numpy as np
            from PIL import Image
            import cv2
            
            # 創建測試圖片
            test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            pil_image = Image.fromarray(test_image)
            
            # 測試 PIL 操作
            resized = pil_image.resize((256, 256))
            
            # 測試 OpenCV 操作
            cv_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(cv_image, 50, 150)
            
            print("   ✅ 基本圖片處理測試通過")
            return True
            
        except Exception as e:
            print(f"   ❌ 圖片處理測試失敗: {e}")
            return False
    
    def test_integrated_launcher(self):
        """測試整合啟動器"""
        print("測試整合啟動器...")
        
        try:
            if not os.path.exists("day3_integrated_launcher.py"):
                print("   ❌ 整合啟動器不存在")
                return False
            
            # 簡單語法檢查
            with open("day3_integrated_launcher.py", 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 編譯檢查
            compile(content, "day3_integrated_launcher.py", "exec")
            
            print("   ✅ 整合啟動器語法正確")
            return True
            
        except Exception as e:
            print(f"   ❌ 整合啟動器測試失敗: {e}")
            return False
    
    def run_all_tests(self):
        """運行所有測試"""
        print("🚀 開始系統測試")
        print("=" * 60)
        print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 定義測試列表
        tests = [
            ("依賴項檢查", self.test_dependencies),
            ("GPU 可用性", self.test_gpu_availability),
            ("文件結構", self.test_file_structure),
            ("模組導入", self.test_module_imports),
            ("FashionCLIP 訪問", self.test_fashion_clip_access),
            ("Stable Diffusion 訪問", self.test_stable_diffusion_access),
            ("配置驗證", self.test_config_validation),
            ("圖片處理", self.test_basic_image_processing),
            ("整合啟動器", self.test_integrated_launcher)
        ]
        
        # 運行測試
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # 生成測試報告
        self.generate_test_report()
    
    def generate_test_report(self):
        """生成測試報告"""
        print("\n" + "=" * 60)
        print("📊 測試報告")
        print("=" * 60)
        
        print(f"總測試數: {self.total_tests}")
        print(f"通過數: {self.passed_tests}")
        print(f"失敗數: {self.total_tests - self.passed_tests}")
        print(f"成功率: {(self.passed_tests / self.total_tests * 100):.1f}%")
        
        print("\n詳細結果:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result == "PASS" else "❌"
            print(f"   {status_icon} {test_name}: {result}")
        
        # 建議
        print("\n💡 建議:")
        if self.passed_tests == self.total_tests:
            print("   🎉 所有測試通過！系統已準備就緒")
            print("   📚 查看 day3_setup_guide.md 開始使用")
        else:
            print("   🔧 修復失敗的測試項目")
            print("   📖 參考安裝指南解決問題")
        
        # 保存報告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"day3_system_test_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Day 3 系統測試報告\n")
            f.write(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"總測試數: {self.total_tests}\n")
            f.write(f"通過數: {self.passed_tests}\n")
            f.write(f"失敗數: {self.total_tests - self.passed_tests}\n")
            f.write(f"成功率: {(self.passed_tests / self.total_tests * 100):.1f}%\n\n")
            
            f.write("詳細結果:\n")
            for test_name, result in self.test_results.items():
                f.write(f"   {test_name}: {result}\n")
        
        print(f"\n📄 測試報告已保存: {report_path}")

def main():
    """主函數"""
    print("🧪 Day 3: Fashion AI Training Suite - 系統測試")
    
    if "--help" in sys.argv:
        print("\n使用方法:")
        print("   python day3_system_test.py           # 運行所有測試")
        print("   python day3_system_test.py --verbose # 顯示詳細錯誤信息")
        print("   python day3_system_test.py --help    # 顯示此幫助信息")
        return
    
    # 創建並運行測試
    tester = SystemTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
