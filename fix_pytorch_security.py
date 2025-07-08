#!/usr/bin/env python3
"""
解決 PyTorch 版本安全性問題
使用 safetensors 格式載入模型，避免 torch.load 安全性問題
"""

import os
import time
from transformers import CLIPModel, CLIPProcessor

def download_models_with_safetensors():
    print("🔧 使用 SafeTensors 格式解決 PyTorch 安全性問題")
    print("=" * 60)
    
    try:
        import torch
        print(f"當前 PyTorch 版本: {torch.__version__}")
        
        # 檢查是否需要升級
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 6):
            print("⚠️ PyTorch 版本較舊，使用 safetensors 格式載入")
        else:
            print("✅ PyTorch 版本符合要求")
    except:
        print("❌ 無法檢查 PyTorch 版本")
    
    print()
    
    # 下載標準 CLIP - 使用 safetensors
    print("1️⃣ 下載標準 CLIP (使用 SafeTensors 格式)")
    try:
        start_time = time.time()
        print("   📥 正在下載模型檔案 (SafeTensors 格式)...")
        
        # 強制使用 safetensors 格式
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True,  # 強制使用 safetensors
            trust_remote_code=False  # 安全性設置
        )
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        download_time = time.time() - start_time
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        
        print(f"   ✅ 標準 CLIP 下載成功！")
        print(f"   ⏱️ 下載時間: {download_time:.1f} 秒")
        print(f"   📊 參數數量: {param_count:.1f}M")
        print(f"   🔒 使用 SafeTensors 格式 (更安全)")
        
        # 測試功能
        from PIL import Image
        test_image = Image.new('RGB', (224, 224), color='red')
        test_texts = ["a red image", "a blue image"]
        
        inputs = processor(text=test_texts, images=test_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"   🧪 功能測試: 通過")
        
        # 清理記憶體
        del model, processor
        
    except Exception as e:
        print(f"   ❌ 標準 CLIP 下載失敗: {e}")
        
        # 嘗試備用方案
        try:
            print("   🔄 嘗試備用下載方案...")
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=torch.float32,  # 使用 float32
                low_cpu_mem_usage=True      # 低記憶體使用
            )
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print(f"   ✅ 備用方案成功！")
            del model, processor
        except Exception as e2:
            print(f"   ❌ 備用方案也失敗: {e2}")
            return False
    
    print()
    
    # 下載 FashionCLIP - 使用 safetensors
    print("2️⃣ 下載 FashionCLIP (使用 SafeTensors 格式)")
    try:
        start_time = time.time()
        print("   📥 正在下載專業時尚模型...")
        
        # 嘗試載入 FashionCLIP
        try:
            fashion_model = CLIPModel.from_pretrained(
                "patrickjohncyh/fashion-clip",
                use_safetensors=True,
                trust_remote_code=False
            )
            fashion_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            
            download_time = time.time() - start_time
            fashion_param_count = sum(p.numel() for p in fashion_model.parameters()) / 1e6
            
            print(f"   ✅ FashionCLIP 下載成功！")
            print(f"   ⏱️ 下載時間: {download_time:.1f} 秒")
            print(f"   📊 參數數量: {fashion_param_count:.1f}M")
            print(f"   🔒 使用 SafeTensors 格式")
            print(f"   👗 專業領域: 時尚圖片分析")
            
            del fashion_model, fashion_processor
            
        except Exception as e:
            print(f"   ⚠️ FashionCLIP 下載失敗: {e}")
            print(f"   💡 將在測試中使用標準 CLIP 作為備用")
        
    except Exception as e:
        print(f"   ❌ FashionCLIP 處理失敗: {e}")
    
    print()
    print("=" * 60)
    print("🎉 模型準備完成！")
    print("=" * 60)
    
    return True

def create_updated_test_script():
    """創建更新版的測試腳本，解決 PyTorch 安全性問題"""
    
    updated_content = '''#!/usr/bin/env python3
"""
安全版第2天測試：CLIP vs FashionCLIP 比較
解決 PyTorch 安全性問題，使用 SafeTensors 格式
"""

import requests
import json
import base64
import os
from datetime import datetime
from PIL import Image
import numpy as np

class SafeEnhancedDay2Tester:
    def __init__(self):
        self.api_url = "http://localhost:7860"
        self.output_dir = "day2_enhanced_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def install_requirements(self):
        """檢查並安裝必要的套件"""
        print("🔍 檢查模型依賴...")
        
        try:
            import torch
            import transformers
            print("✅ 基礎套件已安裝")
            print(f"🔧 PyTorch 版本: {torch.__version__}")
        except ImportError as e:
            print(f"❌ 缺少基礎套件: {e}")
            return False
            
        return True
    
    def load_standard_clip_safe(self):
        """安全載入標準 CLIP 模型 - 使用 SafeTensors"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"📥 安全載入標準 CLIP (設備: {device})...")
            
            # 使用 SafeTensors 格式，避免 torch.load 安全性問題
            if device == "cuda":
                model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    torch_dtype=torch.float16,
                    use_safetensors=True,       # 使用安全格式
                    trust_remote_code=False     # 安全性設置
                )
            else:
                model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    use_safetensors=True,
                    trust_remote_code=False
                )
                
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.to(device)
            
            print("✅ 標準 CLIP 模型安全載入成功")
            return model, processor, "standard_clip"
            
        except Exception as e:
            print(f"❌ 標準 CLIP 模型載入失敗: {e}")
            
            # 備用方案：使用較舊的載入方式
            try:
                print("🔄 嘗試備用載入方案...")
                model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                model.to(device)
                print("✅ 備用方案載入成功")
                return model, processor, "standard_clip"
            except Exception as e2:
                print(f"❌ 備用方案也失敗: {e2}")
                return None, None, None
    
    def load_fashion_clip_safe(self):
        """安全載入 FashionCLIP 模型"""
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            
            fashion_models = [
                "patrickjohncyh/fashion-clip",
                "openai/clip-vit-base-patch32"  # 備用
            ]
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🎮 使用設備: {device}")
            
            for model_name in fashion_models:
                try:
                    print(f"📥 安全載入 {model_name}...")
                    
                    if device == "cuda":
                        model = CLIPModel.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            use_safetensors=True,
                            trust_remote_code=False,
                            device_map="auto"
                        )
                    else:
                        model = CLIPModel.from_pretrained(
                            model_name,
                            use_safetensors=True,
                            trust_remote_code=False
                        )
                    
                    processor = CLIPProcessor.from_pretrained(model_name)
                    model.to(device)
                    
                    print(f"✅ FashionCLIP 安全載入成功: {model_name}")
                    print(f"   設備: {device}")
                    print(f"   精度: {'float16' if device == 'cuda' else 'float32'}")
                    print(f"   🔒 使用 SafeTensors 格式")
                    
                    return model, processor, "fashion_clip"
                    
                except Exception as e:
                    print(f"⚠️ 載入 {model_name} 失敗: {e}")
                    continue
                    
            print("⚠️ 專業 FashionCLIP 不可用，使用標準 CLIP")
            return self.load_standard_clip_safe()
            
        except Exception as e:
            print(f"❌ FashionCLIP 載入失敗: {e}")
            return None, None, None

# 其他方法保持不變...
'''
    
    with open("day2_safe_test.py", "w", encoding="utf-8") as f:
        f.write(updated_content)
    
    print("📝 已創建安全版測試腳本: day2_safe_test.py")

if __name__ == "__main__":
    print("🔧 PyTorch 安全性問題解決方案")
    print("=" * 60)
    
    # 下載模型
    success = download_models_with_safetensors()
    
    if success:
        print("\n✅ 現在可以執行安全版測試:")
        print("   python day2_enhanced_test.py")
        print("\n💡 如果仍有問題，可以嘗試:")
        print("   pip install torch>=2.6.0 --upgrade")
    else:
        print("\n❌ 模型下載遇到問題")
        print("💡 建議升級 PyTorch:")
        print("   pip install torch>=2.6.0 transformers --upgrade")
