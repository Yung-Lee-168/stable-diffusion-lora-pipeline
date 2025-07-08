#!/usr/bin/env python3
"""
簡化版 CLIP 測試 - 用於調試和驗證
"""

import requests
import json
import base64
import os
from datetime import datetime
from PIL import Image

def check_api_status():
    """檢查 API 狀態"""
    print("🔍 檢查 WebUI API 狀態...")
    
    try:
        # 檢查基本連接
        response = requests.get("http://localhost:7860/sdapi/v1/options", timeout=10)
        if response.status_code == 200:
            print("✅ API 連接正常")
            
            # 檢查模型載入狀態
            model_response = requests.get("http://localhost:7860/sdapi/v1/sd-models", timeout=10)
            if model_response.status_code == 200:
                models = model_response.json()
                print(f"✅ 發現 {len(models)} 個可用模型")
                if models:
                    print(f"   當前模型: {models[0].get('title', '未知')}")
                else:
                    print("⚠️ 沒有可用的模型")
                    return False
            
            return True
        else:
            print(f"❌ API 連接失敗，狀態碼: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 無法連接到 WebUI，請確認:")
        print("   1. WebUI 已啟動")
        print("   2. API 模式已啟用 (--api)")
        print("   3. 端口 7860 可用")
        return False
    except Exception as e:
        print(f"❌ 連接檢查失敗: {e}")
        return False

def test_basic_generation():
    """測試基本圖像生成"""
    print("\n🎨 測試基本圖像生成...")
    
    payload = {
        "prompt": "a beautiful woman in elegant dress, high quality",
        "negative_prompt": "low quality, blurry",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg_scale": 7.0,
        "sampler_name": "Euler a"
    }
    
    try:
        print("📤 發送生成請求...")
        response = requests.post("http://localhost:7860/sdapi/v1/txt2img", 
                               json=payload, timeout=120)
        
        print(f"📥 收到回應: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'images' in result and result['images']:
                # 保存圖片
                image_data = base64.b64decode(result['images'][0])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"debug_test_{timestamp}.png"
                
                with open(image_path, "wb") as f:
                    f.write(image_data)
                
                print(f"✅ 圖像生成成功！保存為: {image_path}")
                
                # 檢查圖片大小
                img = Image.open(image_path)
                print(f"   圖片尺寸: {img.size}")
                print(f"   檔案大小: {len(image_data)} bytes")
                
                return True
            else:
                print("❌ 回應中沒有圖像數據")
                print(f"回應內容: {result}")
                return False
        else:
            print(f"❌ 生成失敗，狀態碼: {response.status_code}")
            print(f"錯誤詳情: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 請求超時，可能是:")
        print("   1. 模型載入中")
        print("   2. GPU 記憶體不足")
        print("   3. 生成時間過長")
        return False
    except Exception as e:
        print(f"❌ 生成失敗: {e}")
        return False

def test_clip_models():
    """測試 CLIP 模型載入"""
    print("\n🔍 測試 CLIP 模型載入...")
    
    # 測試標準 CLIP
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 使用設備: {device}")
        
        print("📥 載入標準 CLIP...")
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        
        print("✅ 標準 CLIP 載入成功")
        
        # 簡單測試
        test_labels = ["red dress", "blue shirt", "black pants"]
        
        # 創建一個測試圖片（純色）
        test_img = Image.new('RGB', (224, 224), color='red')
        
        inputs = processor(text=test_labels, images=test_img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        top_idx = probs[0].argmax().item()
        print(f"✅ CLIP 測試成功，最高匹配: {test_labels[top_idx]} ({probs[0][top_idx]:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ CLIP 測試失敗: {e}")
        return False

def main():
    """主要測試流程"""
    print("=" * 50)
    print("  簡化版 CLIP 測試與調試")
    print("=" * 50)
    
    # 檢查 API
    if not check_api_status():
        print("\n❌ API 檢查失敗，請先解決 WebUI 問題")
        return False
    
    # 測試圖像生成
    if not test_basic_generation():
        print("\n❌ 基本圖像生成失敗")
        return False
    
    # 測試 CLIP
    if not test_clip_models():
        print("\n❌ CLIP 模型測試失敗")
        return False
    
    print("\n" + "=" * 50)
    print("✅ 所有測試通過！")
    print("現在可以執行完整的 day2_enhanced_test.py")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    main()
    input("\n按 Enter 鍵結束...")
