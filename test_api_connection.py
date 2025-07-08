#!/usr/bin/env python3
"""
快速 API 連接測試
在開始3天測試之前，驗證 Stable Diffusion WebUI API 是否正常工作
"""

import requests
import json
import sys

def test_api_connection():
    """測試 API 連接"""
    api_url = "http://localhost:7860"
    
    print("🔍 測試 Stable Diffusion WebUI API 連接...")
    print(f"API 地址: {api_url}")
    
    try:
        # 測試基本連接
        response = requests.get(f"{api_url}/sdapi/v1/memory", timeout=10)
        if response.status_code == 200:
            print("✅ API 連接成功")
            
            # 獲取系統信息
            memory_info = response.json()
            print(f"   GPU 記憶體: {memory_info.get('cuda', {}).get('memory', {}).get('total', 'N/A')}")
            
            # 測試模型信息
            models_response = requests.get(f"{api_url}/sdapi/v1/sd-models", timeout=10)
            if models_response.status_code == 200:
                models = models_response.json()
                if models:
                    current_model = models[0].get('title', 'Unknown')
                    print(f"   當前模型: {current_model}")
                    print(f"   可用模型數量: {len(models)}")
                else:
                    print("   ⚠️ 未發現可用模型")
            
            # 測試基本生成 API
            print("\n🎨 測試基本圖片生成 API...")
            test_payload = {
                "prompt": "test image, simple",
                "negative_prompt": "low quality",
                "width": 256,
                "height": 256,
                "steps": 10,
                "cfg_scale": 7,
                "sampler_name": "Euler a"
            }
            
            gen_response = requests.post(f"{api_url}/sdapi/v1/txt2img", 
                                       json=test_payload, timeout=60)
            if gen_response.status_code == 200:
                print("✅ 圖片生成 API 正常工作")
                result = gen_response.json()
                if result.get('images'):
                    print("✅ 成功生成測試圖片")
                    return True
                else:
                    print("❌ 生成結果為空")
                    return False
            else:
                print(f"❌ 圖片生成失敗: HTTP {gen_response.status_code}")
                print(f"   錯誤詳情: {gen_response.text[:200]}")
                return False
                
        else:
            print(f"❌ API 連接失敗: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectRefused:
        print("❌ API 連接被拒絕")
        print("   請確保 Stable Diffusion WebUI 已啟動")
        print("   啟動命令: webui-user.bat")
        return False
    except requests.exceptions.Timeout:
        print("❌ API 連接逾時")
        print("   WebUI 可能正在啟動中，請稍後再試")
        return False
    except Exception as e:
        print(f"❌ API 測試錯誤: {e}")
        return False

def check_environment():
    """檢查環境依賴"""
    print("🔍 檢查環境依賴...")
    
    required_packages = [
        ("requests", "網路請求"),
        ("PIL", "圖片處理"),
        ("torch", "深度學習框架"),
        ("transformers", "Transformer模型")
    ]
    
    all_installed = True
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (未安裝)")
            all_installed = False
    
    if not all_installed:
        print("\n📦 安裝缺失套件:")
        print("pip install requests pillow torch transformers matplotlib pandas")
    
    return all_installed

def main():
    """主要測試流程"""
    print("=" * 60)
    print("  Stable Diffusion WebUI API 連接測試")
    print("=" * 60)
    
    # 檢查環境
    env_ok = check_environment()
    print()
    
    # 測試 API
    api_ok = test_api_connection()
    
    print("\n" + "=" * 60)
    print("📊 測試結果摘要")
    print("=" * 60)
    print(f"環境依賴: {'✅ 正常' if env_ok else '❌ 有問題'}")
    print(f"API 連接: {'✅ 正常' if api_ok else '❌ 有問題'}")
    
    if env_ok and api_ok:
        print("\n🎉 環境檢查通過！可以開始3天可行性測試")
        print("執行命令: python day1_basic_test.py")
        return True
    else:
        print("\n⚠️ 環境檢查未通過，請先解決以上問題")
        if not env_ok:
            print("   • 安裝缺失的 Python 套件")
        if not api_ok:
            print("   • 確保 WebUI 已啟動並開啟 API 模式")
            print("   • 檢查 webui-user.bat 是否包含 --api --listen")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
