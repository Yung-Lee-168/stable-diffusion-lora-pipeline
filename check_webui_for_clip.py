#!/usr/bin/env python3
"""
檢查 Stable Diffusion WebUI 狀態
為 day2_enhanced_test.py 做準備
"""

import requests
import time
import json

def check_webui_status():
    """檢查 WebUI 是否運行並準備就緒"""
    api_url = "http://localhost:7860"
    
    print("🔍 檢查 Stable Diffusion WebUI 狀態")
    print("=" * 50)
    
    # 檢查基本連接
    try:
        print("📡 測試 API 連接...")
        response = requests.get(f"{api_url}/sdapi/v1/options", timeout=5)
        
        if response.status_code == 200:
            print("✅ WebUI API 連接成功")
            
            # 檢查模型載入狀態
            try:
                model_response = requests.get(f"{api_url}/sdapi/v1/sd-models", timeout=5)
                if model_response.status_code == 200:
                    models = model_response.json()
                    print(f"✅ 已載入 {len(models)} 個 SD 模型")
                    
                    # 顯示當前模型
                    current_model_response = requests.get(f"{api_url}/sdapi/v1/options", timeout=5)
                    if current_model_response.status_code == 200:
                        options = current_model_response.json()
                        current_model = options.get("sd_model_checkpoint", "未知")
                        print(f"🎨 當前模型: {current_model}")
                
            except Exception as e:
                print(f"⚠️ 模型檢查失敗: {e}")
            
            # 測試生成功能
            print("\n🧪 測試圖片生成功能...")
            test_payload = {
                "prompt": "test image, simple",
                "negative_prompt": "",
                "width": 256,
                "height": 256,
                "steps": 5,
                "cfg_scale": 7,
                "sampler_name": "Euler a"
            }
            
            try:
                test_response = requests.post(f"{api_url}/sdapi/v1/txt2img", json=test_payload, timeout=30)
                if test_response.status_code == 200:
                    print("✅ 圖片生成功能正常")
                    print("\n🎉 WebUI 完全準備就緒！")
                    print("✅ 現在可以執行: python day2_enhanced_test.py")
                    return True
                else:
                    print(f"❌ 圖片生成測試失敗: {test_response.status_code}")
                    
            except Exception as e:
                print(f"❌ 圖片生成測試失敗: {e}")
                
        else:
            print(f"❌ API 連接失敗: HTTP {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 無法連接到 WebUI")
        print("💡 請確認 WebUI 已啟動並運行在 http://localhost:7860")
        print("💡 使用命令啟動: START_WEBUI_FOR_CLIP_TEST.bat")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ 連接超時")
        print("💡 WebUI 可能正在啟動中，請稍等後重試")
        return False
        
    except Exception as e:
        print(f"❌ 檢查過程發生錯誤: {e}")
        return False
    
    return False

def wait_for_webui(max_wait_time=300):
    """等待 WebUI 啟動完成"""
    print("⏳ 等待 WebUI 啟動...")
    print("💡 這可能需要幾分鐘時間")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        if check_webui_status():
            return True
        
        print("⏳ 等待中... (5秒後重試)")
        time.sleep(5)
    
    print("❌ 等待超時")
    return False

if __name__ == "__main__":
    print("🔍 Stable Diffusion WebUI 狀態檢查工具")
    print("為 CLIP vs FashionCLIP 測試做準備")
    print()
    
    if check_webui_status():
        print("\n🚀 可以開始 CLIP 測試了！")
        print("執行命令: python day2_enhanced_test.py")
    else:
        print("\n❓ 需要啟動 WebUI 嗎？")
        print("1. 執行: START_WEBUI_FOR_CLIP_TEST.bat")
        print("2. 等待 WebUI 啟動完成")
        print("3. 重新執行此檢查腳本")
        print("4. 然後執行: python day2_enhanced_test.py")
