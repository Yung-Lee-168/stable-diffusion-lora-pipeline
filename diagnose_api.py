#!/usr/bin/env python3
"""
診斷和測試腳本
用於檢查 Stable Diffusion API 的各種問題
"""

import sys
import os
import requests
import json
from datetime import datetime

def check_python_environment():
    """檢查 Python 環境"""
    print("🔍 檢查 Python 環境...")
    print(f"Python 版本: {sys.version}")
    print(f"工作目錄: {os.getcwd()}")
    
    # 檢查必要模組
    required_modules = ['requests', 'base64', 'json', 'PIL']
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'PIL':
                import PIL
                print(f"✅ {module} 已安裝 (版本: {PIL.__version__})")
            else:
                __import__(module)
                print(f"✅ {module} 已安裝")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} 未安裝")
    
    if missing_modules:
        print(f"\n需要安裝的模組: {', '.join(missing_modules)}")
        print("執行命令: pip install requests pillow")
        return False
    
    return True

def check_webui_server():
    """檢查 WebUI 服務器狀態"""
    print("\n🔍 檢查 Stable Diffusion WebUI 服務器...")
    
    urls_to_check = [
        "http://localhost:7860/sdapi/v1/options",
        "http://127.0.0.1:7860/sdapi/v1/options",
        "http://localhost:7860",
        "http://127.0.0.1:7860"
    ]
    
    for url in urls_to_check:
        try:
            print(f"測試連接: {url}")
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ 服務器正常運行: {url}")
                return True
            else:
                print(f"⚠️ 服務器回應代碼: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ 無法連接: {url}")
        except requests.exceptions.Timeout:
            print(f"⏱️ 連接超時: {url}")
        except Exception as e:
            print(f"❌ 連接錯誤: {e}")
    
    print("\n❌ 無法連接到 WebUI 服務器")
    print("請確認:")
    print("1. webui-user.bat 是否已執行")
    print("2. 是否看到 'Running on local URL' 訊息")
    print("3. 防火牆是否阻擋連接")
    return False

def check_api_endpoints():
    """檢查 API 端點"""
    print("\n🔍 檢查 API 端點...")
    
    base_url = "http://localhost:7860"
    endpoints = [
        "/sdapi/v1/txt2img",
        "/sdapi/v1/img2img", 
        "/sdapi/v1/models",
        "/sdapi/v1/samplers",
        "/sdapi/v1/options"
    ]
    
    for endpoint in endpoints:
        url = base_url + endpoint
        try:
            if endpoint == "/sdapi/v1/txt2img" or endpoint == "/sdapi/v1/img2img":
                # 這些是 POST 端點，我們只檢查是否存在
                response = requests.options(url, timeout=5)
            else:
                response = requests.get(url, timeout=5)
            
            if response.status_code in [200, 405]:  # 405 表示方法不允許但端點存在
                print(f"✅ {endpoint} 可用")
            else:
                print(f"⚠️ {endpoint} 回應: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint} 錯誤: {e}")

def test_simple_generation():
    """測試簡單圖片生成"""
    print("\n🎨 測試簡單圖片生成...")
    
    url = "http://localhost:7860/sdapi/v1/txt2img"
    payload = {
        "prompt": "a simple test image, red circle on white background",
        "negative_prompt": "complex, detailed",
        "width": 256,
        "height": 256,
        "steps": 10,
        "cfg_scale": 7,
        "sampler_name": "Euler",
        "n_iter": 1,
        "batch_size": 1
    }
    
    try:
        print("發送測試請求...")
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("images"):
                print("✅ 圖片生成測試成功!")
                print(f"生成了 {len(result['images'])} 張圖片")
                
                # 嘗試保存測試圖片
                import base64
                os.makedirs("test_output", exist_ok=True)
                
                image_data = base64.b64decode(result["images"][0])
                test_filename = f"test_output/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                with open(test_filename, 'wb') as f:
                    f.write(image_data)
                
                print(f"測試圖片已保存: {test_filename}")
                return True
            else:
                print("❌ 沒有生成圖片數據")
                return False
        else:
            print(f"❌ API 請求失敗: {response.status_code}")
            print(f"錯誤詳情: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 測試生成失敗: {e}")
        return False

def check_file_permissions():
    """檢查檔案權限"""
    print("\n📁 檢查檔案權限...")
    
    test_dirs = ["generated_images", "test_output", "custom_output"]
    
    for dir_name in test_dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
            
            # 測試寫入權限
            test_file = os.path.join(dir_name, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            
            # 測試讀取權限
            with open(test_file, 'r') as f:
                content = f.read()
            
            # 清理測試檔案
            os.remove(test_file)
            
            print(f"✅ {dir_name} 資料夾權限正常")
            
        except Exception as e:
            print(f"❌ {dir_name} 資料夾權限錯誤: {e}")

def main():
    """主診斷函數"""
    print("🔧 Stable Diffusion API 診斷工具")
    print("=" * 60)
    
    all_checks_passed = True
    
    # 1. 檢查 Python 環境
    if not check_python_environment():
        all_checks_passed = False
    
    # 2. 檢查服務器
    if not check_webui_server():
        all_checks_passed = False
        print("\n⚠️ 服務器未運行，跳過後續測試")
        print("\n請先啟動 webui-user.bat，然後重新運行此診斷工具")
    else:
        # 3. 檢查 API 端點
        check_api_endpoints()
        
        # 4. 測試圖片生成
        if not test_simple_generation():
            all_checks_passed = False
    
    # 5. 檢查檔案權限
    check_file_permissions()
    
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("🎉 所有檢查通過! API 應該可以正常工作")
    else:
        print("⚠️ 發現一些問題，請參考上述建議進行修復")
    
    print("\n如果問題持續存在，請檢查:")
    print("1. Windows 防火牆設定")
    print("2. 防毒軟體是否阻擋")
    print("3. WebUI 控制台是否有錯誤訊息")
    print("4. GPU 記憶體是否足夠")
    
    input("\n按 Enter 鍵退出...")

if __name__ == "__main__":
    main()
