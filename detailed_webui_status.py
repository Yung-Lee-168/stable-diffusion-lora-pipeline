#!/usr/bin/env python3
"""
WebUI 狀態詳細檢查工具
"""

import requests
import json
import time

def detailed_webui_check():
    """詳細檢查 WebUI 狀態"""
    print("=" * 50)
    print("  WebUI 詳細狀態檢查")
    print("=" * 50)
    
    base_urls = [
        "http://localhost:7860",
        "http://127.0.0.1:7860",
        "http://0.0.0.0:7860"
    ]
    
    # 測試不同的端點
    endpoints = [
        "",  # 主頁
        "/docs",  # API 文檔
        "/openapi.json",  # OpenAPI 規範
        "/api/v1/options",  # 舊版 API
        "/sdapi/v1/options",  # 新版 API
        "/sdapi/v1/cmd-flags",  # 命令標誌
        "/sdapi/v1/sd-models",  # 模型列表
        "/sdapi/v1/samplers",  # 採樣器列表
    ]
    
    print("🔍 測試不同的 URL 和端點...")
    
    working_endpoints = []
    
    for base_url in base_urls:
        print(f"\n測試基礎 URL: {base_url}")
        
        for endpoint in endpoints:
            full_url = f"{base_url}{endpoint}"
            try:
                response = requests.get(full_url, timeout=5)
                status = response.status_code
                
                if status == 200:
                    print(f"  ✅ {endpoint or '/'}: {status}")
                    working_endpoints.append(full_url)
                    
                    # 如果是 API 端點，顯示一些內容
                    if endpoint.startswith("/sdapi") or endpoint.startswith("/api"):
                        try:
                            data = response.json()
                            if isinstance(data, dict):
                                keys = list(data.keys())[:5]  # 只顯示前5個鍵
                                print(f"    數據鍵: {keys}")
                            elif isinstance(data, list):
                                print(f"    列表長度: {len(data)}")
                        except:
                            print(f"    內容長度: {len(response.text)}")
                            
                elif status == 404:
                    print(f"  ❌ {endpoint or '/'}: 404 (不存在)")
                else:
                    print(f"  ⚠️ {endpoint or '/'}: {status}")
                    
            except requests.exceptions.ConnectionError:
                print(f"  💀 {endpoint or '/'}: 連接失敗")
            except requests.exceptions.Timeout:
                print(f"  ⏰ {endpoint or '/'}: 超時")
            except Exception as e:
                print(f"  ❌ {endpoint or '/'}: {str(e)}")
    
    print(f"\n📊 結果摘要:")
    print(f"可用端點數量: {len(working_endpoints)}")
    
    if working_endpoints:
        print("✅ 可用端點:")
        for endpoint in working_endpoints:
            print(f"  - {endpoint}")
        
        # 檢查是否有 API 端點
        api_endpoints = [ep for ep in working_endpoints if "/api" in ep]
        if api_endpoints:
            print(f"\n🎯 發現 {len(api_endpoints)} 個 API 端點!")
            return api_endpoints[0].split("/sdapi")[0] if "/sdapi" in api_endpoints[0] else api_endpoints[0].split("/api")[0]
        else:
            print("\n⚠️ 沒有發現 API 端點，WebUI 可能沒有啟用 API 模式")
            return None
    else:
        print("❌ 沒有發現任何可用端點")
        return None

def check_webui_version():
    """檢查 WebUI 版本信息"""
    print("\n🔍 檢查 WebUI 版本...")
    
    try:
        # 嘗試從主頁獲取版本信息
        response = requests.get("http://localhost:7860", timeout=10)
        if response.status_code == 200:
            content = response.text
            
            # 查找版本信息
            if "stable-diffusion-webui" in content.lower():
                print("✅ 確認是 Stable Diffusion WebUI")
            
            # 查找 API 相關信息
            if "api" in content.lower():
                print("✅ 頁面中提到了 API")
            else:
                print("⚠️ 頁面中沒有提到 API")
                
            return True
        else:
            print(f"❌ 無法訪問主頁，狀態碼: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 檢查版本失敗: {e}")
        return False

def test_basic_functionality(base_url):
    """測試基本功能"""
    print(f"\n🧪 測試基本功能 (使用 {base_url})...")
    
    # 測試選項獲取
    try:
        response = requests.get(f"{base_url}/sdapi/v1/options", timeout=10)
        if response.status_code == 200:
            print("✅ 選項 API 可用")
            options = response.json()
            print(f"   選項數量: {len(options)}")
        else:
            print(f"❌ 選項 API 失敗: {response.status_code}")
    except Exception as e:
        print(f"❌ 選項 API 錯誤: {e}")
    
    # 測試模型列表
    try:
        response = requests.get(f"{base_url}/sdapi/v1/sd-models", timeout=10)
        if response.status_code == 200:
            print("✅ 模型列表 API 可用")
            models = response.json()
            print(f"   模型數量: {len(models)}")
            if models:
                print(f"   當前模型: {models[0].get('title', '未知')}")
        else:
            print(f"❌ 模型列表 API 失敗: {response.status_code}")
    except Exception as e:
        print(f"❌ 模型列表 API 錯誤: {e}")
    
    # 測試簡單的文本到圖像
    try:
        print("🎨 測試簡單圖像生成...")
        payload = {
            "prompt": "a simple test image",
            "steps": 10,
            "width": 256,
            "height": 256
        }
        response = requests.post(f"{base_url}/sdapi/v1/txt2img", json=payload, timeout=30)
        if response.status_code == 200:
            print("✅ 圖像生成 API 可用")
            result = response.json()
            if 'images' in result and result['images']:
                print("✅ 成功生成圖像")
                return True
            else:
                print("❌ 沒有返回圖像數據")
        else:
            print(f"❌ 圖像生成失敗: {response.status_code}")
            if response.text:
                print(f"   錯誤: {response.text[:200]}")
    except Exception as e:
        print(f"❌ 圖像生成錯誤: {e}")
    
    return False

def main():
    # 1. 詳細檢查
    working_base_url = detailed_webui_check()
    
    # 2. 檢查版本
    version_ok = check_webui_version()
    
    # 3. 如果找到可用的 API，測試功能
    if working_base_url:
        api_works = test_basic_functionality(working_base_url)
        
        if api_works:
            print(f"\n🎉 WebUI API 完全可用!")
            print(f"使用的 URL: {working_base_url}")
            print("\n下一步:")
            print("1. 執行 python debug_clip_test.py")
            print("2. 執行 python day2_enhanced_test.py")
            return True
        else:
            print(f"\n⚠️ WebUI 在運行，但 API 功能有問題")
    else:
        print(f"\n❌ WebUI API 不可用")
        print("\n可能的解決方案:")
        print("1. 確保 webui-user.bat 中有 --api 參數")
        print("2. 重新啟動 WebUI")
        print("3. 檢查是否有防火牆阻擋")
        print("4. 嘗試使用更新版本的 WebUI")
    
    return False

if __name__ == "__main__":
    success = main()
    input(f"\n{'✅ 成功' if success else '❌ 失敗'}！按 Enter 鍵結束...")
