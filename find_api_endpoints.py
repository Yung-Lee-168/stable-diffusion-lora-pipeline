#!/usr/bin/env python3
"""
從 OpenAPI 文檔中找出正確的 API 端點
"""

import requests
import json

def find_correct_api_endpoints():
    """從 OpenAPI 文檔中找出正確的 API 端點"""
    print("🔍 分析 WebUI OpenAPI 文檔...")
    
    try:
        # 獲取 OpenAPI 規範
        response = requests.get("http://localhost:7860/openapi.json", timeout=10)
        if response.status_code != 200:
            print(f"❌ 無法獲取 OpenAPI 文檔，狀態碼: {response.status_code}")
            return []
        
        openapi_data = response.json()
        
        # 分析路徑
        paths = openapi_data.get("paths", {})
        print(f"📋 發現 {len(paths)} 個 API 端點:")
        
        # 按類別分組端點
        endpoints = {
            "圖像生成": [],
            "模型管理": [],
            "配置": [],
            "其他": []
        }
        
        for path, methods in paths.items():
            print(f"  {path}")
            
            # 分類端點
            if "txt2img" in path or "img2img" in path:
                endpoints["圖像生成"].append(path)
            elif "model" in path or "checkpoint" in path:
                endpoints["模型管理"].append(path)
            elif "option" in path or "config" in path or "setting" in path:
                endpoints["配置"].append(path)
            else:
                endpoints["其他"].append(path)
        
        # 顯示分類結果
        print("\n📊 端點分類:")
        for category, paths in endpoints.items():
            if paths:
                print(f"\n🎯 {category}:")
                for path in paths:
                    print(f"   {path}")
        
        # 尋找關鍵端點
        critical_endpoints = []
        for path in paths.keys():
            if any(keyword in path for keyword in ["txt2img", "options", "models", "samplers"]):
                critical_endpoints.append(path)
        
        return critical_endpoints
        
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        return []

def test_found_endpoints(endpoints):
    """測試找到的端點"""
    print(f"\n🧪 測試 {len(endpoints)} 個關鍵端點...")
    
    working_endpoints = {}
    
    for endpoint in endpoints:
        try:
            # 測試 GET 請求
            url = f"http://localhost:7860{endpoint}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"  ✅ GET {endpoint}: 成功")
                working_endpoints[endpoint] = "GET"
            elif response.status_code == 405:  # Method Not Allowed - 可能需要 POST
                print(f"  🔄 GET {endpoint}: 405 (可能需要 POST)")
                working_endpoints[endpoint] = "POST"
            else:
                print(f"  ❌ GET {endpoint}: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ GET {endpoint}: {str(e)}")
    
    return working_endpoints

def test_txt2img_generation(txt2img_endpoint):
    """測試圖像生成功能"""
    print(f"\n🎨 測試圖像生成: {txt2img_endpoint}")
    
    # 簡單的測試載荷
    test_payload = {
        "prompt": "a simple test image",
        "steps": 10,
        "width": 256,
        "height": 256
    }
    
    try:
        url = f"http://localhost:7860{txt2img_endpoint}"
        response = requests.post(url, json=test_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'images' in result and result['images']:
                print("✅ 圖像生成測試成功！")
                return True
            else:
                print("⚠️ 請求成功但沒有圖像數據")
                print(f"回應內容: {list(result.keys())}")
        else:
            print(f"❌ 圖像生成失敗: {response.status_code}")
            if response.text:
                print(f"錯誤詳情: {response.text[:200]}")
                
    except Exception as e:
        print(f"❌ 圖像生成測試失敗: {e}")
    
    return False

def main():
    print("=" * 50)
    print("  尋找正確的 API 端點")
    print("=" * 50)
    
    # 1. 分析 OpenAPI 文檔
    endpoints = find_correct_api_endpoints()
    
    if not endpoints:
        print("❌ 沒有找到任何端點")
        return
    
    # 2. 測試端點
    working_endpoints = test_found_endpoints(endpoints)
    
    if not working_endpoints:
        print("❌ 沒有可用的端點")
        return
    
    # 3. 尋找圖像生成端點
    txt2img_candidates = [ep for ep in working_endpoints.keys() if "txt2img" in ep]
    
    if txt2img_candidates:
        print(f"\n🎯 找到圖像生成端點: {txt2img_candidates}")
        
        # 測試第一個端點
        if test_txt2img_generation(txt2img_candidates[0]):
            print(f"\n🎉 成功！使用端點: {txt2img_candidates[0]}")
            print(f"基礎 URL: http://localhost:7860")
            print(f"完整圖像生成 URL: http://localhost:7860{txt2img_candidates[0]}")
            
            # 生成更新的配置
            print(f"\n📝 更新你的腳本配置:")
            print(f'self.txt2img_endpoint = "{txt2img_candidates[0]}"')
            
            return True
    
    print("❌ 沒有找到可用的圖像生成端點")
    return False

if __name__ == "__main__":
    success = main()
    input(f"\n{'✅ 成功' if success else '❌ 失敗'}！按 Enter 鍵結束...")
