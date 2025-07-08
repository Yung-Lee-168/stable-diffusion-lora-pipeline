#!/usr/bin/env python3
"""
手動檢查 WebUI 狀態 - 簡單快速的檢查工具
"""

import requests
import webbrowser
import time

def quick_webui_check():
    """快速檢查 WebUI 狀態"""
    print("🔍 快速檢查 WebUI 狀態...")
    
    base_url = "http://localhost:7860"
    
    # 1. 檢查主頁
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ WebUI 主頁可訪問")
            
            # 2. 檢查常見 API 端點
            endpoints = [
                "/sdapi/v1/options",
                "/docs", 
                "/info",
                "/openapi.json"
            ]
            
            working_endpoints = []
            for endpoint in endpoints:
                try:
                    resp = requests.get(f"{base_url}{endpoint}", timeout=3)
                    if resp.status_code == 200:
                        working_endpoints.append(endpoint)
                        print(f"✅ {endpoint} 可用")
                    else:
                        print(f"❌ {endpoint} 不可用 ({resp.status_code})")
                except:
                    print(f"❌ {endpoint} 無法訪問")
            
            # 3. 判斷 API 類型
            if "/sdapi/v1/options" in working_endpoints:
                print("\n🎯 檢測到標準 SD API")
                print("   可以執行: python day2_enhanced_test.py")
                return "standard"
            elif "/docs" in working_endpoints or "/info" in working_endpoints:
                print("\n🎯 檢測到 Gradio API")
                print("   需要使用 Gradio 客戶端")
                return "gradio"
            else:
                print("\n⚠️ 未知的 API 類型")
                return "unknown"
        else:
            print(f"❌ WebUI 主頁無法訪問 ({response.status_code})")
            return "offline"
    except:
        print("❌ WebUI 未運行或無法連接")
        return "offline"

def open_browser():
    """在瀏覽器中打開 WebUI"""
    print("\n🌐 在瀏覽器中打開 WebUI...")
    try:
        webbrowser.open("http://localhost:7860")
        print("✅ 瀏覽器已打開，請檢查 WebUI 是否正常顯示")
    except:
        print("❌ 無法打開瀏覽器")

def main():
    print("=" * 40)
    print("  WebUI 快速狀態檢查")
    print("=" * 40)
    
    status = quick_webui_check()
    
    if status == "offline":
        print("\n💡 建議操作:")
        print("1. 執行 START_WEBUI_AND_WAIT.bat")
        print("2. 或手動執行 webui.bat")
        print("3. 等待看到 'Running on local URL' 訊息")
    elif status == "standard":
        print("\n🎉 準備就緒！可以執行 CLIP 測試")
        open_browser()
    elif status == "gradio":
        print("\n⚠️ 需要特殊處理 Gradio API")
        open_browser()
        print("請檢查瀏覽器中的界面是否正常")
    else:
        print("\n🤔 需要進一步檢查")
        open_browser()
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    main()
    input("\n按 Enter 鍵結束...")
