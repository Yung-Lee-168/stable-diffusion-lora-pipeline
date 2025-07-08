#!/usr/bin/env python3
"""
快速狀態檢查和指導腳本
"""

import requests
import subprocess
import sys
import os

def check_webui_status():
    """檢查 WebUI 狀態並提供指導"""
    print("🔍 檢查 WebUI 狀態...")
    
    # 檢查主頁
    try:
        response = requests.get("http://localhost:7860", timeout=5)
        if response.status_code == 200:
            print("✅ WebUI 在運行中")
            
            # 檢查 API
            try:
                api_response = requests.get("http://localhost:7860/sdapi/v1/options", timeout=5)
                if api_response.status_code == 200:
                    print("✅ API 可用")
                    print("🎯 可以開始測試了！")
                    return "ready"
                else:
                    print("❌ API 不可用")
                    print("💡 建議：重新啟動 WebUI 並確保使用 --api 參數")
                    return "api_disabled"
            except:
                print("❌ API 端點無法訪問")
                return "api_error"
        else:
            print("⚠️ WebUI 回應異常")
            return "webui_error"
    except requests.exceptions.ConnectionError:
        print("❌ WebUI 未運行")
        print("💡 需要先啟動 WebUI")
        return "not_running"
    except Exception as e:
        print(f"❌ 檢查失敗: {e}")
        return "unknown_error"

def provide_guidance(status):
    """根據狀態提供操作指導"""
    print("\n" + "="*50)
    print("📋 操作指導")
    print("="*50)
    
    if status == "ready":
        print("🎯 一切就緒！請執行：")
        print("   python day2_enhanced_test.py")
        
    elif status == "not_running":
        print("🚀 請按照以下步驟啟動 WebUI：")
        print("   1. 運行：.\\START_WEBUI_AND_WAIT.bat")
        print("   2. 等待瀏覽器自動打開")
        print("   3. 再次運行本腳本檢查狀態")
        
    elif status == "api_disabled":
        print("🔧 API 未啟用，請：")
        print("   1. 關閉當前 WebUI")
        print("   2. 運行：.\\COMPLETE_FIX.bat")
        print("   3. 重新啟動 WebUI")
        
    else:
        print("⚠️ 遇到問題，請：")
        print("   1. 重啟 WebUI")
        print("   2. 檢查防火牆設置")
        print("   3. 確認端口 7860 未被佔用")

def main():
    print("=" * 60)
    print("    Stable Diffusion WebUI 快速狀態檢查")
    print("=" * 60)
    
    status = check_webui_status()
    provide_guidance(status)
    
    print("\n" + "="*50)
    print("🔄 如需重新檢查，請再次運行此腳本")
    print("="*50)

if __name__ == "__main__":
    main()
