#!/usr/bin/env python3
"""
檢查 WebUI 是否啟動 - 超簡單版
"""

import requests
import sys

def main():
    print("🔍 檢查 WebUI 狀態...")
    
    try:
        # 檢查主頁
        response = requests.get("http://localhost:7860", timeout=5)
        if response.status_code == 200:
            print("✅ WebUI 正在運行")
            
            # 檢查 API
            try:
                api_response = requests.get("http://localhost:7860/sdapi/v1/options", timeout=5)
                if api_response.status_code == 200:
                    print("✅ API 可用")
                    print("🎯 可以開始測試了！")
                    print("\n執行：python day2_enhanced_test.py")
                    sys.exit(0)
                else:
                    print("❌ API 不可用 - 請確認使用了 --api 參數啟動")
            except:
                print("❌ API 無法連接")
        else:
            print("❌ WebUI 回應異常")
            
    except requests.exceptions.ConnectionError:
        print("❌ WebUI 未啟動")
        
    except Exception as e:
        print(f"❌ 檢查失敗: {e}")
    
    print("\n請先啟動 WebUI：")
    print("1. 雙擊執行：webui-user.bat")
    print("2. 或執行：webui.bat --api")
    print("3. 等待瀏覽器打開並顯示界面")
    print("4. 然後重新運行此檢查")
    sys.exit(1)

if __name__ == "__main__":
    main()
