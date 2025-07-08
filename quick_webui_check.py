#!/usr/bin/env python3
"""
快速 WebUI 狀態檢查工具
"""

import requests
import time
import subprocess
import os

def simple_api_check():
    """簡單的 API 檢查"""
    print("🔍 檢查 WebUI API 狀態...")
    
    urls_to_test = [
        "http://localhost:7860",
        "http://localhost:7860/docs",
        "http://localhost:7860/sdapi/v1/options",
        "http://127.0.0.1:7860",
        "http://127.0.0.1:7860/sdapi/v1/options"
    ]
    
    for url in urls_to_test:
        try:
            print(f"測試: {url}")
            response = requests.get(url, timeout=10)
            print(f"✅ 成功! 狀態碼: {response.status_code}")
            
            if "sdapi/v1/options" in url and response.status_code == 200:
                print("🎉 WebUI API 正常運作！")
                return True
                
        except requests.exceptions.ConnectionError:
            print(f"❌ 連接失敗: {url}")
        except Exception as e:
            print(f"❌ 錯誤: {e}")
    
    return False

def check_webui_running():
    """檢查 WebUI 是否正在運行"""
    print("\n🔍 檢查 WebUI 進程...")
    
    try:
        # 使用 netstat 檢查端口
        result = subprocess.run(['netstat', '-an'], capture_output=True, text=True, shell=True)
        if ':7860' in result.stdout:
            print("✅ 端口 7860 正在被使用")
            return True
        else:
            print("❌ 端口 7860 沒有被使用")
            return False
    except:
        print("⚠️ 無法檢查端口狀態")
        return False

def start_webui():
    """啟動 WebUI"""
    print("\n🚀 啟動 WebUI...")
    
    if os.path.exists("webui.bat"):
        print("執行 webui.bat...")
        subprocess.Popen(["webui.bat"], shell=True, cwd=os.getcwd())
        
        print("⏳ 等待 WebUI 啟動（60秒）...")
        for i in range(12):  # 60秒，每5秒檢查一次
            time.sleep(5)
            if simple_api_check():
                return True
            print(f"   等待中... ({(i+1)*5}秒)")
        
        print("❌ WebUI 啟動超時")
        return False
    else:
        print("❌ 找不到 webui.bat")
        return False

def main():
    print("=" * 40)
    print("  快速 WebUI 診斷工具")
    print("=" * 40)
    
    # 首先檢查 API 是否已經可用
    if simple_api_check():
        print("\n✅ WebUI API 已經正常運作！")
        print("你可以執行 debug_clip_test.py 或 day2_enhanced_test.py")
        return True
    
    # 檢查 WebUI 是否正在運行
    if not check_webui_running():
        print("\n⚠️ WebUI 似乎沒有運行，嘗試啟動...")
        if start_webui():
            print("✅ WebUI 啟動成功！")
            return True
        else:
            print("❌ WebUI 啟動失敗")
            print("\n🔧 手動啟動步驟:")
            print("1. 開啟命令提示字元")
            print("2. 切換到 WebUI 目錄")
            print("3. 執行: webui.bat")
            print("4. 等待看到 'Running on local URL: http://127.0.0.1:7860'")
            return False
    else:
        print("\n⚠️ WebUI 在運行但 API 不可用")
        print("可能的原因:")
        print("1. WebUI 正在啟動中（請等待）")
        print("2. API 模式未啟用")
        print("3. 端口被其他程序佔用")
        return False

if __name__ == "__main__":
    success = main()
    input(f"\n{'成功' if success else '失敗'}！按 Enter 鍵結束...")
