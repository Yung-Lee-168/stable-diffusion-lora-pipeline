#!/usr/bin/env python3
"""
簡化的 WebUI 狀態檢查器
持續監控 WebUI 是否啟動完成
"""

import requests
import time
import sys
from datetime import datetime

def check_webui_status():
    """檢查 WebUI 狀態"""
    urls_to_test = [
        "http://localhost:7860",
        "http://127.0.0.1:7860"
    ]
    
    for url in urls_to_test:
        try:
            # 測試基本連接
            response = requests.get(f"{url}/sdapi/v1/memory", timeout=3)
            if response.status_code == 200:
                return True, url, response.json()
        except:
            continue
    
    return False, None, None

def wait_for_webui():
    """等待 WebUI 啟動"""
    print("🔍 正在檢查 Stable Diffusion WebUI 狀態...")
    print("(如果 WebUI 尚未啟動，請在另一個視窗運行 webui-user.bat)")
    print()
    
    check_count = 0
    while True:
        check_count += 1
        print(f"\r⏳ 檢查中... ({check_count})", end="", flush=True)
        
        is_ready, url, info = check_webui_status()
        
        if is_ready:
            print(f"\n✅ WebUI 已啟動！")
            print(f"   API 地址: {url}")
            
            if info:
                gpu_info = info.get('cuda', {})
                if gpu_info:
                    total_memory = gpu_info.get('memory', {}).get('total', 'Unknown')
                    print(f"   GPU 記憶體: {total_memory}")
            
            # 測試圖片生成 API
            print("\n🎨 測試圖片生成 API...")
            try:
                test_payload = {
                    "prompt": "test",
                    "steps": 1,
                    "width": 64,
                    "height": 64
                }
                response = requests.post(f"{url}/sdapi/v1/txt2img", 
                                       json=test_payload, timeout=30)
                if response.status_code == 200:
                    print("✅ 圖片生成 API 正常")
                    return True
                else:
                    print(f"⚠️ 圖片生成 API 回應異常: {response.status_code}")
                    return False
            except Exception as e:
                print(f"⚠️ 圖片生成 API 測試失敗: {e}")
                return False
        
        time.sleep(2)
        
        # 每30次檢查（約1分鐘）顯示一次提示
        if check_count % 30 == 0:
            print(f"\n💡 提示：如果持續等待，請確認：")
            print("   1. webui-user.bat 是否正在運行")
            print("   2. 是否有錯誤訊息")
            print("   3. 端口 7860 是否被其他程式佔用")
            print("   按 Ctrl+C 可以中斷等待")

def main():
    """主函數"""
    print("=" * 60)
    print("  Stable Diffusion WebUI 狀態檢查器")
    print("=" * 60)
    print(f"檢查時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        if wait_for_webui():
            print("\n🎉 WebUI 完全就緒！現在可以開始 3天測試了。")
            print("\n下一步:")
            print("  python day1_basic_test.py")
        else:
            print("\n⚠️ WebUI 部分功能異常，建議檢查 WebUI 日誌")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ 檢查已中斷")
        print("💡 如需手動驗證，可以：")
        print("   1. 在瀏覽器打開 http://localhost:7860")
        print("   2. 確認 WebUI 界面正常顯示")
        print("   3. 嘗試生成一張測試圖片")
    
    except Exception as e:
        print(f"\n❌ 檢查過程出錯: {e}")
    
    input("\n按 Enter 鍵退出...")

if __name__ == "__main__":
    main()
