#!/usr/bin/env python3
"""
Stable Diffusion WebUI 診斷工具
幫助排查 API 連接問題
"""

import requests
import json
import time
import subprocess
import os
import sys
from datetime import datetime

class WebUIDiagnostic:
    def __init__(self):
        self.api_url = "http://localhost:7860"
        self.alternative_urls = [
            "http://127.0.0.1:7860",
            "http://0.0.0.0:7860"
        ]
        
    def test_connection(self, url, timeout=5):
        """測試指定 URL 的連接"""
        try:
            response = requests.get(f"{url}/sdapi/v1/memory", timeout=timeout)
            return {
                "success": True,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "連接被拒絕"}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "連接逾時"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def check_webui_process(self):
        """檢查 WebUI 進程是否運行"""
        try:
            # Windows 檢查進程
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                  capture_output=True, text=True)
            python_processes = result.stdout
            
            webui_running = False
            if 'python.exe' in python_processes:
                # 進一步檢查是否是 webui 相關進程
                try:
                    result = subprocess.run(['netstat', '-ano'], 
                                          capture_output=True, text=True)
                    if ':7860' in result.stdout:
                        webui_running = True
                except:
                    pass
            
            return webui_running
        except Exception as e:
            print(f"檢查進程時出錯: {e}")
            return False
    
    def check_port_availability(self):
        """檢查端口 7860 是否被佔用"""
        try:
            result = subprocess.run(['netstat', '-ano'], 
                                  capture_output=True, text=True)
            lines = result.stdout.split('\n')
            port_info = []
            
            for line in lines:
                if ':7860' in line:
                    port_info.append(line.strip())
            
            return port_info
        except Exception as e:
            print(f"檢查端口時出錯: {e}")
            return []
    
    def test_all_endpoints(self):
        """測試所有可能的 API 端點"""
        print("🔍 測試 API 連接...")
        print("=" * 50)
        
        for i, url in enumerate([self.api_url] + self.alternative_urls):
            print(f"測試 {i+1}: {url}")
            result = self.test_connection(url)
            
            if result["success"]:
                print(f"✅ 連接成功！")
                print(f"   狀態碼: {result['status_code']}")
                print(f"   響應時間: {result['response_time']:.2f}秒")
                
                # 測試更多端點
                self.test_additional_endpoints(url)
                return True
            else:
                print(f"❌ 連接失敗: {result['error']}")
        
        return False
    
    def test_additional_endpoints(self, base_url):
        """測試其他 API 端點"""
        endpoints = [
            ("/sdapi/v1/sd-models", "模型列表"),
            ("/sdapi/v1/samplers", "採樣器列表"),
            ("/sdapi/v1/cmd-flags", "命令行參數"),
            ("/sdapi/v1/progress", "進度查詢")
        ]
        
        print(f"\n🔍 測試其他 API 端點...")
        for endpoint, description in endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print(f"✅ {description}: 正常")
                else:
                    print(f"⚠️ {description}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {description}: {e}")
    
    def provide_solutions(self):
        """提供解決方案"""
        print("\n" + "=" * 50)
        print("🛠️ 問題診斷和解決方案")
        print("=" * 50)
        
        # 檢查進程
        webui_running = self.check_webui_process()
        print(f"WebUI 進程狀態: {'✅ 運行中' if webui_running else '❌ 未運行'}")
        
        # 檢查端口
        port_info = self.check_port_availability()
        if port_info:
            print("✅ 端口 7860 已被使用:")
            for info in port_info[:3]:  # 只顯示前3個
                print(f"   {info}")
        else:
            print("❌ 端口 7860 未被使用")
        
        print("\n📋 建議的解決步驟:")
        
        if not webui_running:
            print("1. 🚀 啟動 WebUI:")
            print("   • 打開新的命令提示字元")
            print("   • 切換到 WebUI 目錄")
            print("   • 執行: webui-user.bat")
            print("   • 等待看到: 'Running on local URL: http://127.0.0.1:7860'")
        
        print("\n2. 🔧 檢查配置:")
        print("   • 確認 webui-user.bat 包含: --api --listen")
        print("   • 檢查防火牆設定")
        print("   • 確保沒有其他程式佔用 7860 端口")
        
        print("\n3. 🔄 重啟 WebUI:")
        print("   • 關閉現有的 WebUI 視窗 (Ctrl+C)")
        print("   • 等待 5-10 秒")
        print("   • 重新執行 webui-user.bat")
        
        print("\n4. 🌐 測試替代方案:")
        print("   • 嘗試使用 127.0.0.1:7860 而不是 localhost:7860")
        print("   • 檢查 hosts 檔案是否正確")
        
        print("\n5. 📊 系統資源:")
        print("   • 確保有足夠的 GPU 記憶體")
        print("   • 關閉其他佔用 GPU 的程式")
        print("   • 檢查系統記憶體使用情況")
    
    def wait_for_webui(self, max_wait=300):
        """等待 WebUI 啟動"""
        print(f"\n⏳ 等待 WebUI 啟動 (最多等待 {max_wait} 秒)...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if self.test_connection(self.api_url, timeout=2)["success"]:
                print("✅ WebUI 已啟動！")
                return True
            
            elapsed = int(time.time() - start_time)
            print(f"\r等待中... {elapsed}/{max_wait}秒", end="", flush=True)
            time.sleep(5)
        
        print(f"\n❌ 等待逾時 ({max_wait}秒)")
        return False
    
    def run_full_diagnostic(self):
        """運行完整診斷"""
        print("🔍 Stable Diffusion WebUI API 診斷工具")
        print("=" * 50)
        print(f"診斷時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 測試連接
        connection_ok = self.test_all_endpoints()
        
        if connection_ok:
            print("\n🎉 API 連接正常！您可以開始進行3天測試了。")
            print("執行命令: python day1_basic_test.py")
        else:
            # 提供解決方案
            self.provide_solutions()
            
            # 詢問是否等待
            print(f"\n❓ 是否要等待 WebUI 啟動？(Y/N)")
            choice = input("請輸入選擇: ").strip().upper()
            
            if choice == 'Y':
                if self.wait_for_webui():
                    print("🎉 現在可以開始測試了！")
                    return True
                else:
                    print("😞 WebUI 仍未啟動，請手動檢查。")
        
        return connection_ok

def main():
    """主函數"""
    diagnostic = WebUIDiagnostic()
    
    try:
        success = diagnostic.run_full_diagnostic()
        
        if success:
            print("\n" + "=" * 50)
            print("🚀 準備開始 3天可行性測試")
            print("=" * 50)
            print("下一步: python day1_basic_test.py")
        else:
            print("\n" + "=" * 50)
            print("❌ 診斷完成，請按照建議解決問題")
            print("=" * 50)
            
    except KeyboardInterrupt:
        print("\n⏹️ 診斷已中斷")
    except Exception as e:
        print(f"\n❌ 診斷過程出錯: {e}")
    
    input("\n按 Enter 鍵退出...")

if __name__ == "__main__":
    main()
