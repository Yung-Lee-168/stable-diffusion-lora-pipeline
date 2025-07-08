#!/usr/bin/env python3
"""
WebUI API 診斷工具 - 檢查和修復 API 連接問題
"""

import requests
import time
import subprocess
import os
import psutil

def check_webui_process():
    """檢查 WebUI 進程是否正在運行"""
    print("🔍 檢查 WebUI 進程...")
    
    webui_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'webui.py' in cmdline or 'webui.bat' in cmdline:
                webui_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if webui_processes:
        print("✅ 發現 WebUI 進程:")
        for proc in webui_processes:
            print(f"   PID: {proc['pid']}, 命令: {proc['cmdline']}")
        return True
    else:
        print("❌ 沒有發現 WebUI 進程")
        return False

def check_port_7860():
    """檢查端口 7860 是否被佔用"""
    print("🔍 檢查端口 7860...")
    
    for conn in psutil.net_connections():
        if conn.laddr.port == 7860:
            print(f"✅ 端口 7860 被佔用，PID: {conn.pid}")
            try:
                proc = psutil.Process(conn.pid)
                print(f"   進程名稱: {proc.name()}")
                print(f"   命令行: {' '.join(proc.cmdline())}")
            except:
                pass
            return True
    
    print("❌ 端口 7860 未被佔用")
    return False

def test_api_endpoints():
    """測試不同的 API 端點"""
    print("🔍 測試 API 端點...")
    
    base_url = "http://localhost:7860"
    endpoints = [
        "/",
        "/docs",
        "/sdapi/v1/options",
        "/sdapi/v1/cmd-flags",
        "/sdapi/v1/sd-models"
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            results[endpoint] = {
                "status": response.status_code,
                "accessible": True
            }
            print(f"   {endpoint}: {response.status_code}")
        except requests.exceptions.ConnectionError:
            results[endpoint] = {
                "status": "Connection Error",
                "accessible": False
            }
            print(f"   {endpoint}: 連接錯誤")
        except Exception as e:
            results[endpoint] = {
                "status": str(e),
                "accessible": False
            }
            print(f"   {endpoint}: {str(e)}")
    
    return results

def check_webui_config():
    """檢查 WebUI 配置文件"""
    print("🔍 檢查 WebUI 配置...")
    
    config_files = [
        "webui-user.bat",
        "webui-user.sh",
        "config.json",
        "ui-config.json"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ 發現配置文件: {config_file}")
            
            if config_file == "webui-user.bat":
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '--api' in content:
                        print("   ✅ API 模式已啟用")
                    else:
                        print("   ❌ API 模式未啟用")
                        return False
        else:
            print(f"⚠️ 配置文件不存在: {config_file}")
    
    return True

def fix_webui_api():
    """修復 WebUI API 配置"""
    print("🔧 修復 WebUI API 配置...")
    
    webui_user_bat = "webui-user.bat"
    
    if not os.path.exists(webui_user_bat):
        print("❌ webui-user.bat 不存在")
        return False
    
    # 讀取現有配置
    with open(webui_user_bat, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查是否已經有 API 設定
    if '--api' in content:
        print("✅ API 模式已經啟用")
        return True
    
    # 添加 API 設定
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if line.strip().startswith('set COMMANDLINE_ARGS='):
            if '--api' not in line:
                line = line.rstrip() + ' --api'
            new_lines.append(line)
        elif 'COMMANDLINE_ARGS' not in content and line.strip().startswith('@echo off'):
            new_lines.append(line)
            new_lines.append('set COMMANDLINE_ARGS=--api')
        else:
            new_lines.append(line)
    
    # 如果沒有找到 COMMANDLINE_ARGS，添加它
    if 'COMMANDLINE_ARGS' not in content:
        new_lines.insert(-1, 'set COMMANDLINE_ARGS=--api')
    
    # 寫回文件
    with open(webui_user_bat, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print("✅ API 模式已啟用，請重新啟動 WebUI")
    return True

def restart_webui():
    """重新啟動 WebUI"""
    print("🔄 重新啟動 WebUI...")
    
    # 先終止現有進程
    for proc in psutil.process_iter():
        try:
            cmdline = ' '.join(proc.cmdline())
            if 'webui.py' in cmdline or 'webui.bat' in cmdline:
                print(f"   終止進程 PID: {proc.pid}")
                proc.terminate()
                time.sleep(2)
                if proc.is_running():
                    proc.kill()
        except:
            continue
    
    # 啟動新進程
    if os.path.exists("webui.bat"):
        print("   啟動 webui.bat...")
        subprocess.Popen(["webui.bat"], shell=True)
        return True
    elif os.path.exists("webui.py"):
        print("   啟動 webui.py...")
        subprocess.Popen(["python", "webui.py", "--api"], shell=True)
        return True
    else:
        print("❌ 找不到 WebUI 啟動文件")
        return False

def wait_for_api(timeout=120):
    """等待 API 可用"""
    print(f"⏳ 等待 API 啟動（最多 {timeout} 秒）...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://localhost:7860/sdapi/v1/options", timeout=5)
            if response.status_code == 200:
                print("✅ API 已啟動並可用！")
                return True
        except:
            pass
        
        print(".", end="", flush=True)
        time.sleep(5)
    
    print("\n❌ API 啟動超時")
    return False

def main():
    """主診斷流程"""
    print("=" * 50)
    print("  WebUI API 診斷與修復工具")
    print("=" * 50)
    
    # 1. 檢查進程
    process_running = check_webui_process()
    
    # 2. 檢查端口
    port_occupied = check_port_7860()
    
    # 3. 測試 API
    api_results = test_api_endpoints()
    
    # 4. 檢查配置
    config_ok = check_webui_config()
    
    # 5. 分析問題並修復
    if not config_ok:
        print("\n🔧 發現配置問題，正在修復...")
        if fix_webui_api():
            print("✅ 配置已修復")
        else:
            print("❌ 配置修復失敗")
            return False
    
    # 6. 如果需要，重新啟動
    if not api_results.get("/sdapi/v1/options", {}).get("accessible", False):
        print("\n🔄 API 不可用，重新啟動 WebUI...")
        if restart_webui():
            if wait_for_api():
                print("✅ WebUI API 修復完成！")
                return True
            else:
                print("❌ WebUI 啟動失敗")
                return False
        else:
            print("❌ 無法重新啟動 WebUI")
            return False
    else:
        print("\n✅ API 已正常運作！")
        return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 診斷完成！現在你可以:")
        print("   1. 執行 debug_clip_test.py")
        print("   2. 執行 day2_enhanced_test.py")
        print("   3. 訪問 http://localhost:7860")
    else:
        print("\n❌ 診斷失敗，請手動檢查:")
        print("   1. WebUI 是否正確安裝")
        print("   2. Python 環境是否正確")
        print("   3. 是否有足夠的 GPU 記憶體")
    
    input("\n按 Enter 鍵結束...")
