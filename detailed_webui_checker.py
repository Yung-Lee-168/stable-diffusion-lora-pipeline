#!/usr/bin/env python3
"""
詳細的 WebUI 狀態檢查器
檢查 Stable Diffusion WebUI 的所有重要狀態項目
"""

import requests
import json
import time
import subprocess
import os
import sys
from datetime import datetime

class DetailedWebUIChecker:
    def __init__(self):
        self.api_url = "http://localhost:7860"
        self.check_results = {}
        
    def print_header(self, title):
        """打印檢查項目標題"""
        print(f"\n{'='*60}")
        print(f"📋 {title}")
        print(f"{'='*60}")
        
    def check_item(self, item_name, check_function):
        """執行單個檢查項目"""
        print(f"🔍 檢查 {item_name}...", end=" ")
        try:
            result = check_function()
            if result.get("success", False):
                print("✅ 正常")
                self.check_results[item_name] = {"status": "OK", "details": result}
            else:
                print("❌ 異常")
                self.check_results[item_name] = {"status": "FAIL", "details": result}
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            self.check_results[item_name] = {"status": "ERROR", "error": str(e)}
    
    def check_1_basic_connection(self):
        """檢查1: 基本網路連接"""
        try:
            response = requests.get(f"{self.api_url}/sdapi/v1/memory", timeout=5)
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "連接被拒絕"}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "連接逾時"}
    
    def check_2_api_endpoints(self):
        """檢查2: 主要 API 端點"""
        endpoints = [
            "/sdapi/v1/memory",
            "/sdapi/v1/sd-models", 
            "/sdapi/v1/samplers",
            "/sdapi/v1/cmd-flags"
        ]
        
        working_endpoints = 0
        endpoint_details = {}
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.api_url}{endpoint}", timeout=3)
                if response.status_code == 200:
                    working_endpoints += 1
                    endpoint_details[endpoint] = "OK"
                else:
                    endpoint_details[endpoint] = f"HTTP {response.status_code}"
            except:
                endpoint_details[endpoint] = "FAILED"
        
        return {
            "success": working_endpoints == len(endpoints),
            "working_endpoints": working_endpoints,
            "total_endpoints": len(endpoints),
            "details": endpoint_details
        }
    
    def check_3_models(self):
        """檢查3: 可用模型"""
        try:
            response = requests.get(f"{self.api_url}/sdapi/v1/sd-models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                return {
                    "success": len(models) > 0,
                    "model_count": len(models),
                    "current_model": models[0].get("title", "Unknown") if models else None,
                    "all_models": [m.get("title", "Unknown") for m in models[:3]]  # 只顯示前3個
                }
        except:
            pass
        return {"success": False, "error": "無法獲取模型列表"}
    
    def check_4_memory_status(self):
        """檢查4: 記憶體狀態"""
        try:
            response = requests.get(f"{self.api_url}/sdapi/v1/memory", timeout=5)
            if response.status_code == 200:
                memory_info = response.json()
                cuda_info = memory_info.get("cuda", {})
                ram_info = memory_info.get("ram", {})
                
                return {
                    "success": True,
                    "gpu_memory": cuda_info.get("memory", {}),
                    "ram_memory": ram_info,
                    "gpu_available": bool(cuda_info)
                }
        except:
            pass
        return {"success": False, "error": "無法獲取記憶體信息"}
    
    def check_5_generation_test(self):
        """檢查5: 圖片生成測試"""
        test_payload = {
            "prompt": "test",
            "negative_prompt": "",
            "width": 256,
            "height": 256,
            "steps": 5,
            "cfg_scale": 7,
            "sampler_name": "Euler a"
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{self.api_url}/sdapi/v1/txt2img", 
                                   json=test_payload, timeout=60)
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": bool(result.get("images")),
                    "generation_time": generation_time,
                    "image_count": len(result.get("images", []))
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text[:200]
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def check_6_samplers(self):
        """檢查6: 可用採樣器"""
        try:
            response = requests.get(f"{self.api_url}/sdapi/v1/samplers", timeout=5)
            if response.status_code == 200:
                samplers = response.json()
                return {
                    "success": len(samplers) > 0,
                    "sampler_count": len(samplers),
                    "samplers": [s.get("name", "Unknown") for s in samplers[:5]]  # 只顯示前5個
                }
        except:
            pass
        return {"success": False, "error": "無法獲取採樣器列表"}
    
    def check_7_progress_api(self):
        """檢查7: 進度查詢 API"""
        try:
            response = requests.get(f"{self.api_url}/sdapi/v1/progress", timeout=5)
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code
            }
        except:
            return {"success": False, "error": "進度 API 無法訪問"}
    
    def check_8_config(self):
        """檢查8: 配置信息"""
        try:
            response = requests.get(f"{self.api_url}/sdapi/v1/cmd-flags", timeout=5)
            if response.status_code == 200:
                config = response.json()
                return {
                    "success": True,
                    "api_enabled": "--api" in str(config),
                    "listen_enabled": "--listen" in str(config),
                    "config_sample": str(config)[:200]
                }
        except:
            pass
        return {"success": False, "error": "無法獲取配置信息"}
    
    def run_all_checks(self):
        """執行所有檢查項目"""
        print("🔍 詳細的 Stable Diffusion WebUI 狀態檢查")
        print(f"檢查時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 定義所有檢查項目
        checks = [
            ("基本網路連接", self.check_1_basic_connection),
            ("API 端點可用性", self.check_2_api_endpoints),
            ("可用模型", self.check_3_models),
            ("記憶體狀態", self.check_4_memory_status),
            ("圖片生成測試", self.check_5_generation_test),
            ("可用採樣器", self.check_6_samplers),
            ("進度查詢 API", self.check_7_progress_api),
            ("配置信息", self.check_8_config)
        ]
        
        # 執行所有檢查
        self.print_header("執行檢查項目")
        for i, (name, check_func) in enumerate(checks, 1):
            print(f"{i}/8", end=" ")
            self.check_item(name, check_func)
            time.sleep(0.5)  # 短暫延遲避免過度請求
        
        # 顯示詳細結果
        self.show_detailed_results()
        
        # 總結
        self.show_summary()
    
    def show_detailed_results(self):
        """顯示詳細檢查結果"""
        self.print_header("詳細檢查結果")
        
        for item_name, result in self.check_results.items():
            print(f"\n📋 {item_name}:")
            
            if result["status"] == "OK":
                details = result["details"]
                
                if item_name == "基本網路連接":
                    print(f"   ✅ 狀態碼: {details['status_code']}")
                    print(f"   ⏱️ 響應時間: {details['response_time']:.2f}秒")
                
                elif item_name == "API 端點可用性":
                    print(f"   ✅ 可用端點: {details['working_endpoints']}/{details['total_endpoints']}")
                    for endpoint, status in details['details'].items():
                        print(f"      {endpoint}: {status}")
                
                elif item_name == "可用模型":
                    print(f"   ✅ 模型數量: {details['model_count']}")
                    print(f"   🎯 當前模型: {details['current_model']}")
                    if details.get('all_models'):
                        print(f"   📚 可用模型: {', '.join(details['all_models'])}")
                
                elif item_name == "記憶體狀態":
                    if details['gpu_available']:
                        gpu_mem = details['gpu_memory']
                        print(f"   🎮 GPU: 可用")
                        if gpu_mem.get('total'):
                            print(f"      總記憶體: {gpu_mem['total']}")
                        if gpu_mem.get('free'):
                            print(f"      可用記憶體: {gpu_mem['free']}")
                    else:
                        print(f"   ⚠️ GPU: 不可用或信息獲取失敗")
                
                elif item_name == "圖片生成測試":
                    print(f"   ✅ 生成時間: {details['generation_time']:.2f}秒")
                    print(f"   🖼️ 生成圖片數: {details['image_count']}")
                
                elif item_name == "可用採樣器":
                    print(f"   ✅ 採樣器數量: {details['sampler_count']}")
                    print(f"   📋 可用採樣器: {', '.join(details['samplers'])}")
                
                elif item_name == "配置信息":
                    print(f"   ✅ API 啟用: {details['api_enabled']}")
                    print(f"   🌐 Listen 啟用: {details['listen_enabled']}")
            
            elif result["status"] == "FAIL":
                print(f"   ❌ 檢查失敗: {result['details'].get('error', '未知錯誤')}")
            
            else:  # ERROR
                print(f"   🚫 檢查錯誤: {result['error']}")
    
    def show_summary(self):
        """顯示檢查總結"""
        self.print_header("檢查總結")
        
        total_checks = len(self.check_results)
        passed_checks = sum(1 for r in self.check_results.values() if r["status"] == "OK")
        failed_checks = total_checks - passed_checks
        
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        print(f"📊 檢查結果統計:")
        print(f"   總檢查項目: {total_checks}")
        print(f"   通過項目: {passed_checks}")
        print(f"   失敗項目: {failed_checks}")
        print(f"   成功率: {success_rate:.1f}%")
        
        # 給出建議
        print(f"\n💡 建議:")
        if success_rate >= 80:
            print("   🎉 WebUI 狀態良好，可以開始 3天測試！")
            print("   執行命令: python day1_basic_test.py")
        elif success_rate >= 60:
            print("   ⚠️ WebUI 部分功能異常，建議檢查：")
            for item_name, result in self.check_results.items():
                if result["status"] != "OK":
                    print(f"      • {item_name}")
        else:
            print("   ❌ WebUI 狀態不佳，建議：")
            print("      1. 重新啟動 WebUI")
            print("      2. 檢查錯誤日誌")
            print("      3. 確認硬體需求")

def main():
    """主函數"""
    checker = DetailedWebUIChecker()
    
    try:
        checker.run_all_checks()
    except KeyboardInterrupt:
        print("\n\n⏹️ 檢查已中斷")
    except Exception as e:
        print(f"\n❌ 檢查過程出錯: {e}")
    
    input("\n按 Enter 鍵退出...")

if __name__ == "__main__":
    main()
