#!/usr/bin/env python3
"""
第1天：基礎 API 測試和手動提示詞驗證
目標：確保 API 正常工作，測試基本的時尚相關提示詞
"""

import requests
import json
import base64
import os
from datetime import datetime

class Day1Tester:
    def __init__(self):
        self.api_url = "http://localhost:7860"
        self.output_dir = "day1_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def test_api_connection(self):
        """測試 API 連接"""
        try:
            response = requests.get(f"{self.api_url}/sdapi/v1/memory")
            if response.status_code == 200:
                print("✅ API 連接成功")
                return True
            else:
                print(f"❌ API 連接失敗: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API 連接錯誤: {e}")
            return False
    
    def test_basic_generation(self):
        """測試基本圖片生成"""
        payload = {
            "prompt": "a beautiful woman in elegant dress, high fashion, professional photo",
            "negative_prompt": "low quality, blurry, distorted",
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg_scale": 7,
            "sampler_name": "DPM++ 2M Karras"
        }
        
        try:
            response = requests.post(f"{self.api_url}/sdapi/v1/txt2img", json=payload)
            if response.status_code == 200:
                result = response.json()
                # 保存圖片
                image_data = base64.b64decode(result['images'][0])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(self.output_dir, f"basic_test_{timestamp}.png")
                with open(image_path, "wb") as f:
                    f.write(image_data)
                print(f"✅ 基本生成測試成功，圖片保存至: {image_path}")
                return True
            else:
                print(f"❌ 圖片生成失敗: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 生成錯誤: {e}")
            return False
    
    def test_fashion_prompts(self):
        """測試時尚相關提示詞"""
        fashion_prompts = [
            "elegant evening gown, high fashion model, studio lighting, luxury brand style",
            "casual street fashion, modern trendy outfit, urban background",
            "vintage fashion style, 1950s dress, retro aesthetic, classic pose",
            "business attire, professional woman, office fashion, sophisticated look",
            "bohemian style dress, flowing fabric, natural lighting, artistic composition"
        ]
        
        results = []
        for i, prompt in enumerate(fashion_prompts):
            print(f"測試提示詞 {i+1}/{len(fashion_prompts)}: {prompt[:50]}...")
            
            payload = {
                "prompt": f"{prompt}, high quality, detailed, professional photography",
                "negative_prompt": "low quality, blurry, distorted, amateur",
                "width": 512,
                "height": 768,
                "steps": 25,
                "cfg_scale": 7.5,
                "sampler_name": "DPM++ 2M Karras"
            }
            
            try:
                response = requests.post(f"{self.api_url}/sdapi/v1/txt2img", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    # 保存圖片
                    image_data = base64.b64decode(result['images'][0])
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(self.output_dir, f"fashion_test_{i+1}_{timestamp}.png")
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    
                    results.append({
                        "prompt": prompt,
                        "image_path": image_path,
                        "success": True
                    })
                    print(f"✅ 成功生成: {image_path}")
                else:
                    results.append({
                        "prompt": prompt,
                        "error": f"HTTP {response.status_code}",
                        "success": False
                    })
                    print(f"❌ 生成失敗: HTTP {response.status_code}")
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "error": str(e),
                    "success": False
                })
                print(f"❌ 錯誤: {e}")
        
        # 保存結果
        with open(os.path.join(self.output_dir, "day1_fashion_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def run_day1_tests(self):
        """運行第1天的所有測試"""
        print("=" * 50)
        print("第1天測試開始：基礎 API 和時尚提示詞測試")
        print("=" * 50)
        
        # 測試1：API 連接
        if not self.test_api_connection():
            print("❌ API 連接失敗，請確保 WebUI 已啟動並開啟 API 模式")
            return False
        
        # 測試2：基本生成
        if not self.test_basic_generation():
            print("❌ 基本生成測試失敗")
            return False
        
        # 測試3：時尚提示詞
        results = self.test_fashion_prompts()
        successful = sum(1 for r in results if r["success"])
        print(f"\n第1天測試完成：{successful}/{len(results)} 個時尚提示詞測試成功")
        
        # 生成報告
        report = {
            "day": 1,
            "timestamp": datetime.now().isoformat(),
            "api_connection": True,
            "basic_generation": True,
            "fashion_prompts_tested": len(results),
            "fashion_prompts_successful": successful,
            "success_rate": successful / len(results) if results else 0,
            "results": results
        }
        
        with open(os.path.join(self.output_dir, "day1_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 第1天報告已保存至: {os.path.join(self.output_dir, 'day1_report.json')}")
        return True

if __name__ == "__main__":
    tester = Day1Tester()
    tester.run_day1_tests()
