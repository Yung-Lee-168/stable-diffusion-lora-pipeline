#!/usr/bin/env python3
"""
3天 Stable Diffusion + 時尚圖片生成可行性測試計畫
執行此腳本將自動創建所有測試工具和計畫文件
"""

import os
import json
from datetime import datetime, timedelta

def create_day1_scripts():
    """創建第1天的手動測試腳本"""
    
    # Day 1: 基礎 API 測試腳本
    day1_basic_test = '''#!/usr/bin/env python3
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
        print(f"\\n第1天測試完成：{successful}/{len(results)} 個時尚提示詞測試成功")
        
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
        
        print(f"\\n📊 第1天報告已保存至: {os.path.join(self.output_dir, 'day1_report.json')}")
        return True

if __name__ == "__main__":
    tester = Day1Tester()
    tester.run_day1_tests()
'''
    
    with open("day1_basic_test.py", "w", encoding="utf-8") as f:
        f.write(day1_basic_test)
    print("✅ 已創建 day1_basic_test.py")

def create_day2_scripts():
    """創建第2天的自動化測試腳本"""
    
    day2_advanced_test = '''#!/usr/bin/env python3
"""
第2天：進階測試 - 圖片分析和提示詞生成
目標：測試圖片特徵提取和自動提示詞生成
"""

import requests
import json
import base64
import os
from datetime import datetime
from PIL import Image
import numpy as np

class Day2Tester:
    def __init__(self):
        self.api_url = "http://localhost:7860"
        self.output_dir = "day2_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def install_requirements(self):
        """檢查並安裝必要的套件"""
        try:
            import torch
            import transformers
            from transformers import CLIPProcessor, CLIPModel
            print("✅ 所需套件已安裝")
            return True
        except ImportError as e:
            print(f"❌ 缺少必要套件: {e}")
            print("請運行: pip install torch transformers pillow")
            return False
    
    def load_clip_model(self):
        """載入 CLIP 模型用於圖片分析"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✅ CLIP 模型載入成功")
            return model, processor
        except Exception as e:
            print(f"❌ CLIP 模型載入失敗: {e}")
            return None, None
    
    def analyze_image_features(self, image_path, model, processor):
        """分析圖片特徵並生成描述"""
        try:
            from PIL import Image
            import torch
            
            image = Image.open(image_path).convert("RGB")
            
            # 預定義的時尚相關標籤
            fashion_labels = [
                "elegant dress", "casual outfit", "formal wear", "vintage style",
                "modern fashion", "luxury clothing", "street style", "business attire",
                "evening gown", "summer dress", "winter coat", "bohemian style",
                "minimalist fashion", "colorful outfit", "black and white clothing",
                "floral pattern", "solid color", "textured fabric"
            ]
            
            # 使用 CLIP 分析圖片
            inputs = processor(text=fashion_labels, images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # 獲取最相關的標籤
            top_indices = probs[0].topk(5).indices
            top_labels = [fashion_labels[i] for i in top_indices]
            top_scores = [probs[0][i].item() for i in top_indices]
            
            return {
                "top_labels": top_labels,
                "scores": top_scores,
                "analysis_success": True
            }
            
        except Exception as e:
            print(f"❌ 圖片分析失敗: {e}")
            return {"analysis_success": False, "error": str(e)}
    
    def generate_prompt_from_analysis(self, analysis_result):
        """根據分析結果生成 SD 提示詞"""
        if not analysis_result["analysis_success"]:
            return "elegant fashion, high quality, professional photography"
        
        top_labels = analysis_result["top_labels"]
        
        # 構建提示詞
        base_prompt = ", ".join(top_labels[:3])
        enhanced_prompt = f"{base_prompt}, high fashion, professional photography, detailed, high quality, studio lighting"
        
        return enhanced_prompt
    
    def test_image_to_prompt_generation(self):
        """測試圖片分析到提示詞生成的完整流程"""
        # 先生成一些參考圖片用於分析
        reference_prompts = [
            "elegant woman in red evening dress, professional fashion photography",
            "casual street style outfit, modern urban fashion",
            "vintage 1960s fashion, retro style dress"
        ]
        
        results = []
        model, processor = self.load_clip_model()
        
        if model is None:
            print("❌ 無法載入 CLIP 模型，跳過圖片分析測試")
            return []
        
        # 生成參考圖片
        for i, prompt in enumerate(reference_prompts):
            print(f"生成參考圖片 {i+1}/{len(reference_prompts)}...")
            
            payload = {
                "prompt": prompt,
                "negative_prompt": "low quality, blurry, distorted",
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
                    image_path = os.path.join(self.output_dir, f"reference_{i+1}_{timestamp}.png")
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    
                    # 分析生成的圖片
                    print(f"分析圖片特徵...")
                    analysis = self.analyze_image_features(image_path, model, processor)
                    
                    # 根據分析結果生成新提示詞
                    generated_prompt = self.generate_prompt_from_analysis(analysis)
                    
                    # 用生成的提示詞再次生成圖片
                    print(f"使用分析結果生成新圖片...")
                    new_payload = {
                        "prompt": generated_prompt,
                        "negative_prompt": "low quality, blurry, distorted",
                        "width": 512,
                        "height": 768,
                        "steps": 25,
                        "cfg_scale": 7.5,
                        "sampler_name": "DPM++ 2M Karras"
                    }
                    
                    new_response = requests.post(f"{self.api_url}/sdapi/v1/txt2img", json=new_payload)
                    if new_response.status_code == 200:
                        new_result = new_response.json()
                        new_image_data = base64.b64decode(new_result['images'][0])
                        new_image_path = os.path.join(self.output_dir, f"generated_{i+1}_{timestamp}.png")
                        with open(new_image_path, "wb") as f:
                            f.write(new_image_data)
                        
                        results.append({
                            "original_prompt": prompt,
                            "reference_image": image_path,
                            "analysis": analysis,
                            "generated_prompt": generated_prompt,
                            "generated_image": new_image_path,
                            "success": True
                        })
                        print(f"✅ 完成測試 {i+1}")
                    else:
                        results.append({
                            "original_prompt": prompt,
                            "reference_image": image_path,
                            "analysis": analysis,
                            "generated_prompt": generated_prompt,
                            "error": f"New generation failed: {new_response.status_code}",
                            "success": False
                        })
                else:
                    results.append({
                        "original_prompt": prompt,
                        "error": f"Reference generation failed: {response.status_code}",
                        "success": False
                    })
            except Exception as e:
                results.append({
                    "original_prompt": prompt,
                    "error": str(e),
                    "success": False
                })
                print(f"❌ 測試 {i+1} 失敗: {e}")
        
        return results
    
    def run_day2_tests(self):
        """運行第2天的所有測試"""
        print("=" * 50)
        print("第2天測試開始：進階圖片分析和自動提示詞生成")
        print("=" * 50)
        
        # 檢查環境
        if not self.install_requirements():
            print("❌ 環境檢查失敗，請安裝必要套件")
            return False
        
        # 運行圖片分析測試
        results = self.test_image_to_prompt_generation()
        successful = sum(1 for r in results if r["success"])
        
        print(f"\\n第2天測試完成：{successful}/{len(results)} 個測試成功")
        
        # 生成報告
        report = {
            "day": 2,
            "timestamp": datetime.now().isoformat(),
            "tests_run": len(results),
            "tests_successful": successful,
            "success_rate": successful / len(results) if results else 0,
            "results": results
        }
        
        with open(os.path.join(self.output_dir, "day2_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\\n📊 第2天報告已保存至: {os.path.join(self.output_dir, 'day2_report.json')}")
        return True

if __name__ == "__main__":
    tester = Day2Tester()
    tester.run_day2_tests()
'''
    
    with open("day2_advanced_test.py", "w", encoding="utf-8") as f:
        f.write(day2_advanced_test)
    print("✅ 已創建 day2_advanced_test.py")

def create_day3_scripts():
    """創建第3天的評估腳本"""
    
    day3_evaluation = '''#!/usr/bin/env python3
"""
第3天：結果評估和可行性分析
目標：分析前兩天的測試結果，評估整體可行性
"""

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np

class Day3Evaluator:
    def __init__(self):
        self.results_dir = "day3_evaluation"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_test_results(self):
        """載入前兩天的測試結果"""
        results = {}
        
        # 載入第1天結果
        day1_file = "day1_results/day1_report.json"
        if os.path.exists(day1_file):
            with open(day1_file, "r", encoding="utf-8") as f:
                results["day1"] = json.load(f)
        else:
            print("⚠️ 未找到第1天測試結果")
            results["day1"] = None
        
        # 載入第2天結果
        day2_file = "day2_results/day2_report.json"
        if os.path.exists(day2_file):
            with open(day2_file, "r", encoding="utf-8") as f:
                results["day2"] = json.load(f)
        else:
            print("⚠️ 未找到第2天測試結果")
            results["day2"] = None
        
        return results
    
    def analyze_success_rates(self, results):
        """分析成功率"""
        analysis = {
            "day1_success_rate": 0,
            "day2_success_rate": 0,
            "overall_success_rate": 0
        }
        
        if results["day1"]:
            analysis["day1_success_rate"] = results["day1"].get("success_rate", 0)
        
        if results["day2"]:
            analysis["day2_success_rate"] = results["day2"].get("success_rate", 0)
        
        # 計算整體成功率
        rates = []
        if results["day1"]:
            rates.append(results["day1"].get("success_rate", 0))
        if results["day2"]:
            rates.append(results["day2"].get("success_rate", 0))
        
        analysis["overall_success_rate"] = sum(rates) / len(rates) if rates else 0
        
        return analysis
    
    def create_visual_report(self, results, analysis):
        """創建視覺化報告"""
        try:
            import matplotlib.pyplot as plt
            
            # 創建成功率圖表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 每日成功率
            days = []
            rates = []
            if results["day1"]:
                days.append("Day 1\\nBasic Tests")
                rates.append(analysis["day1_success_rate"] * 100)
            if results["day2"]:
                days.append("Day 2\\nAdvanced Tests")
                rates.append(analysis["day2_success_rate"] * 100)
            
            if days:
                ax1.bar(days, rates, color=['#4CAF50', '#2196F3'])
                ax1.set_ylabel('Success Rate (%)')
                ax1.set_title('Daily Test Success Rates')
                ax1.set_ylim(0, 100)
                
                # 添加數值標籤
                for i, rate in enumerate(rates):
                    ax1.text(i, rate + 2, f'{rate:.1f}%', ha='center', va='bottom')
            
            # 整體評估
            overall_rate = analysis["overall_success_rate"] * 100
            colors = ['#4CAF50' if overall_rate >= 80 else '#FF9800' if overall_rate >= 60 else '#F44336']
            ax2.pie([overall_rate, 100 - overall_rate], 
                   labels=[f'Success\\n{overall_rate:.1f}%', f'Issues\\n{100-overall_rate:.1f}%'],
                   colors=[colors[0], '#E0E0E0'],
                   startangle=90)
            ax2.set_title('Overall Feasibility Assessment')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "success_rate_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ 視覺化報告已生成")
            
        except ImportError:
            print("⚠️ matplotlib 未安裝，跳過視覺化報告生成")
        except Exception as e:
            print(f"⚠️ 視覺化報告生成失敗: {e}")
    
    def generate_feasibility_assessment(self, results, analysis):
        """生成可行性評估報告"""
        
        # 基於成功率判斷可行性
        overall_rate = analysis["overall_success_rate"]
        
        if overall_rate >= 0.8:
            feasibility = "HIGH"
            recommendation = "強烈建議繼續開發。系統表現優秀，可以進入生產階段的準備。"
            next_steps = [
                "準備更大規模的時尚圖片數據集",
                "實施 LoRA 或 DreamBooth 微調",
                "開發用戶界面",
                "進行性能優化"
            ]
        elif overall_rate >= 0.6:
            feasibility = "MEDIUM"
            recommendation = "可行性中等。建議先解決發現的問題，然後再繼續開發。"
            next_steps = [
                "分析失敗案例，改進提示詞生成",
                "調整 SD 參數設定",
                "擴充測試數據集",
                "考慮使用更先進的模型"
            ]
        else:
            feasibility = "LOW"
            recommendation = "當前可行性較低。建議重新評估技術方案或尋找替代方法。"
            next_steps = [
                "檢查 SD 模型是否適合時尚領域",
                "考慮使用專門的時尚生成模型",
                "重新設計提示詞策略",
                "評估硬體和環境需求"
            ]
        
        # 技術問題分析
        technical_issues = []
        if results["day1"] and results["day1"].get("success_rate", 0) < 0.8:
            technical_issues.append("基礎 API 生成存在穩定性問題")
        if results["day2"] and results["day2"].get("success_rate", 0) < 0.8:
            technical_issues.append("圖片分析和自動提示詞生成需要改進")
        
        assessment = {
            "feasibility_level": feasibility,
            "overall_success_rate": overall_rate,
            "recommendation": recommendation,
            "next_steps": next_steps,
            "technical_issues": technical_issues,
            "evaluation_date": datetime.now().isoformat()
        }
        
        return assessment
    
    def create_final_report(self, results, analysis, assessment):
        """創建最終報告"""
        
        report = {
            "evaluation_summary": {
                "test_period": "3 days",
                "total_tests_conducted": 0,
                "overall_success_rate": analysis["overall_success_rate"],
                "feasibility_assessment": assessment["feasibility_level"]
            },
            "day_by_day_results": {},
            "technical_analysis": analysis,
            "feasibility_assessment": assessment,
            "conclusions": {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": []
            }
        }
        
        # 統計總測試數
        total_tests = 0
        if results["day1"]:
            report["day_by_day_results"]["day1"] = results["day1"]
            total_tests += results["day1"].get("fashion_prompts_tested", 0)
        
        if results["day2"]:
            report["day_by_day_results"]["day2"] = results["day2"]
            total_tests += results["day2"].get("tests_run", 0)
        
        report["evaluation_summary"]["total_tests_conducted"] = total_tests
        
        # SWOT 分析
        if analysis["overall_success_rate"] >= 0.7:
            report["conclusions"]["strengths"].append("API 集成成功，基礎功能穩定")
        if results["day2"] and results["day2"].get("success_rate", 0) > 0.5:
            report["conclusions"]["strengths"].append("圖片分析和自動提示詞生成展現潛力")
        
        if analysis["overall_success_rate"] < 0.8:
            report["conclusions"]["weaknesses"].append("整體成功率有待提升")
        if not results["day1"] or not results["day2"]:
            report["conclusions"]["weaknesses"].append("測試數據不完整")
        
        report["conclusions"]["opportunities"] = [
            "時尚 AI 市場需求巨大",
            "Stable Diffusion 技術日趨成熟",
            "可與現有時尚平台集成"
        ]
        
        report["conclusions"]["threats"] = [
            "競爭對手可能先行進入市場",
            "技術變化快速",
            "版權和原創性問題"
        ]
        
        return report
    
    def run_evaluation(self):
        """運行第3天評估"""
        print("=" * 50)
        print("第3天評估開始：結果分析和可行性評估")
        print("=" * 50)
        
        # 載入測試結果
        results = self.load_test_results()
        
        # 分析成功率
        analysis = self.analyze_success_rates(results)
        
        # 生成可行性評估
        assessment = self.generate_feasibility_assessment(results, analysis)
        
        # 創建視覺化報告
        self.create_visual_report(results, analysis)
        
        # 創建最終報告
        final_report = self.create_final_report(results, analysis, assessment)
        
        # 保存最終報告
        with open(os.path.join(self.results_dir, "final_feasibility_report.json"), "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        # 輸出摘要
        print("\\n" + "=" * 50)
        print("📊 3天可行性測試結果摘要")
        print("=" * 50)
        print(f"整體成功率: {analysis['overall_success_rate']*100:.1f}%")
        print(f"可行性評估: {assessment['feasibility_level']}")
        print(f"建議: {assessment['recommendation']}")
        print("\\n下一步行動:")
        for i, step in enumerate(assessment['next_steps'], 1):
            print(f"  {i}. {step}")
        
        print(f"\\n📄 完整報告已保存至: {os.path.join(self.results_dir, 'final_feasibility_report.json')}")
        
        return final_report

if __name__ == "__main__":
    evaluator = Day3Evaluator()
    evaluator.run_evaluation()
'''
    
    with open("day3_evaluation.py", "w", encoding="utf-8") as f:
        f.write(day3_evaluation)
    print("✅ 已創建 day3_evaluation.py")

def create_quick_start_guide():
    """創建快速開始指南"""
    
    guide = '''# 3天 Stable Diffusion 時尚圖片生成可行性測試指南

## 🎯 測試目標
在3天內快速評估使用 Stable Diffusion WebUI API 進行時尚圖片生成的技術可行性。

## 📋 前置準備

### 1. 確保環境準備就緒
```bash
# 檢查 Python 環境
python --version

# 安裝必要套件
pip install requests pillow torch transformers matplotlib pandas
```

### 2. 啟動 Stable Diffusion WebUI
```bash
# Windows 用戶
webui-user.bat

# WebUI 啟動後，API 將在 http://localhost:7860 可用
```

## 📅 3天測試計畫

### 第1天：基礎功能測試
**目標**: 驗證 API 基本功能和時尚相關提示詞效果

**執行**:
```bash
python day1_basic_test.py
```

**預期結果**:
- API 連接正常
- 基本圖片生成成功
- 5個時尚提示詞測試完成
- 生成 `day1_results/` 文件夾和報告

### 第2天：進階功能測試
**目標**: 測試圖片分析和自動提示詞生成

**執行**:
```bash
python day2_advanced_test.py
```

**預期結果**:
- CLIP 模型載入成功
- 圖片特徵分析正常
- 自動提示詞生成測試完成
- 生成 `day2_results/` 文件夾和報告

### 第3天：結果評估
**目標**: 分析測試結果，評估整體可行性

**執行**:
```bash
python day3_evaluation.py
```

**預期結果**:
- 生成成功率分析圖表
- 完整的可行性評估報告
- 明確的下一步建議

## 📊 成功標準

### 高可行性 (80%+ 成功率)
- ✅ API 穩定運行
- ✅ 時尚提示詞效果良好
- ✅ 圖片分析準確
- ✅ 自動提示詞生成有效

### 中等可行性 (60-80% 成功率)
- ⚠️ 部分功能需要調整
- ⚠️ 可能需要優化參數
- ⚠️ 建議進一步測試

### 低可行性 (<60% 成功率)
- ❌ 需要重新評估技術方案
- ❌ 考慮替代解決方案

## 🔧 故障排除

### API 連接失敗
1. 確認 WebUI 已啟動
2. 檢查 `webui-user.bat` 中是否包含 `--api --listen`
3. 確認端口 7860 未被佔用

### 圖片生成失敗
1. 檢查顯卡記憶體是否足夠
2. 降低圖片解析度 (512x512)
3. 減少生成步數 (20 steps)

### CLIP 模型載入失敗
1. 確認網路連接正常
2. 手動下載模型：`transformers-cli download openai/clip-vit-base-patch32`

## 📁 輸出文件結構
```
day1_results/
├── day1_report.json          # 第1天測試報告
├── basic_test_*.png          # 基礎測試生成圖片
└── fashion_test_*.png        # 時尚提示詞測試圖片

day2_results/
├── day2_report.json          # 第2天測試報告
├── reference_*.png           # 參考圖片
└── generated_*.png           # 基於分析生成的圖片

day3_evaluation/
├── final_feasibility_report.json  # 最終可行性報告
└── success_rate_analysis.png      # 成功率分析圖表
```

## 🚀 後續發展方向

### 高可行性情況下
1. 擴大測試數據集
2. 實施模型微調 (LoRA/DreamBooth)
3. 開發用戶界面
4. 性能優化

### 中等可行性情況下
1. 問題診斷和修復
2. 參數調優
3. 替代方法評估

### 低可行性情況下
1. 技術方案重新評估
2. 尋找專業時尚生成模型
3. 考慮商業解決方案

## 📞 支援和協助
如果在測試過程中遇到問題，請：
1. 檢查生成的錯誤日誌
2. 確認環境配置正確
3. 參考故障排除部分
4. 記錄詳細的錯誤信息以便後續分析
'''
    
    with open("README_3DAY_TEST.md", "w", encoding="utf-8") as f:
        f.write(guide)
    print("✅ 已創建 README_3DAY_TEST.md")

def create_test_plan():
    """創建3天測試計畫"""
    print("🚀 開始創建3天 Stable Diffusion 時尚圖片生成可行性測試計畫...")
    
    create_day1_scripts()
    create_day2_scripts()
    create_day3_scripts()
    create_quick_start_guide()
    
    # 創建主測試計畫 JSON
    plan = {
        "project_name": "Stable Diffusion 時尚圖片生成可行性測試",
        "duration": "3 days",
        "created_date": datetime.now().isoformat(),
        "objectives": [
            "驗證 Stable Diffusion WebUI API 的穩定性",
            "測試時尚相關提示詞的生成效果",
            "評估圖片分析和自動提示詞生成的可行性",
            "提供明確的技術可行性評估和後續建議"
        ],
        "daily_plan": {
            "day1": {
                "title": "基礎功能測試",
                "script": "day1_basic_test.py",
                "objectives": [
                    "API 連接測試",
                    "基本圖片生成驗證",
                    "時尚提示詞效果測試"
                ],
                "expected_outputs": [
                    "day1_results/day1_report.json",
                    "基礎測試生成圖片",
                    "時尚提示詞測試圖片"
                ]
            },
            "day2": {
                "title": "進階功能測試",
                "script": "day2_advanced_test.py",
                "objectives": [
                    "CLIP 模型圖片分析",
                    "自動提示詞生成",
                    "完整工作流程測試"
                ],
                "expected_outputs": [
                    "day2_results/day2_report.json",
                    "圖片分析結果",
                    "自動生成的圖片"
                ]
            },
            "day3": {
                "title": "結果評估和可行性分析",
                "script": "day3_evaluation.py",
                "objectives": [
                    "測試結果統計分析",
                    "可行性評估",
                    "後續發展建議"
                ],
                "expected_outputs": [
                    "day3_evaluation/final_feasibility_report.json",
                    "成功率分析圖表",
                    "最終建議報告"
                ]
            }
        },
        "success_criteria": {
            "high_feasibility": ">= 80% success rate",
            "medium_feasibility": "60-80% success rate", 
            "low_feasibility": "< 60% success rate"
        },
        "required_packages": [
            "requests",
            "pillow",
            "torch",
            "transformers",
            "matplotlib",
            "pandas",
            "numpy"
        ]
    }
    
    with open("3day_test_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("🎉 3天測試計畫創建完成！")
    print("="*60)
    print("📁 已創建的文件:")
    print("  • day1_basic_test.py - 第1天基礎測試腳本")
    print("  • day2_advanced_test.py - 第2天進階測試腳本") 
    print("  • day3_evaluation.py - 第3天評估腳本")
    print("  • README_3DAY_TEST.md - 完整使用指南")
    print("  • 3day_test_plan.json - 測試計畫配置")
    print("\n🚀 開始測試:")
    print("  1. 確保 Stable Diffusion WebUI 已啟動")
    print("  2. 運行: python day1_basic_test.py")
    print("  3. 運行: python day2_advanced_test.py") 
    print("  4. 運行: python day3_evaluation.py")
    print("\n📖 詳細說明請參考: README_3DAY_TEST.md")

if __name__ == "__main__":
    create_test_plan()
