#!/usr/bin/env python3
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
                days.append("Day 1\nBasic Tests")
                rates.append(analysis["day1_success_rate"] * 100)
            if results["day2"]:
                days.append("Day 2\nAdvanced Tests")
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
                   labels=[f'Success\n{overall_rate:.1f}%', f'Issues\n{100-overall_rate:.1f}%'],
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
        print("\n" + "=" * 50)
        print("📊 3天可行性測試結果摘要")
        print("=" * 50)
        print(f"整體成功率: {analysis['overall_success_rate']*100:.1f}%")
        print(f"可行性評估: {assessment['feasibility_level']}")
        print(f"建議: {assessment['recommendation']}")
        print("\n下一步行動:")
        for i, step in enumerate(assessment['next_steps'], 1):
            print(f"  {i}. {step}")
        
        print(f"\n📄 完整報告已保存至: {os.path.join(self.results_dir, 'final_feasibility_report.json')}")
        
        return final_report

if __name__ == "__main__":
    evaluator = Day3Evaluator()
    evaluator.run_evaluation()
