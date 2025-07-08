#!/usr/bin/env python3
"""
快速技術指標查看器
快速查看最新的 LoRA 調教技術指標，支援即時監控和歷史比較

使用方式：
python quick_metrics_viewer.py [選項]

選項：
--latest        查看最新指標
--history       查看歷史趨勢
--compare       比較多輪結果
--dashboard     生成儀表板
--monitor       開始即時監控
"""

import json
import os
import sys
import argparse
import datetime
from typing import Dict, List, Any, Optional
import glob

class QuickMetricsViewer:
    """快速技術指標查看器"""
    
    def __init__(self, results_dir: str = "test_results"):
        self.results_dir = results_dir
        
    def get_latest_report(self) -> Optional[str]:
        """獲取最新報告路徑"""
        if not os.path.exists(self.results_dir):
            return None
        
        json_files = glob.glob(os.path.join(self.results_dir, "training_report_*.json"))
        if not json_files:
            return None
        
        # 按時間排序，取最新的
        json_files.sort(key=os.path.getmtime, reverse=True)
        return json_files[0]
    
    def load_report(self, report_path: str) -> Optional[Dict[str, Any]]:
        """載入報告"""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 載入報告失敗：{e}")
            return None
    
    def display_latest_metrics(self):
        """顯示最新技術指標"""
        latest_report = self.get_latest_report()
        if not latest_report:
            print("❌ 沒有找到分析報告")
            return
        
        report = self.load_report(latest_report)
        if not report:
            return
        
        print("=" * 60)
        print("🎯 最新 LoRA 調教技術指標")
        print("=" * 60)
        
        # 基本資訊
        print(f"📅 分析時間：{report.get('analysis_time', 'N/A')}")
        print(f"📁 報告檔案：{os.path.basename(latest_report)}")
        print()
        
        # 三基準點評估
        if "benchmark_analysis" in report:
            self._display_benchmark_metrics(report["benchmark_analysis"])
        
        # LoRA 調優指標
        if "lora_tuning" in report:
            self._display_lora_tuning_metrics(report["lora_tuning"])
        
        # 調優目標
        if "tuning_targets" in report:
            self._display_tuning_targets(report["tuning_targets"])
    
    def _display_benchmark_metrics(self, benchmark: Dict[str, Any]):
        """顯示基準指標"""
        print("📊 三基準點性能評估")
        print("-" * 40)
        
        avg_metrics = benchmark.get("average_metrics", {})
        benchmark_comparison = benchmark.get("benchmark_comparison", {})
        
        # 顯示平均指標
        print(f"總損失：{avg_metrics.get('avg_total_loss', 'N/A'):.3f}")
        print(f"視覺相似度：{avg_metrics.get('avg_visual_similarity', 'N/A'):.3f}")
        print(f"FashionCLIP 相似度：{avg_metrics.get('avg_fashion_clip_similarity', 'N/A'):.3f}")
        print(f"色彩相似度：{avg_metrics.get('avg_color_similarity', 'N/A'):.3f}")
        print()
        
        # 顯示與參考值比較
        print("📈 與參考值比較：")
        for key, value in benchmark_comparison.items():
            if isinstance(value, (int, float)):
                status = "📈" if value > 0 else "📉" if value < 0 else "➡️"
                print(f"  {key}: {status} {value:+.3f}")
        print()
        
        # 顯示性能分布
        if "performance_distribution" in benchmark:
            dist = benchmark["performance_distribution"]
            total = sum(dist.values())
            print("🏆 性能分布：")
            for level, count in dist.items():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"  {level.capitalize()}: {count} ({percentage:.1f}%)")
        print()
    
    def _display_lora_tuning_metrics(self, lora_tuning: Dict[str, Any]):
        """顯示 LoRA 調優指標"""
        print("🎯 LoRA 調優專業指標")
        print("-" * 40)
        
        # 訓練效率
        if "training_efficiency" in lora_tuning:
            eff = lora_tuning["training_efficiency"]
            print(f"訓練效率：{eff.get('grade', 'N/A').upper()} (分數: {eff.get('score', 'N/A'):.3f})")
            print(f"  效率比率：{eff.get('efficiency_ratio', 'N/A'):.3f}")
        
        # 生成品質
        if "generation_quality" in lora_tuning:
            qual = lora_tuning["generation_quality"]
            print(f"生成品質：{qual.get('grade', 'N/A').upper()} (分數: {qual.get('score', 'N/A'):.3f})")
            print(f"  平均 SSIM：{qual.get('average_ssim', 'N/A'):.3f}")
        
        # 特徵保持
        if "feature_preservation" in lora_tuning:
            feat = lora_tuning["feature_preservation"]
            print(f"特徵保持：{feat.get('grade', 'N/A').upper()} (分數: {feat.get('score', 'N/A'):.3f})")
            print(f"  FashionCLIP 相似度：{feat.get('fashion_clip_similarity', 'N/A'):.3f}")
        
        # 整體分數
        overall_score = lora_tuning.get("overall_tuning_score", 0)
        if overall_score >= 0.9:
            grade = "EXCELLENT ✨"
        elif overall_score >= 0.7:
            grade = "GOOD ✅"
        elif overall_score >= 0.5:
            grade = "AVERAGE ⚠️"
        else:
            grade = "POOR ❌"
        
        print(f"整體調優分數：{overall_score:.3f} ({grade})")
        print()
    
    def _display_tuning_targets(self, tuning_targets: Dict[str, Any]):
        """顯示調優目標"""
        print("🎯 下一輪調優目標")
        print("-" * 40)
        
        # 當前表現
        if "current_performance" in tuning_targets:
            current = tuning_targets["current_performance"]
            print("📊 當前表現：")
            for key, value in current.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.3f}")
        
        # 目標指標
        if "target_metrics" in tuning_targets:
            targets = tuning_targets["target_metrics"]
            print("🎯 目標指標：")
            for key, value in targets.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.3f}")
        
        # 參數建議
        if "parameter_suggestions" in tuning_targets:
            params = tuning_targets["parameter_suggestions"]
            print("⚙️ 參數建議：")
            for key, value in params.items():
                print(f"  {key}: {value}")
        print()
    
    def display_history(self, limit: int = 5):
        """顯示歷史趨勢"""
        if not os.path.exists(self.results_dir):
            print("❌ 沒有找到結果目錄")
            return
        
        json_files = glob.glob(os.path.join(self.results_dir, "training_report_*.json"))
        if not json_files:
            print("❌ 沒有找到歷史報告")
            return
        
        # 按時間排序
        json_files.sort(key=os.path.getmtime, reverse=True)
        json_files = json_files[:limit]
        
        print("=" * 60)
        print(f"📈 最近 {min(len(json_files), limit)} 次調教歷史")
        print("=" * 60)
        
        for i, file_path in enumerate(json_files):
            report = self.load_report(file_path)
            if not report:
                continue
            
            print(f"\n第 {i+1} 次調教：{os.path.basename(file_path)}")
            print(f"時間：{report.get('analysis_time', 'N/A')}")
            
            # 關鍵指標
            if "benchmark_analysis" in report:
                avg_metrics = report["benchmark_analysis"].get("average_metrics", {})
                print(f"總損失：{avg_metrics.get('avg_total_loss', 'N/A'):.3f}")
                print(f"視覺相似度：{avg_metrics.get('avg_visual_similarity', 'N/A'):.3f}")
                print(f"FashionCLIP 相似度：{avg_metrics.get('avg_fashion_clip_similarity', 'N/A'):.3f}")
            
            if "lora_tuning" in report:
                overall_score = report["lora_tuning"].get("overall_tuning_score", 0)
                print(f"整體分數：{overall_score:.3f}")
            
            print("-" * 30)
    
    def compare_latest_reports(self, count: int = 3):
        """比較最新幾次的報告"""
        if not os.path.exists(self.results_dir):
            print("❌ 沒有找到結果目錄")
            return
        
        json_files = glob.glob(os.path.join(self.results_dir, "training_report_*.json"))
        if len(json_files) < 2:
            print("❌ 需要至少 2 個報告才能比較")
            return
        
        # 按時間排序
        json_files.sort(key=os.path.getmtime, reverse=True)
        json_files = json_files[:count]
        
        print("=" * 60)
        print(f"🔍 最近 {min(len(json_files), count)} 次調教比較")
        print("=" * 60)
        
        # 載入報告
        reports = []
        for file_path in json_files:
            report = self.load_report(file_path)
            if report:
                reports.append({
                    "file": os.path.basename(file_path),
                    "data": report
                })
        
        if len(reports) < 2:
            print("❌ 無法載入足夠的報告進行比較")
            return
        
        # 比較關鍵指標
        print("📊 關鍵指標比較：")
        print("-" * 40)
        
        metrics_to_compare = [
            ("總損失", "benchmark_analysis.average_metrics.avg_total_loss"),
            ("視覺相似度", "benchmark_analysis.average_metrics.avg_visual_similarity"),
            ("FashionCLIP 相似度", "benchmark_analysis.average_metrics.avg_fashion_clip_similarity"),
            ("整體分數", "lora_tuning.overall_tuning_score")
        ]
        
        for metric_name, metric_path in metrics_to_compare:
            print(f"\n{metric_name}：")
            values = []
            for report in reports:
                value = self._get_nested_value(report["data"], metric_path)
                if value is not None:
                    values.append(value)
                    print(f"  {report['file']}: {value:.3f}")
            
            # 顯示改善情況
            if len(values) >= 2:
                if metric_name == "總損失":
                    change = values[0] - values[1]  # 損失降低是好的
                    status = "改善 📈" if change < 0 else "惡化 📉" if change > 0 else "持平 ➡️"
                else:
                    change = values[0] - values[1]  # 其他指標提高是好的
                    status = "改善 📈" if change > 0 else "惡化 📉" if change < 0 else "持平 ➡️"
                
                print(f"  變化：{change:+.3f} ({status})")
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Optional[float]:
        """獲取嵌套字典的值"""
        keys = path.split(".")
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current if isinstance(current, (int, float)) else None
    
    def start_monitoring(self):
        """開始即時監控"""
        print("🔍 開始即時監控...")
        print("按 Ctrl+C 停止監控")
        
        try:
            import time
            from lora_tuning_monitor import LoRATuningMonitor
            
            monitor = LoRATuningMonitor(self.results_dir)
            
            while True:
                # 檢查是否有新報告
                report = monitor.load_latest_report()
                if report:
                    metrics = monitor.extract_metrics(report)
                    if metrics:
                        monitor.update_metrics(metrics)
                        
                        # 生成儀表板
                        dashboard_path = os.path.join(self.results_dir, 
                                                    f"realtime_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                        monitor.generate_dashboard(dashboard_path)
                
                time.sleep(30)  # 每30秒檢查一次
                
        except KeyboardInterrupt:
            print("\n⏹️ 監控已停止")
        except ImportError:
            print("❌ 監控功能需要 lora_tuning_monitor 模組")

def main():
    parser = argparse.ArgumentParser(description="快速技術指標查看器")
    parser.add_argument("--latest", action="store_true", help="查看最新指標")
    parser.add_argument("--history", action="store_true", help="查看歷史趨勢")
    parser.add_argument("--compare", action="store_true", help="比較多輪結果")
    parser.add_argument("--monitor", action="store_true", help="開始即時監控")
    parser.add_argument("--results-dir", default="test_results", help="結果目錄路徑")
    parser.add_argument("--limit", type=int, default=5, help="歷史記錄限制數量")
    
    args = parser.parse_args()
    
    viewer = QuickMetricsViewer(args.results_dir)
    
    if args.latest:
        viewer.display_latest_metrics()
    elif args.history:
        viewer.display_history(args.limit)
    elif args.compare:
        viewer.compare_latest_reports(args.limit)
    elif args.monitor:
        viewer.start_monitoring()
    else:
        # 預設顯示最新指標
        viewer.display_latest_metrics()

if __name__ == "__main__":
    main()
