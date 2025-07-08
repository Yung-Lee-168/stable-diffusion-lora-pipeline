#!/usr/bin/env python3
"""
LoRA 調優監控儀表板
即時監控 LoRA 調優過程，顯示性能指標變化趨勢

功能：
1. 即時監控訓練進度
2. 顯示性能指標趨勢
3. 自動預警和建議
4. 生成調優歷史報告
5. 支援多輪次比較
"""

import json
import os
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import argparse
from dataclasses import dataclass
import threading
import queue

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class PerformanceMetrics:
    """性能指標數據類"""
    timestamp: str
    iteration: int
    total_loss: float
    visual_similarity: float
    fashion_clip_similarity: float
    color_similarity: float
    overall_score: float
    training_efficiency: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'iteration': self.iteration,
            'total_loss': self.total_loss,
            'visual_similarity': self.visual_similarity,
            'fashion_clip_similarity': self.fashion_clip_similarity,
            'color_similarity': self.color_similarity,
            'overall_score': self.overall_score,
            'training_efficiency': self.training_efficiency
        }

class LoRATuningMonitor:
    """LoRA 調優監控器"""
    
    def __init__(self, monitor_dir: str = "test_results"):
        self.monitor_dir = monitor_dir
        self.metrics_history: List[PerformanceMetrics] = []
        self.alert_thresholds = {
            'total_loss_max': 0.8,
            'visual_similarity_min': 0.2,
            'fashion_clip_similarity_min': 0.3,
            'overall_score_min': 0.4
        }
        self.is_monitoring = False
        self.monitor_thread = None
        self.update_queue = queue.Queue()
        
    def load_latest_report(self) -> Optional[Dict[str, Any]]:
        """載入最新的分析報告"""
        try:
            if not os.path.exists(self.monitor_dir):
                return None
            
            json_files = [f for f in os.listdir(self.monitor_dir) 
                         if f.startswith("training_report_") and f.endswith(".json")]
            
            if not json_files:
                return None
            
            # 按時間排序，取最新的
            json_files.sort(reverse=True)
            latest_file = os.path.join(self.monitor_dir, json_files[0])
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 載入報告失敗：{e}")
            return None
    
    def extract_metrics(self, report: Dict[str, Any]) -> Optional[PerformanceMetrics]:
        """從報告中提取性能指標"""
        try:
            # 從 benchmark_analysis 提取
            benchmark = report.get("benchmark_analysis", {})
            avg_metrics = benchmark.get("average_metrics", {})
            
            # 從 lora_tuning 提取
            lora_tuning = report.get("lora_tuning", {})
            
            # 從 tuning_targets 提取迭代次數
            iteration = len(self.metrics_history) + 1
            
            metrics = PerformanceMetrics(
                timestamp=report.get("analysis_time", datetime.datetime.now().isoformat()),
                iteration=iteration,
                total_loss=avg_metrics.get("avg_total_loss", 1.0),
                visual_similarity=avg_metrics.get("avg_visual_similarity", 0.0),
                fashion_clip_similarity=avg_metrics.get("avg_fashion_clip_similarity", 0.0),
                color_similarity=avg_metrics.get("avg_color_similarity", 0.0),
                overall_score=lora_tuning.get("overall_tuning_score", 0.0),
                training_efficiency=lora_tuning.get("training_efficiency", {}).get("score", 0.0)
            )
            
            return metrics
        except Exception as e:
            print(f"❌ 提取指標失敗：{e}")
            return None
    
    def check_alerts(self, metrics: PerformanceMetrics) -> List[str]:
        """檢查預警條件"""
        alerts = []
        
        if metrics.total_loss > self.alert_thresholds['total_loss_max']:
            alerts.append(f"🚨 總損失過高：{metrics.total_loss:.3f} > {self.alert_thresholds['total_loss_max']}")
        
        if metrics.visual_similarity < self.alert_thresholds['visual_similarity_min']:
            alerts.append(f"⚠️ 視覺相似度過低：{metrics.visual_similarity:.3f} < {self.alert_thresholds['visual_similarity_min']}")
        
        if metrics.fashion_clip_similarity < self.alert_thresholds['fashion_clip_similarity_min']:
            alerts.append(f"⚠️ FashionCLIP 相似度過低：{metrics.fashion_clip_similarity:.3f} < {self.alert_thresholds['fashion_clip_similarity_min']}")
        
        if metrics.overall_score < self.alert_thresholds['overall_score_min']:
            alerts.append(f"❌ 整體分數過低：{metrics.overall_score:.3f} < {self.alert_thresholds['overall_score_min']}")
        
        return alerts
    
    def generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """生成調優建議"""
        recommendations = []
        
        # 基於當前指標生成建議
        if metrics.total_loss > 0.7:
            recommendations.append("🔧 建議大幅降低學習率（0.0002-0.0003）")
            recommendations.append("📊 建議增加訓練步數（200-300步）")
        elif metrics.total_loss > 0.5:
            recommendations.append("🔧 建議適度降低學習率（0.0003-0.0005）")
        
        if metrics.visual_similarity < 0.3:
            recommendations.append("👁️ 建議提高視覺損失權重（0.3-0.4）")
            recommendations.append("📸 建議提高訓練解析度（768x768）")
        
        if metrics.fashion_clip_similarity < 0.4:
            recommendations.append("🎨 建議提高 FashionCLIP 權重（0.7-0.8）")
            recommendations.append("📝 建議優化訓練數據標籤")
        
        if metrics.training_efficiency < 0.5:
            recommendations.append("🚀 建議使用更高效的訓練策略")
            recommendations.append("💾 建議啟用快取機制")
        
        # 基於趨勢生成建議
        if len(self.metrics_history) >= 2:
            prev_metrics = self.metrics_history[-1]
            
            # 檢查是否有改善
            if metrics.total_loss > prev_metrics.total_loss:
                recommendations.append("📈 損失增加：建議檢查學習率設定")
            
            if metrics.overall_score < prev_metrics.overall_score:
                recommendations.append("📉 整體分數下降：建議回到上一輪參數")
        
        return recommendations
    
    def update_metrics(self, metrics: PerformanceMetrics):
        """更新指標歷史"""
        self.metrics_history.append(metrics)
        
        # 檢查預警
        alerts = self.check_alerts(metrics)
        if alerts:
            print(f"\n⚠️ 第 {metrics.iteration} 輪預警：")
            for alert in alerts:
                print(f"   {alert}")
        
        # 生成建議
        recommendations = self.generate_recommendations(metrics)
        if recommendations:
            print(f"\n💡 第 {metrics.iteration} 輪建議：")
            for rec in recommendations:
                print(f"   {rec}")
        
        print(f"\n📊 第 {metrics.iteration} 輪性能：")
        print(f"   總損失：{metrics.total_loss:.3f}")
        print(f"   視覺相似度：{metrics.visual_similarity:.3f}")
        print(f"   FashionCLIP 相似度：{metrics.fashion_clip_similarity:.3f}")
        print(f"   整體分數：{metrics.overall_score:.3f}")
    
    def generate_dashboard(self, save_path: str = "tuning_dashboard.png"):
        """生成監控儀表板"""
        if not self.metrics_history:
            print("❌ 沒有可視化的數據")
            return
        
        # 準備數據
        iterations = [m.iteration for m in self.metrics_history]
        total_losses = [m.total_loss for m in self.metrics_history]
        visual_sims = [m.visual_similarity for m in self.metrics_history]
        fashion_sims = [m.fashion_clip_similarity for m in self.metrics_history]
        overall_scores = [m.overall_score for m in self.metrics_history]
        
        # 創建圖表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LoRA 調優監控儀表板', fontsize=16, fontweight='bold')
        
        # 1. 總損失趨勢
        ax1.plot(iterations, total_losses, 'r-o', linewidth=2, markersize=6)
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='良好閾值')
        ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='優秀閾值')
        ax1.set_title('總損失趨勢')
        ax1.set_xlabel('迭代次數')
        ax1.set_ylabel('總損失')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 視覺相似度趨勢
        ax2.plot(iterations, visual_sims, 'b-o', linewidth=2, markersize=6)
        ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='一般閾值')
        ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='良好閾值')
        ax2.set_title('視覺相似度趨勢')
        ax2.set_xlabel('迭代次數')
        ax2.set_ylabel('視覺相似度')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. FashionCLIP 相似度趨勢
        ax3.plot(iterations, fashion_sims, 'g-o', linewidth=2, markersize=6)
        ax3.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='一般閾值')
        ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='良好閾值')
        ax3.set_title('FashionCLIP 相似度趨勢')
        ax3.set_xlabel('迭代次數')
        ax3.set_ylabel('FashionCLIP 相似度')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 整體分數趨勢
        ax4.plot(iterations, overall_scores, 'purple', marker='o', linewidth=2, markersize=6)
        ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='一般閾值')
        ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='良好閾值')
        ax4.set_title('整體分數趨勢')
        ax4.set_xlabel('迭代次數')
        ax4.set_ylabel('整體分數')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 儀表板已保存：{save_path}")
        return save_path
    
    def generate_comparison_report(self) -> str:
        """生成比較報告"""
        if not self.metrics_history:
            return "沒有可比較的數據"
        
        report = f"""
# LoRA 調優監控報告
生成時間：{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 調優歷史總覽
總迭代次數：{len(self.metrics_history)}

## 性能指標比較
"""
        
        # 找出最佳和最差的迭代
        best_overall = max(self.metrics_history, key=lambda x: x.overall_score)
        worst_overall = min(self.metrics_history, key=lambda x: x.overall_score)
        
        best_loss = min(self.metrics_history, key=lambda x: x.total_loss)
        best_visual = max(self.metrics_history, key=lambda x: x.visual_similarity)
        best_fashion = max(self.metrics_history, key=lambda x: x.fashion_clip_similarity)
        
        report += f"""
### 最佳表現
- 最佳整體分數：第 {best_overall.iteration} 輪（{best_overall.overall_score:.3f}）
- 最低總損失：第 {best_loss.iteration} 輪（{best_loss.total_loss:.3f}）
- 最高視覺相似度：第 {best_visual.iteration} 輪（{best_visual.visual_similarity:.3f}）
- 最高 FashionCLIP 相似度：第 {best_fashion.iteration} 輪（{best_fashion.fashion_clip_similarity:.3f}）

### 最差表現
- 最差整體分數：第 {worst_overall.iteration} 輪（{worst_overall.overall_score:.3f}）

### 改善幅度
"""
        
        if len(self.metrics_history) >= 2:
            first = self.metrics_history[0]
            last = self.metrics_history[-1]
            
            loss_improvement = (first.total_loss - last.total_loss) / first.total_loss * 100
            visual_improvement = (last.visual_similarity - first.visual_similarity) / max(first.visual_similarity, 0.001) * 100
            fashion_improvement = (last.fashion_clip_similarity - first.fashion_clip_similarity) / max(first.fashion_clip_similarity, 0.001) * 100
            overall_improvement = (last.overall_score - first.overall_score) / max(first.overall_score, 0.001) * 100
            
            report += f"""
- 總損失改善：{loss_improvement:+.1f}%
- 視覺相似度改善：{visual_improvement:+.1f}%
- FashionCLIP 相似度改善：{fashion_improvement:+.1f}%
- 整體分數改善：{overall_improvement:+.1f}%
"""
        
        report += f"""
## 詳細數據

| 迭代 | 總損失 | 視覺相似度 | FashionCLIP | 整體分數 | 時間 |
|------|--------|------------|-------------|----------|------|
"""
        
        for metrics in self.metrics_history:
            report += f"| {metrics.iteration} | {metrics.total_loss:.3f} | {metrics.visual_similarity:.3f} | {metrics.fashion_clip_similarity:.3f} | {metrics.overall_score:.3f} | {metrics.timestamp[:19]} |\n"
        
        return report
    
    def start_monitoring(self, interval: int = 30):
        """開始監控"""
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    # 載入最新報告
                    report = self.load_latest_report()
                    if report:
                        metrics = self.extract_metrics(report)
                        if metrics:
                            # 檢查是否是新的迭代
                            if not self.metrics_history or metrics.timestamp != self.metrics_history[-1].timestamp:
                                self.update_metrics(metrics)
                                self.generate_dashboard()
                    
                    time.sleep(interval)
                except Exception as e:
                    print(f"❌ 監控錯誤：{e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print(f"🔍 開始監控 LoRA 調優，間隔 {interval} 秒")
    
    def stop_monitoring(self):
        """停止監控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("⏹️ 監控已停止")
    
    def save_history(self, filename: str = None):
        """保存監控歷史"""
        if not filename:
            filename = f"tuning_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        history_data = {
            "metadata": {
                "total_iterations": len(self.metrics_history),
                "monitoring_start": self.metrics_history[0].timestamp if self.metrics_history else None,
                "monitoring_end": self.metrics_history[-1].timestamp if self.metrics_history else None
            },
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "alert_thresholds": self.alert_thresholds
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 監控歷史已保存：{filename}")
        return filename

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="LoRA 調優監控工具")
    parser.add_argument("--monitor_dir", type=str, default="test_results", help="監控目錄")
    parser.add_argument("--interval", type=int, default=30, help="監控間隔（秒）")
    parser.add_argument("--mode", type=str, choices=["monitor", "report", "dashboard"], default="monitor", help="運行模式")
    parser.add_argument("--output", type=str, default="tuning_dashboard.png", help="輸出文件名")
    
    args = parser.parse_args()
    
    # 創建監控器
    monitor = LoRATuningMonitor(args.monitor_dir)
    
    if args.mode == "monitor":
        # 即時監控模式
        try:
            monitor.start_monitoring(args.interval)
            print("按 Ctrl+C 停止監控")
            
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 收到停止信號")
            monitor.stop_monitoring()
            
            # 保存歷史
            history_file = monitor.save_history()
            
            # 生成最終報告
            report = monitor.generate_comparison_report()
            report_file = f"tuning_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"📋 最終報告已保存：{report_file}")
    
    elif args.mode == "report":
        # 報告模式
        report = monitor.load_latest_report()
        if report:
            metrics = monitor.extract_metrics(report)
            if metrics:
                monitor.update_metrics(metrics)
                
                comparison_report = monitor.generate_comparison_report()
                print(comparison_report)
                
                # 保存報告
                report_file = f"tuning_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(comparison_report)
                
                print(f"📋 報告已保存：{report_file}")
        else:
            print("❌ 找不到分析報告")
    
    elif args.mode == "dashboard":
        # 儀表板模式
        # 載入所有歷史報告
        if os.path.exists(args.monitor_dir):
            json_files = [f for f in os.listdir(args.monitor_dir) 
                         if f.startswith("training_report_") and f.endswith(".json")]
            json_files.sort()
            
            for json_file in json_files:
                file_path = os.path.join(args.monitor_dir, json_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                metrics = monitor.extract_metrics(report)
                if metrics:
                    monitor.update_metrics(metrics)
            
            # 生成儀表板
            dashboard_file = monitor.generate_dashboard(args.output)
            print(f"📊 儀表板已生成：{dashboard_file}")
        else:
            print("❌ 找不到監控目錄")

if __name__ == "__main__":
    main()
