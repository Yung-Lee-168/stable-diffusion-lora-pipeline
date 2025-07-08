#!/usr/bin/env python3
"""
LoRA 訓練進度監控器
在 train_network.py 訓練過程中即時監控和記錄技術指標

功能：
1. 監控訓練損失和學習率
2. 記錄中間檢查點的性能指標
3. 提供早停機制建議
4. 生成訓練進度報告
5. 決定是否繼續到推理階段
"""

import os
import json
import time
import datetime
import subprocess
import sys
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import threading
import queue
import logging

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class TrainingProgress:
    """訓練進度數據類"""
    step: int
    epoch: int
    loss: float
    lr: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'epoch': self.epoch,
            'loss': self.loss,
            'lr': self.lr,
            'timestamp': self.timestamp
        }

class LoRATrainingMonitor:
    """LoRA 訓練監控器"""
    
    def __init__(self, 
                 training_dir: str = ".",
                 log_dir: str = "training_logs",
                 checkpoint_interval: int = 100,
                 early_stop_patience: int = 500):
        self.training_dir = training_dir
        self.log_dir = log_dir
        self.checkpoint_interval = checkpoint_interval
        self.early_stop_patience = early_stop_patience
        
        # 建立日誌目錄
        os.makedirs(log_dir, exist_ok=True)
        
        # 訓練數據記錄
        self.training_history: List[TrainingProgress] = []
        self.loss_history: List[float] = []
        self.lr_history: List[float] = []
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        
        # 監控閾值
        self.thresholds = {
            'loss_spike_threshold': 2.0,  # 損失突然增加的閾值
            'loss_plateau_threshold': 0.001,  # 損失平穩的閾值
            'min_loss_improvement': 0.01,  # 最小損失改善
            'target_loss': 0.1,  # 目標損失值
            'max_lr_decay': 0.1,  # 最大學習率衰減
        }
        
        # 設定日誌
        self.setup_logging()
        
    def setup_logging(self):
        """設定日誌系統"""
        log_file = os.path.join(self.log_dir, f"training_monitor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def parse_training_log(self, log_line: str) -> Optional[TrainingProgress]:
        """解析訓練日誌行"""
        try:
            # 解析類似這樣的日誌行：
            # "epoch 1, step 100, loss: 0.1234, lr: 0.0001"
            if "loss:" in log_line and "lr:" in log_line:
                parts = log_line.split()
                
                # 提取 epoch
                epoch = 0
                if "epoch" in log_line:
                    epoch_idx = parts.index("epoch") + 1
                    if epoch_idx < len(parts):
                        epoch = int(parts[epoch_idx].rstrip(','))
                
                # 提取 step
                step = 0
                if "step" in log_line:
                    step_idx = parts.index("step") + 1
                    if step_idx < len(parts):
                        step = int(parts[step_idx].rstrip(','))
                
                # 提取 loss
                loss = 0.0
                if "loss:" in log_line:
                    loss_idx = parts.index("loss:") + 1
                    if loss_idx < len(parts):
                        loss = float(parts[loss_idx].rstrip(','))
                
                # 提取 lr
                lr = 0.0
                if "lr:" in log_line:
                    lr_idx = parts.index("lr:") + 1
                    if lr_idx < len(parts):
                        lr = float(parts[lr_idx].rstrip(','))
                
                return TrainingProgress(
                    step=step,
                    epoch=epoch,
                    loss=loss,
                    lr=lr,
                    timestamp=datetime.datetime.now().isoformat()
                )
                
        except Exception as e:
            self.logger.warning(f"解析日誌行失敗: {log_line}, 錯誤: {e}")
            
        return None
        
    def update_training_progress(self, progress: TrainingProgress):
        """更新訓練進度"""
        self.training_history.append(progress)
        self.loss_history.append(progress.loss)
        self.lr_history.append(progress.lr)
        
        # 檢查是否有改善
        if progress.loss < self.best_loss:
            improvement = self.best_loss - progress.loss
            if improvement >= self.thresholds['min_loss_improvement']:
                self.best_loss = progress.loss
                self.steps_without_improvement = 0
                self.logger.info(f"🎯 損失改善: {improvement:.4f}, 新最佳: {self.best_loss:.4f}")
            else:
                self.steps_without_improvement += 1
        else:
            self.steps_without_improvement += 1
            
        # 記錄進度
        self.logger.info(f"📊 Step {progress.step}: Loss={progress.loss:.4f}, LR={progress.lr:.6f}")
        
        # 檢查預警條件
        self.check_training_alerts(progress)
        
        # 定期保存進度
        if progress.step % self.checkpoint_interval == 0:
            self.save_progress_checkpoint(progress.step)
            
    def check_training_alerts(self, progress: TrainingProgress):
        """檢查訓練預警"""
        alerts = []
        
        # 檢查損失突然增加
        if len(self.loss_history) >= 2:
            current_loss = progress.loss
            prev_loss = self.loss_history[-2]
            
            if current_loss > prev_loss * self.thresholds['loss_spike_threshold']:
                alerts.append(f"⚠️ 損失突然增加: {prev_loss:.4f} → {current_loss:.4f}")
                
        # 檢查學習率異常
        if progress.lr < self.thresholds['max_lr_decay']:
            alerts.append(f"⚠️ 學習率過低: {progress.lr:.6f}")
            
        # 檢查早停條件
        if self.steps_without_improvement >= self.early_stop_patience:
            alerts.append(f"🛑 建議早停: {self.steps_without_improvement} 步無改善")
            
        # 檢查目標達成
        if progress.loss <= self.thresholds['target_loss']:
            alerts.append(f"🎯 達到目標損失: {progress.loss:.4f}")
            
        # 輸出預警
        for alert in alerts:
            self.logger.warning(alert)
            
    def save_progress_checkpoint(self, step: int):
        """保存進度檢查點"""
        checkpoint_data = {
            'step': step,
            'timestamp': datetime.datetime.now().isoformat(),
            'training_history': [p.to_dict() for p in self.training_history],
            'best_loss': self.best_loss,
            'steps_without_improvement': self.steps_without_improvement,
            'thresholds': self.thresholds
        }
        
        checkpoint_file = os.path.join(self.log_dir, f"training_checkpoint_{step}.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"💾 保存檢查點: {checkpoint_file}")
        
    def generate_training_report(self) -> Dict[str, Any]:
        """生成訓練報告"""
        if not self.training_history:
            return {}
            
        report = {
            'training_summary': {
                'total_steps': len(self.training_history),
                'start_time': self.training_history[0].timestamp,
                'end_time': self.training_history[-1].timestamp,
                'best_loss': self.best_loss,
                'final_loss': self.training_history[-1].loss,
                'final_lr': self.training_history[-1].lr
            },
            'training_metrics': {
                'loss_improvement': self.training_history[0].loss - self.best_loss,
                'loss_reduction_rate': (self.training_history[0].loss - self.best_loss) / self.training_history[0].loss * 100,
                'average_loss': sum(self.loss_history) / len(self.loss_history),
                'loss_variance': np.var(self.loss_history),
                'steps_without_improvement': self.steps_without_improvement
            },
            'training_evaluation': self.evaluate_training_performance(),
            'recommendations': self.generate_training_recommendations()
        }
        
        return report
        
    def evaluate_training_performance(self) -> Dict[str, Any]:
        """評估訓練表現"""
        if not self.training_history:
            return {}
            
        # 計算訓練效率
        loss_improvement = self.training_history[0].loss - self.best_loss
        steps_taken = len(self.training_history)
        efficiency = loss_improvement / steps_taken if steps_taken > 0 else 0
        
        # 計算收斂性
        recent_losses = self.loss_history[-10:] if len(self.loss_history) >= 10 else self.loss_history
        loss_stability = 1.0 - (np.std(recent_losses) / np.mean(recent_losses)) if recent_losses else 0
        
        # 評估等級
        performance_grade = "poor"
        if self.best_loss <= self.thresholds['target_loss']:
            performance_grade = "excellent"
        elif loss_improvement >= 0.1:
            performance_grade = "good"
        elif loss_improvement >= 0.05:
            performance_grade = "average"
            
        return {
            'efficiency': efficiency,
            'loss_stability': loss_stability,
            'performance_grade': performance_grade,
            'convergence_rate': self.calculate_convergence_rate(),
            'training_effectiveness': min(efficiency * 100, 100)
        }
        
    def calculate_convergence_rate(self) -> float:
        """計算收斂率"""
        if len(self.loss_history) < 10:
            return 0.0
            
        # 計算最近 10 步的平均改善率
        recent_losses = self.loss_history[-10:]
        improvements = []
        
        for i in range(1, len(recent_losses)):
            if recent_losses[i-1] > 0:
                improvement = (recent_losses[i-1] - recent_losses[i]) / recent_losses[i-1]
                improvements.append(improvement)
                
        return np.mean(improvements) if improvements else 0.0
        
    def generate_training_recommendations(self) -> List[str]:
        """生成訓練建議"""
        recommendations = []
        
        if not self.training_history:
            return recommendations
            
        # 基於損失改善的建議
        loss_improvement = self.training_history[0].loss - self.best_loss
        if loss_improvement < 0.01:
            recommendations.append("🔧 損失改善不足，建議調整學習率或增加訓練步數")
            
        # 基於收斂性的建議
        convergence_rate = self.calculate_convergence_rate()
        if convergence_rate < 0.001:
            recommendations.append("📈 收斂緩慢，建議檢查數據質量或調整優化器")
            
        # 基於早停的建議
        if self.steps_without_improvement >= self.early_stop_patience // 2:
            recommendations.append("⏹️ 可能需要早停，或嘗試降低學習率")
            
        # 基於目標達成的建議
        if self.best_loss <= self.thresholds['target_loss']:
            recommendations.append("✅ 已達到目標損失，可以進行推理測試")
        elif self.best_loss <= self.thresholds['target_loss'] * 2:
            recommendations.append("🎯 接近目標損失，建議繼續微調")
            
        return recommendations
        
    def should_continue_to_inference(self) -> bool:
        """判斷是否應該繼續到推理階段"""
        if not self.training_history:
            return False
            
        # 檢查是否達到目標損失
        if self.best_loss <= self.thresholds['target_loss']:
            return True
            
        # 檢查是否有足夠的改善
        if len(self.training_history) >= 2:
            loss_improvement = self.training_history[0].loss - self.best_loss
            if loss_improvement >= 0.05:  # 至少 5% 的改善
                return True
                
        # 檢查是否已經穩定
        if self.steps_without_improvement >= self.early_stop_patience:
            return True
            
        return False
        
    def generate_training_charts(self, output_path: str = "training_progress.png"):
        """生成訓練圖表"""
        if not self.training_history:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LoRA 訓練進度監控', fontsize=16, fontweight='bold')
        
        steps = [p.step for p in self.training_history]
        
        # 1. 損失曲線
        ax1.plot(steps, self.loss_history, 'b-', linewidth=2, label='Training Loss')
        ax1.axhline(y=self.best_loss, color='r', linestyle='--', alpha=0.7, label=f'Best Loss: {self.best_loss:.4f}')
        ax1.axhline(y=self.thresholds['target_loss'], color='g', linestyle='--', alpha=0.7, label=f'Target: {self.thresholds["target_loss"]:.4f}')
        ax1.set_title('訓練損失曲線')
        ax1.set_xlabel('步數')
        ax1.set_ylabel('損失值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 學習率曲線
        ax2.plot(steps, self.lr_history, 'g-', linewidth=2, label='Learning Rate')
        ax2.set_title('學習率變化')
        ax2.set_xlabel('步數')
        ax2.set_ylabel('學習率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 損失改善分布
        if len(self.loss_history) >= 10:
            loss_improvements = []
            for i in range(1, len(self.loss_history)):
                improvement = self.loss_history[i-1] - self.loss_history[i]
                loss_improvements.append(improvement)
                
            ax3.hist(loss_improvements, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax3.set_title('損失改善分布')
            ax3.set_xlabel('損失改善值')
            ax3.set_ylabel('頻率')
            ax3.grid(True, alpha=0.3)
        
        # 4. 訓練效率
        if len(self.training_history) >= 10:
            window_size = 10
            efficiency_values = []
            window_steps = []
            
            for i in range(window_size, len(self.training_history)):
                window_losses = self.loss_history[i-window_size:i]
                improvement = window_losses[0] - window_losses[-1]
                efficiency = improvement / window_size
                efficiency_values.append(efficiency)
                window_steps.append(steps[i])
                
            ax4.plot(window_steps, efficiency_values, 'purple', linewidth=2, label='訓練效率')
            ax4.set_title('訓練效率趨勢')
            ax4.set_xlabel('步數')
            ax4.set_ylabel('效率值')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 訓練圖表已保存: {output_path}")
        
    def monitor_training_process(self, training_command: str, max_wait_time: int = 3600):
        """監控訓練過程"""
        self.logger.info(f"🚀 開始監控訓練過程: {training_command}")
        
        try:
            # 啟動訓練過程
            process = subprocess.Popen(
                training_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            start_time = time.time()
            
            # 監控輸出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    # 解析訓練日誌
                    progress = self.parse_training_log(output.strip())
                    if progress:
                        self.update_training_progress(progress)
                    
                    # 輸出原始日誌
                    print(output.strip())
                    
                # 檢查超時
                if time.time() - start_time > max_wait_time:
                    self.logger.warning(f"⏰ 訓練超時 ({max_wait_time} 秒)")
                    process.terminate()
                    break
                    
            # 等待完成
            return_code = process.wait()
            
            # 生成最終報告
            report = self.generate_training_report()
            report_file = os.path.join(self.log_dir, f"training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            # 生成圖表
            chart_file = os.path.join(self.log_dir, f"training_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            self.generate_training_charts(chart_file)
            
            self.logger.info(f"✅ 訓練監控完成, 返回碼: {return_code}")
            self.logger.info(f"📋 報告已保存: {report_file}")
            
            return return_code == 0, report
            
        except Exception as e:
            self.logger.error(f"❌ 監控訓練失敗: {e}")
            return False, {}

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA 訓練進度監控器")
    parser.add_argument("--training-command", required=True, help="訓練命令")
    parser.add_argument("--log-dir", default="training_logs", help="日誌目錄")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="檢查點間隔")
    parser.add_argument("--early-stop-patience", type=int, default=500, help="早停耐心值")
    parser.add_argument("--max-wait-time", type=int, default=3600, help="最大等待時間")
    
    args = parser.parse_args()
    
    # 建立監控器
    monitor = LoRATrainingMonitor(
        log_dir=args.log_dir,
        checkpoint_interval=args.checkpoint_interval,
        early_stop_patience=args.early_stop_patience
    )
    
    # 開始監控
    success, report = monitor.monitor_training_process(args.training_command, args.max_wait_time)
    
    if success:
        print("✅ 訓練完成")
        
        # 檢查是否應該繼續推理
        if monitor.should_continue_to_inference():
            print("🎯 建議繼續進行推理測試")
            return 0
        else:
            print("⚠️ 建議調整參數後重新訓練")
            return 1
    else:
        print("❌ 訓練失敗")
        return 2

if __name__ == "__main__":
    sys.exit(main())
