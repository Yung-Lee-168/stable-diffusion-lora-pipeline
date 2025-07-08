#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: Training Monitor and Visualizer
訓練監控和可視化工具 - 實時監控訓練進度、損失變化和生成品質
"""

import os
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np
from datetime import datetime, timedelta
import time
from PIL import Image
from collections import deque
import threading
import logging

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 嘗試導入 seaborn，如果失敗則使用默認樣式
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    print("⚠️  Seaborn 未安裝，使用 matplotlib 默認樣式")
    HAS_SEABORN = False

class TrainingMonitor:
    """訓練監控器"""
    
    def __init__(self, checkpoint_dir="day3_finetuning_results/checkpoints", 
                 output_dir="day3_finetuning_results/monitoring"):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 監控數據
        self.losses = deque(maxlen=1000)  # 最近1000個損失值
        self.timestamps = deque(maxlen=1000)
        self.epochs = deque(maxlen=1000)
        self.steps = deque(maxlen=1000)
        self.learning_rates = deque(maxlen=1000)
        
        # GPU 監控
        self.gpu_memory = deque(maxlen=100)
        self.gpu_utilization = deque(maxlen=100)
        
        # 訓練狀態
        self.start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.is_training = False
        
        # 日誌設置
        self.setup_logging()
        
    def setup_logging(self):
        """設置日誌"""
        log_path = os.path.join(self.output_dir, "training_monitor.log")
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self, total_epochs=50):
        """開始監控"""
        self.start_time = datetime.now()
        self.total_epochs = total_epochs
        self.is_training = True
        
        print("🔍 開始訓練監控...")
        self.logger.info(f"開始監控，預計訓練 {total_epochs} epochs")
    
    def update_training_metrics(self, epoch, step, loss, lr=None):
        """更新訓練指標"""
        current_time = datetime.now()
        
        self.losses.append(loss)
        self.timestamps.append(current_time)
        self.epochs.append(epoch)
        self.steps.append(step)
        
        if lr is not None:
            self.learning_rates.append(lr)
        
        self.current_epoch = epoch
        
        # 記錄到日誌
        self.logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
    
    def update_gpu_metrics(self):
        """更新 GPU 指標"""
        if torch.cuda.is_available():
            try:
                # GPU 記憶體使用率
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_percent = (memory_used / memory_total) * 100
                
                self.gpu_memory.append(memory_percent)
                
                # GPU 利用率 (簡化估算)
                utilization = min(100, memory_percent * 1.2)  # 簡化的利用率估算
                self.gpu_utilization.append(utilization)
                
            except Exception as e:
                self.logger.warning(f"GPU 指標更新失敗: {e}")
    
    def calculate_eta(self):
        """計算預估剩餘時間"""
        if not self.start_time or self.current_epoch == 0:
            return "未知"
        
        elapsed = datetime.now() - self.start_time
        progress = self.current_epoch / self.total_epochs
        
        if progress > 0:
            total_time = elapsed / progress
            remaining_time = total_time - elapsed
            
            # 格式化時間
            hours = int(remaining_time.total_seconds() // 3600)
            minutes = int((remaining_time.total_seconds() % 3600) // 60)
            
            return f"{hours:02d}:{minutes:02d}"
        
        return "未知"
    
    def generate_training_plots(self):
        """生成訓練圖表"""
        if len(self.losses) < 2:
            return
        
        # 創建圖表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stable Diffusion Fine-tuning 訓練監控', fontsize=16, fontweight='bold')
        
        # 1. 損失曲線
        ax1 = axes[0, 0]
        if len(self.losses) > 0:
            steps_array = np.array(list(self.steps))
            losses_array = np.array(list(self.losses))
            
            ax1.plot(steps_array, losses_array, 'b-', linewidth=1, alpha=0.7)
            
            # 平滑曲線
            if len(losses_array) > 10:
                from scipy.ndimage import uniform_filter1d
                smoothed = uniform_filter1d(losses_array, size=min(20, len(losses_array)//5))
                ax1.plot(steps_array, smoothed, 'r-', linewidth=2, label='平滑曲線')
            
            ax1.set_title('訓練損失曲線')
            ax1.set_xlabel('訓練步數')
            ax1.set_ylabel('損失值')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 2. 學習率變化
        ax2 = axes[0, 1]
        if len(self.learning_rates) > 0:
            steps_lr = np.array(list(self.steps)[-len(self.learning_rates):])
            lr_array = np.array(list(self.learning_rates))
            
            ax2.plot(steps_lr, lr_array, 'g-', linewidth=2)
            ax2.set_title('學習率變化')
            ax2.set_xlabel('訓練步數')
            ax2.set_ylabel('學習率')
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 3. GPU 使用率
        ax3 = axes[1, 0]
        if len(self.gpu_memory) > 0:
            time_points = np.arange(len(self.gpu_memory))
            memory_array = np.array(list(self.gpu_memory))
            util_array = np.array(list(self.gpu_utilization))
            
            ax3.plot(time_points, memory_array, 'orange', linewidth=2, label='記憶體使用率')
            ax3.plot(time_points, util_array, 'purple', linewidth=2, label='GPU 利用率')
            ax3.set_title('GPU 監控')
            ax3.set_xlabel('時間點')
            ax3.set_ylabel('使用率 (%)')
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. 訓練進度
        ax4 = axes[1, 1]
        progress = (self.current_epoch / self.total_epochs) * 100 if self.total_epochs > 0 else 0
        
        # 進度條
        ax4.barh(0, progress, height=0.3, color='lightblue', alpha=0.7)
        ax4.barh(0, 100-progress, height=0.3, left=progress, color='lightgray', alpha=0.3)
        
        ax4.set_xlim(0, 100)
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_title(f'訓練進度: {progress:.1f}%')
        ax4.set_xlabel('完成百分比')
        ax4.text(50, 0, f'{self.current_epoch}/{self.total_epochs} epochs', 
                ha='center', va='center', fontweight='bold')
        
        # 移除 y 軸
        ax4.set_yticks([])
        
        # 添加統計信息
        if len(self.losses) > 0:
            current_loss = self.losses[-1]
            min_loss = min(self.losses)
            avg_loss = np.mean(list(self.losses))
            eta = self.calculate_eta()
            
            stats_text = f"""當前損失: {current_loss:.4f}
最低損失: {min_loss:.4f}
平均損失: {avg_loss:.4f}
預估剩餘: {eta}"""
            
            fig.text(0.02, 0.02, stats_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存圖表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.output_dir, f"training_monitor_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_loss_comparison(self, checkpoints_data):
        """生成不同配置的損失比較圖"""
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (config_name, data) in enumerate(checkpoints_data.items()):
            if 'losses' in data and len(data['losses']) > 0:
                steps = np.arange(len(data['losses']))
                plt.plot(steps, data['losses'], 
                        color=colors[i % len(colors)], 
                        linewidth=2, 
                        label=config_name,
                        alpha=0.8)
        
        plt.title('不同配置訓練損失比較', fontsize=14, fontweight='bold')
        plt.xlabel('訓練步數')
        plt.ylabel('損失值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存比較圖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(self.output_dir, f"loss_comparison_{timestamp}.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return comparison_path
    
    def save_training_summary(self):
        """保存訓練摘要"""
        if len(self.losses) == 0:
            return
        
        summary = {
            "training_start": self.start_time.isoformat() if self.start_time else None,
            "training_end": datetime.now().isoformat(),
            "total_epochs": self.total_epochs,
            "current_epoch": self.current_epoch,
            "total_steps": len(self.losses),
            "final_loss": float(self.losses[-1]),
            "min_loss": float(min(self.losses)),
            "avg_loss": float(np.mean(list(self.losses))),
            "loss_std": float(np.std(list(self.losses))),
            "training_duration": str(datetime.now() - self.start_time) if self.start_time else None
        }
        
        # 保存到 JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.output_dir, f"training_summary_{timestamp}.json")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"📊 訓練摘要已保存: {summary_path}")
        return summary_path

class ValidationImageAnalyzer:
    """驗證圖片分析器"""
    
    def __init__(self, validation_dir="day3_finetuning_results/validation_images"):
        self.validation_dir = validation_dir
        self.analysis_results = {}
    
    def analyze_generated_images(self):
        """分析生成的驗證圖片"""
        if not os.path.exists(self.validation_dir):
            print(f"⚠️  驗證圖片目錄不存在: {self.validation_dir}")
            return
        
        # 找到所有驗證圖片
        validation_images = []
        for file in os.listdir(self.validation_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                validation_images.append(os.path.join(self.validation_dir, file))
        
        if not validation_images:
            print("⚠️  沒有找到驗證圖片")
            return
        
        print(f"🖼️  分析 {len(validation_images)} 張驗證圖片...")
        
        # 分析每張圖片
        analysis_results = []
        
        for img_path in validation_images:
            try:
                image = Image.open(img_path)
                analysis = self.analyze_single_image(image, img_path)
                analysis_results.append(analysis)
                
            except Exception as e:
                print(f"❌ 圖片分析失敗 {img_path}: {e}")
        
        # 生成分析報告
        self.generate_analysis_report(analysis_results)
        
        return analysis_results
    
    def analyze_single_image(self, image, image_path):
        """分析單張圖片"""
        # 基本信息
        width, height = image.size
        
        # 轉換為 numpy 數組
        img_array = np.array(image)
        
        # 顏色分析
        if len(img_array.shape) == 3:
            # RGB 圖片
            r_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            b_mean = np.mean(img_array[:, :, 2])
            
            # 亮度分析
            brightness = np.mean(img_array)
            
            # 對比度分析 (標準差)
            contrast = np.std(img_array)
            
            # 顏色豐富度 (顏色直方圖的熵)
            hist_r = np.histogram(img_array[:, :, 0], bins=256)[0]
            hist_g = np.histogram(img_array[:, :, 1], bins=256)[0]
            hist_b = np.histogram(img_array[:, :, 2], bins=256)[0]
            
            # 計算熵
            def calculate_entropy(hist):
                hist = hist[hist > 0]  # 移除零值
                prob = hist / hist.sum()
                return -np.sum(prob * np.log2(prob))
            
            color_entropy = (calculate_entropy(hist_r) + 
                           calculate_entropy(hist_g) + 
                           calculate_entropy(hist_b)) / 3
        else:
            # 灰度圖片
            r_mean = g_mean = b_mean = np.mean(img_array)
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            color_entropy = 0
        
        # 邊緣檢測 (Canny)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        return {
            "image_path": os.path.basename(image_path),
            "size": [width, height],
            "color_means": [float(r_mean), float(g_mean), float(b_mean)],
            "brightness": float(brightness),
            "contrast": float(contrast),
            "color_entropy": float(color_entropy),
            "edge_density": float(edge_density),
            "analysis_time": datetime.now().isoformat()
        }
    
    def generate_analysis_report(self, analysis_results):
        """生成分析報告"""
        if not analysis_results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建分析圖表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('驗證圖片品質分析', fontsize=16, fontweight='bold')
        
        # 提取數據
        brightness_values = [r["brightness"] for r in analysis_results]
        contrast_values = [r["contrast"] for r in analysis_results]
        entropy_values = [r["color_entropy"] for r in analysis_results]
        edge_values = [r["edge_density"] for r in analysis_results]
        
        # 1. 亮度分布
        axes[0, 0].hist(brightness_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('亮度分布')
        axes[0, 0].set_xlabel('亮度值')
        axes[0, 0].set_ylabel('頻率')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 對比度分布
        axes[0, 1].hist(contrast_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('對比度分布')
        axes[0, 1].set_xlabel('對比度值')
        axes[0, 1].set_ylabel('頻率')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 顏色豐富度分布
        axes[1, 0].hist(entropy_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_title('顏色豐富度分布')
        axes[1, 0].set_xlabel('熵值')
        axes[1, 0].set_ylabel('頻率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 邊緣密度分布
        axes[1, 1].hist(edge_values, bins=20, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('邊緣密度分布')
        axes[1, 1].set_xlabel('邊緣密度')
        axes[1, 1].set_ylabel('頻率')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        chart_path = os.path.join(os.path.dirname(self.validation_dir), 
                                 f"validation_analysis_{timestamp}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存 JSON 報告
        report_path = os.path.join(os.path.dirname(self.validation_dir), 
                                  f"validation_report_{timestamp}.json")
        
        report = {
            "analysis_time": datetime.now().isoformat(),
            "total_images": len(analysis_results),
            "statistics": {
                "brightness": {
                    "mean": float(np.mean(brightness_values)),
                    "std": float(np.std(brightness_values)),
                    "min": float(np.min(brightness_values)),
                    "max": float(np.max(brightness_values))
                },
                "contrast": {
                    "mean": float(np.mean(contrast_values)),
                    "std": float(np.std(contrast_values)),
                    "min": float(np.min(contrast_values)),
                    "max": float(np.max(contrast_values))
                },
                "color_entropy": {
                    "mean": float(np.mean(entropy_values)),
                    "std": float(np.std(entropy_values)),
                    "min": float(np.min(entropy_values)),
                    "max": float(np.max(entropy_values))
                },
                "edge_density": {
                    "mean": float(np.mean(edge_values)),
                    "std": float(np.std(edge_values)),
                    "min": float(np.min(edge_values)),
                    "max": float(np.max(edge_values))
                }
            },
            "individual_results": analysis_results
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📊 驗證分析報告已保存: {report_path}")
        print(f"📈 分析圖表已保存: {chart_path}")

def main():
    """主函數 - 監控工具演示"""
    print("🔍 訓練監控和可視化工具")
    print("=" * 50)
    
    # 創建監控器
    monitor = TrainingMonitor()
    analyzer = ValidationImageAnalyzer()
    
    # 模擬一些訓練數據
    print("📊 生成模擬訓練數據...")
    monitor.start_monitoring(total_epochs=20)
    
    # 模擬訓練過程
    for epoch in range(5):
        for step in range(10):
            # 模擬損失值
            loss = 0.8 * np.exp(-step * 0.1) + 0.1 + np.random.normal(0, 0.05)
            lr = 1e-4 * (0.95 ** epoch)
            
            monitor.update_training_metrics(epoch, step, loss, lr)
            monitor.update_gpu_metrics()
    
    # 生成監控圖表
    plot_path = monitor.generate_training_plots()
    print(f"📈 監控圖表已生成: {plot_path}")
    
    # 保存訓練摘要
    summary_path = monitor.save_training_summary()
    
    # 分析驗證圖片（如果存在）
    print("\n🖼️  分析驗證圖片...")
    analyzer.analyze_generated_images()

if __name__ == "__main__":
    # 安裝必需的包
    try:
        import scipy
    except ImportError:
        print("⚠️  需要安裝 scipy: pip install scipy")
    
    try:
        import cv2
    except ImportError:
        print("⚠️  需要安裝 opencv: pip install opencv-python")
    
    main()
