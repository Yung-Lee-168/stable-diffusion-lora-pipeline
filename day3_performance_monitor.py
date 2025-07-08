#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3 系統效能監控與優化工具
🎯 監控系統效能，提供優化建議，確保大規模處理的穩定性

功能：
1. 系統資源監控
2. 處理效能分析
3. 記憶體使用優化
4. 批次處理效率評估
"""

import os
import time
import psutil
import json
from datetime import datetime
from day3_fashion_training import FashionTrainingPipeline

class PerformanceMonitor:
    """效能監控器"""
    
    def __init__(self):
        self.pipeline = FashionTrainingPipeline()
        self.monitoring_data = []
        self.start_time = None
        
    def monitor_system_resources(self):
        """監控系統資源"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_free_gb': disk.free / (1024**3)
        }
    
    def benchmark_feature_extraction(self, image_path, iterations=3):
        """特徵提取效能基準測試"""
        print(f"🔬 特徵提取效能測試 (x{iterations})")
        
        times = []
        memory_usage = []
        
        for i in range(iterations):
            # 記錄開始狀態
            start_memory = psutil.virtual_memory().percent
            start_time = time.time()
            
            # 執行特徵提取
            try:
                features = self.pipeline.extract_fashion_features(image_path)
                extraction_time = time.time() - start_time
                
                # 記錄結束狀態
                end_memory = psutil.virtual_memory().percent
                
                times.append(extraction_time)
                memory_usage.append(end_memory - start_memory)
                
                print(f"   第 {i+1} 次: {extraction_time:.3f}s, 記憶體變化: {end_memory-start_memory:+.1f}%")
                
            except Exception as e:
                print(f"   第 {i+1} 次失敗: {e}")
                continue
        
        if times:
            avg_time = sum(times) / len(times)
            avg_memory = sum(memory_usage) / len(memory_usage)
            
            print(f"📊 平均效能: {avg_time:.3f}s/張, 記憶體影響: {avg_memory:+.1f}%")
            
            return {
                'avg_extraction_time': avg_time,
                'avg_memory_impact': avg_memory,
                'times': times,
                'memory_changes': memory_usage
            }
        
        return None
    
    def estimate_batch_capacity(self):
        """估算批次處理容量"""
        print("📊 估算批次處理容量")
        
        # 獲取系統資源
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        
        # 估算每張圖片的記憶體需求 (基於經驗值)
        estimated_memory_per_image_mb = 50  # MB
        
        # 保留系統記憶體緩衝
        buffer_ratio = 0.7  # 使用70%的可用記憶體
        usable_memory_gb = available_memory_gb * buffer_ratio
        
        # 計算建議的批次大小
        recommended_batch_size = int((usable_memory_gb * 1024) / estimated_memory_per_image_mb)
        
        print(f"💾 系統記憶體狀況:")
        print(f"   總記憶體: {memory.total / (1024**3):.1f} GB")
        print(f"   可用記憶體: {available_memory_gb:.1f} GB")
        print(f"   建議使用: {usable_memory_gb:.1f} GB")
        print(f"📦 建議批次大小: {recommended_batch_size} 張圖片")
        
        if recommended_batch_size < 5:
            print("⚠️ 警告: 系統記憶體可能不足，建議釋放記憶體或減少並行處理")
        elif recommended_batch_size > 50:
            print("🚀 系統資源充足，可以進行大規模批次處理")
        
        return {
            'available_memory_gb': available_memory_gb,
            'recommended_batch_size': recommended_batch_size,
            'memory_status': 'sufficient' if recommended_batch_size >= 10 else 'limited'
        }
    
    def optimize_processing_strategy(self, total_images):
        """優化處理策略"""
        print(f"🎯 為 {total_images} 張圖片優化處理策略")
        
        capacity = self.estimate_batch_capacity()
        recommended_batch = capacity['recommended_batch_size']
        
        if total_images <= recommended_batch:
            strategy = "single_batch"
            batches = [total_images]
            print(f"✅ 策略: 單批次處理 ({total_images} 張)")
        else:
            strategy = "multi_batch"
            num_batches = (total_images + recommended_batch - 1) // recommended_batch
            batches = []
            
            for i in range(num_batches):
                start_idx = i * recommended_batch
                end_idx = min((i + 1) * recommended_batch, total_images)
                batch_size = end_idx - start_idx
                batches.append(batch_size)
            
            print(f"📦 策略: 多批次處理 ({num_batches} 批次)")
            for i, batch_size in enumerate(batches):
                print(f"   批次 {i+1}: {batch_size} 張")
        
        return {
            'strategy': strategy,
            'batches': batches,
            'total_batches': len(batches),
            'estimated_total_time': self.estimate_processing_time(total_images)
        }
    
    def estimate_processing_time(self, num_images):
        """估算處理時間"""
        # 基於之前的測試數據估算
        avg_time_per_image = 2.0  # 秒 (包含特徵提取和提示詞生成)
        
        total_time_seconds = num_images * avg_time_per_image
        
        if total_time_seconds < 60:
            time_str = f"{total_time_seconds:.0f} 秒"
        elif total_time_seconds < 3600:
            time_str = f"{total_time_seconds/60:.1f} 分鐘"
        else:
            time_str = f"{total_time_seconds/3600:.1f} 小時"
        
        return {
            'total_seconds': total_time_seconds,
            'formatted': time_str
        }
    
    def run_performance_analysis(self):
        """運行完整的效能分析"""
        print("🔍 Day 3 系統效能分析")
        print("=" * 50)
        
        # 1. 系統資源檢查
        print("\n1️⃣ 系統資源檢查")
        resources = self.monitor_system_resources()
        
        print(f"💻 CPU 使用率: {resources['cpu_percent']:.1f}%")
        print(f"💾 記憶體使用率: {resources['memory_percent']:.1f}%")
        print(f"📁 可用磁碟空間: {resources['disk_free_gb']:.1f} GB")
        
        # 2. 特徵提取效能測試
        print("\n2️⃣ 特徵提取效能測試")
        test_images = [f for f in os.listdir("day1_results") 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if test_images:
            test_image = os.path.join("day1_results", test_images[0])
            benchmark_result = self.benchmark_feature_extraction(test_image)
        else:
            print("❌ 沒有可用的測試圖片")
            benchmark_result = None
        
        # 3. 批次處理容量評估
        print("\n3️⃣ 批次處理容量評估")
        capacity = self.estimate_batch_capacity()
        
        # 4. 處理策略建議
        print("\n4️⃣ 處理策略建議")
        total_images = len(test_images)
        if total_images > 0:
            strategy = self.optimize_processing_strategy(total_images)
            print(f"⏱️ 預計總處理時間: {strategy['estimated_total_time']['formatted']}")
        
        # 5. 優化建議
        print("\n5️⃣ 優化建議")
        self.provide_optimization_recommendations(resources, capacity, benchmark_result)
        
        # 保存分析結果
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'system_resources': resources,
            'batch_capacity': capacity,
            'benchmark_result': benchmark_result,
            'total_images': total_images,
            'optimization_strategy': strategy if total_images > 0 else None
        }
        
        # 保存到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"day3_performance_analysis_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 分析結果已保存至: {output_file}")
        
        return analysis_result
    
    def provide_optimization_recommendations(self, resources, capacity, benchmark):
        """提供優化建議"""
        recommendations = []
        
        # 記憶體建議
        if resources['memory_percent'] > 80:
            recommendations.append("⚠️ 記憶體使用率過高，建議關閉其他應用程式")
        elif resources['memory_percent'] < 50:
            recommendations.append("✅ 記憶體資源充足，可以增加並行處理")
        
        # CPU建議
        if resources['cpu_percent'] > 90:
            recommendations.append("⚠️ CPU負載過高，建議降低並行度")
        elif resources['cpu_percent'] < 30:
            recommendations.append("🚀 CPU資源充足，可以啟用更多並行處理")
        
        # 磁碟空間建議
        if resources['disk_free_gb'] < 5:
            recommendations.append("⚠️ 磁碟空間不足，建議清理臨時文件")
        
        # 批次大小建議
        if capacity['memory_status'] == 'limited':
            recommendations.append("📦 建議使用小批次處理，避免記憶體不足")
        else:
            recommendations.append("📦 系統可支援大批次處理，提升效率")
        
        # 效能建議
        if benchmark and benchmark['avg_extraction_time'] > 3.0:
            recommendations.append("⏱️ 特徵提取較慢，可能需要硬體升級或優化")
        elif benchmark and benchmark['avg_extraction_time'] < 1.0:
            recommendations.append("⚡ 特徵提取效能優秀")
        
        if not recommendations:
            recommendations.append("✅ 系統效能良好，無需特別優化")
        
        for rec in recommendations:
            print(f"   {rec}")
    
    def quick_health_check(self):
        """快速健康檢查"""
        print("🏥 系統健康快速檢查")
        print("-" * 30)
        
        # 檢查重要組件
        checks = {
            'day1_results 目錄': os.path.exists('day1_results'),
            'day3_fashion_training.py': os.path.exists('day3_fashion_training.py'),
            '測試圖片': len([f for f in os.listdir('day1_results') if f.endswith(('.jpg', '.png'))]) > 0 if os.path.exists('day1_results') else False
        }
        
        all_good = True
        for check_name, status in checks.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {check_name}")
            if not status:
                all_good = False
        
        # 資源檢查
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        memory_ok = memory.percent < 90
        cpu_ok = cpu < 95
        
        print(f"{'✅' if memory_ok else '⚠️'} 記憶體使用率: {memory.percent:.1f}%")
        print(f"{'✅' if cpu_ok else '⚠️'} CPU使用率: {cpu:.1f}%")
        
        overall_health = all_good and memory_ok and cpu_ok
        
        print(f"\n🎯 整體系統狀態: {'健康 ✅' if overall_health else '需要注意 ⚠️'}")
        
        return overall_health

def main():
    """主函數"""
    print("🔍 Day 3 效能監控與優化工具")
    print("確保系統在大規模處理時保持最佳效能")
    print("=" * 60)
    
    monitor = PerformanceMonitor()
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "health":
            monitor.quick_health_check()
        elif sys.argv[1] == "full":
            monitor.run_performance_analysis()
        else:
            print("用法: python day3_performance_monitor.py [health|full]")
    else:
        # 默認運行健康檢查
        monitor.quick_health_check()
        
        print(f"\n🔬 要運行完整效能分析，請執行:")
        print(f"   python day3_performance_monitor.py full")

if __name__ == "__main__":
    main()
