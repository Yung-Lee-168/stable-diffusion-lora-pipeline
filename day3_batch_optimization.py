#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3 大規模批次優化測試腳本
🎯 擴大測試規模，進行全面的提示詞優化分析

功能：
1. 批次處理所有可用圖片 (目標：20張以上)
2. 自動對比不同 prompt 配置
3. 生成詳細的優化報告
4. 效能分析與建議
"""

import os
import json
import time
from datetime import datetime
from day3_fashion_training import FashionTrainingPipeline
import shutil

class BatchOptimizationTester:
    """大規模批次優化測試器"""
    
    def __init__(self):
        self.pipeline = FashionTrainingPipeline()
        self.results_dir = "day3_batch_results"
        self.ensure_results_dir()
        
    def ensure_results_dir(self):
        """確保結果目錄存在"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
    def expand_test_dataset(self):
        """擴大測試數據集"""
        print("🔍 擴大測試數據集...")
        
        # 檢查現有圖片
        day1_images = self.get_image_files("day1_results")
        backup_images = self.get_image_files("day1_results/BackUp")
        
        print(f"📊 現有圖片統計:")
        print(f"   day1_results: {len(day1_images)} 張")
        print(f"   backup: {len(backup_images)} 張")
        
        # 如果圖片數量不足，從 backup 複製更多圖片
        total_available = len(day1_images) + len(backup_images)
        
        if len(day1_images) < 10 and len(backup_images) > 0:
            print("📁 從 backup 複製額外圖片...")
            
            for img in backup_images[:5]:  # 複製最多5張
                if img.endswith('.png'):
                    src = os.path.join("day1_results/BackUp", img)
                    # 重命名為 jpg 格式的名稱
                    new_name = f"backup_{img.replace('.png', '.jpg')}"
                    dst = os.path.join("day1_results", new_name)
                    
                    try:
                        shutil.copy2(src, dst)
                        print(f"   ✓ 複製: {img} → {new_name}")
                    except Exception as e:
                        print(f"   ❌ 複製失敗: {e}")
        
        # 重新統計
        final_images = self.get_image_files("day1_results")
        print(f"📈 最終測試集: {len(final_images)} 張圖片")
        return final_images
        
    def get_image_files(self, directory):
        """獲取目錄中的圖片文件"""
        if not os.path.exists(directory):
            return []
            
        return [f for f in os.listdir(directory) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                and not f.startswith('generated_')]
    
    def run_comprehensive_test(self):
        """運行全面測試"""
        print("🚀 Day 3 大規模批次優化測試")
        print("=" * 60)
        
        # 擴大數據集
        test_images = self.expand_test_dataset()
        
        if len(test_images) < 5:
            print("❌ 測試圖片數量不足 (需要至少5張)")
            return None
            
        # 測試配置
        configs_to_test = [
            "minimal_prompt",      # 最簡配置
            "high_confidence_only", # 高置信度配置  
            "default"              # 標準配置
        ]
        
        print(f"\n🧪 測試配置: {', '.join(configs_to_test)}")
        print(f"📊 測試圖片數量: {len(test_images)}")
        
        # 開始批次測試
        batch_results = {}
        start_time = time.time()
        
        for i, config in enumerate(configs_to_test, 1):
            print(f"\n🔄 [{i}/{len(configs_to_test)}] 測試配置: {config}")
            print("-" * 40)
            
            config_results = self.test_single_config(test_images, config)
            batch_results[config] = config_results
            
            # 顯示進度
            progress = i / len(configs_to_test) * 100
            print(f"📈 總體進度: {progress:.1f}%")
            
        total_time = time.time() - start_time
        
        # 生成報告
        self.generate_batch_report(batch_results, test_images, total_time)
        
        return batch_results
    
    def test_single_config(self, test_images, config_name):
        """測試單一配置"""
        results = {
            'config_name': config_name,
            'images_tested': len(test_images),
            'successful_generations': 0,
            'total_fashionclip_similarity': 0,
            'prompt_lengths': [],
            'processing_times': [],
            'detailed_results': []
        }
        
        # 設定配置
        self.pipeline.set_prompt_config(config_name)
        
        for i, image_file in enumerate(test_images, 1):
            print(f"   🖼️ [{i}/{len(test_images)}] 處理: {image_file}")
            
            image_path = os.path.join("day1_results", image_file)
            start_time = time.time()
            
            try:
                # 提取特徵並生成提示詞
                features = self.pipeline.extract_fashion_features(image_path)
                structured_features = self.pipeline.structure_features(features)
                prompt = self.pipeline.features_to_prompt(structured_features)
                
                # 記錄結果
                processing_time = time.time() - start_time
                
                results['prompt_lengths'].append(len(prompt))
                results['processing_times'].append(processing_time)
                results['successful_generations'] += 1
                
                # 詳細結果
                detailed_result = {
                    'image': image_file,
                    'prompt': prompt,
                    'prompt_length': len(prompt),
                    'processing_time': processing_time,
                    'features_count': len([k for k, v in structured_features.items() if v and k != 'overall_style'])
                }
                
                results['detailed_results'].append(detailed_result)
                
                print(f"      ✓ 成功 ({processing_time:.2f}s, {len(prompt)} 字符)")
                
            except Exception as e:
                print(f"      ❌ 失敗: {e}")
                
        # 計算平均值
        if results['prompt_lengths']:
            results['avg_prompt_length'] = sum(results['prompt_lengths']) / len(results['prompt_lengths'])
            results['avg_processing_time'] = sum(results['processing_times']) / len(results['processing_times'])
        
        return results
    
    def generate_batch_report(self, batch_results, test_images, total_time):
        """生成批次測試報告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 報告
        json_report_path = os.path.join(self.results_dir, f"batch_optimization_{timestamp}.json")
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        # Markdown 報告
        md_report_path = os.path.join(self.results_dir, f"batch_optimization_report_{timestamp}.md")
        self.generate_markdown_report(batch_results, test_images, total_time, md_report_path)
        
        print(f"\n📋 報告已生成:")
        print(f"   📄 JSON: {json_report_path}")
        print(f"   📖 Markdown: {md_report_path}")
        
    def generate_markdown_report(self, batch_results, test_images, total_time, report_path):
        """生成 Markdown 格式的詳細報告"""
        
        report_content = f"""# Day 3 大規模批次優化測試報告

## 🎯 測試概覽
- **測試時間**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}
- **測試圖片數量**: {len(test_images)} 張
- **測試配置數量**: {len(batch_results)} 種
- **總耗時**: {total_time:.2f} 秒

## 📊 配置對比分析

| 配置 | 成功率 | 平均提示詞長度 | 平均處理時間 | 特徵豐富度 |
|------|--------|----------------|--------------|------------|
"""
        
        for config_name, results in batch_results.items():
            success_rate = (results['successful_generations'] / results['images_tested']) * 100
            avg_length = results.get('avg_prompt_length', 0)
            avg_time = results.get('avg_processing_time', 0)
            
            # 計算特徵豐富度 (平均特徵數量)
            if results['detailed_results']:
                avg_features = sum(r['features_count'] for r in results['detailed_results']) / len(results['detailed_results'])
            else:
                avg_features = 0
            
            report_content += f"| {config_name} | {success_rate:.1f}% | {avg_length:.0f} 字符 | {avg_time:.2f}s | {avg_features:.1f} 特徵 |\n"
        
        report_content += f"""

## 🔍 詳細分析

### 配置特性對比
"""
        
        for config_name, results in batch_results.items():
            report_content += f"""
#### {config_name} 配置
- **成功處理**: {results['successful_generations']}/{results['images_tested']} 張
- **提示詞長度範圍**: {min(results['prompt_lengths']) if results['prompt_lengths'] else 0} - {max(results['prompt_lengths']) if results['prompt_lengths'] else 0} 字符
- **處理時間範圍**: {min(results['processing_times']):.2f} - {max(results['processing_times']):.2f} 秒

**範例提示詞**:
"""
            if results['detailed_results']:
                example = results['detailed_results'][0]
                report_content += f"```\n{example['prompt']}\n```\n"
        
        # 優化建議
        report_content += f"""
## 💡 優化建議

### 基於測試結果的建議

"""
        
        # 找出最佳配置
        best_config = None
        best_balance_score = 0
        
        for config_name, results in batch_results.items():
            if results['successful_generations'] > 0:
                success_rate = results['successful_generations'] / results['images_tested']
                avg_time = results.get('avg_processing_time', 0)
                
                # 平衡分數：成功率高、處理時間短
                balance_score = success_rate * 100 - avg_time * 10
                
                if balance_score > best_balance_score:
                    best_balance_score = balance_score
                    best_config = config_name
        
        if best_config:
            report_content += f"""
#### 🏆 推薦配置: {best_config}
- 在成功率和效率之間達到最佳平衡
- 建議作為預設配置使用

"""
        
        report_content += f"""
#### 📈 擴展建議
1. **增加測試圖片**: 當前 {len(test_images)} 張，建議擴展至 50+ 張
2. **多樣化測試**: 包含不同風格、年齡、場合的時尚圖片
3. **生成品質評估**: 加入實際圖片生成與視覺相似度評估
4. **A/B 測試**: 針對特定場景進行精細化配置優化

#### 🔧 技術優化
- 考慮引入特徵重要性評分
- 優化 FashionCLIP 特徵提取效率
- 實施智能特徵篩選策略

---
*報告生成時間: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}*
*Day 3 Fashion Training Pipeline - 大規模批次優化測試*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def quick_demo(self):
        """快速演示"""
        print("🎬 快速演示 - 提示詞優化效果")
        print("=" * 40)
        
        test_images = self.get_image_files("day1_results")
        if not test_images:
            print("❌ 沒有可用的測試圖片")
            return
            
        demo_image = os.path.join("day1_results", test_images[0])
        print(f"🖼️ 演示圖片: {test_images[0]}")
        
        configs = ["minimal_prompt", "default"]
        
        for config in configs:
            print(f"\n🔧 配置: {config}")
            self.pipeline.set_prompt_config(config)
            
            features = self.pipeline.extract_fashion_features(demo_image)
            structured_features = self.pipeline.structure_features(features)
            prompt = self.pipeline.features_to_prompt(structured_features)
            
            print(f"📝 提示詞 ({len(prompt)} 字符):")
            print(f"   {prompt}")

def main():
    """主函數"""
    print("🎯 Day 3 大規模批次優化測試")
    print("擴大測試規模，全面優化提示詞策略")
    print("=" * 60)
    
    tester = BatchOptimizationTester()
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            tester.quick_demo()
        elif sys.argv[1] == "full":
            tester.run_comprehensive_test()
        else:
            print("用法: python day3_batch_optimization.py [demo|full]")
    else:
        # 默認運行快速演示
        tester.quick_demo()
        
        print(f"\n🚀 要運行完整測試，請執行:")
        print(f"   python day3_batch_optimization.py full")

if __name__ == "__main__":
    main()
