#!/usr/bin/env python3
"""
LoRA 調優自動化腳本
根據分析結果自動調整 LoRA 訓練參數，並生成下一輪訓練配置

功能：
1. 讀取分析報告
2. 自動計算最佳參數
3. 生成訓練配置文件
4. 提供參數調整建議
5. 支援多輪迭代優化
"""

import json
import os
import datetime
import argparse
from typing import Dict, Any, List, Tuple

class LoRATuningOptimizer:
    """LoRA 調優優化器"""
    
    def __init__(self):
        self.base_config = {
            "learning_rate": 0.0005,
            "steps": 100,
            "batch_size": 1,
            "resolution": "512x512",
            "network_dim": 32,
            "network_alpha": 32,
            "clip_skip": 2,
            "mixed_precision": "fp16",
            "save_every_n_epochs": 1,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 0,
            "noise_offset": 0.0,
            "adaptive_noise_scale": 0.0,
            "cache_latents": True,
            "cache_text_encoder_outputs": True,
            "enable_bucket": True,
            "bucket_no_upscale": True,
            "bucket_reso_steps": 64,
            "min_bucket_reso": 256,
            "max_bucket_reso": 1024
        }
        
        self.optimization_history = []
        self.iteration_count = 0
        
    def load_analysis_report(self, report_path: str) -> Dict[str, Any]:
        """載入分析報告"""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            print(f"✅ 成功載入分析報告：{report_path}")
            return report
        except Exception as e:
            print(f"❌ 載入分析報告失敗：{e}")
            return {}
    
    def analyze_current_performance(self, report: Dict[str, Any]) -> Dict[str, float]:
        """分析當前性能指標"""
        performance = {
            "total_loss": 1.0,
            "visual_similarity": 0.0,
            "fashion_clip_similarity": 0.0,
            "color_similarity": 0.0,
            "overall_score": 0.0,
            "training_efficiency": 0.0
        }
        
        # 從 benchmark_analysis 提取指標
        if "benchmark_analysis" in report:
            benchmark = report["benchmark_analysis"]
            avg_metrics = benchmark.get("average_metrics", {})
            
            performance["total_loss"] = avg_metrics.get("avg_total_loss", 1.0)
            performance["visual_similarity"] = avg_metrics.get("avg_visual_similarity", 0.0)
            performance["fashion_clip_similarity"] = avg_metrics.get("avg_fashion_clip_similarity", 0.0)
            performance["color_similarity"] = avg_metrics.get("avg_color_similarity", 0.0)
        
        # 從 lora_tuning 提取整體分數
        if "lora_tuning" in report:
            lora_tuning = report["lora_tuning"]
            performance["overall_score"] = lora_tuning.get("overall_tuning_score", 0.0)
            
            # 訓練效率
            if "training_efficiency" in lora_tuning:
                performance["training_efficiency"] = lora_tuning["training_efficiency"].get("score", 0.0)
        
        return performance
    
    def calculate_optimal_parameters(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """根據性能指標計算最佳參數"""
        config = self.base_config.copy()
        
        total_loss = performance["total_loss"]
        visual_sim = performance["visual_similarity"]
        fashion_sim = performance["fashion_clip_similarity"]
        overall_score = performance["overall_score"]
        efficiency = performance["training_efficiency"]
        
        # 學習率調整策略
        if total_loss > 0.8:
            config["learning_rate"] = 0.0002  # 大幅降低
        elif total_loss > 0.6:
            config["learning_rate"] = 0.0003  # 適度降低
        elif total_loss > 0.4:
            config["learning_rate"] = 0.0005  # 標準值
        elif total_loss > 0.2:
            config["learning_rate"] = 0.0008  # 稍微提高
        else:
            config["learning_rate"] = 0.001   # 可以更積極
        
        # 訓練步數調整
        if fashion_sim < 0.3:
            config["steps"] = 300  # 需要更多訓練
        elif fashion_sim < 0.5:
            config["steps"] = 200  # 適度增加
        elif fashion_sim < 0.7:
            config["steps"] = 150  # 標準訓練
        else:
            config["steps"] = 100  # 保持效率
        
        # 網路維度調整
        if overall_score < 0.4:
            config["network_dim"] = 64   # 增加容量
            config["network_alpha"] = 64
        elif overall_score < 0.6:
            config["network_dim"] = 48   # 適度增加
            config["network_alpha"] = 48
        elif overall_score > 0.8:
            config["network_dim"] = 16   # 減少過擬合
            config["network_alpha"] = 16
        
        # 解析度調整
        if visual_sim < 0.3:
            config["resolution"] = "768x768"  # 提高解析度
        elif visual_sim < 0.5:
            config["resolution"] = "640x640"  # 適度提高
        else:
            config["resolution"] = "512x512"  # 標準解析度
        
        # 學習率調度器
        if total_loss > 0.6:
            config["lr_scheduler"] = "cosine_with_restarts"
            config["lr_warmup_steps"] = config["steps"] // 10
        elif efficiency < 0.5:
            config["lr_scheduler"] = "polynomial"
            config["lr_warmup_steps"] = 0
        else:
            config["lr_scheduler"] = "cosine"
            config["lr_warmup_steps"] = 0
        
        # 雜訊偏移調整
        if visual_sim < 0.4:
            config["noise_offset"] = 0.1  # 增加變化
        elif visual_sim > 0.7:
            config["noise_offset"] = 0.0  # 保持穩定
        
        # 混合精度
        if efficiency < 0.6:
            config["mixed_precision"] = "bf16"  # 更高效率
        else:
            config["mixed_precision"] = "fp16"  # 平衡品質
        
        return config
    
    def generate_training_weights(self, performance: Dict[str, float]) -> Dict[str, float]:
        """生成訓練權重配置"""
        weights = {
            "visual_weight": 0.2,
            "fashion_clip_weight": 0.6,
            "color_weight": 0.2,
            "text_encoder_weight": 1.0,
            "unet_weight": 1.0
        }
        
        visual_sim = performance["visual_similarity"]
        fashion_sim = performance["fashion_clip_similarity"]
        color_sim = performance["color_similarity"]
        
        # 動態調整權重
        if visual_sim < 0.3:
            weights["visual_weight"] = 0.4  # 加強視覺
            weights["fashion_clip_weight"] = 0.4
            weights["color_weight"] = 0.2
        elif fashion_sim < 0.4:
            weights["visual_weight"] = 0.15  # 加強特徵
            weights["fashion_clip_weight"] = 0.7
            weights["color_weight"] = 0.15
        elif color_sim < 0.3:
            weights["visual_weight"] = 0.15  # 加強色彩
            weights["fashion_clip_weight"] = 0.55
            weights["color_weight"] = 0.3
        
        # 文本編碼器權重
        if fashion_sim < 0.4:
            weights["text_encoder_weight"] = 0.5  # 減少文本編碼器學習
        elif fashion_sim > 0.7:
            weights["text_encoder_weight"] = 0.2  # 進一步減少
        
        return weights
    
    def create_optimization_plan(self, performance: Dict[str, float]) -> List[Dict[str, Any]]:
        """創建多輪優化計劃"""
        plan = []
        
        total_loss = performance["total_loss"]
        overall_score = performance["overall_score"]
        
        # 根據當前表現決定輪數
        if overall_score < 0.4:
            iterations = 4  # 需要多輪優化
        elif overall_score < 0.6:
            iterations = 3  # 適度優化
        elif overall_score < 0.8:
            iterations = 2  # 微調
        else:
            iterations = 1  # 維持
        
        for i in range(iterations):
            # 每輪逐步調整
            iteration_performance = performance.copy()
            
            # 模擬改善
            iteration_performance["total_loss"] *= (0.8 ** (i + 1))
            iteration_performance["visual_similarity"] *= (1.1 ** (i + 1))
            iteration_performance["fashion_clip_similarity"] *= (1.05 ** (i + 1))
            iteration_performance["overall_score"] *= (1.1 ** (i + 1))
            
            # 限制在合理範圍
            iteration_performance["total_loss"] = max(0.1, iteration_performance["total_loss"])
            iteration_performance["visual_similarity"] = min(0.9, iteration_performance["visual_similarity"])
            iteration_performance["fashion_clip_similarity"] = min(0.9, iteration_performance["fashion_clip_similarity"])
            iteration_performance["overall_score"] = min(1.0, iteration_performance["overall_score"])
            
            config = self.calculate_optimal_parameters(iteration_performance)
            weights = self.generate_training_weights(iteration_performance)
            
            plan.append({
                "iteration": i + 1,
                "target_performance": iteration_performance,
                "config": config,
                "weights": weights,
                "description": f"第 {i + 1} 輪優化：{'基礎改善' if i == 0 else '進階調整' if i == 1 else '細微調整' if i == 2 else '精細優化'}"
            })
        
        return plan
    
    def save_optimization_configs(self, plan: List[Dict[str, Any]], output_dir: str = "optimization_configs"):
        """保存優化配置文件"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = []
        
        for iteration_plan in plan:
            iteration = iteration_plan["iteration"]
            config = iteration_plan["config"]
            weights = iteration_plan["weights"]
            
            # 保存訓練配置
            config_file = os.path.join(output_dir, f"lora_config_iter_{iteration}_{timestamp}.json")
            full_config = {
                "metadata": {
                    "iteration": iteration,
                    "timestamp": timestamp,
                    "description": iteration_plan["description"]
                },
                "training_config": config,
                "loss_weights": weights,
                "target_performance": iteration_plan["target_performance"]
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=2, ensure_ascii=False)
            
            saved_files.append(config_file)
            
            # 生成命令行參數
            cmd_file = os.path.join(output_dir, f"lora_command_iter_{iteration}_{timestamp}.txt")
            command = self.generate_training_command(config, weights)
            
            with open(cmd_file, 'w', encoding='utf-8') as f:
                f.write(command)
            
            saved_files.append(cmd_file)
        
        # 保存優化計劃總覽
        plan_file = os.path.join(output_dir, f"optimization_plan_{timestamp}.json")
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        
        saved_files.append(plan_file)
        
        return saved_files
    
    def generate_training_command(self, config: Dict[str, Any], weights: Dict[str, float]) -> str:
        """生成訓練命令"""
        command = "python train_lora.py"
        
        # 基本參數
        command += f" --learning_rate {config['learning_rate']}"
        command += f" --max_train_steps {config['steps']}"
        command += f" --train_batch_size {config['batch_size']}"
        command += f" --resolution {config['resolution']}"
        command += f" --network_dim {config['network_dim']}"
        command += f" --network_alpha {config['network_alpha']}"
        command += f" --clip_skip {config['clip_skip']}"
        command += f" --mixed_precision {config['mixed_precision']}"
        command += f" --save_every_n_epochs {config['save_every_n_epochs']}"
        command += f" --lr_scheduler {config['lr_scheduler']}"
        
        # 可選參數
        if config.get('lr_warmup_steps', 0) > 0:
            command += f" --lr_warmup_steps {config['lr_warmup_steps']}"
        
        if config.get('noise_offset', 0) > 0:
            command += f" --noise_offset {config['noise_offset']}"
        
        # 權重參數
        command += f" --visual_weight {weights['visual_weight']}"
        command += f" --fashion_clip_weight {weights['fashion_clip_weight']}"
        command += f" --color_weight {weights['color_weight']}"
        command += f" --text_encoder_lr {config['learning_rate'] * weights['text_encoder_weight']}"
        command += f" --unet_lr {config['learning_rate'] * weights['unet_weight']}"
        
        # 效能優化
        if config.get('cache_latents', True):
            command += " --cache_latents"
        if config.get('cache_text_encoder_outputs', True):
            command += " --cache_text_encoder_outputs"
        if config.get('enable_bucket', True):
            command += " --enable_bucket"
        if config.get('bucket_no_upscale', True):
            command += " --bucket_no_upscale"
        
        command += f" --bucket_reso_steps {config.get('bucket_reso_steps', 64)}"
        command += f" --min_bucket_reso {config.get('min_bucket_reso', 256)}"
        command += f" --max_bucket_reso {config.get('max_bucket_reso', 1024)}"
        
        return command
    
    def generate_optimization_report(self, plan: List[Dict[str, Any]], performance: Dict[str, float]) -> str:
        """生成優化報告"""
        report = f"""
# LoRA 調優優化報告
生成時間：{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 當前性能分析
- 總損失：{performance['total_loss']:.3f}
- 視覺相似度：{performance['visual_similarity']:.3f}
- FashionCLIP 相似度：{performance['fashion_clip_similarity']:.3f}
- 色彩相似度：{performance['color_similarity']:.3f}
- 整體分數：{performance['overall_score']:.3f}
- 訓練效率：{performance['training_efficiency']:.3f}

## 優化計劃（共 {len(plan)} 輪）
"""
        
        for i, iteration_plan in enumerate(plan):
            config = iteration_plan["config"]
            weights = iteration_plan["weights"]
            target = iteration_plan["target_performance"]
            
            report += f"""
### 第 {i + 1} 輪：{iteration_plan['description']}

**目標性能：**
- 總損失：{target['total_loss']:.3f}
- 視覺相似度：{target['visual_similarity']:.3f}
- FashionCLIP 相似度：{target['fashion_clip_similarity']:.3f}

**訓練參數：**
- 學習率：{config['learning_rate']}
- 訓練步數：{config['steps']}
- 網路維度：{config['network_dim']}
- 解析度：{config['resolution']}
- 調度器：{config['lr_scheduler']}

**損失權重：**
- 視覺權重：{weights['visual_weight']:.2f}
- FashionCLIP 權重：{weights['fashion_clip_weight']:.2f}
- 色彩權重：{weights['color_weight']:.2f}
"""
        
        report += f"""
## 執行建議
1. 按順序執行各輪配置
2. 每輪訓練後執行分析腳本
3. 根據結果調整下一輪參數
4. 保存每輪的最佳檢查點

## 成功標準
- 總損失 < 0.4
- 視覺相似度 > 0.5
- FashionCLIP 相似度 > 0.6
- 整體分數 > 0.7
"""
        
        return report

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="LoRA 調優自動化工具")
    parser.add_argument("--report", type=str, help="分析報告 JSON 文件路徑")
    parser.add_argument("--output_dir", type=str, default="optimization_configs", help="輸出目錄")
    parser.add_argument("--iterations", type=int, default=None, help="指定優化輪數")
    
    args = parser.parse_args()
    
    # 如果沒有指定報告，尋找最新的
    if not args.report:
        report_dir = "test_results"
        if os.path.exists(report_dir):
            json_files = [f for f in os.listdir(report_dir) if f.startswith("training_report_") and f.endswith(".json")]
            if json_files:
                json_files.sort(reverse=True)
                args.report = os.path.join(report_dir, json_files[0])
                print(f"📄 使用最新報告：{args.report}")
            else:
                print("❌ 找不到分析報告文件")
                return
        else:
            print("❌ 找不到 test_results 目錄")
            return
    
    # 創建優化器
    optimizer = LoRATuningOptimizer()
    
    # 載入報告
    report = optimizer.load_analysis_report(args.report)
    if not report:
        print("❌ 無法載入分析報告")
        return
    
    # 分析性能
    performance = optimizer.analyze_current_performance(report)
    print(f"📊 當前性能：總損失={performance['total_loss']:.3f}, 整體分數={performance['overall_score']:.3f}")
    
    # 創建優化計劃
    plan = optimizer.create_optimization_plan(performance)
    print(f"📋 優化計劃：{len(plan)} 輪調優")
    
    # 保存配置
    saved_files = optimizer.save_optimization_configs(plan, args.output_dir)
    print(f"💾 已保存 {len(saved_files)} 個配置文件到：{args.output_dir}")
    
    # 生成報告
    optimization_report = optimizer.generate_optimization_report(plan, performance)
    report_file = os.path.join(args.output_dir, f"optimization_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(optimization_report)
    
    print(f"📋 優化報告已保存：{report_file}")
    print("✅ LoRA 調優配置生成完成")
    
    # 顯示下一步建議
    print("\n🚀 執行建議：")
    for i, iteration_plan in enumerate(plan):
        config_file = [f for f in saved_files if f.endswith(f"iter_{i + 1}_*.json")][0]
        cmd_file = [f for f in saved_files if f.endswith(f"iter_{i + 1}_*.txt")][0]
        print(f"   {i + 1}. 使用配置：{config_file}")
        print(f"      命令文件：{cmd_file}")

if __name__ == "__main__":
    main()
