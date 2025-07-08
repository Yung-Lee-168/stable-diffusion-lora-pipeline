#!/usr/bin/env python3
"""
LoRA 調優完整流程腳本
整合訓練、推理、分析、優化和監控的完整自動化流程

功能：
1. 自動化 LoRA 訓練流程
2. 推理測試圖片生成
3. 結果分析與指標計算
4. 參數自動優化
5. 監控與預警
6. 多輪迭代調優
"""

import os
import sys
import json
import time
import datetime
import subprocess
import argparse
import shutil
from typing import Dict, List, Any, Optional
import threading
from pathlib import Path

class LoRAOptimizationPipeline:
    """LoRA 調優完整流程"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.pipeline_dir = self.base_dir / "auto_test_pipeline"
        self.results_dir = self.pipeline_dir / "test_results"
        self.configs_dir = self.pipeline_dir / "optimization_configs"
        self.models_dir = self.base_dir / "models" / "Lora"
        
        # 創建必要目錄
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        # 流程狀態
        self.current_iteration = 0
        self.max_iterations = 5
        self.target_metrics = {
            "total_loss": 0.4,
            "visual_similarity": 0.5,
            "fashion_clip_similarity": 0.6,
            "overall_score": 0.7
        }
        
        # 歷史記錄
        self.iteration_history = []
        self.best_iteration = None
        self.best_score = 0.0
        
        # 腳本路徑
        self.scripts = {
            "train": self.pipeline_dir / "train_lora.py",
            "infer": self.pipeline_dir / "infer_lora.py",
            "analyze": self.pipeline_dir / "analyze_results.py",
            "optimize": self.pipeline_dir / "lora_tuning_optimizer.py",
            "monitor": self.pipeline_dir / "lora_tuning_monitor.py"
        }
    
    def check_environment(self) -> bool:
        """檢查環境和依賴"""
        print("🔍 檢查環境...")
        
        # 檢查腳本存在性
        missing_scripts = []
        for name, script_path in self.scripts.items():
            if not script_path.exists():
                missing_scripts.append(f"{name}: {script_path}")
        
        if missing_scripts:
            print("❌ 缺少必要腳本：")
            for script in missing_scripts:
                print(f"   {script}")
            return False
        
        # 檢查訓練數據
        train_data_dir = self.pipeline_dir / "lora_train_set" / "10_test"
        if not train_data_dir.exists():
            print(f"❌ 找不到訓練數據目錄：{train_data_dir}")
            return False
        
        # 檢查圖片數量
        image_files = list(train_data_dir.glob("*.jpg"))
        if len(image_files) < 5:
            print(f"❌ 訓練圖片不足，需要至少 5 張，當前只有 {len(image_files)} 張")
            return False
        
        print(f"✅ 環境檢查通過，訓練數據：{len(image_files)} 張圖片")
        return True
    
    def run_script(self, script_name: str, args: List[str] = None) -> tuple:
        """執行腳本"""
        if script_name not in self.scripts:
            return False, f"未知腳本：{script_name}"
        
        script_path = self.scripts[script_name]
        command = [sys.executable, str(script_path)]
        
        if args:
            command.extend(args)
        
        print(f"🚀 執行：{' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
                timeout=1800  # 30分鐘超時
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "腳本執行超時"
        except Exception as e:
            return False, f"執行錯誤：{str(e)}"
    
    def load_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """載入最新分析結果"""
        try:
            json_files = list(self.results_dir.glob("training_report_*.json"))
            if not json_files:
                return None
            
            # 按時間排序
            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_file = json_files[0]
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 載入分析結果失敗：{e}")
            return None
    
    def extract_performance_metrics(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """提取性能指標"""
        metrics = {
            "total_loss": 1.0,
            "visual_similarity": 0.0,
            "fashion_clip_similarity": 0.0,
            "color_similarity": 0.0,
            "overall_score": 0.0,
            "training_efficiency": 0.0
        }
        
        # 從 benchmark_analysis 提取
        if "benchmark_analysis" in analysis:
            benchmark = analysis["benchmark_analysis"]
            avg_metrics = benchmark.get("average_metrics", {})
            
            metrics["total_loss"] = avg_metrics.get("avg_total_loss", 1.0)
            metrics["visual_similarity"] = avg_metrics.get("avg_visual_similarity", 0.0)
            metrics["fashion_clip_similarity"] = avg_metrics.get("avg_fashion_clip_similarity", 0.0)
            metrics["color_similarity"] = avg_metrics.get("avg_color_similarity", 0.0)
        
        # 從 lora_tuning 提取
        if "lora_tuning" in analysis:
            lora_tuning = analysis["lora_tuning"]
            metrics["overall_score"] = lora_tuning.get("overall_tuning_score", 0.0)
            
            if "training_efficiency" in lora_tuning:
                metrics["training_efficiency"] = lora_tuning["training_efficiency"].get("score", 0.0)
        
        return metrics
    
    def check_convergence(self, metrics: Dict[str, float]) -> bool:
        """檢查是否達到收斂條件"""
        conditions = [
            metrics["total_loss"] <= self.target_metrics["total_loss"],
            metrics["visual_similarity"] >= self.target_metrics["visual_similarity"],
            metrics["fashion_clip_similarity"] >= self.target_metrics["fashion_clip_similarity"],
            metrics["overall_score"] >= self.target_metrics["overall_score"]
        ]
        
        return sum(conditions) >= 3  # 至少滿足3個條件
    
    def load_optimization_config(self, iteration: int) -> Optional[Dict[str, Any]]:
        """載入優化配置"""
        try:
            config_files = list(self.configs_dir.glob(f"lora_config_iter_{iteration}_*.json"))
            if not config_files:
                return None
            
            # 取最新的配置
            config_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            config_file = config_files[0]
            
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 載入優化配置失敗：{e}")
            return None
    
    def backup_best_model(self, iteration: int):
        """備份最佳模型"""
        try:
            # 找到最新的模型文件
            model_files = list(self.models_dir.glob("*.safetensors"))
            if not model_files:
                return
            
            # 按時間排序
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_model = model_files[0]
            
            # 備份到結果目錄
            backup_name = f"best_model_iter_{iteration}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.safetensors"
            backup_path = self.results_dir / backup_name
            
            shutil.copy2(latest_model, backup_path)
            print(f"💾 最佳模型已備份：{backup_path}")
            
        except Exception as e:
            print(f"❌ 備份模型失敗：{e}")
    
    def run_single_iteration(self, iteration: int, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """執行單次迭代"""
        print(f"\n{'='*60}")
        print(f"🔄 開始第 {iteration} 輪調優")
        print(f"{'='*60}")
        
        iteration_start_time = time.time()
        iteration_result = {
            "iteration": iteration,
            "start_time": datetime.datetime.now().isoformat(),
            "config": config,
            "steps": [],
            "metrics": {},
            "success": False,
            "error": None
        }
        
        # 步驟1：訓練
        print(f"\n🎯 步驟1：LoRA 訓練...")
        train_args = []
        if config and "training_config" in config:
            train_config = config["training_config"]
            train_args.extend([
                "--learning_rate", str(train_config.get("learning_rate", 0.0005)),
                "--max_train_steps", str(train_config.get("steps", 100)),
                "--resolution", str(train_config.get("resolution", "512x512"))
            ])
        
        success, output = self.run_script("train", train_args)
        iteration_result["steps"].append({
            "step": "train",
            "success": success,
            "output": output[:500] if output else None,
            "error": output if not success else None
        })
        
        if not success:
            print(f"❌ 訓練失敗：{output}")
            iteration_result["error"] = f"訓練失敗：{output}"
            return iteration_result
        
        print("✅ 訓練完成")
        
        # 步驟2：推理
        print(f"\n🎨 步驟2：推理測試...")
        success, output = self.run_script("infer")
        iteration_result["steps"].append({
            "step": "infer",
            "success": success,
            "output": output[:500] if output else None,
            "error": output if not success else None
        })
        
        if not success:
            print(f"❌ 推理失敗：{output}")
            iteration_result["error"] = f"推理失敗：{output}"
            return iteration_result
        
        print("✅ 推理完成")
        
        # 步驟3：分析
        print(f"\n📊 步驟3：結果分析...")
        success, output = self.run_script("analyze")
        iteration_result["steps"].append({
            "step": "analyze",
            "success": success,
            "output": output[:500] if output else None,
            "error": output if not success else None
        })
        
        if not success:
            print(f"❌ 分析失敗：{output}")
            iteration_result["error"] = f"分析失敗：{output}"
            return iteration_result
        
        print("✅ 分析完成")
        
        # 載入分析結果
        analysis = self.load_latest_analysis()
        if analysis:
            metrics = self.extract_performance_metrics(analysis)
            iteration_result["metrics"] = metrics
            
            print(f"\n📈 第 {iteration} 輪性能指標：")
            print(f"   總損失：{metrics['total_loss']:.3f}")
            print(f"   視覺相似度：{metrics['visual_similarity']:.3f}")
            print(f"   FashionCLIP 相似度：{metrics['fashion_clip_similarity']:.3f}")
            print(f"   整體分數：{metrics['overall_score']:.3f}")
            
            # 檢查是否為最佳結果
            if metrics["overall_score"] > self.best_score:
                self.best_score = metrics["overall_score"]
                self.best_iteration = iteration
                self.backup_best_model(iteration)
                print(f"🎉 新的最佳結果！整體分數：{self.best_score:.3f}")
        
        # 步驟4：生成下一輪優化配置
        if iteration < self.max_iterations:
            print(f"\n🔧 步驟4：生成優化配置...")
            success, output = self.run_script("optimize")
            iteration_result["steps"].append({
                "step": "optimize",
                "success": success,
                "output": output[:500] if output else None,
                "error": output if not success else None
            })
            
            if not success:
                print(f"⚠️ 優化配置生成失敗：{output}")
            else:
                print("✅ 優化配置生成完成")
        
        iteration_result["success"] = True
        iteration_result["duration"] = time.time() - iteration_start_time
        iteration_result["end_time"] = datetime.datetime.now().isoformat()
        
        return iteration_result
    
    def run_complete_optimization(self, max_iterations: int = 5) -> Dict[str, Any]:
        """執行完整的優化流程"""
        print(f"🚀 開始 LoRA 調優完整流程")
        print(f"📋 最大迭代次數：{max_iterations}")
        print(f"🎯 目標指標：{self.target_metrics}")
        
        self.max_iterations = max_iterations
        optimization_start_time = time.time()
        
        # 總結報告
        optimization_report = {
            "start_time": datetime.datetime.now().isoformat(),
            "max_iterations": max_iterations,
            "target_metrics": self.target_metrics,
            "iterations": [],
            "best_iteration": None,
            "best_score": 0.0,
            "convergence_achieved": False,
            "total_duration": 0.0,
            "final_metrics": {}
        }
        
        # 執行迭代
        for iteration in range(1, max_iterations + 1):
            # 載入優化配置
            config = None
            if iteration > 1:
                config = self.load_optimization_config(iteration)
            
            # 執行單次迭代
            iteration_result = self.run_single_iteration(iteration, config)
            optimization_report["iterations"].append(iteration_result)
            
            # 檢查是否成功
            if not iteration_result["success"]:
                print(f"❌ 第 {iteration} 輪失敗，停止優化")
                break
            
            # 檢查收斂
            if "metrics" in iteration_result and iteration_result["metrics"]:
                metrics = iteration_result["metrics"]
                if self.check_convergence(metrics):
                    optimization_report["convergence_achieved"] = True
                    optimization_report["final_metrics"] = metrics
                    print(f"🎉 達到收斂條件，提前結束優化")
                    break
            
            # 等待一段時間再進行下一輪
            if iteration < max_iterations:
                print(f"\n⏳ 等待 10 秒後開始第 {iteration + 1} 輪...")
                time.sleep(10)
        
        # 完成優化
        optimization_report["total_duration"] = time.time() - optimization_start_time
        optimization_report["end_time"] = datetime.datetime.now().isoformat()
        optimization_report["best_iteration"] = self.best_iteration
        optimization_report["best_score"] = self.best_score
        
        # 保存優化報告
        report_file = self.results_dir / f"optimization_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"🎉 LoRA 調優完成")
        print(f"{'='*60}")
        print(f"📊 總迭代次數：{len(optimization_report['iterations'])}")
        print(f"🏆 最佳迭代：第 {self.best_iteration} 輪")
        print(f"⭐ 最佳分數：{self.best_score:.3f}")
        print(f"🎯 是否收斂：{'是' if optimization_report['convergence_achieved'] else '否'}")
        print(f"⏱️ 總耗時：{optimization_report['total_duration']:.1f} 秒")
        print(f"📋 詳細報告：{report_file}")
        
        return optimization_report
    
    def generate_final_summary(self, report: Dict[str, Any]) -> str:
        """生成最終總結"""
        summary = f"""
# LoRA 調優完整流程總結報告

## 基本信息
- 開始時間：{report['start_time']}
- 結束時間：{report['end_time']}
- 總耗時：{report['total_duration']:.1f} 秒
- 最大迭代次數：{report['max_iterations']}
- 實際迭代次數：{len(report['iterations'])}

## 優化結果
- 最佳迭代：第 {report['best_iteration']} 輪
- 最佳分數：{report['best_score']:.3f}
- 是否收斂：{'是' if report['convergence_achieved'] else '否'}

## 目標達成情況
"""
        
        if report.get("final_metrics"):
            metrics = report["final_metrics"]
            target = report["target_metrics"]
            
            for metric, value in metrics.items():
                if metric in target:
                    target_value = target[metric]
                    achieved = value >= target_value if metric != "total_loss" else value <= target_value
                    status = "✅" if achieved else "❌"
                    summary += f"- {metric}: {value:.3f} / {target_value:.3f} {status}\n"
        
        summary += f"""
## 迭代歷史
| 輪次 | 總損失 | 視覺相似度 | FashionCLIP | 整體分數 | 狀態 |
|------|--------|------------|-------------|----------|------|
"""
        
        for iteration in report["iterations"]:
            metrics = iteration.get("metrics", {})
            status = "✅" if iteration["success"] else "❌"
            summary += f"| {iteration['iteration']} | {metrics.get('total_loss', 0):.3f} | {metrics.get('visual_similarity', 0):.3f} | {metrics.get('fashion_clip_similarity', 0):.3f} | {metrics.get('overall_score', 0):.3f} | {status} |\n"
        
        summary += f"""
## 建議
"""
        
        if report["convergence_achieved"]:
            summary += "🎉 調優成功達到收斂條件，建議使用最佳迭代的模型進行部署。\n"
        else:
            summary += "⚠️ 調優未達到收斂條件，建議：\n"
            summary += "1. 檢查訓練數據質量\n"
            summary += "2. 調整目標指標\n"
            summary += "3. 增加迭代次數\n"
            summary += "4. 優化超參數設定\n"
        
        return summary

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="LoRA 調優完整流程")
    parser.add_argument("--max_iterations", type=int, default=5, help="最大迭代次數")
    parser.add_argument("--target_loss", type=float, default=0.4, help="目標總損失")
    parser.add_argument("--target_visual", type=float, default=0.5, help="目標視覺相似度")
    parser.add_argument("--target_fashion", type=float, default=0.6, help="目標FashionCLIP相似度")
    parser.add_argument("--target_overall", type=float, default=0.7, help="目標整體分數")
    parser.add_argument("--base_dir", type=str, default=".", help="基礎目錄")
    
    args = parser.parse_args()
    
    # 創建流程實例
    pipeline = LoRAOptimizationPipeline(args.base_dir)
    
    # 設定目標指標
    pipeline.target_metrics = {
        "total_loss": args.target_loss,
        "visual_similarity": args.target_visual,
        "fashion_clip_similarity": args.target_fashion,
        "overall_score": args.target_overall
    }
    
    # 檢查環境
    if not pipeline.check_environment():
        print("❌ 環境檢查失敗，無法繼續")
        return
    
    # 執行完整優化
    try:
        report = pipeline.run_complete_optimization(args.max_iterations)
        
        # 生成最終總結
        summary = pipeline.generate_final_summary(report)
        summary_file = pipeline.results_dir / f"final_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n📋 最終總結已保存：{summary_file}")
        
    except KeyboardInterrupt:
        print("\n🛑 收到中斷信號，正在停止...")
    except Exception as e:
        print(f"❌ 流程執行失敗：{e}")

if __name__ == "__main__":
    main()
