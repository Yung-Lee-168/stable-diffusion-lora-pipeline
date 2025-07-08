#!/usr/bin/env python3
"""
批量訓練助手 - 將大量圖片分批進行 LoRA 訓練
"""

import os
import shutil
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

class BatchTrainingHelper:
    def __init__(self, source_dir: str, batch_size: int = 50):
        self.source_dir = Path(source_dir)
        self.batch_size = batch_size
        self.work_dir = Path("batch_training_work")
        self.results_dir = Path("batch_training_results")
        
        # 創建工作目錄
        self.work_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
    def prepare_batches(self) -> List[Tuple[int, List[str]]]:
        """準備分批訓練的圖片列表"""
        if not self.source_dir.exists():
            raise FileNotFoundError(f"源目錄不存在: {self.source_dir}")
            
        # 獲取所有圖片文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            image_files.extend(self.source_dir.glob(f'*{ext}'))
            image_files.extend(self.source_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError(f"在 {self.source_dir} 中沒有找到圖片文件")
        
        print(f"📁 找到 {len(image_files)} 張圖片")
        
        # 分批
        batches = []
        for i in range(0, len(image_files), self.batch_size):
            batch_files = image_files[i:i + self.batch_size]
            batches.append((i // self.batch_size + 1, [str(f) for f in batch_files]))
        
        print(f"📊 將分成 {len(batches)} 批，每批約 {self.batch_size} 張")
        return batches
    
    def setup_batch_training_dir(self, batch_num: int, batch_files: List[str]) -> str:
        """設置批次訓練目錄"""
        batch_dir = self.work_dir / f"batch_{batch_num}"
        train_dir = batch_dir / "lora_train_set" / "10_test"
        
        # 清理並創建目錄
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # 複製圖片到批次目錄
        for img_path in batch_files:
            img_file = Path(img_path)
            if img_file.exists():
                shutil.copy2(img_file, train_dir / img_file.name)
        
        print(f"📁 批次 {batch_num} 準備完成: {len(batch_files)} 張圖片")
        return str(batch_dir)
    
    def run_batch_training(self, batch_num: int, batch_dir: str, continue_training: bool = False) -> dict:
        """執行批次訓練"""
        print(f"\n🚀 開始批次 {batch_num} 訓練...")
        
        # 切換到批次目錄
        original_cwd = os.getcwd()
        os.chdir(batch_dir)
        
        try:
            # 準備訓練命令
            script_path = Path(original_cwd) / "train_lora_monitored.py"
            
            cmd = [
                "python", str(script_path),
                "--new" if not continue_training else "--continue"
            ]
            
            print(f"📝 執行命令: {' '.join(cmd)}")
            
            # 執行訓練
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=14400  # 4小時超時
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # 檢查結果
            success = result.returncode == 0
            
            # 移動結果到結果目錄
            lora_output = Path("lora_output")
            batch_result_dir = self.results_dir / f"batch_{batch_num}"
            
            if lora_output.exists():
                if batch_result_dir.exists():
                    shutil.rmtree(batch_result_dir)
                shutil.move(str(lora_output), str(batch_result_dir))
            
            # 返回結果
            return {
                "batch_num": batch_num,
                "success": success,
                "training_time": training_time,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "result_dir": str(batch_result_dir) if batch_result_dir.exists() else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "batch_num": batch_num,
                "success": False,
                "training_time": 14400,
                "error": "訓練超時（4小時）",
                "returncode": -1
            }
        except Exception as e:
            return {
                "batch_num": batch_num,
                "success": False,
                "training_time": 0,
                "error": str(e),
                "returncode": -1
            }
        finally:
            os.chdir(original_cwd)
    
    def merge_models(self, batch_results: List[dict]) -> str:
        """合併多個批次的模型（簡單版本）"""
        print("\n🔄 合併批次訓練結果...")
        
        # 找到所有成功的批次
        successful_batches = [r for r in batch_results if r["success"] and r.get("result_dir")]
        
        if not successful_batches:
            print("❌ 沒有成功的批次可以合併")
            return ""
        
        # 創建合併結果目錄
        merged_dir = self.results_dir / "merged_result"
        merged_dir.mkdir(exist_ok=True)
        
        # 簡單合併：複製第一個成功的模型作為基礎
        # 在實際應用中，這裡需要更復雜的模型合併邏輯
        first_batch = successful_batches[0]
        first_batch_dir = Path(first_batch["result_dir"])
        
        for item in first_batch_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, merged_dir / item.name)
        
        print(f"📁 合併結果保存到: {merged_dir}")
        print("⚠️  注意：當前使用簡單合併策略，實際應用中需要更復雜的模型融合")
        
        return str(merged_dir)
    
    def run_full_batch_training(self, continue_training: bool = False) -> dict:
        """執行完整的批次訓練流程"""
        print("🚀 開始批次訓練流程...")
        
        # 準備批次
        batches = self.prepare_batches()
        
        # 執行所有批次
        batch_results = []
        total_start_time = time.time()
        
        for batch_num, batch_files in batches:
            print(f"\n{'='*60}")
            print(f"📦 處理批次 {batch_num}/{len(batches)}")
            print(f"{'='*60}")
            
            # 設置批次目錄
            batch_dir = self.setup_batch_training_dir(batch_num, batch_files)
            
            # 執行訓練
            result = self.run_batch_training(batch_num, batch_dir, continue_training)
            batch_results.append(result)
            
            # 顯示結果
            if result["success"]:
                print(f"✅ 批次 {batch_num} 完成 ({result['training_time']:.1f}秒)")
            else:
                print(f"❌ 批次 {batch_num} 失敗: {result.get('error', '未知錯誤')}")
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        # 統計結果
        successful_batches = [r for r in batch_results if r["success"]]
        success_rate = len(successful_batches) / len(batch_results) * 100
        
        print(f"\n{'='*60}")
        print("📊 批次訓練完成統計")
        print(f"{'='*60}")
        print(f"總批次數: {len(batch_results)}")
        print(f"成功批次: {len(successful_batches)}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"總時間: {total_time/3600:.1f} 小時")
        
        # 合併結果
        merged_result = ""
        if successful_batches:
            merged_result = self.merge_models(batch_results)
        
        # 保存完整報告
        report = {
            "total_batches": len(batch_results),
            "successful_batches": len(successful_batches),
            "success_rate": success_rate,
            "total_time": total_time,
            "batch_results": batch_results,
            "merged_result": merged_result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        report_file = self.results_dir / "batch_training_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 完整報告已保存: {report_file}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description="LoRA 批次訓練助手")
    parser.add_argument("source_dir", help="源圖片目錄")
    parser.add_argument("--batch-size", type=int, default=50, help="每批次圖片數量 (默認: 50)")
    parser.add_argument("--continue", action="store_true", help="從檢查點繼續訓練")
    
    args = parser.parse_args()
    
    # 檢查源目錄
    if not os.path.exists(args.source_dir):
        print(f"❌ 源目錄不存在: {args.source_dir}")
        return 1
    
    # 創建批次訓練助手
    helper = BatchTrainingHelper(args.source_dir, args.batch_size)
    
    try:
        # 執行批次訓練
        report = helper.run_full_batch_training(getattr(args, 'continue'))
        
        if report["success_rate"] > 0:
            print(f"\n🎉 批次訓練完成！成功率: {report['success_rate']:.1f}%")
            return 0
        else:
            print(f"\n❌ 批次訓練失敗！")
            return 1
            
    except Exception as e:
        print(f"❌ 批次訓練過程中發生錯誤: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
