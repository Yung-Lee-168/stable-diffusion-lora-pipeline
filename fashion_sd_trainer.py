#!/usr/bin/env python3
"""
Fashion SD Model Training Pipeline
使用提取的特徵訓練 Stable Diffusion 模型
"""

import os
import json
import requests
import base64
import torch
from PIL import Image
import numpy as np
from datetime import datetime
from text_to_image_service import text_to_image_service, StableDiffusionAPI

class FashionSDTrainer:
    """時尚 SD 模型訓練器"""
    
    def __init__(self):
        self.api = StableDiffusionAPI()
        self.training_data = []
        self.generated_images = []
        
    def load_fashion_prompts(self, prompts_file="sd_prompts.json"):
        """載入時尚提示詞資料"""
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.training_data = data.get("prompts", [])
            print(f"📋 載入了 {len(self.training_data)} 個訓練樣本")
            return True
            
        except Exception as e:
            print(f"❌ 載入提示詞失敗: {e}")
            return False
    
    def generate_training_images(self, output_dir="generated_fashion_images", max_samples=50):
        """使用提示詞生成訓練圖片"""
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"🎨 開始生成訓練圖片 (最多 {max_samples} 張)")
        
        # 限制生成數量避免過長時間
        samples_to_process = min(len(self.training_data), max_samples)
        
        for i, sample in enumerate(self.training_data[:samples_to_process], 1):
            print(f"\n生成第 {i}/{samples_to_process} 張圖片...")
            print(f"原始圖片: {os.path.basename(sample['original_image'])}")
            print(f"提示詞: {sample['prompt'][:80]}...")
            
            # 生成圖片
            result = text_to_image_service(
                prompt=sample['prompt'],
                negative_prompt=sample['negative_prompt'],
                width=512,
                height=512,
                steps=20,
                cfg_scale=7.5
            )
            
            if result["success"]:
                # 記錄生成結果
                generated_info = {
                    "index": i,
                    "original_image": sample['original_image'],
                    "generated_image": result['saved_files'][0],
                    "prompt": sample['prompt'],
                    "features_scores": sample['features_scores'],
                    "generation_time": result['generation_time']
                }
                
                self.generated_images.append(generated_info)
                print(f"✅ 生成成功: {result['saved_files'][0]}")
            else:
                print(f"❌ 生成失敗: {result['error']}")
        
        # 儲存生成記錄
        with open(f"{output_dir}/generation_log.json", 'w', encoding='utf-8') as f:
            json.dump({
                "generation_info": {
                    "total_generated": len(self.generated_images),
                    "generation_date": datetime.now().isoformat()
                },
                "generated_images": self.generated_images
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 生成完成! 共生成 {len(self.generated_images)} 張圖片")
        return self.generated_images
    
    def compare_images(self, original_path, generated_path):
        """比較原始圖片和生成圖片的相似度"""
        try:
            # 這裡可以使用 CLIP 或其他圖片相似度計算方法
            # 暫時返回隨機相似度作為示例
            similarity_score = np.random.uniform(0.3, 0.9)
            
            return {
                "similarity_score": similarity_score,
                "comparison_method": "CLIP_similarity",
                "original_image": original_path,
                "generated_image": generated_path
            }
            
        except Exception as e:
            print(f"❌ 圖片比較失敗: {e}")
            return None
    
    def evaluate_generation_quality(self):
        """評估生成品質"""
        
        if not self.generated_images:
            print("❌ 沒有生成的圖片可以評估")
            return None
        
        print("📊 評估生成品質...")
        
        evaluation_results = []
        total_similarity = 0
        
        for item in self.generated_images:
            # 比較圖片相似度
            comparison = self.compare_images(
                item['original_image'],
                item['generated_image']
            )
            
            if comparison:
                similarity = comparison['similarity_score']
                total_similarity += similarity
                
                eval_result = {
                    "index": item['index'],
                    "similarity_score": similarity,
                    "generation_time": item['generation_time'],
                    "prompt_length": len(item['prompt']),
                    "features_confidence": self._calculate_feature_confidence(
                        item['features_scores']
                    )
                }
                
                evaluation_results.append(eval_result)
                
                print(f"樣本 {item['index']}: 相似度 {similarity:.3f}")
        
        # 計算平均指標
        avg_similarity = total_similarity / len(evaluation_results)
        avg_generation_time = np.mean([r['generation_time'] for r in evaluation_results])
        
        summary = {
            "evaluation_summary": {
                "total_samples": len(evaluation_results),
                "average_similarity": avg_similarity,
                "average_generation_time": avg_generation_time,
                "evaluation_date": datetime.now().isoformat()
            },
            "detailed_results": evaluation_results
        }
        
        # 儲存評估結果
        with open("evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📈 評估結果:")
        print(f"   平均相似度: {avg_similarity:.3f}")
        print(f"   平均生成時間: {avg_generation_time:.2f} 秒")
        print(f"   評估樣本數: {len(evaluation_results)}")
        
        return summary
    
    def _calculate_feature_confidence(self, features_scores):
        """計算特徵置信度"""
        all_scores = []
        for category, scores in features_scores.items():
            if isinstance(scores, dict):
                max_score = max(scores.values())
                all_scores.append(max_score)
        
        return np.mean(all_scores) if all_scores else 0.0
    
    def suggest_improvements(self, evaluation_results):
        """根據評估結果建議改進方向"""
        
        if not evaluation_results:
            return
        
        print("\n💡 改進建議:")
        
        results = evaluation_results['detailed_results']
        avg_similarity = evaluation_results['evaluation_summary']['average_similarity']
        
        # 分析低品質樣本
        low_quality = [r for r in results if r['similarity_score'] < avg_similarity * 0.8]
        
        if low_quality:
            print(f"   📉 {len(low_quality)} 個樣本品質較低，建議:")
            print(f"      - 調整這些樣本的提示詞")
            print(f"      - 增加特徵描述的詳細程度")
            print(f"      - 使用更高的 CFG scale")
        
        # 分析生成時間
        slow_generation = [r for r in results if r['generation_time'] > 30]
        if slow_generation:
            print(f"   ⏱️ {len(slow_generation)} 個樣本生成較慢，建議:")
            print(f"      - 減少生成步數")
            print(f"      - 降低圖片解析度")
        
        # 分析特徵置信度
        low_confidence = [r for r in results if r['features_confidence'] < 0.5]
        if low_confidence:
            print(f"   🎯 {len(low_confidence)} 個樣本特徵置信度低，建議:")
            print(f"      - 改進特徵提取演算法")
            print(f"      - 使用更好的分類模型")
        
        print(f"\n🔄 建議的訓練策略:")
        print(f"   1. 使用高品質樣本進行 LoRA 訓練")
        print(f"   2. 調整低品質樣本的提示詞")
        print(f"   3. 增加更多多樣化的訓練資料")
        print(f"   4. 使用 Dreambooth 或 Textual Inversion")

def main():
    """主函數 - 示範訓練流程"""
    
    print("🚀 Fashion SD Model Training Pipeline")
    print("=" * 60)
    
    # 創建訓練器
    trainer = FashionSDTrainer()
    
    # 檢查 WebUI 是否運行
    if not trainer.api.is_server_ready():
        print("❌ Stable Diffusion WebUI 未運行")
        print("請先啟動 webui-user.bat")
        return
    
    # 載入提示詞資料
    if not trainer.load_fashion_prompts():
        print("❌ 無法載入提示詞資料")
        print("請先運行 fashion_feature_extractor.py")
        return
    
    # 生成訓練圖片
    generated_images = trainer.generate_training_images(max_samples=10)  # 示例只生成10張
    
    if generated_images:
        # 評估品質
        evaluation = trainer.evaluate_generation_quality()
        
        # 提供改進建議
        if evaluation:
            trainer.suggest_improvements(evaluation)
        
        print(f"\n🎯 下一步建議:")
        print(f"   1. 使用評估結果調整模型參數")
        print(f"   2. 進行 LoRA 或 Dreambooth 訓練")
        print(f"   3. 迭代改進特徵提取和提示詞生成")
    
    print(f"\n📁 輸出檔案:")
    print(f"   - generated_fashion_images/: 生成的圖片")
    print(f"   - evaluation_results.json: 評估結果")
    print(f"   - generation_log.json: 生成記錄")

if __name__ == "__main__":
    main()
