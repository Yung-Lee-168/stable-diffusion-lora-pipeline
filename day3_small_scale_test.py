#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: Small Scale Test
小規模測試 - 驗證 FashionCLIP 核心功能與流程

🎯 測試目標：
- 驗證 FashionCLIP 模型載入與推理
- 測試特徵提取準確性
- 驗證 SD 圖片生成流程
- 測試相似度計算
- 驗證完整流程穩定性

📋 測試範圍：
- 單張圖片完整流程測試
- 特徵提取詳細驗證
- 生成品質評估
- 性能指標統計
"""

import os
import json
import time
from datetime import datetime
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import requests
import base64
import io

class Day3SmallScaleTest:
    def __init__(self):
        print("🧪 Day 3 小規模測試初始化...")
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }
        
        # 測試配置
        self.test_config = {
            "test_image": "day1_results/p (1).jpg",  # 選擇第一張圖片進行測試
            "output_dir": "day3_small_test_results",
            "api_url": "http://localhost:7860"
        }
        
        os.makedirs(self.test_config["output_dir"], exist_ok=True)
        
        # 載入模型
        self.init_fashion_clip()
        
        # 時尚分類定義
        self.categories = {
            "Gender": ["male", "female"],
            "Age": ["child", "teenager", "young adult", "adult", "senior"],
            "Season": ["spring", "summer", "autumn", "winter"],
            "Occasion": ["casual", "formal", "business", "sport", "party", "beach", "wedding", "date", "travel", "home"],
            "Upper Body": ["t-shirt", "shirt", "jacket", "coat", "sweater", "blazer", "hoodie", "tank top", "blouse", "dress"],
            "Lower Body": ["jeans", "trousers", "shorts", "skirt", "leggings", "cargo pants", "sweatpants", "culottes", "capris", "dress"]
        }
        
        self.detailed_features = {
            "Dress Style": ["A-line dress", "sheath dress", "wrap dress", "maxi dress", "midi dress", "mini dress", "bodycon dress", "shift dress", "empire waist dress", "fit and flare dress", "slip dress", "shirt dress", "sweater dress"],
            "Fabric Texture": ["cotton fabric", "silk material", "denim texture", "leather finish", "wool texture", "linen fabric", "chiffon material", "velvet texture", "knit fabric", "lace material", "satin finish", "corduroy texture"],
            "Pattern Details": ["solid color", "striped pattern", "floral print", "polka dots", "geometric pattern", "animal print", "plaid pattern", "paisley design", "abstract print", "tie-dye pattern", "checkered pattern"]
        }
    
    def init_fashion_clip(self):
        """初始化 FashionCLIP 模型"""
        print("📥 載入 FashionCLIP 模型...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        try:
            self.fashion_clip_model = CLIPModel.from_pretrained(
                "patrickjohncyh/fashion-clip",
                torch_dtype=torch_dtype
            ).to(device)
            self.fashion_clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            
            print(f"✅ FashionCLIP 載入成功 - 設備: {device}, 精度: {torch_dtype}")
            
            # 記錄模型信息
            self.test_results["model_info"] = {
                "device": device,
                "dtype": str(torch_dtype),
                "model_loaded": True
            }
            
        except Exception as e:
            print(f"❌ FashionCLIP 載入失敗: {e}")
            self.fashion_clip_model = None
            self.fashion_clip_processor = None
            self.test_results["model_info"] = {
                "model_loaded": False,
                "error": str(e)
            }
    
    def test_01_model_availability(self):
        """測試 1: 模型可用性檢查"""
        print("\n🧪 測試 1: 模型可用性檢查")
        test_result = {
            "test_name": "model_availability",
            "start_time": time.time(),
            "status": "running"
        }
        
        try:
            if self.fashion_clip_model is None:
                raise Exception("FashionCLIP 模型未載入")
            
            # 檢查模型參數
            total_params = sum(p.numel() for p in self.fashion_clip_model.parameters())
            trainable_params = sum(p.numel() for p in self.fashion_clip_model.parameters() if p.requires_grad)
            
            test_result.update({
                "status": "passed",
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_device": next(self.fashion_clip_model.parameters()).device.type,
                "model_dtype": str(next(self.fashion_clip_model.parameters()).dtype)
            })
            
            print(f"✅ 模型參數總數: {total_params:,}")
            print(f"✅ 可訓練參數: {trainable_params:,}")
            
        except Exception as e:
            test_result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"❌ 測試失敗: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results["tests"].append(test_result)
        return test_result["status"] == "passed"
    
    def test_02_image_loading(self):
        """測試 2: 圖片載入與預處理"""
        print("\n🧪 測試 2: 圖片載入與預處理")
        test_result = {
            "test_name": "image_loading",
            "start_time": time.time(),
            "status": "running"
        }
        
        try:
            image_path = self.test_config["test_image"]
            if not os.path.exists(image_path):
                raise Exception(f"測試圖片不存在: {image_path}")
            
            # 載入圖片
            self.test_image = Image.open(image_path).convert("RGB")
            
            test_result.update({
                "status": "passed",
                "image_path": image_path,
                "image_size": self.test_image.size,
                "image_mode": self.test_image.mode,
                "image_format": self.test_image.format
            })
            
            print(f"✅ 圖片載入成功: {self.test_image.size}")
            
        except Exception as e:
            test_result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"❌ 測試失敗: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results["tests"].append(test_result)
        return test_result["status"] == "passed"
    
    def test_03_feature_extraction(self):
        """測試 3: 特徵提取詳細測試"""
        print("\n🧪 測試 3: FashionCLIP 特徵提取")
        test_result = {
            "test_name": "feature_extraction",
            "start_time": time.time(),
            "status": "running",
            "categories_tested": 0,
            "categories_successful": 0,
            "feature_details": {}
        }
        
        try:
            device = next(self.fashion_clip_model.parameters()).device
            model_dtype = next(self.fashion_clip_model.parameters()).dtype
            
            all_categories = {**self.categories, **self.detailed_features}
            
            for category_name, labels in all_categories.items():
                test_result["categories_tested"] += 1
                
                try:
                    # 準備輸入
                    inputs = self.fashion_clip_processor(
                        text=labels, 
                        images=self.test_image, 
                        return_tensors="pt", 
                        padding=True
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 處理數據類型
                    if model_dtype == torch.float16:
                        for key in inputs:
                            if inputs[key].dtype == torch.float32:
                                inputs[key] = inputs[key].half()
                    
                    # 模型推理
                    start_inference = time.time()
                    with torch.no_grad():
                        outputs = self.fashion_clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)
                    inference_time = time.time() - start_inference
                    
                    # 獲取結果
                    top_indices = probs[0].topk(min(3, len(labels))).indices
                    top_labels = [labels[i] for i in top_indices]
                    top_scores = [probs[0][i].item() for i in top_indices]
                    
                    test_result["feature_details"][category_name] = {
                        "top_labels": top_labels,
                        "top_scores": top_scores,
                        "confidence": max(top_scores),
                        "inference_time": inference_time,
                        "num_labels": len(labels)
                    }
                    
                    test_result["categories_successful"] += 1
                    print(f"   ✅ {category_name}: {top_labels[0]} ({max(top_scores):.3f})")
                    
                except Exception as e:
                    test_result["feature_details"][category_name] = {
                        "error": str(e)
                    }
                    print(f"   ❌ {category_name}: 失敗 - {e}")
            
            # 計算成功率
            success_rate = test_result["categories_successful"] / test_result["categories_tested"]
            test_result.update({
                "status": "passed" if success_rate > 0.8 else "partial",
                "success_rate": success_rate,
                "total_inference_time": sum(
                    detail.get("inference_time", 0) 
                    for detail in test_result["feature_details"].values()
                    if "inference_time" in detail
                )
            })
            
            print(f"✅ 特徵提取成功率: {success_rate:.2%}")
            
        except Exception as e:
            test_result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"❌ 測試失敗: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results["tests"].append(test_result)
        return test_result["status"] in ["passed", "partial"]
    
    def test_04_prompt_generation(self):
        """測試 4: 提示詞生成"""
        print("\n🧪 測試 4: 提示詞生成")
        test_result = {
            "test_name": "prompt_generation",
            "start_time": time.time(),
            "status": "running"
        }
        
        try:
            # 從上一個測試獲取特徵
            feature_test = next(
                (t for t in self.test_results["tests"] if t["test_name"] == "feature_extraction"), 
                None
            )
            
            if not feature_test or "feature_details" not in feature_test:
                raise Exception("無法獲取特徵提取結果")
            
            features = feature_test["feature_details"]
            
            # 生成結構化特徵
            def get_best_feature(category_name, default=""):
                if category_name in features and "top_labels" in features[category_name]:
                    return features[category_name]["top_labels"][0]
                return default
            
            # 提取基本特徵
            gender = get_best_feature("Gender", "person")
            age = get_best_feature("Age", "adult")
            upper_body = get_best_feature("Upper Body", "clothing")
            lower_body = get_best_feature("Lower Body", "")
            occasion = get_best_feature("Occasion", "casual")
            
            # 構建提示詞
            prompt_parts = []
            
            # 人物描述
            if gender != "person":
                person_desc = f"{age} {gender}"
            else:
                person_desc = age
            prompt_parts.append(person_desc)
            
            # 服裝描述
            clothing_desc = f"wearing {upper_body}"
            if lower_body and lower_body != upper_body:
                clothing_desc += f" and {lower_body}"
            prompt_parts.append(clothing_desc)
            
            # 場合
            if occasion:
                prompt_parts.append(occasion)
            
            # 品質標籤
            prompt_parts.extend([
                "high quality", "detailed", "professional photography", 
                "fashion photography", "studio lighting"
            ])
            
            final_prompt = ", ".join(prompt_parts)
            negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy"
            
            test_result.update({
                "status": "passed",
                "prompt": final_prompt,
                "negative_prompt": negative_prompt,
                "prompt_length": len(final_prompt),
                "components": {
                    "gender": gender,
                    "age": age,
                    "upper_body": upper_body,
                    "lower_body": lower_body,
                    "occasion": occasion
                }
            })
            
            print(f"✅ 生成提示詞: {final_prompt}")
            
        except Exception as e:
            test_result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"❌ 測試失敗: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results["tests"].append(test_result)
        return test_result["status"] == "passed"
    
    def test_05_sd_generation(self):
        """測試 5: Stable Diffusion 圖片生成"""
        print("\n🧪 測試 5: SD 圖片生成")
        test_result = {
            "test_name": "sd_generation",
            "start_time": time.time(),
            "status": "running"
        }
        
        try:
            # 從提示詞測試獲取提示詞
            prompt_test = next(
                (t for t in self.test_results["tests"] if t["test_name"] == "prompt_generation"), 
                None
            )
            
            if not prompt_test or "prompt" not in prompt_test:
                raise Exception("無法獲取提示詞")
            
            # SD API 請求
            payload = {
                "prompt": prompt_test["prompt"],
                "negative_prompt": prompt_test["negative_prompt"],
                "steps": 20,  # 測試用較少步數
                "cfg_scale": 7.5,
                "width": 512,
                "height": 512,
                "sampler_name": "DPM++ 2M Karras",
                "batch_size": 1,
                "n_iter": 1
            }
            
            print("🎨 向 SD API 發送請求...")
            response = requests.post(
                f"{self.test_config['api_url']}/sdapi/v1/txt2img",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("images"):
                    # 保存生成的圖片
                    img_data = base64.b64decode(result["images"][0])
                    self.generated_image = Image.open(io.BytesIO(img_data))
                    
                    output_path = os.path.join(
                        self.test_config["output_dir"], 
                        "test_generated_image.png"
                    )
                    self.generated_image.save(output_path)
                    
                    test_result.update({
                        "status": "passed",
                        "generated_image_path": output_path,
                        "generated_image_size": self.generated_image.size,
                        "api_response_time": time.time() - test_result["start_time"]
                    })
                    
                    print(f"✅ 圖片生成成功: {output_path}")
                else:
                    raise Exception("API 響應中沒有圖片數據")
            else:
                raise Exception(f"API 請求失敗: {response.status_code}")
            
        except Exception as e:
            test_result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"❌ 測試失敗: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results["tests"].append(test_result)
        return test_result["status"] == "passed"
    
    def test_06_similarity_calculation(self):
        """測試 6: 相似度計算"""
        print("\n🧪 測試 6: 相似度計算")
        test_result = {
            "test_name": "similarity_calculation",
            "start_time": time.time(),
            "status": "running"
        }
        
        try:
            if not hasattr(self, 'generated_image'):
                raise Exception("沒有生成的圖片進行比較")
            
            # FashionCLIP 相似度計算
            device = next(self.fashion_clip_model.parameters()).device
            model_dtype = next(self.fashion_clip_model.parameters()).dtype
            
            inputs = self.fashion_clip_processor(
                images=[self.generated_image, self.test_image], 
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 確保數據類型一致
            if model_dtype == torch.float16:
                for key in inputs:
                    if inputs[key].dtype == torch.float32:
                        inputs[key] = inputs[key].half()
            
            with torch.no_grad():
                image_features = self.fashion_clip_model.get_image_features(**inputs)
                from sklearn.metrics.pairwise import cosine_similarity
                fashion_similarity = cosine_similarity(
                    image_features[0:1].cpu().numpy(), 
                    image_features[1:2].cpu().numpy()
                )[0][0]
            
            test_result.update({
                "status": "passed",
                "fashion_clip_similarity": float(fashion_similarity),
                "similarity_threshold_met": fashion_similarity > 0.3  # 設定閾值
            })
            
            print(f"✅ FashionCLIP 相似度: {fashion_similarity:.3f}")
            
        except Exception as e:
            test_result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"❌ 測試失敗: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results["tests"].append(test_result)
        return test_result["status"] == "passed"
    
    def run_all_tests(self):
        """執行所有測試"""
        print("🚀 開始 Day 3 小規模測試...")
        print("=" * 60)
        
        # 測試序列
        tests = [
            ("模型可用性", self.test_01_model_availability),
            ("圖片載入", self.test_02_image_loading),
            ("特徵提取", self.test_03_feature_extraction),
            ("提示詞生成", self.test_04_prompt_generation),
            ("SD 圖片生成", self.test_05_sd_generation),
            ("相似度計算", self.test_06_similarity_calculation)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            if test_func():
                passed_tests += 1
            
            # 短暫暫停避免資源衝突
            time.sleep(1)
        
        # 生成測試摘要
        self.generate_test_summary(passed_tests, total_tests)
        
        # 保存測試結果
        self.save_test_results()
        
        return passed_tests == total_tests
    
    def generate_test_summary(self, passed_tests, total_tests):
        """生成測試摘要"""
        print(f"\n📊 測試摘要")
        print("=" * 40)
        
        success_rate = passed_tests / total_tests
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "overall_status": "PASSED" if success_rate == 1.0 else "PARTIAL" if success_rate >= 0.7 else "FAILED"
        }
        
        print(f"✅ 通過測試: {passed_tests}/{total_tests}")
        print(f"📈 成功率: {success_rate:.1%}")
        print(f"🎯 整體狀態: {self.test_results['summary']['overall_status']}")
        
        # 詳細測試結果
        for test in self.test_results["tests"]:
            status_icon = "✅" if test["status"] == "passed" else "⚠️" if test["status"] == "partial" else "❌"
            print(f"{status_icon} {test['test_name']}: {test['status']} ({test['duration']:.2f}s)")
    
    def save_test_results(self):
        """保存測試結果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 結果
        json_path = os.path.join(
            self.test_config["output_dir"], 
            f"small_scale_test_results_{timestamp}.json"
        )
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 測試結果已保存: {json_path}")
        
        # 生成簡要報告
        report_path = os.path.join(
            self.test_config["output_dir"], 
            f"test_report_{timestamp}.md"
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Day 3 小規模測試報告\n\n")
            f.write(f"**測試時間**: {self.test_results['timestamp']}\n")
            f.write(f"**成功率**: {self.test_results['summary']['success_rate']:.1%}\n")
            f.write(f"**整體狀態**: {self.test_results['summary']['overall_status']}\n\n")
            
            f.write("## 測試詳情\n\n")
            for test in self.test_results["tests"]:
                f.write(f"### {test['test_name']}\n")
                f.write(f"- 狀態: {test['status']}\n")
                f.write(f"- 耗時: {test['duration']:.2f}s\n")
                if "error" in test:
                    f.write(f"- 錯誤: {test['error']}\n")
                f.write("\n")
        
        print(f"📋 測試報告已生成: {report_path}")

def main():
    """主測試函數"""
    print("🧪 Day 3: 小規模測試")
    print("專業 FashionCLIP 系統核心功能驗證")
    print("=" * 50)
    
    # 執行測試
    tester = Day3SmallScaleTest()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 所有測試通過！系統準備就緒。")
    else:
        print("\n⚠️ 部分測試未通過，請檢查具體問題。")
    
    return success

if __name__ == "__main__":
    main()
