#!/usr/bin/env python3
"""
3天快速可行性測試計劃
Day 1: 基礎設置和手動測試
Day 2: 自動化 Pipeline
Day 3: 評估和總結
"""

import os
import json
from datetime import datetime

def create_3day_plan():
    """創建3天測試計劃"""
    
    plan = {
        "project_timeline": "3 Days Feasibility Test",
        "start_date": datetime.now().isoformat(),
        
        "day_1": {
            "title": "基礎設置和概念驗證",
            "duration": "8 小時",
            "goals": [
                "環境設置完成",
                "手動測試 5-10 張圖片",
                "驗證基本 pipeline 可行性"
            ],
            "tasks": [
                {
                    "task": "環境準備",
                    "time": "1 小時",
                    "details": "安裝 CLIP、設置資料夾、準備測試圖片"
                },
                {
                    "task": "手動特徵標註",
                    "time": "2 小時", 
                    "details": "為 10 張時尚圖片手動標註特徵"
                },
                {
                    "task": "提示詞生成測試",
                    "time": "2 小時",
                    "details": "根據特徵生成 SD 提示詞，測試生成效果"
                },
                {
                    "task": "初步比較",
                    "time": "2 小時",
                    "details": "肉眼比較原圖和生成圖，記錄觀察"
                },
                {
                    "task": "問題識別",
                    "time": "1 小時",
                    "details": "記錄遇到的問題和改進方向"
                }
            ]
        },
        
        "day_2": {
            "title": "自動化流程實作",
            "duration": "8 小時",
            "goals": [
                "CLIP 特徵提取自動化",
                "批次生成測試",
                "建立評估指標"
            ],
            "tasks": [
                {
                    "task": "實作簡化版 CLIP 特徵提取",
                    "time": "3 小時",
                    "details": "使用 CLIP 自動分析圖片特徵"
                },
                {
                    "task": "批次生成 pipeline",
                    "time": "2 小時",
                    "details": "自動化生成 20-30 張圖片"
                },
                {
                    "task": "建立評估方法",
                    "time": "2 小時",
                    "details": "CLIP 相似度計算、人工評分表"
                },
                {
                    "task": "資料記錄",
                    "time": "1 小時",
                    "details": "整理實驗資料和結果"
                }
            ]
        },
        
        "day_3": {
            "title": "評估和結論",
            "duration": "6 小時",
            "goals": [
                "量化評估結果",
                "識別改進方向",
                "制定後續計劃"
            ],
            "tasks": [
                {
                    "task": "量化分析",
                    "time": "2 小時",
                    "details": "計算相似度、統計成功率"
                },
                {
                    "task": "定性評估",
                    "time": "2 小時",
                    "details": "分析失敗案例、識別問題模式"
                },
                {
                    "task": "可行性報告",
                    "time": "1.5 小時",
                    "details": "撰寫測試報告和建議"
                },
                {
                    "task": "後續計劃",
                    "time": "0.5 小時",
                    "details": "制定完整實作計劃"
                }
            ]
        },
        
        "success_criteria": {
            "minimum_viable": [
                "能夠提取基本服裝特徵",
                "生成的圖片與原圖有相似的服裝類型",
                "至少 30% 的生成圖片在視覺上相關"
            ],
            "ideal_outcome": [
                "特徵提取準確度 > 70%",
                "生成圖片風格一致性 > 60%", 
                "整個 pipeline 可以自動運行"
            ]
        },
        
        "risk_mitigation": {
            "time_shortage": "專注核心功能，跳過美化",
            "technical_issues": "準備備用方案（手動標註）",
            "quality_issues": "降低期望，專注可行性驗證"
        }
    }
    
    return plan

def create_day1_script():
    """Day 1: 快速概念驗證腳本"""
    
    script_content = '''#!/usr/bin/env python3
"""
Day 1: 快速概念驗證
使用最簡單的方法測試可行性
"""

import os
import json
import requests
import base64
from datetime import datetime
from text_to_image_service import text_to_image_service

class QuickFashionTest:
    """快速時尚測試"""
    
    def __init__(self):
        self.test_data = []
        
    def manual_feature_annotation(self, image_path):
        """手動特徵標註 (Day 1 快速方法)"""
        
        print(f"\\n📸 分析圖片: {os.path.basename(image_path)}")
        print("請根據圖片內容回答以下問題:")
        
        # 簡化的手動標註
        features = {}
        
        # 性別
        gender = input("性別 (male/female/unisex): ").strip().lower()
        features["gender"] = gender if gender in ["male", "female", "unisex"] else "unisex"
        
        # 上衣
        top = input("上衣類型 (shirt/t-shirt/jacket/sweater/other): ").strip().lower()
        features["top"] = top if top else "shirt"
        
        # 下身
        bottom = input("下身類型 (jeans/trousers/skirt/dress/shorts/other): ").strip().lower()
        features["bottom"] = bottom if bottom else "jeans"
        
        # 風格
        style = input("風格 (casual/formal/sporty/elegant/other): ").strip().lower()
        features["style"] = style if style else "casual"
        
        # 顏色
        colors = input("主要顏色 (多個用逗號分隔): ").strip()
        features["colors"] = colors if colors else "neutral"
        
        return features
    
    def generate_prompt_from_features(self, features):
        """從特徵生成提示詞"""
        
        prompt_parts = []
        
        # 構建基本描述
        if features.get("gender") == "female":
            prompt_parts.append("woman")
        elif features.get("gender") == "male":
            prompt_parts.append("man")
        else:
            prompt_parts.append("person")
        
        # 添加服裝
        if features.get("top"):
            prompt_parts.append(f"wearing {features['top']}")
        
        if features.get("bottom"):
            prompt_parts.append(f"and {features['bottom']}")
        
        # 添加風格
        if features.get("style"):
            prompt_parts.append(f"{features['style']} style")
        
        # 添加顏色
        if features.get("colors"):
            prompt_parts.append(f"{features['colors']} colors")
        
        # 組合提示詞
        main_prompt = " ".join(prompt_parts)
        main_prompt += ", fashion photography, high quality, detailed, studio lighting"
        
        negative_prompt = "blurry, low quality, distorted, deformed, multiple people"
        
        return main_prompt, negative_prompt
    
    def test_single_image(self, image_path):
        """測試單張圖片"""
        
        print(f"\\n{'='*60}")
        print(f"測試圖片: {image_path}")
        print(f"{'='*60}")
        
        # 1. 手動特徵標註
        features = self.manual_feature_annotation(image_path)
        
        # 2. 生成提示詞
        prompt, negative_prompt = self.generate_prompt_from_features(features)
        print(f"\\n生成的提示詞: {prompt}")
        
        # 3. 生成圖片
        print("\\n🎨 生成圖片...")
        result = text_to_image_service(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=512,
            steps=20
        )
        
        # 4. 記錄結果
        test_record = {
            "original_image": image_path,
            "features": features,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "timestamp": datetime.now().isoformat()
        }
        
        if result["success"]:
            test_record["generated_image"] = result["saved_files"][0]
            test_record["generation_time"] = result["generation_time"]
            test_record["status"] = "success"
            print(f"✅ 生成成功: {result['saved_files'][0]}")
        else:
            test_record["error"] = result["error"]
            test_record["status"] = "failed"
            print(f"❌ 生成失敗: {result['error']}")
        
        self.test_data.append(test_record)
        
        # 5. 人工評估
        if result["success"]:
            print("\\n👀 請打開生成的圖片進行比較")
            similarity = input("相似度評分 (1-10, 10最相似): ").strip()
            try:
                test_record["human_similarity_score"] = int(similarity)
            except:
                test_record["human_similarity_score"] = 5
        
        return test_record
    
    def run_day1_test(self, test_images_dir="test_images"):
        """執行 Day 1 測試"""
        
        print("🚀 Day 1: 快速可行性測試")
        print("=" * 60)
        
        # 檢查測試圖片
        if not os.path.exists(test_images_dir):
            os.makedirs(test_images_dir)
            print(f"⚠️ 請將 5-10 張時尚圖片放入 {test_images_dir}/ 目錄")
            return
        
        image_files = [f for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"⚠️ 在 {test_images_dir}/ 中沒有找到圖片檔案")
            return
        
        print(f"📋 找到 {len(image_files)} 張測試圖片")
        
        # 測試每張圖片
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(test_images_dir, image_file)
            print(f"\\n📊 進度: {i}/{len(image_files)}")
            
            try:
                self.test_single_image(image_path)
            except KeyboardInterrupt:
                print("\\n⏹️ 測試中斷")
                break
            except Exception as e:
                print(f"❌ 測試失敗: {e}")
        
        # 儲存結果
        self.save_day1_results()
    
    def save_day1_results(self):
        """儲存 Day 1 結果"""
        
        results = {
            "test_info": {
                "test_date": datetime.now().isoformat(),
                "test_type": "Day 1 Manual Feasibility Test",
                "total_samples": len(self.test_data)
            },
            "results": self.test_data
        }
        
        with open("day1_test_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 簡單統計
        successful = [r for r in self.test_data if r.get("status") == "success"]
        if successful:
            avg_similarity = sum(r.get("human_similarity_score", 0) for r in successful) / len(successful)
            avg_time = sum(r.get("generation_time", 0) for r in successful) / len(successful)
            
            print(f"\\n📊 Day 1 測試總結:")
            print(f"   總測試數: {len(self.test_data)}")
            print(f"   成功生成: {len(successful)}")
            print(f"   成功率: {len(successful)/len(self.test_data)*100:.1f}%")
            print(f"   平均相似度: {avg_similarity:.1f}/10")
            print(f"   平均生成時間: {avg_time:.1f} 秒")
            
            if avg_similarity >= 6:
                print("\\n🎉 初步結果良好，建議繼續 Day 2 測試")
            else:
                print("\\n⚠️ 需要改進提示詞生成方法")
        
        print(f"\\n💾 結果已保存: day1_test_results.json")

def main():
    """Day 1 主程式"""
    
    tester = QuickFashionTest()
    tester.run_day1_test()

if __name__ == "__main__":
    main()
'''
    
    return script_content

def create_day2_script():
    """Day 2: 自動化測試腳本"""
    
    script_content = '''#!/usr/bin/env python3
"""
Day 2: 自動化特徵提取和批次測試
"""

import os
import json
import torch
import clip
import requests
from PIL import Image
from datetime import datetime
from text_to_image_service import text_to_image_service

class AutomatedFashionTest:
    """自動化時尚測試"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.test_results = []
        
        # 簡化的特徵詞彙
        self.feature_vocabulary = {
            "clothing_type": ["dress", "shirt", "t-shirt", "jacket", "sweater", "jeans", "trousers", "skirt"],
            "style": ["casual", "formal", "sporty", "elegant", "vintage"],
            "color": ["black", "white", "blue", "red", "green", "brown", "gray", "colorful"]
        }
    
    def extract_features_auto(self, image_path):
        """自動特徵提取"""
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            features = {}
            
            # 對每個特徵類別計算相似度
            for category, words in self.feature_vocabulary.items():
                text_prompts = [f"a photo of {word} clothing" for word in words]
                text_inputs = clip.tokenize(text_prompts).to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    text_features = self.model.encode_text(text_inputs)
                    
                    similarities = torch.cosine_similarity(image_features, text_features, dim=1)
                    
                    # 取最高分的特徵
                    best_idx = similarities.argmax().item()
                    best_score = similarities[best_idx].item()
                    
                    features[category] = {
                        "value": words[best_idx],
                        "confidence": best_score
                    }
            
            return features
            
        except Exception as e:
            print(f"❌ 特徵提取失敗: {e}")
            return None
    
    def generate_prompt_auto(self, features):
        """自動生成提示詞"""
        
        prompt_parts = ["person wearing"]
        
        # 添加服裝類型
        clothing = features.get("clothing_type", {})
        if clothing.get("confidence", 0) > 0.3:
            prompt_parts.append(clothing["value"])
        
        # 添加風格
        style = features.get("style", {})
        if style.get("confidence", 0) > 0.3:
            prompt_parts.append(f"{style['value']} style")
        
        # 添加顏色
        color = features.get("color", {})
        if color.get("confidence", 0) > 0.3:
            prompt_parts.append(f"{color['value']} color")
        
        prompt = " ".join(prompt_parts)
        prompt += ", fashion photography, high quality, detailed"
        
        negative_prompt = "blurry, low quality, distorted, multiple people"
        
        return prompt, negative_prompt
    
    def calculate_clip_similarity(self, image1_path, image2_path):
        """計算兩張圖片的 CLIP 相似度"""
        
        try:
            # 載入圖片
            image1 = Image.open(image1_path).convert('RGB')
            image2 = Image.open(image2_path).convert('RGB')
            
            # 預處理
            image1_input = self.preprocess(image1).unsqueeze(0).to(self.device)
            image2_input = self.preprocess(image2).unsqueeze(0).to(self.device)
            
            # 計算特徵
            with torch.no_grad():
                features1 = self.model.encode_image(image1_input)
                features2 = self.model.encode_image(image2_input)
                
                # 計算相似度
                similarity = torch.cosine_similarity(features1, features2, dim=1)
                return similarity.item()
                
        except Exception as e:
            print(f"❌ 相似度計算失敗: {e}")
            return 0.0
    
    def test_image_automated(self, image_path):
        """自動化測試單張圖片"""
        
        print(f"\\n🔍 自動分析: {os.path.basename(image_path)}")
        
        # 1. 自動特徵提取
        features = self.extract_features_auto(image_path)
        if not features:
            return None
        
        print("📊 提取的特徵:")
        for category, data in features.items():
            print(f"   {category}: {data['value']} (信心度: {data['confidence']:.3f})")
        
        # 2. 生成提示詞
        prompt, negative_prompt = self.generate_prompt_auto(features)
        print(f"\\n📝 生成的提示詞: {prompt}")
        
        # 3. 生成圖片
        result = text_to_image_service(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=512,
            steps=20
        )
        
        test_record = {
            "original_image": image_path,
            "features": features,
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        }
        
        if result["success"]:
            generated_path = result["saved_files"][0]
            test_record["generated_image"] = generated_path
            test_record["generation_time"] = result["generation_time"]
            
            # 4. 計算相似度
            similarity = self.calculate_clip_similarity(image_path, generated_path)
            test_record["clip_similarity"] = similarity
            test_record["status"] = "success"
            
            print(f"✅ 生成成功: {generated_path}")
            print(f"🎯 CLIP 相似度: {similarity:.3f}")
        else:
            test_record["error"] = result["error"]
            test_record["status"] = "failed"
            print(f"❌ 生成失敗: {result['error']}")
        
        return test_record
    
    def run_day2_test(self, test_images_dir="test_images"):
        """執行 Day 2 自動化測試"""
        
        print("🤖 Day 2: 自動化特徵提取測試")
        print("=" * 60)
        
        image_files = [f for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"⚠️ 在 {test_images_dir}/ 中沒有找到圖片")
            return
        
        print(f"📋 處理 {len(image_files)} 張圖片")
        
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(test_images_dir, image_file)
            print(f"\\n📊 進度: {i}/{len(image_files)}")
            
            try:
                result = self.test_image_automated(image_path)
                if result:
                    self.test_results.append(result)
            except Exception as e:
                print(f"❌ 處理失敗: {e}")
        
        self.save_day2_results()
    
    def save_day2_results(self):
        """儲存 Day 2 結果"""
        
        results = {
            "test_info": {
                "test_date": datetime.now().isoformat(),
                "test_type": "Day 2 Automated Feature Extraction",
                "total_samples": len(self.test_results)
            },
            "results": self.test_results
        }
        
        with open("day2_test_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 統計分析
        successful = [r for r in self.test_results if r.get("status") == "success"]
        
        if successful:
            similarities = [r["clip_similarity"] for r in successful]
            avg_similarity = sum(similarities) / len(similarities)
            
            # 特徵信心度統計
            confidence_scores = []
            for r in successful:
                for feature_data in r["features"].values():
                    confidence_scores.append(feature_data["confidence"])
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            print(f"\\n📊 Day 2 測試總結:")
            print(f"   總測試數: {len(self.test_results)}")
            print(f"   成功生成: {len(successful)}")
            print(f"   平均 CLIP 相似度: {avg_similarity:.3f}")
            print(f"   平均特徵信心度: {avg_confidence:.3f}")
            
            good_results = len([s for s in similarities if s > 0.6])
            print(f"   高相似度樣本 (>0.6): {good_results}/{len(successful)}")
            
            if avg_similarity > 0.5:
                print("\\n🎉 自動化方法顯示良好潛力")
            else:
                print("\\n⚠️ 需要改進特徵提取或提示詞生成")
        
        print(f"\\n💾 結果已保存: day2_test_results.json")

def main():
    """Day 2 主程式"""
    
    tester = AutomatedFashionTest()
    tester.run_day2_test()

if __name__ == "__main__":
    main()
'''
    
    return script_content

def main():
    """建立3天測試計劃"""
    
    print("⚡ 3天快速可行性測試計劃")
    print("=" * 60)
    
    # 創建計劃
    plan = create_3day_plan()
    
    # 儲存計劃
    with open("3day_feasibility_plan.json", 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    
    # 創建 Day 1 腳本
    day1_script = create_day1_script()
    with open("day1_quick_test.py", 'w', encoding='utf-8') as f:
        f.write(day1_script)
    
    # 創建 Day 2 腳本
    day2_script = create_day2_script()
    with open("day2_automated_test.py", 'w', encoding='utf-8') as f:
        f.write(day2_script)
    
    # 顯示計劃
    print("📋 測試計劃概覽:")
    for day_key in ["day_1", "day_2", "day_3"]:
        day_info = plan[day_key]
        print(f"\\n{day_info['title']} ({day_info['duration']}):")
        for goal in day_info['goals']:
            print(f"   🎯 {goal}")
    
    print(f"\\n📁 已創建檔案:")
    print(f"   📄 3day_feasibility_plan.json - 詳細計劃")
    print(f"   🐍 day1_quick_test.py - Day 1 測試腳本")
    print(f"   🐍 day2_automated_test.py - Day 2 測試腳本")
    
    print(f"\\n🚀 立即開始:")
    print(f"   1. 創建 test_images/ 資料夾")
    print(f"   2. 放入 5-10 張時尚圖片")
    print(f"   3. 執行: python day1_quick_test.py")
    
    print(f"\\n⚠️ 成功標準:")
    for criteria in plan["success_criteria"]["minimum_viable"]:
        print(f"   ✓ {criteria}")

if __name__ == "__main__":
    main()
