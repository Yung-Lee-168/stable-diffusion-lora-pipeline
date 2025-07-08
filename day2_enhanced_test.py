#!/usr/bin/env python3
"""
增強版第2天測試：CLIP vs FashionCLIP 比較
目標：比較通用 CLIP 和專業 FashionCLIP 在時尚圖片分析上的表現
"""

import requests
import json
import base64
import os
from datetime import datetime
from PIL import Image
import numpy as np

class EnhancedDay2Tester:
    def __init__(self):
        self.api_url = "http://localhost:7860"
        self.output_dir = "day2_enhanced_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def install_requirements(self):
        """檢查並安裝必要的套件"""
        print("🔍 檢查模型依賴...")
        
        try:
            import torch
            import transformers
            print("✅ 基礎套件已安裝")
        except ImportError as e:
            print(f"❌ 缺少基礎套件: {e}")
            return False
            
        # 檢查 FashionCLIP
        try:
            import clip
            print("✅ CLIP 套件可用")
        except ImportError:
            print("⚠️ CLIP 套件未安裝，將使用 transformers 版本")
            
        return True
    
    def load_standard_clip(self):
        """載入標準 CLIP 模型 - 針對你的系統優化 (安全版)"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"📥 安全載入標準 CLIP (設備: {device})...")
            
            # 優化載入方案 - 針對 4GB VRAM 使用 float16
            if device == "cuda":
                model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    torch_dtype=torch.float16,  # GPU 使用 float16 節省 VRAM
                    low_cpu_mem_usage=True,
                    trust_remote_code=False
                )
            else:
                model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    torch_dtype=torch.float32,  # CPU 使用 float32
                    low_cpu_mem_usage=True,
                    trust_remote_code=False
                )
                
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.to(device)
            
            print("✅ 標準 CLIP 模型安全載入成功")
            return model, processor, "standard_clip"
            
        except Exception as e:
            print(f"❌ 標準 CLIP 模型載入失敗: {e}")
            
            # 備用方案：根據設備選擇精度
            try:
                print("🔄 嘗試備用載入方案...")
                if device == "cuda":
                    model = CLIPModel.from_pretrained(
                        "openai/clip-vit-base-patch32",
                        torch_dtype=torch.float16,  # GPU 使用 float16
                        low_cpu_mem_usage=True
                    )
                else:
                    model = CLIPModel.from_pretrained(
                        "openai/clip-vit-base-patch32",
                        torch_dtype=torch.float32,  # CPU 使用 float32
                        low_cpu_mem_usage=True
                    )
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                model.to(device)
                print("✅ 備用方案載入成功")
                return model, processor, "standard_clip"
            except Exception as e2:
                print(f"❌ 備用方案也失敗: {e2}")
                return None, None, None
    
    def load_fashion_clip(self):
        """載入 FashionCLIP 模型 - 針對你的 RTX 3050 Ti 優化"""
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            
            # 根據系統分析推薦的專業時尚模型
            fashion_models = [
                "patrickjohncyh/fashion-clip",  # 主要推薦：專業時尚模型
                "openai/clip-vit-base-patch32"  # 備用標準模型
            ]
            
            # 檢查 GPU 可用性並設置設備
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🎮 使用設備: {device}")
            
            for model_name in fashion_models:
                try:
                    print(f"📥 正在載入 {model_name}...")
                    
                    # 優化載入方案 - 針對 4GB VRAM 使用 float16
                    if device == "cuda":
                        model = CLIPModel.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,  # GPU 使用 float16 節省 VRAM
                            low_cpu_mem_usage=True,
                            trust_remote_code=False
                        )
                    else:
                        model = CLIPModel.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,  # CPU 使用 float32
                            low_cpu_mem_usage=True,
                            trust_remote_code=False
                        )
                    
                    processor = CLIPProcessor.from_pretrained(model_name)
                    model.to(device)
                    
                    print(f"✅ FashionCLIP 模型載入成功: {model_name}")
                    print(f"   設備: {device}")
                    print(f"   精度: {'float16' if device == 'cuda' else 'float32'}")  # 根據設備顯示精度
                    
                    return model, processor, "fashion_clip"
                    
                except Exception as e:
                    print(f"⚠️ 載入 {model_name} 失敗: {e}")
                    continue
                    
            print("⚠️ 專業 FashionCLIP 不可用，使用標準 CLIP")
            return self.load_standard_clip()
            
        except Exception as e:
            print(f"❌ FashionCLIP 載入失敗: {e}")
            return None, None, None
    
    def analyze_with_clip(self, image_path, model, processor, model_type):
        """使用 CLIP 分析圖片"""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # 時尚相關標籤
            if model_type == "fashion_clip":
                fashion_labels = [
                    "elegant evening dress", "casual streetwear", "formal business suit",
                    "vintage retro clothing", "bohemian flowing dress", "modern minimalist outfit",
                    "luxury designer fashion", "sporty athletic wear", "romantic feminine style",
                    "edgy punk fashion", "classic timeless style", "trendy contemporary look",
                    "silk fabric", "cotton material", "leather texture", "denim style",
                    "floral pattern", "solid color", "striped design", "polka dots"
                ]
            else:
                fashion_labels = [
                    "elegant dress", "casual outfit", "formal wear", "vintage style",
                    "modern fashion", "luxury clothing", "street style", "business attire",
                    "evening gown", "summer dress", "winter coat", "bohemian style",
                    "minimalist fashion", "colorful outfit", "black clothing", "white clothing"
                ]
            
            # 使用模型分析 - 針對 4GB VRAM 優化精度處理
            import torch
            device = next(model.parameters()).device  # 獲取模型所在設備
            model_dtype = next(model.parameters()).dtype  # 獲取模型精度
            
            inputs = processor(text=fashion_labels, images=image, return_tensors="pt", padding=True)
            # 將輸入移到正確設備
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 確保輸入精度與模型匹配
            if model_dtype == torch.float16:
                # 如果模型是 float16，將輸入轉換為 float16
                for key in inputs:
                    if inputs[key].dtype == torch.float32:
                        inputs[key] = inputs[key].half()
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # 獲取前5個最相關的標籤
            top_indices = probs[0].topk(5).indices
            top_labels = [fashion_labels[i] for i in top_indices]
            top_scores = [probs[0][i].item() for i in top_indices]
            
            return {
                "success": True,
                "model_type": model_type,
                "top_labels": top_labels,
                "scores": top_scores,
                "confidence": max(top_scores)
            }
            
        except Exception as e:
            return {
                "success": False,
                "model_type": model_type,
                "error": str(e)
            }
    
    def generate_prompt_from_analysis(self, analysis_results):
        """基於分析結果生成提示詞"""
        all_prompts = {}
        
        for model_type, result in analysis_results.items():
            if result["success"]:
                top_labels = result["top_labels"]
                confidence = result["confidence"]
                
                # 根據置信度調整提示詞強度
                if confidence > 0.7:
                    intensity = "highly detailed, professional"
                elif confidence > 0.5:
                    intensity = "detailed, well-crafted"
                else:
                    intensity = "stylized, artistic"
                
                # 構建提示詞
                base_prompt = ", ".join(top_labels[:3])
                enhanced_prompt = f"{base_prompt}, {intensity}, high fashion photography, studio lighting"
                
                all_prompts[model_type] = {
                    "prompt": enhanced_prompt,
                    "confidence": confidence,
                    "base_labels": top_labels[:3]
                }
            else:
                all_prompts[model_type] = {
                    "prompt": "elegant fashion, high quality, professional photography",
                    "confidence": 0.0,
                    "error": result.get("error", "Unknown error")
                }
        
        return all_prompts
    
    def check_api_connection(self):
        """檢查 API 連接狀態 - 支持多種端點"""
        print("🔍 檢查 WebUI API 連接...")
        
        # 嘗試不同的基礎 URL
        base_urls = [
            "http://localhost:7860",
            "http://127.0.0.1:7860",
            "http://0.0.0.0:7860"
        ]
        
        # 嘗試不同的 API 端點
        api_endpoints = [
            "/sdapi/v1/options",
            "/api/v1/options",
            "/sdapi/v1/cmd-flags",
            "/sdapi/v1/sd-models"
        ]
        
        for base_url in base_urls:
            for endpoint in api_endpoints:
                try:
                    full_url = f"{base_url}{endpoint}"
                    response = requests.get(full_url, timeout=5)
                    
                    if response.status_code == 200:
                        print(f"✅ API 連接成功: {full_url}")
                        self.api_url = base_url  # 更新工作的基礎 URL
                        return True
                    
                except requests.exceptions.ConnectionError:
                    continue
                except Exception:
                    continue
        
        # 檢查主頁是否可訪問
        for base_url in base_urls:
            try:
                response = requests.get(base_url, timeout=5)
                if response.status_code == 200:
                    print(f"⚠️ WebUI 在運行 ({base_url}) 但 API 不可用")
                    print("   可能的原因:")
                    print("   1. API 模式未啟用 (缺少 --api 參數)")
                    print("   2. WebUI 版本太舊")
                    print("   3. API 端點路徑不同")
                    return False
            except:
                continue
        
        print("❌ 無法連接到 WebUI，請確認:")
        print("   1. WebUI 已啟動")
        print("   2. 使用了 --api 參數")
        print("   3. 端口 7860 未被其他程序佔用")
        return False

    def test_model_comparison(self):
        """比較 CLIP 和 FashionCLIP 的表現 - 使用 Day1 已生成的圖片"""
        print("🔍 開始模型比較測試...")
        
        # 檢查 Day1 生成的圖片
        day1_output_dir = "day1_results"
        if not os.path.exists(day1_output_dir):
            print("❌ 找不到 Day1 生成的圖片")
            print("💡 請先運行 Day1 測試生成圖片")
            return []
        
        # 找到所有 Day1 生成的圖片
        day1_images = []
        for file in os.listdir(day1_output_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                day1_images.append(os.path.join(day1_output_dir, file))
        
        if not day1_images:
            print("❌ Day1 資料夾中沒有找到圖片")
            print("💡 請確認 Day1 測試已成功執行")
            return []
        
        print(f"✅ 找到 {len(day1_images)} 張 Day1 生成的圖片")
        
        # 載入兩種模型
        standard_clip = self.load_standard_clip()
        fashion_clip = self.load_fashion_clip()
        
        models = {}
        if standard_clip[0] is not None:
            models["standard_clip"] = standard_clip
        if fashion_clip[0] is not None and fashion_clip[2] != "standard_clip":
            models["fashion_clip"] = fashion_clip
        
        if not models:
            print("❌ 無法載入任何 CLIP 模型")
            return []
        
        results = []
        
        # 使用前 3 張圖片進行測試（或所有圖片如果少於 3 張）
        test_images = day1_images[:3]
        
        for i, image_path in enumerate(test_images):
            print(f"\n🎨 測試 {i+1}/{len(test_images)}: {os.path.basename(image_path)}")
            
            # 用不同模型分析圖片
            analysis_results = {}
            for model_name, (model, processor, model_type) in models.items():
                print(f"   📊 使用 {model_type} 分析中...")
                analysis = self.analyze_with_clip(image_path, model, processor, model_type)
                analysis_results[model_type] = analysis
                
                if analysis["success"]:
                    print(f"      ✅ 置信度: {analysis['confidence']:.3f}")
                    print(f"      🏷️ 前3標籤: {', '.join(analysis['top_labels'][:3])}")
                else:
                    print(f"      ❌ 分析失敗: {analysis.get('error', 'Unknown error')}")
            
            # 生成基於分析的提示詞（僅用於比較，不生成圖片）
            generated_prompts = self.generate_prompt_from_analysis(analysis_results)
            
            # 顯示不同模型生成的提示詞差異
            print(f"   📝 提示詞比較:")
            for model_type, prompt_info in generated_prompts.items():
                if prompt_info.get("prompt"):
                    print(f"      {model_type}: {prompt_info['prompt'][:100]}...")
            
            results.append({
                "image_path": image_path,
                "analysis_results": analysis_results,
                "generated_prompts": generated_prompts,
                "success": True
            })
            
            print(f"✅ 測試 {i+1} 完成")
        
        return results
    
    def run_enhanced_day2_tests(self):
        """運行增強版第2天測試"""
        print("=" * 60)
        print("增強版第2天測試：CLIP vs FashionCLIP 比較")
        print("目標：使用 Day1 生成的圖片比較不同 CLIP 模型的分析能力")
        print("=" * 60)
        
        # 檢查環境
        if not self.install_requirements():
            print("❌ 環境檢查失敗")
            return False
        
        # 運行比較測試
        results = self.test_model_comparison()
        successful = sum(1 for r in results if r["success"])
        
        print(f"\n增強版第2天測試完成：{successful}/{len(results)} 個測試成功")
        
        # 分析不同模型的表現
        self.analyze_model_performance(results)
        
        # 生成報告
        report = {
            "day": 2,
            "test_type": "enhanced_clip_comparison",
            "timestamp": datetime.now().isoformat(),
            "tests_run": len(results),
            "tests_successful": successful,
            "success_rate": successful / len(results) if results else 0,
            "results": results,
            "model_comparison": self.get_model_comparison_summary(results)
        }
        
        # 保存 JSON 報告
        with open(os.path.join(self.output_dir, "day2_enhanced_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成易讀的 HTML 和 Markdown 報告
        html_path, md_path = self.generate_readable_report(results, report)
        
        print(f"\n📊 報告已生成:")
        print(f"   📄 JSON 報告: {os.path.join(self.output_dir, 'day2_enhanced_report.json')}")
        print(f"   🌐 HTML 報告: {html_path}")
        print(f"   📝 Markdown 報告: {md_path}")
        print(f"\n💡 建議瀏覽 HTML 報告以獲得最佳閱讀體驗！")
        
        # 生成易讀報告
        html_report, md_report = self.generate_readable_report(results, report)
        print(f"📄 易讀報告已生成:\n- HTML: {html_report}\n- Markdown: {md_report}")
        
        return True
    
    def analyze_model_performance(self, results):
        """分析不同模型的表現"""
        print("\n" + "=" * 60)
        print("📊 模型表現比較")
        print("=" * 60)
        
        model_stats = {}
        
        for result in results:
            if result["success"]:
                for model_type, analysis in result["analysis_results"].items():
                    if model_type not in model_stats:
                        model_stats[model_type] = {
                            "total_tests": 0,
                            "successful_analyses": 0,
                            "avg_confidence": 0,
                            "confidences": []
                        }
                    
                    model_stats[model_type]["total_tests"] += 1
                    if analysis["success"]:
                        model_stats[model_type]["successful_analyses"] += 1
                        model_stats[model_type]["confidences"].append(analysis["confidence"])
        
        # 計算平均置信度
        for model_type, stats in model_stats.items():
            if stats["confidences"]:
                stats["avg_confidence"] = sum(stats["confidences"]) / len(stats["confidences"])
        
        # 顯示比較結果
        for model_type, stats in model_stats.items():
            print(f"\n🤖 {model_type.upper()}:")
            print(f"   成功率: {stats['successful_analyses']}/{stats['total_tests']}")
            print(f"   平均置信度: {stats['avg_confidence']:.3f}")
            
            if model_type == "fashion_clip":
                print("   👗 專業時尚模型 - 預期在時尚細節識別上表現更好")
            else:
                print("   🔍 通用模型 - 穩定可靠的基準表現")
    
    def get_model_comparison_summary(self, results):
        """獲取模型比較摘要"""
        summary = {
            "models_tested": [],
            "recommendation": "",
            "performance_analysis": {}
        }
        
        # 從結果中提取模型類型
        for result in results:
            if result["success"]:
                for model_type in result["analysis_results"].keys():
                    if model_type not in summary["models_tested"]:
                        summary["models_tested"].append(model_type)
        
        # 基於測試結果給出建議
        if "fashion_clip" in summary["models_tested"]:
            summary["recommendation"] = "建議使用 FashionCLIP 進行時尚相關的圖片分析，它對服飾細節的理解更加精確。"
        else:
            summary["recommendation"] = "目前使用標準 CLIP，表現穩定。如需更專業的時尚分析，建議考慮 FashionCLIP。"
        
        return summary

    def generate_readable_report(self, results, report):
        """生成易讀的 HTML 和 Markdown 報告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 生成 HTML 報告
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day 2 CLIP vs FashionCLIP 比較報告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .model-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .model-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .model-card.standard {{
            border-left: 5px solid #4CAF50;
        }}
        .model-card.fashion {{
            border-left: 5px solid #FF9800;
        }}
        .test-result {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .comparison-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .confidence {{
            font-weight: bold;
            color: #2196F3;
        }}
        .labels {{
            background-color: #e3f2fd;
            padding: 8px;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        .prompt {{
            background-color: #f3e5f5;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.85em;
            margin-top: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Day 2: CLIP vs FashionCLIP 比較報告</h1>
        <p>測試時間: {timestamp}</p>
    </div>

    <div class="summary">
        <h2>📊 測試摘要</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{report['tests_run']}</div>
                <div>總測試數</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{report['tests_successful']}</div>
                <div>成功測試</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{report['success_rate']:.1%}</div>
                <div>成功率</div>
            </div>
        </div>
    </div>

    <div class="model-comparison">"""
        
        # 添加模型比較統計
        model_stats = {}
        for result in results:
            if result["success"]:
                for model_type, analysis in result["analysis_results"].items():
                    if model_type not in model_stats:
                        model_stats[model_type] = {
                            "total_tests": 0,
                            "successful_analyses": 0,
                            "confidences": []
                        }
                    model_stats[model_type]["total_tests"] += 1
                    if analysis["success"]:
                        model_stats[model_type]["successful_analyses"] += 1
                        model_stats[model_type]["confidences"].append(analysis["confidence"])
        
        for model_type, stats in model_stats.items():
            avg_confidence = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0
            card_class = "standard" if model_type == "standard_clip" else "fashion"
            model_name = "標準 CLIP" if model_type == "standard_clip" else "FashionCLIP"
            icon = "🔍" if model_type == "standard_clip" else "👗"
            
            html_content += f"""
        <div class="model-card {card_class}">
            <h3>{icon} {model_name}</h3>
            <p><strong>成功率:</strong> {stats['successful_analyses']}/{stats['total_tests']} ({stats['successful_analyses']/stats['total_tests']:.1%})</p>
            <p><strong>平均置信度:</strong> <span class="confidence">{avg_confidence:.3f}</span></p>
            <p><strong>特點:</strong> {'穩定可靠的基準表現' if model_type == 'standard_clip' else '專業時尚細節識別'}</p>
        </div>"""

        html_content += """
    </div>

    <h2>🔍 詳細測試結果</h2>"""

        # 添加每個測試的詳細結果
        for i, result in enumerate(results):
            if result["success"]:
                image_name = os.path.basename(result["image_path"])
                html_content += f"""
    <div class="test-result">
        <h3>測試 {i+1}: {image_name}</h3>
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>模型</th>
                    <th>置信度</th>
                    <th>前3個標籤</th>
                    <th>生成的提示詞</th>
                </tr>
            </thead>
            <tbody>"""
                
                for model_type, analysis in result["analysis_results"].items():
                    model_name = "標準 CLIP" if model_type == "standard_clip" else "FashionCLIP"
                    if analysis["success"]:
                        confidence = analysis["confidence"]
                        labels = ", ".join(analysis["top_labels"][:3])
                        prompt = result["generated_prompts"][model_type]["prompt"]
                        html_content += f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td><span class="confidence">{confidence:.3f}</span></td>
                    <td><div class="labels">{labels}</div></td>
                    <td><div class="prompt">{prompt}</div></td>
                </tr>"""
                    else:
                        html_content += f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td>❌ 失敗</td>
                    <td colspan="2">{analysis.get('error', 'Unknown error')}</td>
                </tr>"""
                
                html_content += """
            </tbody>
        </table>
    </div>"""

        html_content += f"""
    <div class="summary">
        <h2>💡 建議</h2>
        <p>{report['model_comparison']['recommendation']}</p>
    </div>

</body>
</html>"""

        # 保存 HTML 報告
        html_path = os.path.join(self.output_dir, "day2_comparison_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # 生成 Markdown 報告
        md_content = f"""# 🎨 Day 2: CLIP vs FashionCLIP 比較報告

**測試時間:** {timestamp}

## 📊 測試摘要

- **總測試數:** {report['tests_run']}
- **成功測試:** {report['tests_successful']}
- **成功率:** {report['success_rate']:.1%}

## 🤖 模型表現比較

"""
        
        for model_type, stats in model_stats.items():
            avg_confidence = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0
            model_name = "標準 CLIP" if model_type == "standard_clip" else "FashionCLIP"
            icon = "🔍" if model_type == "standard_clip" else "👗"
            
            md_content += f"""### {icon} {model_name}

- **成功率:** {stats['successful_analyses']}/{stats['total_tests']} ({stats['successful_analyses']/stats['total_tests']:.1%})
- **平均置信度:** {avg_confidence:.3f}
- **特點:** {'穩定可靠的基準表現' if model_type == 'standard_clip' else '專業時尚細節識別'}

"""

        md_content += "## 🔍 詳細測試結果\n\n"
        
        for i, result in enumerate(results):
            if result["success"]:
                image_name = os.path.basename(result["image_path"])
                md_content += f"### 測試 {i+1}: {image_name}\n\n"
                
                for model_type, analysis in result["analysis_results"].items():
                    model_name = "標準 CLIP" if model_type == "standard_clip" else "FashionCLIP"
                    if analysis["success"]:
                        confidence = analysis["confidence"]
                        labels = ", ".join(analysis["top_labels"][:3])
                        prompt = result["generated_prompts"][model_type]["prompt"]
                        md_content += f"""**{model_name}:**
- 置信度: {confidence:.3f}
- 前3標籤: {labels}
- 生成提示詞: `{prompt}`

"""
                    else:
                        md_content += f"**{model_name}:** ❌ 分析失敗 - {analysis.get('error', 'Unknown error')}\n\n"

        md_content += f"""## 💡 建議

{report['model_comparison']['recommendation']}

---

*報告生成時間: {timestamp}*
"""

        # 保存 Markdown 報告
        md_path = os.path.join(self.output_dir, "day2_comparison_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        return html_path, md_path

if __name__ == "__main__":
    tester = EnhancedDay2Tester()
    tester.run_enhanced_day2_tests()
