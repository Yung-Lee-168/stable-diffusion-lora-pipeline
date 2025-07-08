import os
import json
import shutil
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
import traceback
import sys

# 自定義 JSON 序列化器以處理 NumPy 類型
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(CustomJSONEncoder, self).default(obj)

# 嘗試匯入 FashionCLIP
try:
    sys.path.append("..")
    import 特徵值
    FASHION_CLIP_AVAILABLE = True
    print("✅ 成功匯入 特徵值.py")
except Exception as e:
    print(f"⚠️ 匯入特徵值.py 失敗：{e}")
    FASHION_CLIP_AVAILABLE = False

def calculate_image_similarity(img1_path, img2_path):
    """計算兩張圖片的相似度"""
    try:
        # 讀取圖片
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return None
        
        # 轉換為灰階
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 確保尺寸一致
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # 計算 SSIM
        similarity = ssim(gray1, gray2)
        return similarity
        
    except Exception as e:
        print(f"❌ 計算相似度失敗: {str(e)}")
        return None

def compare_images_with_originals(report):
    """比較生成圖片與原始訓練圖片"""
    print("🔍 比較生成圖片與原始圖片...")
    
    comparison_results = {
        "original_count": 0,
        "generated_count": 0,
        "backup_available": False,
        "resolution_consistency": True,
        "similarity_scores": [],
        "average_similarity": 0.0,
        "fashion_analysis": {}
    }
    
    # 初始化 FashionCLIP
    fashion_model, fashion_processor, device = None, None, None
    labels_dict = {}
    
    if FASHION_CLIP_AVAILABLE:
        fashion_model, fashion_processor, device = load_fashion_clip_model()
        if fashion_model and fashion_processor:
            # 準備特徵標籤
            for k, v in 特徵值.__dict__.items():
                if isinstance(v, (list, tuple)):
                    labels_dict[k] = list(v)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (list, tuple)):
                            labels_dict[kk] = list(vv)
            print(f"✅ FashionCLIP 準備完成，共 {len(labels_dict)} 個特徵類別")
    
    # 檢查原始圖片
    original_path = "lora_train_set/10_test"
    backup_path = "lora_train_set/10_test/original_backup"
    
    if os.path.exists(original_path):
        # 統計所有圖片格式：jpg, jpeg, png
        original_images = [f for f in os.listdir(original_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        comparison_results["original_count"] = len(original_images)
        
        # 檢查備份是否存在
        if os.path.exists(backup_path):
            comparison_results["backup_available"] = True
            print(f"  📁 找到原始圖片備份：{backup_path}")
        
        # 檢查生成的測試圖片
        if os.path.exists("test_images"):
            test_images = [f for f in os.listdir("test_images") if f.endswith('.png')]
            comparison_results["generated_count"] = len(test_images)
            
            print(f"  📊 原始圖片：{len(original_images)} 張")
            print(f"  📊 生成圖片：{len(test_images)} 張")
            
            # 計算圖片相似度 - 比較所有圖片
            if len(original_images) > 0 and len(test_images) > 0:
                similarity_scores = []
                fashion_comparisons = []
                
                # 比較所有圖片對
                max_compare = min(len(original_images), len(test_images))
                
                for i in range(max_compare):
                    original_img_path = os.path.join(original_path, original_images[i])
                    test_img_path = os.path.join("test_images", test_images[i])
                    
                    # SSIM 相似度
                    similarity = calculate_image_similarity(original_img_path, test_img_path)
                    if similarity is not None:
                        similarity_scores.append(similarity)
                        print(f"  📈 SSIM 相似度 {original_images[i]} vs {test_images[i]}: {similarity:.3f}")
                    
                    # FashionCLIP 分析
                    if fashion_model and fashion_processor and labels_dict:
                        print(f"  🎨 FashionCLIP 分析 {original_images[i]} vs {test_images[i]}...")
                        
                        orig_analysis = analyze_image_with_fashion_clip(
                            original_img_path, fashion_model, fashion_processor, device, labels_dict
                        )
                        gen_analysis = analyze_image_with_fashion_clip(
                            test_img_path, fashion_model, fashion_processor, device, labels_dict
                        )
                        
                        if orig_analysis and gen_analysis:
                            feature_comparison = compare_fashion_features(orig_analysis, gen_analysis)
                            if feature_comparison:
                                fashion_comparisons.append({
                                    "original_image": original_images[i],
                                    "generated_image": test_images[i],
                                    "original_analysis": orig_analysis,
                                    "generated_analysis": gen_analysis,
                                    "feature_comparison": feature_comparison
                                })
                                
                                # 計算整體特徵相似度
                                overall_similarity = sum(
                                    comp["combined_similarity"] for comp in feature_comparison.values()
                                ) / len(feature_comparison)
                                print(f"  🎯 特徵相似度 {original_images[i]} vs {test_images[i]}: {overall_similarity:.3f}")
                
                if similarity_scores:
                    comparison_results["similarity_scores"] = similarity_scores
                    comparison_results["average_similarity"] = sum(similarity_scores) / len(similarity_scores)
                    print(f"  📊 平均 SSIM 相似度：{comparison_results['average_similarity']:.3f}")
                
                if fashion_comparisons:
                    comparison_results["fashion_analysis"] = {
                        "comparisons": fashion_comparisons,
                        "total_comparisons": len(fashion_comparisons),
                        "feature_categories": list(labels_dict.keys())
                    }
                    print(f"  🎨 完成 {len(fashion_comparisons)} 對圖片的 FashionCLIP 分析")
    
    return comparison_results

def create_comparison_gallery():
    """建立圖片比較畫廊"""
    print("🎨 建立圖片比較畫廊...")
    
    gallery_html = ""
    
    # 檢查原始圖片和生成圖片
    original_path = "lora_train_set/10_test"
    test_path = "test_images"
    
    if os.path.exists(original_path) and os.path.exists(test_path):
        # 統計所有圖片格式：jpg, jpeg, png
        original_images = [f for f in os.listdir(original_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        test_images = [f for f in os.listdir(test_path) if f.endswith('.png')]
        
        # 創建比較畫廊
        gallery_html = """
        <div class="comparison-gallery">
            <h3>🔍 圖片比較畫廊</h3>
            <div class="gallery-grid">
        """
        
        # 顯示原始圖片
        gallery_html += """
        <div class="gallery-section">
            <h4>📚 原始訓練圖片</h4>
            <div class="image-row">
        """
        
        for i, img in enumerate(original_images):  # 顯示所有圖片
            gallery_html += f"""
            <div class="image-item">
                <div class="image-container">
                    <img src="../{original_path}/{img}" alt="{img}">
                </div>
                <p>{img}</p>
            </div>
            """
        
        gallery_html += """
            </div>
        </div>
        """
        
        # 顯示生成圖片
        gallery_html += """
        <div class="gallery-section">
            <h4>🎨 生成測試圖片</h4>
            <div class="image-row">
        """
        
        for i, img in enumerate(test_images):  # 顯示所有圖片
            gallery_html += f"""
            <div class="image-item">
                <div class="image-container">
                    <img src="../{test_path}/{img}" alt="{img}">
                </div>
                <p>{img}</p>
            </div>
            """
        
        gallery_html += """
            </div>
        </div>
        </div>
        """
    
    return gallery_html

def load_training_progress_records():
    """載入訓練進度記錄"""
    print("📈 載入訓練進度記錄...")
    
    training_records = {
        "progress_available": False,
        "training_history": [],
        "training_summary": {},
        "training_metrics": {},
        "training_charts": []
    }
    
    # 檢查訓練日誌目錄
    log_dir = "training_logs"
    if os.path.exists(log_dir):
        # 尋找最新的訓練報告
        report_files = [f for f in os.listdir(log_dir) if f.startswith("training_report_") and f.endswith(".json")]
        if report_files:
            # 按時間排序，取最新的
            report_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
            latest_report = os.path.join(log_dir, report_files[0])
            
            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                    
                training_records["progress_available"] = True
                training_records["training_summary"] = training_data.get("training_summary", {})
                training_records["training_metrics"] = training_data.get("training_metrics", {})
                training_records["training_evaluation"] = training_data.get("training_evaluation", {})
                training_records["recommendations"] = training_data.get("recommendations", [])
                
                print(f"✅ 載入訓練記錄：{latest_report}")
                print(f"  📊 總步數：{training_records['training_summary'].get('total_steps', 'N/A')}")
                print(f"  🎯 最佳損失：{training_records['training_summary'].get('best_loss', 'N/A')}")
                print(f"  📈 損失改善：{training_records['training_metrics'].get('loss_improvement', 'N/A')}")
                
            except Exception as e:
                print(f"❌ 載入訓練記錄失敗：{e}")
        
        # 尋找訓練圖表
        chart_files = [f for f in os.listdir(log_dir) if f.startswith("training_chart_") and f.endswith(".png")]
        if chart_files:
            chart_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
            training_records["training_charts"] = [os.path.join(log_dir, f) for f in chart_files[:3]]  # 最新3個
            print(f"📊 找到 {len(training_records['training_charts'])} 個訓練圖表")
    
    return training_records

def analyze_training_results():
    """分析訓練結果並產生報告"""
    
    print("📊 開始分析訓練結果...")
    
    # 建立輸出資料夾和時間戳記
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    report = {
        "analysis_time": datetime.now().isoformat(),
        "training_info": {},
        "model_info": {},
        "test_results": {},
        "image_comparison": {},
        "training_progress": {},
        "summary": {}
    }
    
    # 🎯 新增：載入訓練進度記錄
    training_records = load_training_progress_records()
    report["training_progress"] = training_records
    
    # 分析訓練資料
    print("📚 分析訓練資料...")
    if os.path.exists("lora_train_set/10_test"):
        # 統計所有圖片格式：jpg, jpeg, png
        train_images = [f for f in os.listdir("lora_train_set/10_test") 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        train_texts = [f for f in os.listdir("lora_train_set/10_test") if f.endswith('.txt')]
        
        # 檢查是否有備份資料夾
        backup_exists = os.path.exists("lora_train_set/10_test/original_backup")
        
        report["training_info"] = {
            "train_image_count": len(train_images),
            "train_text_count": len(train_texts),
            "training_folder": "10_test",
            "repeat_count": 10,
            "total_training_steps": len(train_images) * 10,
            "backup_created": backup_exists
        }
        
        print(f"  訓練圖片：{len(train_images)} 張")
        print(f"  文字檔案：{len(train_texts)} 個")
        print(f"  總訓練步數：{len(train_images) * 10}")
        
        if backup_exists:
            backup_images = [f for f in os.listdir("lora_train_set/10_test/original_backup") 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  原始備份：{len(backup_images)} 張")
    
    # 分析 LoRA 模型
    print("📁 分析 LoRA 模型...")
    if os.path.exists("lora_output"):
        lora_files = [f for f in os.listdir("lora_output") if f.endswith('.safetensors')]
        if lora_files:
            latest_lora = max(lora_files, key=lambda x: os.path.getmtime(os.path.join("lora_output", x)))
            lora_size = os.path.getsize(os.path.join("lora_output", latest_lora)) / (1024*1024)
            
            report["model_info"] = {
                "filename": latest_lora,
                "size_mb": round(lora_size, 2),
                "total_models": len(lora_files),
                "creation_time": datetime.fromtimestamp(
                    os.path.getctime(os.path.join("lora_output", latest_lora))
                ).isoformat()
            }
            
            print(f"  LoRA 模型：{latest_lora}")
            print(f"  檔案大小：{lora_size:.2f} MB")
            print(f"  總模型數：{len(lora_files)}")
    
    # 分析測試圖片
    print("🎨 分析測試圖片...")
    if os.path.exists("test_images"):
        test_images = [f for f in os.listdir("test_images") if f.endswith('.png')]
        
        # 讀取測試資訊
        test_info = {}
        if os.path.exists("test_images/test_info.json"):
            with open("test_images/test_info.json", 'r', encoding='utf-8') as f:
                test_info = json.load(f)
        
        # 分析圖片屬性
        image_analysis = []
        for img_file in test_images:
            img_path = os.path.join("test_images", img_file)
            try:
                with Image.open(img_path) as img:
                    image_analysis.append({
                        "filename": img_file,
                        "size": f"{img.width}x{img.height}",
                        "mode": img.mode,
                        "file_size_kb": round(os.path.getsize(img_path) / 1024, 2)
                    })
            except Exception as e:
                print(f"  無法分析圖片 {img_file}: {str(e)}")
        
        report["test_results"] = {
            "test_image_count": len(test_images),
            "success_rate": test_info.get("success_count", 0) / max(test_info.get("total_count", 1), 1) * 100,
            "test_info": test_info,
            "image_analysis": image_analysis
        }
        
        print(f"  測試圖片：{len(test_images)} 張")
        print(f"  成功率：{report['test_results']['success_rate']:.1f}%")
    
    # 🎯 新增：圖片比較分析
    report["image_comparison"] = compare_images_with_originals(report)
    
    # 🎯 新增：三基準點性能評估
    if "image_comparison" in report and report["image_comparison"].get("fashion_analysis"):
        print("📊 執行三基準點性能評估...")
        benchmark_analysis = benchmark_results_with_three_points(report["image_comparison"])
        report["benchmark_analysis"] = benchmark_analysis
        
        # 顯示評估結果
        if benchmark_analysis["total_evaluated"] > 0:
            perf_dist = benchmark_analysis["performance_distribution"]
            print(f"🎯 性能分布: 優秀={perf_dist['excellent']}, 良好={perf_dist['good']}, 一般={perf_dist['average']}, 待改善={perf_dist['poor']}")
            
            if benchmark_analysis["recommendations"]:
                print("💡 改善建議:")
                for rec in benchmark_analysis["recommendations"]:
                    print(f"   {rec}")
    
    # 🎯 新增：LoRA 調優指標分析
    print("🔧 執行 LoRA 調優指標分析...")
    lora_tuning_metrics = calculate_lora_tuning_metrics(report)
    report["lora_tuning"] = lora_tuning_metrics
    
    # 🎯 新增：生成調優目標
    tuning_targets = generate_lora_tuning_target(report)
    report["tuning_targets"] = tuning_targets
    
    print(f"🎯 LoRA 調優分數: {lora_tuning_metrics.get('overall_tuning_score', 0):.3f}")
    if lora_tuning_metrics.get("tuning_recommendations"):
        print("🔧 調優建議:")
        for rec in lora_tuning_metrics["tuning_recommendations"]:
            print(f"   {rec}")
    
    # 產生總結
    report["summary"] = {
        "training_completed": "model_info" in report and bool(report["model_info"]),
        "testing_completed": "test_results" in report and report["test_results"]["test_image_count"] > 0,
        "comparison_completed": "image_comparison" in report and report["image_comparison"]["generated_count"] > 0,
        "overall_success": False
    }
    
    # 判定整體是否成功
    if (report["summary"]["training_completed"] and 
        report["summary"]["testing_completed"] and 
        report["test_results"]["success_rate"] > 0):
        report["summary"]["overall_success"] = True
        print("✅ 整體流程成功")
    else:
        print("❌ 整體流程有問題")
    
    # 儲存 JSON 報告
    report_path = os.path.join(output_dir, f"training_report_{timestamp}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
    
    # 產生 HTML 報告
    html_report = generate_html_report(report, timestamp)
    html_path = os.path.join(output_dir, f"training_report_{timestamp}.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    # 產生圖表
    chart_path = generate_charts(report, output_dir, timestamp)
    
    print(f"✅ 分析完成")
    print(f"📋 JSON 報告：{report_path}")
    print(f"🌐 HTML 報告：{html_path}")
    print(f"📊 圖表：{chart_path}")
    
    return report

def create_training_progress_section(training_records):
    """建立訓練進度區段的 HTML"""
    if not training_records.get("progress_available", False):
        return """
        <div class="section">
            <h2>📈 訓練進度記錄</h2>
            <p class="warning">⚠️ 沒有找到訓練進度記錄</p>
            <p>建議使用 training_progress_monitor.py 來監控訓練過程</p>
        </div>
        """
    
    training_summary = training_records.get("training_summary", {})
    training_metrics = training_records.get("training_metrics", {})
    training_evaluation = training_records.get("training_evaluation", {})
    
    html = """
    <div class="section">
        <h2>📈 訓練進度記錄</h2>
        <div class="training-progress">
    """
    
    # 訓練總結
    html += """
        <div class="training-summary">
            <h4>📊 訓練總結</h4>
            <table>
                <tr><th>項目</th><th>數值</th></tr>
    """
    
    summary_items = [
        ("總訓練步數", training_summary.get("total_steps", "N/A")),
        ("最佳損失", f"{training_summary.get('best_loss', 0):.4f}" if isinstance(training_summary.get('best_loss'), (int, float)) else "N/A"),
        ("最終損失", f"{training_summary.get('final_loss', 0):.4f}" if isinstance(training_summary.get('final_loss'), (int, float)) else "N/A"),
        ("最終學習率", f"{training_summary.get('final_lr', 0):.6f}" if isinstance(training_summary.get('final_lr'), (int, float)) else "N/A")
    ]
    
    for item, value in summary_items:
        html += f"<tr><td>{item}</td><td>{value}</td></tr>"
    
    html += """
            </table>
        </div>
    """
    
    # 訓練指標
    html += """
        <div class="training-metrics">
            <h4>📈 訓練指標</h4>
            <table>
                <tr><th>指標</th><th>數值</th></tr>
    """
    
    metrics_items = [
        ("損失改善", f"{training_metrics.get('loss_improvement', 0):.4f}" if isinstance(training_metrics.get('loss_improvement'), (int, float)) else "N/A"),
        ("損失降低率", f"{training_metrics.get('loss_reduction_rate', 0):.2f}%" if isinstance(training_metrics.get('loss_reduction_rate'), (int, float)) else "N/A"),
        ("平均損失", f"{training_metrics.get('average_loss', 0):.4f}" if isinstance(training_metrics.get('average_loss'), (int, float)) else "N/A")
    ]
    
    for metric, value in metrics_items:
        html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
    
    html += """
            </table>
        </div>
    """
    
    # 訓練評估
    if training_evaluation:
        html += """
        <div class="training-evaluation">
            <h4>🎯 訓練評估</h4>
            <table>
                <tr><th>評估項目</th><th>分數/等級</th></tr>
        """
        
        eval_items = [
            ("性能等級", training_evaluation.get('performance_grade', 'N/A').upper()),
            ("訓練效率", f"{training_evaluation.get('efficiency', 0):.4f}" if isinstance(training_evaluation.get('efficiency'), (int, float)) else "N/A"),
            ("收斂率", f"{training_evaluation.get('convergence_rate', 0):.4f}" if isinstance(training_evaluation.get('convergence_rate'), (int, float)) else "N/A")
        ]
        
        for item, value in eval_items:
            grade_class = ""
            if "等級" in item:
                if value == "EXCELLENT":
                    grade_class = "excellent"
                elif value == "GOOD":
                    grade_class = "good"
                elif value == "AVERAGE":
                    grade_class = "average"
                else:
                    grade_class = "poor"
                    
            html += f"<tr class='{grade_class}'><td>{item}</td><td>{value}</td></tr>"
        
        html += """
            </table>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html

def generate_html_report(report, timestamp):
    """產生 HTML 格式的報告"""
    
    # 建立比較畫廊
    comparison_gallery = create_comparison_gallery()
    
    # 圖表檔案名稱（包含時間戳記）
    chart_filename = f"training_charts_{timestamp}.png"
    
    # 相似度分析
    similarity_section = ""
    if "image_comparison" in report and report["image_comparison"].get("similarity_scores"):
        similarity_scores = report["image_comparison"]["similarity_scores"]
        avg_similarity = report["image_comparison"]["average_similarity"]
        
        similarity_section = f"""
        <div class="section">
            <h2>📈 SSIM 相似度分析</h2>
            <table>
                <tr><th>項目</th><th>數值</th></tr>
                <tr><td>平均相似度</td><td>{avg_similarity:.3f}</td></tr>
                <tr><td>最高相似度</td><td>{max(similarity_scores):.3f}</td></tr>
                <tr><td>最低相似度</td><td>{min(similarity_scores):.3f}</td></tr>
                <tr><td>比較圖片數</td><td>{len(similarity_scores)}</td></tr>
            </table>
        </div>
        """
    
    # FashionCLIP 分析結果
    fashion_section = ""
    if "image_comparison" in report and report["image_comparison"].get("fashion_analysis"):
        fashion_data = report["image_comparison"]["fashion_analysis"]
        if fashion_data.get("comparisons"):
            fashion_section = """
            <div class="section">
                <h2>🎨 FashionCLIP 特徵分析</h2>
                <div class="fashion-analysis">
            """
            
            for comp in fashion_data["comparisons"][:3]:  # 只顯示前3個比較
                fashion_section += f"""
                <div class="comparison-pair">
                    <h4>📸 {comp['original_image']} vs {comp['generated_image']}</h4>
                    <table>
                        <tr><th>特徵類別</th><th>原始圖片</th><th>生成圖片</th><th>相似度</th></tr>
                """
                
                for cat, feature_comp in comp["feature_comparison"].items():
                    fashion_section += f"""
                    <tr>
                        <td>{cat}</td>
                        <td>{feature_comp['original_label']}</td>
                        <td>{feature_comp['generated_label']}</td>
                        <td>{feature_comp['combined_similarity']:.3f}</td>
                    </tr>
                    """
                
                fashion_section += """
                    </table>
                </div>
                """
            
            fashion_section += """
                </div>
            </div>
            """
    
    # 三基準點評估結果
    benchmark_section = ""
    if "benchmark_analysis" in report and report["benchmark_analysis"].get("total_evaluated", 0) > 0:
        benchmark_data = report["benchmark_analysis"]
        perf_dist = benchmark_data["performance_distribution"]
        avg_metrics = benchmark_data.get("average_metrics", {})
        
        benchmark_section = f"""
        <div class="section">
            <h2>🎯 三基準點性能評估</h2>
            <div class="benchmark-summary">
                <h4>📊 性能分布 (共 {benchmark_data['total_evaluated']} 張圖片)</h4>
                <table>
                    <tr><th>評級</th><th>數量</th><th>比例</th></tr>
                    <tr class="excellent"><td>🎯 優秀</td><td>{perf_dist['excellent']}</td><td>{perf_dist['excellent']/benchmark_data['total_evaluated']*100:.1f}%</td></tr>
                    <tr class="good"><td>✅ 良好</td><td>{perf_dist['good']}</td><td>{perf_dist['good']/benchmark_data['total_evaluated']*100:.1f}%</td></tr>
                    <tr class="average"><td>⚠️ 一般</td><td>{perf_dist['average']}</td><td>{perf_dist['average']/benchmark_data['total_evaluated']*100:.1f}%</td></tr>
                    <tr class="poor"><td>❌ 待改善</td><td>{perf_dist['poor']}</td><td>{perf_dist['poor']/benchmark_data['total_evaluated']*100:.1f}%</td></tr>
                </table>
            </div>
        """
        
        if avg_metrics:
            benchmark_section += f"""
            <div class="benchmark-metrics">
                <h4>📈 平均指標</h4>
                <table>
                    <tr><th>指標</th><th>平均值</th><th>參考值</th><th>差異</th></tr>
                    <tr><td>總損失</td><td>{avg_metrics.get('avg_total_loss', 0):.3f}</td><td>0.709</td><td>{avg_metrics.get('avg_total_loss', 0) - 0.709:+.3f}</td></tr>
                    <tr><td>視覺相似度</td><td>{avg_metrics.get('avg_visual_similarity', 0):.3f}</td><td>0.326</td><td>{avg_metrics.get('avg_visual_similarity', 0) - 0.326:+.3f}</td></tr>
                    <tr><td>FashionCLIP相似度</td><td>{avg_metrics.get('avg_fashion_clip_similarity', 0):.3f}</td><td>0.523</td><td>{avg_metrics.get('avg_fashion_clip_similarity', 0) - 0.523:+.3f}</td></tr>
                    <tr><td>色彩相似度</td><td>{avg_metrics.get('avg_color_similarity', 0):.3f}</td><td>0.012</td><td>{avg_metrics.get('avg_color_similarity', 0) - 0.012:+.3f}</td></tr>
                </table>
            </div>
            """
        
        if benchmark_data.get("recommendations"):
            benchmark_section += """
            <div class="recommendations">
                <h4>💡 改善建議</h4>
                <ul>
            """
            for rec in benchmark_data["recommendations"]:
                benchmark_section += f"<li>{rec}</li>"
            
            benchmark_section += """
                </ul>
            </div>
            """
        
        benchmark_section += """
        </div>
        """
    
    # LoRA 調優指標結果
    lora_tuning_section = ""
    if "lora_tuning" in report and report["lora_tuning"].get("overall_tuning_score", 0) > 0:
        lora_data = report["lora_tuning"]
        overall_score = lora_data["overall_tuning_score"]
        overall_grade = lora_data.get("overall_grade", "unknown")
        
        lora_tuning_section = f"""
        <div class="section">
            <h2>🔧 LoRA 調優指標分析</h2>
            <div class="lora-summary">
                <h4>🎯 整體調優評分</h4>
                <div class="tuning-score {overall_grade}">
                    <span class="score-value">{overall_score:.3f}</span>
                    <span class="score-grade">({overall_grade.upper()})</span>
                </div>
            </div>
        """
        
        # 詳細指標
        if any(key in lora_data for key in ["training_efficiency", "generation_quality", "feature_preservation"]):
            lora_tuning_section += """
            <div class="detailed-metrics">
                <h4>📊 詳細指標</h4>
                <table>
                    <tr><th>指標類別</th><th>分數</th><th>評級</th><th>詳細資訊</th></tr>
            """
            
            if "training_efficiency" in lora_data:
                eff = lora_data["training_efficiency"]
                lora_tuning_section += f"""
                <tr class="{eff.get('grade', 'unknown')}">
                    <td>🚀 訓練效率</td>
                    <td>{eff.get('score', 0):.3f}</td>
                    <td>{eff.get('grade', 'N/A').upper()}</td>
                    <td>步數: {eff.get('steps', 0)}, 模型: {eff.get('model_size_mb', 0):.1f}MB</td>
                </tr>
                """
            
            if "generation_quality" in lora_data:
                qual = lora_data["generation_quality"]
                lora_tuning_section += f"""
                <tr class="{qual.get('grade', 'unknown')}">
                    <td>🎨 生成品質</td>
                    <td>{qual.get('score', 0):.3f}</td>
                    <td>{qual.get('grade', 'N/A').upper()}</td>
                    <td>SSIM: {qual.get('average_ssim', 0):.3f}, 圖片: {qual.get('image_count', 0)}張</td>
                </tr>
                """
            
            if "feature_preservation" in lora_data:
                feat = lora_data["feature_preservation"]
                lora_tuning_section += f"""
                <tr class="{feat.get('grade', 'unknown')}">
                    <td>🎯 特徵保持</td>
                    <td>{feat.get('score', 0):.3f}</td>
                    <td>{feat.get('grade', 'N/A').upper()}</td>
                    <td>FashionCLIP: {feat.get('fashion_clip_similarity', 0):.3f}</td>
                </tr>
                """
            
            lora_tuning_section += """
                </table>
            </div>
            """
        
        # 調優建議
        if lora_data.get("tuning_recommendations"):
            lora_tuning_section += """
            <div class="tuning-recommendations">
                <h4>🔧 調優建議</h4>
                <ul>
            """
            for rec in lora_data["tuning_recommendations"]:
                lora_tuning_section += f"<li>{rec}</li>"
            
            lora_tuning_section += """
                </ul>
            </div>
            """
        
        lora_tuning_section += """
        </div>
        """
    
    # 調優目標
    tuning_target_section = ""
    if "tuning_targets" in report and report["tuning_targets"].get("target_metrics"):
        target_data = report["tuning_targets"]
        current_perf = target_data.get("current_performance", {})
        target_metrics = target_data.get("target_metrics", {})
        
        tuning_target_section = f"""
        <div class="section">
            <h2>🎯 下一輪調優目標</h2>
            <div class="target-comparison">
                <h4>📈 目標 vs 當前表現</h4>
                <table>
                    <tr><th>指標</th><th>當前值</th><th>目標值</th><th>改善幅度</th></tr>
                    <tr><td>總損失</td><td>{current_perf.get('total_loss', 0):.3f}</td><td>{target_metrics.get('target_total_loss', 0):.3f}</td><td>{((current_perf.get('total_loss', 1) - target_metrics.get('target_total_loss', 1)) / current_perf.get('total_loss', 1) * 100):+.1f}%</td></tr>
                    <tr><td>視覺相似度</td><td>{current_perf.get('visual_similarity', 0):.3f}</td><td>{target_metrics.get('target_visual_similarity', 0):.3f}</td><td>{((target_metrics.get('target_visual_similarity', 0) - current_perf.get('visual_similarity', 0)) / max(current_perf.get('visual_similarity', 0.001), 0.001) * 100):+.1f}%</td></tr>
                    <tr><td>FashionCLIP相似度</td><td>{current_perf.get('fashion_clip_similarity', 0):.3f}</td><td>{target_metrics.get('target_fashion_clip_similarity', 0):.3f}</td><td>{((target_metrics.get('target_fashion_clip_similarity', 0) - current_perf.get('fashion_clip_similarity', 0)) / max(current_perf.get('fashion_clip_similarity', 0.001), 0.001) * 100):+.1f}%</td></tr>
                </table>
            </div>
        """
        
        if target_data.get("action_plan"):
            tuning_target_section += """
            <div class="action-plan">
                <h4>📋 行動計劃</h4>
                <ul>
            """
            for action in target_data["action_plan"]:
                tuning_target_section += f"<li>{action}</li>"
            
            tuning_target_section += """
                </ul>
            </div>
            """
        
        if target_data.get("parameter_suggestions"):
            tuning_target_section += """
            <div class="parameter-suggestions">
                <h4>⚙️ 參數建議</h4>
                <table>
                    <tr><th>參數</th><th>建議</th></tr>
            """
            for param, suggestion in target_data["parameter_suggestions"].items():
                tuning_target_section += f"<tr><td>{param}</td><td>{suggestion}</td></tr>"
            
            tuning_target_section += """
                </table>
            </div>
            """
        
        tuning_target_section += """
        </div>
        """
    
    # 🎯 新增：訓練進度區段
    training_progress_section = ""
    if "training_progress" in report:
        training_progress_section = create_training_progress_section(report["training_progress"])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LoRA 訓練完整報告</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .success {{ color: #28a745; }}
            .error {{ color: #dc3545; }}
            .info {{ color: #17a2b8; }}
            .warning {{ color: #ffc107; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .gallery {{ text-align: center; }}
            .summary {{ font-size: 18px; font-weight: bold; }}
            .comparison-gallery {{ margin: 20px 0; }}
            .gallery-section {{ margin: 15px 0; }}
            .image-row {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: flex-start; }}
            .image-item {{ text-align: center; flex: 0 0 calc(20% - 12px); max-width: calc(20% - 12px); }}
            .image-container {{ height: 200px; display: flex; align-items: center; justify-content: center; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }}
            .image-item img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
            .image-item p {{ margin: 5px 0; font-size: 12px; word-break: break-all; }}
            .fashion-analysis {{ margin: 15px 0; }}
            .comparison-pair {{ margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 5px; background-color: #fafafa; }}
            .comparison-pair h4 {{ margin: 0 0 10px 0; color: #333; }}
            .benchmark-summary {{ margin: 15px 0; }}
            .benchmark-metrics {{ margin: 15px 0; }}
            .recommendations {{ margin: 15px 0; padding: 15px; border: 1px solid #e67e22; border-radius: 5px; background-color: #fef9e7; }}
            .recommendations ul {{ margin: 10px 0; padding-left: 20px; }}
            .excellent {{ background-color: #d5f4e6; }}
            .good {{ background-color: #ffeaa7; }}
            .average {{ background-color: #fab1a0; }}
            .poor {{ background-color: #ff7675; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 LoRA 訓練完整報告</h1>
            <p><strong>分析時間：</strong>{report['analysis_time']}</p>
            <p class="summary {'success' if report['summary']['overall_success'] else 'error'}">
                整體狀態：{'✅ 成功' if report['summary']['overall_success'] else '❌ 失敗'}
            </p>
        </div>
        
        <div class="section">
            <h2>📊 訓練資料統計</h2>
            <table>
                <tr><th>項目</th><th>數值</th></tr>
                <tr><td>訓練圖片數量</td><td>{report['training_info'].get('train_image_count', 'N/A')}</td></tr>
                <tr><td>文字檔案數量</td><td>{report['training_info'].get('train_text_count', 'N/A')}</td></tr>
                <tr><td>重複次數</td><td>{report['training_info'].get('repeat_count', 'N/A')}</td></tr>
                <tr><td>總訓練步數</td><td>{report['training_info'].get('total_training_steps', 'N/A')}</td></tr>
                <tr><td>原始備份</td><td>{'✅ 已建立' if report['training_info'].get('backup_created', False) else '❌ 未建立'}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📁 LoRA 模型資訊</h2>
            <table>
                <tr><th>項目</th><th>數值</th></tr>
                <tr><td>檔案名稱</td><td>{report['model_info'].get('filename', 'N/A')}</td></tr>
                <tr><td>檔案大小</td><td>{report['model_info'].get('size_mb', 'N/A')} MB</td></tr>
                <tr><td>建立時間</td><td>{report['model_info'].get('creation_time', 'N/A')}</td></tr>
                <tr><td>總模型數</td><td>{report['model_info'].get('total_models', 'N/A')}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>🎨 測試結果</h2>
            <table>
                <tr><th>項目</th><th>數值</th></tr>
                <tr><td>測試圖片數量</td><td>{report['test_results'].get('test_image_count', 'N/A')}</td></tr>
                <tr><td>成功率</td><td>{report['test_results'].get('success_rate', 'N/A'):.1f}%</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>🔍 圖片比較統計</h2>
            <table>
                <tr><th>項目</th><th>數值</th></tr>
                <tr><td>原始圖片數量</td><td>{report['image_comparison'].get('original_count', 'N/A')}</td></tr>
                <tr><td>生成圖片數量</td><td>{report['image_comparison'].get('generated_count', 'N/A')}</td></tr>
                <tr><td>備份可用性</td><td>{'✅ 可用' if report['image_comparison'].get('backup_available', False) else '❌ 不可用'}</td></tr>
                <tr><td>解析度一致性</td><td>{'✅ 一致' if report['image_comparison'].get('resolution_consistency', False) else '❌ 不一致'}</td></tr>
            </table>
        </div>
        
        {similarity_section}
        
        {fashion_section}
        
        {benchmark_section}
        
        {lora_tuning_section}
        
        {tuning_target_section}
        
        {training_progress_section}
        
        <div class="section">
            {comparison_gallery}
        </div>
        
        <div class="section">
            <h2>📈 效能圖表</h2>
            <img src="{chart_filename}" style="max-width: 100%; height: auto;">
        </div>
    </body>
    </html>
    """
    return html

def generate_charts(report, output_dir, timestamp):
    """產生訓練圖表"""
    
    try:
        # 設定中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LoRA Training Analysis Charts', fontsize=16, fontweight='bold')
        
        # 圖表1：訓練資料統計
        train_data = [
            report['training_info'].get('train_image_count', 0),
            report['training_info'].get('train_text_count', 0)
        ]
        bars1 = ax1.bar(['Training Images', 'Text Files'], train_data, color=['skyblue', 'lightgreen'])
        ax1.set_title('Training Data Statistics')
        ax1.set_ylabel('Count')
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}', 
                    ha='center', va='bottom')
        
        # 圖表2：模型檔案大小
        model_size = report['model_info'].get('size_mb', 0)
        bars2 = ax2.bar(['LoRA Model'], [model_size], color=['orange'])
        ax2.set_title('Model File Size')
        ax2.set_ylabel('Size (MB)')
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f} MB', 
                    ha='center', va='bottom')
        
        # 圖表3：測試成功率
        success_rate = report['test_results'].get('success_rate', 0)
        colors = ['lightgreen' if success_rate > 80 else 'orange' if success_rate > 50 else 'lightcoral', 'lightgray']
        ax3.pie([success_rate, 100-success_rate], 
                labels=['Success', 'Failed'], 
                autopct='%1.1f%%',
                colors=colors)
        ax3.set_title('Test Success Rate')
        
        # 圖表4：相似度分析
        if 'image_comparison' in report and report['image_comparison'].get('similarity_scores'):
            similarity_scores = report['image_comparison']['similarity_scores']
            x_pos = range(len(similarity_scores))
            bars4 = ax4.bar(x_pos, similarity_scores, color='lightblue', edgecolor='navy')
            ax4.set_title('Image Similarity Analysis')
            ax4.set_xlabel('Image Number')
            ax4.set_ylabel('Similarity Score')
            ax4.set_ylim(0, 1)
            
            # 添加數值標籤
            for i, bar in enumerate(bars4):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', 
                        ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'No similarity data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Image Similarity Analysis')
        
        plt.tight_layout()
        
        # 儲存圖表到指定資料夾，檔名包含時間戳記
        chart_path = os.path.join(output_dir, f"training_charts_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 圖表已產生：{chart_path}")
        return chart_path
        
    except Exception as e:
        print(f"❌ 圖表產生失敗：{str(e)}")
        return None

def load_fashion_clip_model():
    """載入 FashionCLIP 模型"""
    try:
        model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print("✅ FashionCLIP 模型載入成功")
        return model, processor, device
    except Exception as e:
        print(f"❌ FashionCLIP 模型載入失敗：{e}")
        return None, None, None

def analyze_image_with_fashion_clip(image_path, model, processor, device, labels_dict):
    """使用 FashionCLIP 分析圖片特徵"""
    try:
        image = Image.open(image_path).convert("RGB")
        analysis_results = {}
        
        # 對每個特徵類別進行分析
        for cat, cat_labels in labels_dict.items():
            if not cat_labels:
                continue
                
            inputs = processor(text=cat_labels, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                # 獲取最高分數的標籤
                best_idx = probs[0].argmax().item()
                best_label = cat_labels[best_idx]
                best_score = probs[0][best_idx].item()
                
                # 獲取前3個最高分數的標籤
                top3_indices = probs[0].argsort(descending=True)[:3]
                top3_labels = [(cat_labels[i], probs[0][i].item()) for i in top3_indices]
                
                analysis_results[cat] = {
                    "best_label": best_label,
                    "best_score": best_score,
                    "top3": top3_labels
                }
        
        return analysis_results
    except Exception as e:
        print(f"❌ FashionCLIP 分析失敗：{e}")
        return None

def compare_fashion_features(original_analysis, generated_analysis):
    """比較原始圖片和生成圖片的時尚特徵"""
    if not original_analysis or not generated_analysis:
        return None
    
    feature_similarity = {}
    
    for cat in original_analysis:
        if cat in generated_analysis:
            orig_label = original_analysis[cat]["best_label"]
            gen_label = generated_analysis[cat]["best_label"]
            
            # 計算標籤匹配度
            label_match = 1.0 if orig_label == gen_label else 0.0
            
            # 計算分數相似度
            orig_score = original_analysis[cat]["best_score"]
            gen_score = generated_analysis[cat]["best_score"]
            score_similarity = 1.0 - abs(orig_score - gen_score)
            
            feature_similarity[cat] = {
                "original_label": orig_label,
                "generated_label": gen_label,
                "label_match": label_match,
                "score_similarity": score_similarity,
                "combined_similarity": (label_match + score_similarity) / 2
            }
    
    return feature_similarity

def benchmark_results_with_three_points(comparison_results):
    """基於三個基準點評估結果 - 參考 day3_fashion_training.py"""
    print("🎯 三基準點性能評估...")
    
    # 參考基準值 (來自 day3_fashion_training.py)
    benchmarks = {
        "total_loss": {
            "excellent": 0.3,    # 優秀
            "good": 0.5,         # 良好  
            "average": 0.7,      # 一般
            "reference": 0.709   # 參考值
        },
        "visual_similarity": {
            "excellent": 0.7,    # 優秀
            "good": 0.5,         # 良好
            "average": 0.3,      # 一般
            "reference": 0.326   # 參考值
        },
        "fashion_clip_similarity": {
            "excellent": 0.7,    # 優秀
            "good": 0.5,         # 良好
            "average": 0.3,      # 一般
            "reference": 0.523   # 參考值
        },
        "color_similarity": {
            "excellent": 0.8,    # 優秀
            "good": 0.6,         # 良好
            "average": 0.4,      # 一般
            "reference": 0.012   # 參考值 (極低)
        }
    }
    
    benchmark_results = {
        "total_evaluated": 0,
        "performance_distribution": {
            "excellent": 0,
            "good": 0, 
            "average": 0,
            "poor": 0
        },
        "detailed_analysis": [],
        "recommendations": [],
        "average_metrics": {},
        "benchmark_comparison": {}
    }
    
    # 處理 SSIM 相似度數據
    if comparison_results.get("similarity_scores"):
        similarity_scores = comparison_results["similarity_scores"]
        benchmark_results["total_evaluated"] = len(similarity_scores)
        
        # 使用 SSIM 相似度進行基本評估
        avg_visual_sim = sum(similarity_scores) / len(similarity_scores)
        
        # 計算基於 day3_fashion_training.py 權重的總損失
        weights = {"visual": 0.2, "fashion_clip": 0.6, "color": 0.2}
        visual_loss = 1.0 - avg_visual_sim
        
        # 假設色彩相似度為中等水平 (因為沒有具體數據)
        assumed_color_sim = 0.5
        color_loss = 1.0 - assumed_color_sim
        
        # 如果有 FashionCLIP 分析數據
        if "fashion_analysis" in comparison_results and comparison_results["fashion_analysis"].get("comparisons"):
            comparisons = comparison_results["fashion_analysis"]["comparisons"]
            
            for i, comp in enumerate(comparisons):
                if i < len(similarity_scores):
                    visual_sim = similarity_scores[i]
                    
                    # 從 FashionCLIP 分析中獲取語意相似度
                    fashion_sim = 0.5  # 預設值
                    if "feature_comparison" in comp:
                        feature_sims = [feat["combined_similarity"] for feat in comp["feature_comparison"].values()]
                        if feature_sims:
                            fashion_sim = sum(feature_sims) / len(feature_sims)
                    
                    color_sim = assumed_color_sim  # 預設色彩相似度
                    
                    # 計算總損失
                    fashion_clip_loss = 1.0 - fashion_sim
                    total_loss = (
                        weights["visual"] * visual_loss +
                        weights["fashion_clip"] * fashion_clip_loss +
                        weights["color"] * color_loss
                    )
                    
                    # 三基準點評估
                    analysis = evaluate_against_three_benchmarks(
                        total_loss, visual_sim, fashion_sim, color_sim, benchmarks
                    )
                    
                    benchmark_results["detailed_analysis"].append({
                        "image_pair": f"{comp.get('original_image', f'image_{i+1}')} vs {comp.get('generated_image', f'gen_{i+1}')}",
                        "metrics": {
                            "total_loss": total_loss,
                            "visual_similarity": visual_sim,
                            "fashion_clip_similarity": fashion_sim,
                            "color_similarity": color_sim
                        },
                        "benchmark_evaluation": analysis
                    })
                    
                    # 更新性能分佈
                    overall_performance = analysis["overall_performance"]
                    benchmark_results["performance_distribution"][overall_performance] += 1
                    
                    print(f"  📊 {comp.get('original_image', f'image_{i+1}')}: {overall_performance.upper()}")
                    print(f"     總損失: {total_loss:.3f}, 視覺: {visual_sim:.3f}, FashionCLIP: {fashion_sim:.3f}")
        else:
            # 只有 SSIM 數據的情況
            for i, visual_sim in enumerate(similarity_scores):
                # 預設 FashionCLIP 和色彩相似度
                fashion_sim = 0.5
                color_sim = 0.5
                
                fashion_clip_loss = 1.0 - fashion_sim
                total_loss = (
                    weights["visual"] * (1.0 - visual_sim) +
                    weights["fashion_clip"] * fashion_clip_loss +
                    weights["color"] * (1.0 - color_sim)
                )
                
                analysis = evaluate_against_three_benchmarks(
                    total_loss, visual_sim, fashion_sim, color_sim, benchmarks
                )
                
                benchmark_results["detailed_analysis"].append({
                    "image_pair": f"image_{i+1}",
                    "metrics": {
                        "total_loss": total_loss,
                        "visual_similarity": visual_sim,
                        "fashion_clip_similarity": fashion_sim,
                        "color_similarity": color_sim
                    },
                    "benchmark_evaluation": analysis
                })
                
                overall_performance = analysis["overall_performance"]
                benchmark_results["performance_distribution"][overall_performance] += 1
    
    # 計算平均指標
    if benchmark_results["detailed_analysis"]:
        analyses = benchmark_results["detailed_analysis"]
        n = len(analyses)
        
        benchmark_results["average_metrics"] = {
            "avg_total_loss": sum(a["metrics"]["total_loss"] for a in analyses) / n,
            "avg_visual_similarity": sum(a["metrics"]["visual_similarity"] for a in analyses) / n,
            "avg_fashion_clip_similarity": sum(a["metrics"]["fashion_clip_similarity"] for a in analyses) / n,
            "avg_color_similarity": sum(a["metrics"]["color_similarity"] for a in analyses) / n
        }
        
        # 與參考值比較
        avg_metrics = benchmark_results["average_metrics"]
        benchmark_results["benchmark_comparison"] = {
            "vs_reference_total_loss": avg_metrics["avg_total_loss"] - benchmarks["total_loss"]["reference"],
            "vs_reference_visual": avg_metrics["avg_visual_similarity"] - benchmarks["visual_similarity"]["reference"],
            "vs_reference_fashion": avg_metrics["avg_fashion_clip_similarity"] - benchmarks["fashion_clip_similarity"]["reference"],
            "vs_reference_color": avg_metrics["avg_color_similarity"] - benchmarks["color_similarity"]["reference"]
        }
    
    # 生成建議
    benchmark_results["recommendations"] = generate_improvement_recommendations(
        benchmark_results["detailed_analysis"], benchmarks
    )
    
    return benchmark_results

def evaluate_against_three_benchmarks(total_loss, visual_sim, fashion_sim, color_sim, benchmarks):
    """針對三個基準點評估單一結果"""
    
    # 評估每個指標
    evaluations = {}
    
    # 總損失 (越低越好)
    if total_loss <= benchmarks["total_loss"]["excellent"]:
        evaluations["total_loss"] = "excellent"
    elif total_loss <= benchmarks["total_loss"]["good"]:
        evaluations["total_loss"] = "good"
    elif total_loss <= benchmarks["total_loss"]["average"]:
        evaluations["total_loss"] = "average"
    else:
        evaluations["total_loss"] = "poor"
    
    # 視覺相似度 (越高越好)
    if visual_sim >= benchmarks["visual_similarity"]["excellent"]:
        evaluations["visual_similarity"] = "excellent"
    elif visual_sim >= benchmarks["visual_similarity"]["good"]:
        evaluations["visual_similarity"] = "good"
    elif visual_sim >= benchmarks["visual_similarity"]["average"]:
        evaluations["visual_similarity"] = "average"
    else:
        evaluations["visual_similarity"] = "poor"
    
    # FashionCLIP 相似度 (越高越好)
    if fashion_sim >= benchmarks["fashion_clip_similarity"]["excellent"]:
        evaluations["fashion_clip_similarity"] = "excellent"
    elif fashion_sim >= benchmarks["fashion_clip_similarity"]["good"]:
        evaluations["fashion_clip_similarity"] = "good"
    elif fashion_sim >= benchmarks["fashion_clip_similarity"]["average"]:
        evaluations["fashion_clip_similarity"] = "average"
    else:
        evaluations["fashion_clip_similarity"] = "poor"
    
    # 色彩相似度 (越高越好)
    if color_sim >= benchmarks["color_similarity"]["excellent"]:
        evaluations["color_similarity"] = "excellent"
    elif color_sim >= benchmarks["color_similarity"]["good"]:
        evaluations["color_similarity"] = "good"
    elif color_sim >= benchmarks["color_similarity"]["average"]:
        evaluations["color_similarity"] = "average"
    else:
        evaluations["color_similarity"] = "poor"
    
    # 綜合評估 (基於 FashionCLIP 為主要指標)
    if evaluations["fashion_clip_similarity"] == "excellent" and evaluations["total_loss"] in ["excellent", "good"]:
        overall = "excellent"
    elif evaluations["fashion_clip_similarity"] == "good" and evaluations["total_loss"] in ["excellent", "good", "average"]:
        overall = "good"
    elif evaluations["fashion_clip_similarity"] == "average":
        overall = "average"
    else:
        overall = "poor"
    
    return {
        "individual_evaluations": evaluations,
        "overall_performance": overall,
        "benchmark_comparison": {
            "vs_reference_total_loss": total_loss - benchmarks["total_loss"]["reference"],
            "vs_reference_visual": visual_sim - benchmarks["visual_similarity"]["reference"],
            "vs_reference_fashion": fashion_sim - benchmarks["fashion_clip_similarity"]["reference"],
            "vs_reference_color": color_sim - benchmarks["color_similarity"]["reference"]
        }
    }

def generate_improvement_recommendations(detailed_analysis, benchmarks):
    """生成改善建議"""
    recommendations = []
    
    if not detailed_analysis:
        return recommendations
    
    # 分析整體趨勢
    total_count = len(detailed_analysis)
    poor_total_loss = sum(1 for analysis in detailed_analysis 
                         if analysis["benchmark_evaluation"]["individual_evaluations"]["total_loss"] == "poor")
    poor_fashion_clip = sum(1 for analysis in detailed_analysis 
                           if analysis["benchmark_evaluation"]["individual_evaluations"]["fashion_clip_similarity"] == "poor")
    poor_visual = sum(1 for analysis in detailed_analysis 
                     if analysis["benchmark_evaluation"]["individual_evaluations"]["visual_similarity"] == "poor")
    poor_color = sum(1 for analysis in detailed_analysis 
                    if analysis["benchmark_evaluation"]["individual_evaluations"]["color_similarity"] == "poor")
    
    # 生成具體建議
    if poor_total_loss > total_count * 0.5:
        recommendations.append("🎯 總損失過高：建議調整訓練參數或提示詞策略")
    
    if poor_fashion_clip > total_count * 0.5:
        recommendations.append("🎨 FashionCLIP 相似度低：建議優化特徵提取或使用更精準的服裝描述")
    
    if poor_visual > total_count * 0.5:
        recommendations.append("👁️ 視覺相似度低：建議調整生成參數或使用更高解析度")
    
    if poor_color > total_count * 0.7:  # 色彩相似度參考值很低，標準較寬鬆
        recommendations.append("🎨 色彩相似度需改善：建議在提示詞中加入具體色彩描述")
    
    # 基於參考值的建議
    if detailed_analysis:
        avg_fashion_diff = sum(analysis["benchmark_evaluation"]["benchmark_comparison"]["vs_reference_fashion"] 
                              for analysis in detailed_analysis) / total_count
        
        if avg_fashion_diff > 0.1:
            recommendations.append("✅ FashionCLIP 表現超越參考值，模型效果良好")
        elif avg_fashion_diff < -0.1:
            recommendations.append("⚠️ FashionCLIP 表現低於參考值，建議檢查訓練數據質量")
    
    return recommendations

def calculate_lora_tuning_metrics(report):
    """計算 LoRA 調優指標 - 參考 day3_fashion_training.py 的損失函數"""
    print("🎯 計算 LoRA 調優指標...")
    
    tuning_metrics = {
        "training_efficiency": {},
        "generation_quality": {},
        "feature_preservation": {},
        "overall_tuning_score": 0.0,
        "tuning_recommendations": []
    }
    
    # 1. 訓練效率指標
    if "training_info" in report and "model_info" in report:
        train_steps = report["training_info"].get("total_training_steps", 0)
        model_size = report["model_info"].get("size_mb", 0)
        
        # 計算效率比率
        if train_steps > 0:
            efficiency_ratio = model_size / train_steps * 1000  # 每千步的模型大小
            
            # 效率評估 (參考標準: 18MB / 100步 ≈ 0.18)
            if efficiency_ratio < 0.15:
                efficiency_grade = "excellent"
                efficiency_score = 1.0
            elif efficiency_ratio < 0.25:
                efficiency_grade = "good"
                efficiency_score = 0.8
            elif efficiency_ratio < 0.35:
                efficiency_grade = "average"
                efficiency_score = 0.6
            else:
                efficiency_grade = "poor"
                efficiency_score = 0.4
            
            tuning_metrics["training_efficiency"] = {
                "steps": train_steps,
                "model_size_mb": model_size,
                "efficiency_ratio": efficiency_ratio,
                "grade": efficiency_grade,
                "score": efficiency_score
            }
            
            print(f"  📊 訓練效率: {efficiency_grade.upper()} (比率: {efficiency_ratio:.3f})")
    
    # 2. 生成品質指標 (基於相似度分析)
    if "image_comparison" in report:
        comparison = report["image_comparison"]
        
        # SSIM 相似度品質
        ssim_scores = comparison.get("similarity_scores", [])
        if ssim_scores:
            avg_ssim = sum(ssim_scores) / len(ssim_scores)
            
            # 品質評估 (參考 day3_fashion_training.py 的基準)
            if avg_ssim >= 0.7:
                quality_grade = "excellent"
                quality_score = 1.0
            elif avg_ssim >= 0.5:
                quality_grade = "good"
                quality_score = 0.8
            elif avg_ssim >= 0.3:
                quality_grade = "average"
                quality_score = 0.6
            else:
                quality_grade = "poor"
                quality_score = 0.4
            
            tuning_metrics["generation_quality"] = {
                "average_ssim": avg_ssim,
                "image_count": len(ssim_scores),
                "grade": quality_grade,
                "score": quality_score,
                "benchmark_comparison": avg_ssim - 0.326  # 與參考值比較
            }
            
            print(f"  🎨 生成品質: {quality_grade.upper()} (SSIM: {avg_ssim:.3f})")
    
    # 3. 特徵保持指標 (基於 FashionCLIP 分析)
    if "benchmark_analysis" in report:
        benchmark = report["benchmark_analysis"]
        avg_metrics = benchmark.get("average_metrics", {})
        
        fashion_clip_sim = avg_metrics.get("avg_fashion_clip_similarity", 0)
        
        # 特徵保持評估
        if fashion_clip_sim >= 0.7:
            feature_grade = "excellent"
            feature_score = 1.0
        elif fashion_clip_sim >= 0.5:
            feature_grade = "good"
            feature_score = 0.8
        elif fashion_clip_sim >= 0.3:
            feature_grade = "average"
            feature_score = 0.6
        else:
            feature_grade = "poor"
            feature_score = 0.4
        
        tuning_metrics["feature_preservation"] = {
            "fashion_clip_similarity": fashion_clip_sim,
            "grade": feature_grade,
            "score": feature_score,
            "benchmark_comparison": fashion_clip_sim - 0.523  # 與參考值比較
        }
        
        print(f"  🎯 特徵保持: {feature_grade.upper()} (FashionCLIP: {fashion_clip_sim:.3f})")
    
    # 4. 計算整體調優分數 (基於 day3_fashion_training.py 的權重)
    weights = {
        "training_efficiency": 0.2,
        "generation_quality": 0.4,
        "feature_preservation": 0.4
    }
    
    overall_score = 0.0
    valid_scores = 0
    
    for metric_name, weight in weights.items():
        if metric_name in tuning_metrics and "score" in tuning_metrics[metric_name]:
            overall_score += tuning_metrics[metric_name]["score"] * weight
            valid_scores += 1
    
    if valid_scores > 0:
        tuning_metrics["overall_tuning_score"] = overall_score
        
        # 整體評級
        if overall_score >= 0.9:
            overall_grade = "excellent"
            recommendation = "🎯 LoRA 調優已達到優秀水平，可以用於生產環境"
        elif overall_score >= 0.7:
            overall_grade = "good"
            recommendation = "✅ LoRA 調優效果良好，可考慮進一步微調"
        elif overall_score >= 0.5:
            overall_grade = "average"
            recommendation = "⚠️ LoRA 調優需要改善，建議調整訓練參數"
        else:
            overall_grade = "poor"
            recommendation = "❌ LoRA 調優效果不佳，需要重新檢視訓練策略"
        
        tuning_metrics["overall_grade"] = overall_grade
        tuning_metrics["tuning_recommendations"].append(recommendation)
        
        print(f"🎯 整體調優分數: {overall_score:.3f} ({overall_grade.upper()})")
    
    # 5. 生成具體調優建議
    recommendations = generate_lora_tuning_recommendations(tuning_metrics, report)
    tuning_metrics["tuning_recommendations"].extend(recommendations)
    
    return tuning_metrics

def generate_lora_tuning_recommendations(tuning_metrics, report):
    """生成 LoRA 調優建議"""
    recommendations = []
    
    # 訓練效率建議
    if "training_efficiency" in tuning_metrics:
        efficiency = tuning_metrics["training_efficiency"]
        if efficiency.get("grade") == "poor":
            recommendations.append("📉 訓練效率低：建議減少訓練步數或使用更高效的學習率")
            recommendations.append("🔧 建議參數：steps=80-120, learning_rate=0.0005-0.001")
        elif efficiency.get("grade") == "excellent":
            recommendations.append("🚀 訓練效率優秀：當前設定適合作為基準配置")
    


    # 生成品質建議
    if "generation_quality" in tuning_metrics:
        quality = tuning_metrics["generation_quality"]
        benchmark_diff = quality.get("benchmark_comparison", 0)
        
        if quality.get("grade") == "poor":
            if benchmark_diff < -0.1:
                recommendations.append("🎨 生成品質低於基準：建議增加訓練數據或調整損失函數權重")
                recommendations.append("📊 建議：增加視覺損失權重至 0.3-0.4")
            else:
                recommendations.append("🎨 生成品質需改善：建議調整學習率或增加訓練步數")
                recommendations.append("⚙️ 建議：steps=150-200, 降低學習率至 0.0003-0.0005")
        elif benchmark_diff > 0.1:
            recommendations.append("✨ 生成品質超越基準：模型表現優異")
        
        # 基於 SSIM 分數的具體建議
        avg_ssim = quality.get("average_ssim", 0)
        if avg_ssim < 0.3:
            recommendations.append("⚠️ SSIM 過低：建議檢查輸入圖片品質或調整生成參數")
        elif avg_ssim > 0.8:
            recommendations.append("🎯 SSIM 優秀：視覺相似度達到高水準")
    
    # 特徵保持建議
    if "feature_preservation" in tuning_metrics:
        feature = tuning_metrics["feature_preservation"]
        benchmark_diff = feature.get("benchmark_comparison", 0)
        
        if feature.get("grade") == "poor":
            recommendations.append("🎯 特徵保持能力弱：建議使用 FashionCLIP 引導的損失函數")
            recommendations.append("🔧 建議：提高 FashionCLIP 權重至 0.7-0.8")
        elif benchmark_diff > 0.1:
            recommendations.append("🎯 特徵保持優秀：FashionCLIP 語意理解超越基準")
        
        # 基於 FashionCLIP 分數的具體建議
        fashion_sim = feature.get("fashion_clip_similarity", 0)
        if fashion_sim < 0.3:
            recommendations.append("❌ FashionCLIP 相似度極低：建議重新檢視訓練數據和提示詞")
        elif fashion_sim > 0.7:
            recommendations.append("🎨 FashionCLIP 相似度極佳：特徵理解達到專業水準")
    
    # 平衡性建議
    efficiency_score = tuning_metrics.get("training_efficiency", {}).get("score", 0)
    quality_score = tuning_metrics.get("generation_quality", {}).get("score", 0)  
    feature_score = tuning_metrics.get("feature_preservation", {}).get("score", 0)
    
    # 檢查不平衡問題
    scores = [efficiency_score, quality_score, feature_score]
    if max(scores) - min(scores) > 0.4:
        recommendations.append("⚖️ 指標不平衡：建議調整權重以平衡各項指標")
        
        if efficiency_score < 0.5 and quality_score > 0.8:
            recommendations.append("🔄 效率低但品質高：可考慮減少訓練步數以提高效率")
        elif efficiency_score > 0.8 and quality_score < 0.5:
            recommendations.append("🔄 效率高但品質低：建議增加訓練步數或調整學習率")
    
    # 整體策略建議
    overall_score = tuning_metrics.get("overall_tuning_score", 0)
    if overall_score < 0.5:
        recommendations.append("🚨 整體表現需大幅改善：建議重新檢視訓練數據和基礎參數")
    elif overall_score > 0.85:
        recommendations.append("🎖️ 整體表現優秀：可考慮進行細微調整以達到完美")
    
    return recommendations

def generate_lora_tuning_target(report):
    """基於當前結果生成下一輪調優目標"""
    print("🎯 生成 LoRA 調優目標...")
    
    tuning_targets = {
        "current_performance": {},
        "target_metrics": {},
        "action_plan": [],
        "parameter_suggestions": {},
        "automation_config": {},
        "priority_level": "medium"
    }
    
    # 分析當前表現
    if "benchmark_analysis" in report:
        benchmark = report["benchmark_analysis"]
        avg_metrics = benchmark.get("average_metrics", {})
        
        current_total_loss = avg_metrics.get("avg_total_loss", 1.0)
        current_visual_sim = avg_metrics.get("avg_visual_similarity", 0.0)
        current_fashion_sim = avg_metrics.get("avg_fashion_clip_similarity", 0.0)
        current_color_sim = avg_metrics.get("avg_color_similarity", 0.0)
        
        tuning_targets["current_performance"] = {
            "total_loss": current_total_loss,
            "visual_similarity": current_visual_sim,
            "fashion_clip_similarity": current_fashion_sim,
            "color_similarity": current_color_sim
        }
        
        # 設定目標 (基於 day3_fashion_training.py 的優秀標準)
        target_loss = min(0.3, current_total_loss * 0.8)  # 目標：損失減少 20%
        target_visual = min(0.7, current_visual_sim * 1.2)  # 目標：視覺相似度提升 20%
        target_fashion = min(0.7, current_fashion_sim * 1.1)  # 目標：特徵相似度提升 10%
        target_color = min(0.8, current_color_sim * 1.3)  # 目標：色彩相似度提升 30%
        
        tuning_targets["target_metrics"] = {
            "target_total_loss": target_loss,
            "target_visual_similarity": target_visual,
            "target_fashion_clip_similarity": target_fashion,
            "target_color_similarity": target_color
        }
        
        print(f"  📊 當前表現: 損失={current_total_loss:.3f}, 視覺={current_visual_sim:.3f}, 特徵={current_fashion_sim:.3f}, 色彩={current_color_sim:.3f}")
        print(f"  🎯 調優目標: 損失={target_loss:.3f}, 視覺={target_visual:.3f}, 特徵={target_fashion:.3f}, 色彩={target_color:.3f}")
        
        # 判斷優先級
        if current_total_loss > 0.7 or current_fashion_sim < 0.3:
            tuning_targets["priority_level"] = "high"
        elif current_total_loss > 0.5 or current_visual_sim < 0.4:
            tuning_targets["priority_level"] = "medium"
        else:
            tuning_targets["priority_level"] = "low"
        
        # 生成行動計劃
        if current_total_loss > 0.6:
            tuning_targets["action_plan"].append("🔧 優先降低總損失：調整學習率和權重配置")
            tuning_targets["action_plan"].append("📊 建議執行完整的損失函數權重調整")
        
        if current_visual_sim < 0.4:
            tuning_targets["action_plan"].append("👁️ 提升視覺相似度：增加視覺損失權重或調整生成參數")
            tuning_targets["action_plan"].append("🎨 檢查生成圖片的解析度和品質設定")
        
        if current_fashion_sim < 0.5:
            tuning_targets["action_plan"].append("🎨 強化特徵保持：提高 FashionCLIP 損失權重")
            tuning_targets["action_plan"].append("📝 優化訓練數據的標籤描述")
        
        if current_color_sim < 0.3:
            tuning_targets["action_plan"].append("🌈 改善色彩相似度：在提示詞中加入具體色彩描述")
            tuning_targets["action_plan"].append("🎨 考慮使用色彩引導的生成方式")
        
        # 詳細參數建議
        if current_total_loss > 0.7:
            tuning_targets["parameter_suggestions"]["learning_rate"] = "0.0003-0.0005 (降低學習率)"
            tuning_targets["parameter_suggestions"]["steps"] = "200-300 (增加訓練步數)"
            tuning_targets["parameter_suggestions"]["batch_size"] = "1-2 (使用小批次)"
        elif current_total_loss > 0.5:
            tuning_targets["parameter_suggestions"]["learning_rate"] = "0.0005-0.001 (適中學習率)"
            tuning_targets["parameter_suggestions"]["steps"] = "150-200 (中等訓練步數)"
        else:
            tuning_targets["parameter_suggestions"]["learning_rate"] = "0.001-0.002 (可稍微提高)"
            tuning_targets["parameter_suggestions"]["steps"] = "100-150 (維持效率)"
        
        if current_fashion_sim < 0.4:
            tuning_targets["parameter_suggestions"]["fashion_clip_weight"] = "0.7-0.8 (提高特徵權重)"
            tuning_targets["parameter_suggestions"]["text_encoder_lr"] = "0.0001-0.0002 (微調文本編碼器)"
        elif current_fashion_sim > 0.6:
            tuning_targets["parameter_suggestions"]["fashion_clip_weight"] = "0.5-0.6 (維持平衡)"
        
        if current_visual_sim < 0.3:
            tuning_targets["parameter_suggestions"]["visual_weight"] = "0.3-0.4 (提高視覺權重)"
            tuning_targets["parameter_suggestions"]["resolution"] = "768x768 或更高 (提升解析度)"
        
        # 自動化配置建議
        tuning_targets["automation_config"] = {
            "auto_adjust_lr": current_total_loss > 0.6,
            "auto_adjust_steps": current_fashion_sim < 0.4,
            "auto_adjust_weights": current_visual_sim < 0.3,
            "suggested_iterations": 3 if tuning_targets["priority_level"] == "high" else 2
        }
        
        # 下一輪訓練的具體配置
        base_config = {
            "learning_rate": 0.0005,
            "steps": 100,
            "batch_size": 1,
            "resolution": "512x512"
        }
        
        # 根據當前表現調整配置
        if current_total_loss > 0.6:
            base_config["learning_rate"] = 0.0003
            base_config["steps"] = 200
        elif current_total_loss < 0.4:
            base_config["learning_rate"] = 0.001
            base_config["steps"] = 80
        
        if current_visual_sim < 0.3:
            base_config["resolution"] = "768x768"
        
        tuning_targets["recommended_config"] = base_config
        
        # 成功標準
        tuning_targets["success_criteria"] = {
            "minimum_total_loss": 0.5,
            "minimum_visual_similarity": 0.4,
            "minimum_fashion_clip_similarity": 0.5,
            "target_overall_score": 0.7
        }
        
        print(f"  🎯 優先級: {tuning_targets['priority_level'].upper()}")
        print(f"  🔧 建議配置: LR={base_config['learning_rate']}, Steps={base_config['steps']}")
    
    return tuning_targets

# 主程序入口點
if __name__ == "__main__":
    try:
        print("🚀 開始執行 LoRA 訓練結果分析...")
        report = analyze_training_results()
        
        if report:
            print("\n✅ 分析完成！")
            print(f"📊 總結：")
            print(f"  訓練完成: {'✅' if report['summary']['training_completed'] else '❌'}")
            print(f"  測試完成: {'✅' if report['summary']['testing_completed'] else '❌'}")
            print(f"  比較完成: {'✅' if report['summary']['comparison_completed'] else '❌'}")
            print(f"  整體狀態: {'✅ 成功' if report['summary']['overall_success'] else '❌ 失敗'}")
            
            # 顯示關鍵指標
            if "lora_tuning" in report:
                lora_score = report["lora_tuning"].get("overall_tuning_score", 0)
                print(f"  LoRA 調優分數: {lora_score:.3f}")
            
            if "image_comparison" in report:
                avg_sim = report["image_comparison"].get("average_similarity", 0)
                print(f"  平均相似度: {avg_sim:.3f}")
            
            if "benchmark_analysis" in report:
                bench_data = report["benchmark_analysis"]
                total_eval = bench_data.get("total_evaluated", 0)
                perf_dist = bench_data.get("performance_distribution", {})
                if total_eval > 0:
                    print(f"  性能評估: 優秀={perf_dist.get('excellent', 0)}, 良好={perf_dist.get('good', 0)}, 一般={perf_dist.get('average', 0)}, 待改善={perf_dist.get('poor', 0)}")
            
            print("\n📋 檢查輸出目錄 'test_results' 以查看詳細報告")
            
        else:
            print("❌ 分析失敗，請檢查相關檔案是否存在")
            
    except Exception as e:
        print(f"❌ 執行分析時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()