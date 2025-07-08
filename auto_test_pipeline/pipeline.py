import subprocess
import sys
import os
from datetime import datetime

def run_full_pipeline():
    """執行完整的 LoRA 訓練 + 測試 + 分析 pipeline"""
    
    print("🚀 開始完整 LoRA 訓練流程...")
    print(f"⏰ 開始時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 檢查必要檔案
    required_files = ["train_lora.py", "infer_lora.py", "analyze_results.py"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ 找不到必要檔案：{file}")
            return False
    
    # 第一階段：訓練 LoRA
    print("\n📚 第一階段：LoRA 訓練")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, "train_lora.py"], 
                              capture_output=False, text=True)
        if result.returncode != 0:
            print("❌ LoRA 訓練失敗")
            return False
        print("✅ LoRA 訓練完成")
    except Exception as e:
        print(f"❌ LoRA 訓練過程發生錯誤：{str(e)}")
        return False
    
    # 第二階段：產生測試圖片
    print("\n🎨 第二階段：產生測試圖片")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, "infer_lora.py"], 
                              capture_output=False, text=True)
        if result.returncode != 0:
            print("❌ 測試圖片產生失敗")
            return False
        print("✅ 測試圖片產生完成")
    except Exception as e:
        print(f"❌ 測試圖片產生過程發生錯誤：{str(e)}")
        return False
    
    # 第三階段：分析結果
    print("\n📊 第三階段：分析結果")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, "analyze_results.py"], 
                              capture_output=False, text=True)
        if result.returncode != 0:
            print("❌ 結果分析失敗")
            return False
        print("✅ 結果分析完成")
    except Exception as e:
        print(f"❌ 結果分析過程發生錯誤：{str(e)}")
        return False
    
    # 完成總結
    print("\n" + "=" * 60)
    print("🎉 完整 LoRA 訓練流程執行完成！")
    print("=" * 60)
    
    print("\n📁 產生的檔案：")
    if os.path.exists("lora_output"):
        lora_files = [f for f in os.listdir("lora_output") if f.endswith('.safetensors')]
        print(f"  📦 LoRA 模型：{len(lora_files)} 個")
        for lora in lora_files:
            print(f"    - {lora}")
    
    if os.path.exists("test_images"):
        test_images = [f for f in os.listdir("test_images") if f.endswith('.png')]
        print(f"  🎨 測試圖片：{len(test_images)} 張")
    
    if os.path.exists("training_report.html"):
        print(f"  📋 HTML 報告：training_report.html")
    
    if os.path.exists("training_report.json"):
        print(f"  📊 JSON 報告：training_report.json")
    
    if os.path.exists("training_charts.png"):
        print(f"  📈 分析圖表：training_charts.png")
    
    print(f"\n⏰ 完成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🌐 開啟 training_report.html 查看完整報告")
    
    return True

if __name__ == "__main__":
    success = run_full_pipeline()
    sys.exit(0 if success else 1)
