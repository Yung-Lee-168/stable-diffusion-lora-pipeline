#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fashion AI Complete Package - 基本使用範例
展示如何使用 Fashion AI 系統的核心功能

功能演示：
1. 圖片分析
2. 提示詞生成
3. 圖片生成
4. 批次處理
"""

import os
import sys
import json
from pathlib import Path

# 添加專案路徑
sys.path.append(str(Path(__file__).parent.parent))

from core.fashion_analyzer import FashionTrainingPipeline
from core.prompt_generator import FashionPromptGenerator, PromptStyle
from core.webui_connector import ColabEnvironmentSetup
from utils.system_check import SystemTester

def example_image_analysis():
    """範例：圖片分析"""
    print("=" * 60)
    print("🔍 圖片分析範例")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = FashionTrainingPipeline()
    
    # 範例圖片路徑
    sample_image = Path(__file__).parent / "sample_images" / "dress_sample.jpg"
    
    if sample_image.exists():
        print(f"📸 分析圖片: {sample_image}")
        
        # 執行分析
        result = analyzer.analyze_image(str(sample_image))
        
        print("\n📊 分析結果:")
        print(f"類別: {result.get('category', 'Unknown')}")
        print(f"風格: {result.get('style', 'Unknown')}")
        print(f"顏色: {result.get('colors', [])}")
        print(f"材質: {result.get('materials', [])}")
        print(f"置信度: {result.get('confidence', 0.0):.2f}")
        
        # 保存結果
        output_path = Path(__file__).parent.parent / "data" / "output" / "analysis_result.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 結果已保存至: {output_path}")
        
    else:
        print(f"❌ 找不到範例圖片: {sample_image}")
        print("請將圖片檔案放在 examples/sample_images/ 目錄下")

def example_prompt_generation():
    """範例：提示詞生成"""
    print("\n" + "=" * 60)
    print("✍️ 提示詞生成範例")
    print("=" * 60)
    
    # 模擬分析結果
    analysis_result = {
        'category': 'dress',
        'style': 'elegant',
        'colors': ['red', 'black'],
        'materials': ['silk', 'lace'],
        'detailed_features': ['long sleeves', 'v-neck', 'floor length'],
        'confidence': 0.85
    }
    
    print("📝 基於分析結果生成提示詞:")
    print(f"輸入: {analysis_result}")
    
    # 初始化提示詞生成器
    generator = FashionPromptGenerator()
    
    # 生成多種風格的提示詞
    styles = [PromptStyle.MINIMAL, PromptStyle.DETAILED, PromptStyle.ARTISTIC]
    
    for style in styles:
        result = generator.generate_prompt(analysis_result, style)
        
        print(f"\n🎨 {style.value.upper()} 風格:")
        print(f"正面提示詞: {result['positive_prompt']}")
        print(f"負面提示詞: {result['negative_prompt']}")
        
        # 保存結果
        output_path = Path(__file__).parent.parent / "data" / "output" / f"prompt_{style.value}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

def example_image_generation():
    """範例：圖片生成"""
    print("\n" + "=" * 60)
    print("🎨 圖片生成範例")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = FashionTrainingPipeline()
    
    # 測試提示詞
    prompts = [
        "elegant red dress, silk fabric, long sleeves, v-neck, floor length, professional photography",
        "casual blue jeans, cotton fabric, relaxed fit, street style, natural lighting",
        "formal black suit, wool fabric, tailored fit, business style, studio lighting"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🖼️ 生成圖片 {i}/3:")
        print(f"提示詞: {prompt}")
        
        try:
            # 生成圖片
            result = analyzer.generate_image(prompt)
            
            if result and 'image_path' in result:
                print(f"✅ 圖片已生成: {result['image_path']}")
                
                # 複製到輸出目錄
                output_path = Path(__file__).parent.parent / "data" / "output" / f"generated_{i}.png"
                import shutil
                shutil.copy2(result['image_path'], output_path)
                
                print(f"💾 圖片已保存至: {output_path}")
            else:
                print("❌ 圖片生成失敗")
                
        except Exception as e:
            print(f"❌ 生成失敗: {e}")

def example_batch_processing():
    """範例：批次處理"""
    print("\n" + "=" * 60)
    print("📦 批次處理範例")
    print("=" * 60)
    
    # 準備批次處理的圖片
    input_dir = Path(__file__).parent / "sample_images"
    output_dir = Path(__file__).parent.parent / "data" / "output" / "batch_results"
    
    # 創建輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"❌ 輸入目錄不存在: {input_dir}")
        print("請創建 examples/sample_images/ 目錄並放入圖片檔案")
        return
    
    # 獲取圖片檔案
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(input_dir.glob(ext))
    
    if not image_files:
        print(f"❌ 在 {input_dir} 中找不到圖片檔案")
        return
    
    print(f"📁 輸入目錄: {input_dir}")
    print(f"📁 輸出目錄: {output_dir}")
    print(f"📸 找到 {len(image_files)} 個圖片檔案")
    
    # 初始化分析器
    analyzer = FashionTrainingPipeline()
    
    # 批次處理
    results = []
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n🔄 處理 {i}/{len(image_files)}: {image_file.name}")
        
        try:
            # 分析圖片
            result = analyzer.analyze_image(str(image_file))
            
            # 添加檔案資訊
            result['source_file'] = str(image_file)
            result['processed_at'] = str(Path(__file__).parent.parent)
            
            results.append({
                'file': image_file.name,
                'success': True,
                'result': result
            })
            
            print(f"✅ 分析完成: {result.get('category', 'Unknown')}")
            
        except Exception as e:
            results.append({
                'file': image_file.name,
                'success': False,
                'error': str(e)
            })
            print(f"❌ 分析失敗: {e}")
    
    # 保存批次結果
    batch_result_path = output_dir / "batch_analysis_results.json"
    with open(batch_result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 批次結果已保存至: {batch_result_path}")
    
    # 統計結果
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n📊 批次處理統計:")
    print(f"成功: {successful}")
    print(f"失敗: {failed}")
    print(f"總計: {len(results)}")

def example_system_check():
    """範例：系統檢查"""
    print("\n" + "=" * 60)
    print("🔍 系統檢查範例")
    print("=" * 60)
    
    # 初始化系統檢查器
    checker = SystemTester()
    
    # 檢查 GPU
    print("🔧 檢查 GPU 狀態...")
    gpu_info = checker.check_gpu()
    if gpu_info:
        print(f"✅ GPU: {gpu_info['name']}")
        print(f"💾 VRAM: {gpu_info['memory_gb']:.1f} GB")
    else:
        print("❌ 沒有可用的 GPU")
    
    # 檢查 WebUI API
    print("\n🌐 檢查 WebUI API 連接...")
    try:
        import requests
        response = requests.get("http://localhost:7860/sdapi/v1/options", timeout=5)
        if response.status_code == 200:
            print("✅ WebUI API 連接正常")
        else:
            print(f"❌ WebUI API 連接失敗: {response.status_code}")
    except Exception as e:
        print(f"❌ WebUI API 連接失敗: {e}")
    
    # 檢查模型
    print("\n📦 檢查模型狀態...")
    model_dir = Path(__file__).parent.parent / "models"
    if model_dir.exists():
        print(f"✅ 模型目錄存在: {model_dir}")
    else:
        print(f"❌ 模型目錄不存在: {model_dir}")
    
    # 檢查配置
    print("\n⚙️ 檢查配置檔案...")
    config_dir = Path(__file__).parent.parent / "config"
    config_files = ['default_config.yaml', 'api_config.yaml', 'model_config.yaml']
    
    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} 不存在")

def main():
    """主函數 - 執行所有範例"""
    print("🚀 Fashion AI Complete Package - 使用範例")
    print("=" * 80)
    
    # 創建必要目錄
    data_dir = Path(__file__).parent.parent / "data"
    for subdir in ['input', 'output', 'cache']:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 系統檢查
        example_system_check()
        
        # 2. 圖片分析
        example_image_analysis()
        
        # 3. 提示詞生成
        example_prompt_generation()
        
        # 4. 圖片生成（需要 WebUI API）
        print("\n⚠️ 圖片生成範例需要 WebUI API 運行")
        user_input = input("是否執行圖片生成範例？(y/n): ").lower().strip()
        if user_input in ['y', 'yes']:
            example_image_generation()
        
        # 5. 批次處理
        example_batch_processing()
        
    except KeyboardInterrupt:
        print("\n👋 範例執行被中斷")
    except Exception as e:
        print(f"\n❌ 範例執行失敗: {e}")
    
    print("\n🎉 範例執行完成！")
    print("查看 data/output/ 目錄以獲取結果檔案")

if __name__ == "__main__":
    main()
