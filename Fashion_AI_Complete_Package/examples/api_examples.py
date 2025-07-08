#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fashion AI Complete Package - API 使用範例
展示如何使用 Fashion AI 系統的 REST API

功能演示：
1. 圖片分析 API
2. 圖片生成 API
3. 批次處理 API
4. 系統狀態 API
"""

import requests
import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

class FashionAPIClient:
    """Fashion AI API 客戶端"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Fashion-AI-Client/1.0'
        })
    
    def check_status(self) -> Dict[str, Any]:
        """檢查系統狀態"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "available": False}
    
    def upload_image(self, image_path: str) -> Dict[str, Any]:
        """上傳圖片"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(
                    f"{self.base_url}/upload", 
                    files=files
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """分析圖片"""
        try:
            # 先上傳圖片
            upload_result = self.upload_image(image_path)
            if not upload_result.get('success'):
                return upload_result
            
            # 分析圖片
            data = {'filepath': upload_result['filepath']}
            response = self.session.post(
                f"{self.base_url}/api/v1/analyze",
                json=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def generate_image(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成圖片"""
        try:
            data = {
                'prompt': prompt,
                'options': options or {}
            }
            response = self.session.post(
                f"{self.base_url}/api/v1/generate",
                json=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def batch_process(self, file_paths: List[str], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """批次處理"""
        try:
            data = {
                'files': file_paths,
                'options': options or {}
            }
            response = self.session.post(
                f"{self.base_url}/api/v1/batch",
                json=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_config(self) -> Dict[str, Any]:
        """獲取配置"""
        try:
            response = self.session.get(f"{self.base_url}/config")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def download_file(self, filename: str, save_path: str) -> bool:
        """下載檔案"""
        try:
            response = self.session.get(f"{self.base_url}/download/{filename}")
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"下載失敗: {e}")
            return False

def example_api_status_check():
    """範例：API 狀態檢查"""
    print("=" * 60)
    print("🔍 API 狀態檢查範例")
    print("=" * 60)
    
    client = FashionAPIClient()
    
    print("🌐 檢查 API 狀態...")
    status = client.check_status()
    
    if status.get('available', False):
        print("✅ API 服務正常運行")
        print(f"分析器狀態: {'✅' if status.get('analyzer') else '❌'}")
        print(f"WebUI API: {'✅' if status.get('webui_api') else '❌'}")
        print(f"GPU 可用: {'✅' if status.get('gpu_available') else '❌'}")
    else:
        print("❌ API 服務不可用")
        print(f"錯誤: {status.get('error', 'Unknown')}")
    
    # 獲取配置
    print("\n⚙️ 獲取系統配置...")
    config = client.get_config()
    if 'error' not in config:
        print(f"Web 端口: {config.get('web_port', 'Unknown')}")
        print(f"WebUI URL: {config.get('webui_url', 'Unknown')}")
        print(f"最大圖片尺寸: {config.get('max_image_size', 'Unknown')}")
    else:
        print(f"❌ 獲取配置失敗: {config['error']}")

def example_api_image_analysis():
    """範例：API 圖片分析"""
    print("\n" + "=" * 60)
    print("🔍 API 圖片分析範例")
    print("=" * 60)
    
    client = FashionAPIClient()
    
    # 範例圖片路徑
    sample_image = Path(__file__).parent / "sample_images" / "dress_sample.jpg"
    
    if not sample_image.exists():
        print(f"❌ 找不到範例圖片: {sample_image}")
        print("請將圖片檔案放在 examples/sample_images/ 目錄下")
        return
    
    print(f"📸 分析圖片: {sample_image}")
    
    # 分析圖片
    result = client.analyze_image(str(sample_image))
    
    if result.get('success'):
        analysis = result['result']
        print("\n📊 分析結果:")
        print(f"類別: {analysis.get('category', 'Unknown')}")
        print(f"風格: {analysis.get('style', 'Unknown')}")
        print(f"顏色: {analysis.get('colors', [])}")
        print(f"材質: {analysis.get('materials', [])}")
        print(f"置信度: {analysis.get('confidence', 0.0):.2f}")
        
        # 保存結果
        output_file = result.get('output_file')
        if output_file:
            print(f"\n💾 結果檔案: {output_file}")
            
            # 下載結果檔案
            save_path = Path(__file__).parent.parent / "data" / "output" / "api_analysis_result.json"
            if client.download_file(output_file, str(save_path)):
                print(f"✅ 結果已下載至: {save_path}")
            else:
                print("❌ 下載結果檔案失敗")
    else:
        print(f"❌ 分析失敗: {result.get('error', 'Unknown')}")

def example_api_image_generation():
    """範例：API 圖片生成"""
    print("\n" + "=" * 60)
    print("🎨 API 圖片生成範例")
    print("=" * 60)
    
    client = FashionAPIClient()
    
    # 測試提示詞
    prompts = [
        {
            "prompt": "elegant red dress, silk fabric, long sleeves, professional photography",
            "options": {
                "steps": 20,
                "cfg_scale": 7.5,
                "width": 512,
                "height": 512
            }
        },
        {
            "prompt": "casual blue jeans, cotton fabric, street style, natural lighting",
            "options": {
                "steps": 25,
                "cfg_scale": 8.0,
                "width": 512,
                "height": 512
            }
        }
    ]
    
    for i, prompt_data in enumerate(prompts, 1):
        print(f"\n🖼️ 生成圖片 {i}/{len(prompts)}:")
        print(f"提示詞: {prompt_data['prompt']}")
        print(f"參數: {prompt_data['options']}")
        
        # 生成圖片
        result = client.generate_image(prompt_data['prompt'], prompt_data['options'])
        
        if result.get('success'):
            image_url = result.get('image_url')
            print(f"✅ 圖片生成成功: {image_url}")
            
            # 下載生成的圖片
            if image_url:
                filename = image_url.split('/')[-1]
                save_path = Path(__file__).parent.parent / "data" / "output" / f"api_generated_{i}.png"
                
                # 從 URL 下載
                try:
                    response = requests.get(f"http://localhost:8080{image_url}")
                    response.raise_for_status()
                    
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"💾 圖片已保存至: {save_path}")
                except Exception as e:
                    print(f"❌ 下載圖片失敗: {e}")
        else:
            print(f"❌ 生成失敗: {result.get('error', 'Unknown')}")
        
        # 等待一段時間避免 API 過載
        time.sleep(2)

def example_api_batch_processing():
    """範例：API 批次處理"""
    print("\n" + "=" * 60)
    print("📦 API 批次處理範例")
    print("=" * 60)
    
    client = FashionAPIClient()
    
    # 準備批次處理的圖片
    input_dir = Path(__file__).parent / "sample_images"
    
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
    print(f"📸 找到 {len(image_files)} 個圖片檔案")
    
    # 上傳所有圖片並獲取檔案路徑
    file_paths = []
    for image_file in image_files:
        print(f"⬆️ 上傳: {image_file.name}")
        upload_result = client.upload_image(str(image_file))
        
        if upload_result.get('success'):
            file_paths.append(upload_result['filepath'])
            print(f"✅ 上傳成功")
        else:
            print(f"❌ 上傳失敗: {upload_result.get('error', 'Unknown')}")
    
    if not file_paths:
        print("❌ 沒有成功上傳的圖片")
        return
    
    # 執行批次處理
    print(f"\n🔄 開始批次處理 {len(file_paths)} 個圖片...")
    
    batch_options = {
        "include_details": True,
        "save_individual_results": True
    }
    
    result = client.batch_process(file_paths, batch_options)
    
    if result.get('success'):
        results = result['results']
        batch_file = result.get('batch_file')
        
        print(f"\n📊 批次處理完成:")
        successful = sum(1 for r in results if r.get('success'))
        failed = len(results) - successful
        
        print(f"成功: {successful}")
        print(f"失敗: {failed}")
        print(f"總計: {len(results)}")
        
        if batch_file:
            print(f"\n💾 批次結果檔案: {batch_file}")
            
            # 下載批次結果
            save_path = Path(__file__).parent.parent / "data" / "output" / "api_batch_results.json"
            if client.download_file(batch_file, str(save_path)):
                print(f"✅ 批次結果已下載至: {save_path}")
            else:
                print("❌ 下載批次結果失敗")
    else:
        print(f"❌ 批次處理失敗: {result.get('error', 'Unknown')}")

def example_api_workflow():
    """範例：完整的 API 工作流程"""
    print("\n" + "=" * 60)
    print("🔄 完整 API 工作流程範例")
    print("=" * 60)
    
    client = FashionAPIClient()
    
    # 1. 檢查系統狀態
    print("步驟 1: 檢查系統狀態")
    status = client.check_status()
    if not status.get('available', False):
        print("❌ 系統不可用，停止執行")
        return
    print("✅ 系統可用")
    
    # 2. 分析圖片
    print("\n步驟 2: 分析圖片")
    sample_image = Path(__file__).parent / "sample_images" / "dress_sample.jpg"
    
    if not sample_image.exists():
        print("❌ 找不到範例圖片，跳過分析步驟")
        return
    
    analysis_result = client.analyze_image(str(sample_image))
    if not analysis_result.get('success'):
        print("❌ 圖片分析失敗，停止執行")
        return
    
    analysis = analysis_result['result']
    print(f"✅ 分析完成: {analysis.get('category', 'Unknown')}")
    
    # 3. 基於分析結果生成新圖片
    print("\n步驟 3: 基於分析結果生成新圖片")
    
    # 構建提示詞
    category = analysis.get('category', 'clothing')
    style = analysis.get('style', 'elegant')
    colors = analysis.get('colors', ['red'])
    
    prompt = f"{category}, {style} style, {', '.join(colors[:2])} colors, professional photography, high quality"
    
    print(f"生成提示詞: {prompt}")
    
    generation_result = client.generate_image(prompt)
    if generation_result.get('success'):
        print("✅ 圖片生成成功")
        
        # 下載生成的圖片
        image_url = generation_result.get('image_url')
        if image_url:
            save_path = Path(__file__).parent.parent / "data" / "output" / "workflow_generated.png"
            try:
                response = requests.get(f"http://localhost:8080{image_url}")
                response.raise_for_status()
                
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"💾 生成的圖片已保存至: {save_path}")
            except Exception as e:
                print(f"❌ 下載圖片失敗: {e}")
    else:
        print("❌ 圖片生成失敗")
    
    print("\n🎉 工作流程完成！")

def main():
    """主函數 - 執行所有 API 範例"""
    print("🚀 Fashion AI Complete Package - API 使用範例")
    print("=" * 80)
    
    # 創建必要目錄
    output_dir = Path(__file__).parent.parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. API 狀態檢查
        example_api_status_check()
        
        # 2. 圖片分析
        example_api_image_analysis()
        
        # 3. 圖片生成（需要 WebUI API）
        print("\n⚠️ 圖片生成範例需要 WebUI API 運行")
        user_input = input("是否執行圖片生成範例？(y/n): ").lower().strip()
        if user_input in ['y', 'yes']:
            example_api_image_generation()
        
        # 4. 批次處理
        example_api_batch_processing()
        
        # 5. 完整工作流程
        print("\n⚠️ 完整工作流程範例需要 WebUI API 運行")
        user_input = input("是否執行完整工作流程範例？(y/n): ").lower().strip()
        if user_input in ['y', 'yes']:
            example_api_workflow()
        
    except KeyboardInterrupt:
        print("\n👋 API 範例執行被中斷")
    except Exception as e:
        print(f"\n❌ API 範例執行失敗: {e}")
    
    print("\n🎉 API 範例執行完成！")
    print("查看 data/output/ 目錄以獲取結果檔案")

if __name__ == "__main__":
    main()
