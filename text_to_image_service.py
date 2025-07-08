#!/usr/bin/env python3
"""
Stable Diffusion WebUI API 完整解決方案
實現：輸入文字 -> 生成圖片 -> 回傳圖片檔案

功能特點：
1. 簡單易用的 API 封裝
2. 錯誤處理和重試機制
3. 多種輸出格式支援
4. 詳細的狀態監控
"""

import requests
import base64
import json
import time
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import io
from PIL import Image

class StableDiffusionAPI:
    """Stable Diffusion WebUI API 客戶端"""
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        """
        初始化 API 客戶端
        
        Args:
            base_url: WebUI 服務器地址
        """
        self.base_url = base_url.rstrip('/')
        self.txt2img_url = f"{self.base_url}/sdapi/v1/txt2img"
        self.img2img_url = f"{self.base_url}/sdapi/v1/img2img"
        self.options_url = f"{self.base_url}/sdapi/v1/options"
        self.models_url = f"{self.base_url}/sdapi/v1/sd-models"
        
        # 預設參數
        self.default_params = {
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg_scale": 7,
            "sampler_name": "Euler",
            "negative_prompt": "blurry, low quality, watermark, text, deformed, mutated",
            "seed": -1,
            "n_iter": 1,
            "batch_size": 1
        }
    
    def is_server_ready(self) -> bool:
        """檢查服務器是否就緒"""
        try:
            response = requests.get(self.options_url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_server(self, timeout: int = 300) -> bool:
        """等待服務器就緒"""
        print("🔍 檢查服務器狀態...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_server_ready():
                print("✅ 服務器就緒")
                return True
            
            print("⏳ 等待服務器啟動...", end='\r')
            time.sleep(2)
        
        print(f"\n❌ 服務器在 {timeout} 秒內未就緒")
        return False
    
    def get_models(self) -> List[Dict]:
        """獲取可用模型列表"""
        try:
            response = requests.get(self.models_url, timeout=10)
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []
    
    def generate_image(self, 
                      prompt: str,
                      negative_prompt: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        生成圖片的核心函數
        
        Args:
            prompt: 圖片描述文字
            negative_prompt: 負向描述（可選）
            **kwargs: 其他生成參數
        
        Returns:
            包含圖片數據和元信息的字典
        """
        
        # 合併參數
        params = self.default_params.copy()
        params.update(kwargs)
        params["prompt"] = prompt
        
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        
        # 確保服務器就緒
        if not self.is_server_ready():
            print("❌ 服務器未就緒")
            return {"success": False, "error": "Server not ready"}
        
        print(f"🎨 開始生成圖片...")
        print(f"📝 描述: {prompt}")
        if negative_prompt:
            print(f"🚫 排除: {negative_prompt}")
        print(f"📐 尺寸: {params['width']}x{params['height']}")
        print(f"🔢 步數: {params['steps']}")
        
        try:
            start_time = time.time()
            
            # 發送 API 請求
            response = requests.post(
                self.txt2img_url, 
                json=params, 
                timeout=300
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ 生成成功! 耗時: {generation_time:.2f} 秒")
                
                return {
                    "success": True,
                    "images": result["images"],
                    "parameters": result["parameters"],
                    "info": result["info"],
                    "generation_time": generation_time
                }
            else:
                error_msg = f"API 錯誤: {response.status_code}"
                print(f"❌ {error_msg}")
                return {"success": False, "error": error_msg, "details": response.text}
                
        except requests.exceptions.Timeout:
            error_msg = "請求超時"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"生成失敗: {str(e)}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
    
    def save_images(self, 
                   result: Dict[str, Any], 
                   output_dir: str = "generated_images",
                   prefix: str = "generated") -> List[str]:
        """
        保存生成的圖片
        
        Args:
            result: generate_image() 的返回結果
            output_dir: 輸出目錄
            prefix: 檔案名前綴
        
        Returns:
            保存的檔案路徑列表
        """
        if not result.get("success"):
            return []
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, img_data in enumerate(result["images"]):
            try:
                # 解碼 base64 圖片
                image_bytes = base64.b64decode(img_data)
                
                # 生成檔案名
                filename = f"{prefix}_{timestamp}_{i+1}.png"
                filepath = os.path.join(output_dir, filename)
                
                # 保存圖片
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                saved_files.append(filepath)
                print(f"💾 圖片已保存: {filepath}")
                
            except Exception as e:
                print(f"❌ 保存圖片失敗: {e}")
        
        return saved_files
    
    def generate_and_save(self, 
                         prompt: str, 
                         output_path: Optional[str] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        一鍵生成並保存圖片
        
        Args:
            prompt: 圖片描述
            output_path: 指定輸出路徑（可選）
            **kwargs: 其他參數
        
        Returns:
            完整的結果信息
        """
        # 生成圖片
        result = self.generate_image(prompt, **kwargs)
        
        if result["success"]:
            # 保存圖片
            if output_path:
                output_dir = os.path.dirname(output_path) or "."
                prefix = os.path.splitext(os.path.basename(output_path))[0]
            else:
                output_dir = "generated_images"
                prefix = "generated"
            
            saved_files = self.save_images(result, output_dir, prefix)
            result["saved_files"] = saved_files
        
        return result


def text_to_image_service(prompt: str, 
                         output_path: Optional[str] = None,
                         **generation_params) -> Dict[str, Any]:
    """
    文字轉圖片服務函數 - 這是您要的核心功能
    
    Args:
        prompt: 輸入的文字描述
        output_path: 輸出圖片路徑（可選）
        **generation_params: 額外的生成參數
    
    Returns:
        服務結果字典
    """
    
    # 創建 API 客戶端
    api = StableDiffusionAPI()
    
    # 檢查服務器
    if not api.wait_for_server(timeout=30):
        return {
            "success": False,
            "error": "Stable Diffusion WebUI 服務器未啟動",
            "message": "請先執行 webui-user.bat 啟動服務器"
        }
    
    # 生成並保存圖片
    result = api.generate_and_save(prompt, output_path, **generation_params)
    
    return result


def main_interactive():
    """互動式主程式"""
    print("🎨 Stable Diffusion 文字轉圖片服務")
    print("=" * 50)
    
    api = StableDiffusionAPI()
    
    # 檢查服務器狀態
    if not api.wait_for_server():
        print("\n請先啟動 Stable Diffusion WebUI:")
        print("1. 執行 webui-user.bat")
        print("2. 等待看到 'Running on local URL' 訊息")
        print("3. 重新運行此程式")
        return
    
    # 顯示可用模型
    models = api.get_models()
    if models:
        print(f"\n📋 當前模型: {models[0].get('model_name', 'Unknown')}")
    
    print("\n" + "=" * 50)
    print("請輸入圖片描述，程式將自動生成圖片")
    print("輸入 'quit' 退出程式")
    print("=" * 50)
    
    while True:
        try:
            # 獲取用戶輸入
            user_prompt = input("\n📝 請描述您想要的圖片: ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 再見!")
                break
            
            if not user_prompt:
                print("請輸入有效的描述")
                continue
            
            # 生成圖片
            result = text_to_image_service(user_prompt)
            
            if result["success"]:
                print(f"🎉 成功生成 {len(result['saved_files'])} 張圖片")
                for file_path in result["saved_files"]:
                    print(f"   📁 {file_path}")
            else:
                print(f"❌ 生成失敗: {result['error']}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 程式已中斷")
            break
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")


if __name__ == "__main__":
    main_interactive()
