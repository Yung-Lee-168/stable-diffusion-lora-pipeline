#!/usr/bin/env python3
"""
Stable Diffusion WebUI API 客戶端
用於透過 API 呼叫生成圖片
"""

import requests
import base64
import json
import time
from datetime import datetime
import os

class SDWebUIAPI:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url
        self.txt2img_url = f"{base_url}/sdapi/v1/txt2img"
        self.img2img_url = f"{base_url}/sdapi/v1/img2img"
        self.models_url = f"{base_url}/sdapi/v1/sd-models"
        self.samplers_url = f"{base_url}/sdapi/v1/samplers"
        
    def check_server_status(self):
        """檢查服務器是否正常運行"""
        try:
            response = requests.get(f"{self.base_url}/sdapi/v1/options", timeout=10)
            if response.status_code == 200:
                print("✅ 服務器運行正常")
                return True
            else:
                print(f"❌ 服務器回應異常: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ 無法連接到服務器: {e}")
            return False
    
    def get_available_models(self):
        """獲取可用的模型列表"""
        try:
            response = requests.get(self.models_url)
            if response.status_code == 200:
                models = response.json()
                print("📋 可用模型:")
                for model in models:
                    print(f"  - {model['model_name']}")
                return models
            else:
                print(f"無法獲取模型列表: {response.status_code}")
                return []
        except Exception as e:
            print(f"獲取模型列表失敗: {e}")
            return []
    
    def get_available_samplers(self):
        """獲取可用的採樣器列表"""
        try:
            response = requests.get(self.samplers_url)
            if response.status_code == 200:
                samplers = response.json()
                print("🎛️ 可用採樣器:")
                for sampler in samplers:
                    print(f"  - {sampler['name']}")
                return samplers
            else:
                print(f"無法獲取採樣器列表: {response.status_code}")
                return []
        except Exception as e:
            print(f"獲取採樣器列表失敗: {e}")
            return []
    
    def text_to_image(self, prompt, negative_prompt="", width=512, height=512, 
                     steps=20, cfg_scale=7, sampler_name="Euler", seed=-1):
        """文字轉圖像"""
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler_name,
            "seed": seed,
            "n_iter": 1,
            "batch_size": 1,
            "save_images": False,
            "send_images": True
        }
        
        print(f"🎨 開始生成圖像...")
        print(f"📝 提示詞: {prompt}")
        print(f"🚫 負向提示: {negative_prompt}")
        print(f"📐 尺寸: {width}x{height}")
        print(f"🔢 步數: {steps}")
        
        try:
            start_time = time.time()
            response = requests.post(self.txt2img_url, json=payload, timeout=300)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                generation_time = end_time - start_time
                
                print(f"✅ 圖像生成成功! 耗時: {generation_time:.2f} 秒")
                
                # 保存圖像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = "generated_images"
                os.makedirs(output_dir, exist_ok=True)
                
                for i, img_data in enumerate(result['images']):
                    image_data = base64.b64decode(img_data)
                    filename = f"{output_dir}/txt2img_{timestamp}_{i+1}.png"
                    
                    with open(filename, 'wb') as f:
                        f.write(image_data)
                    
                    print(f"💾 圖像已保存: {filename}")
                
                return result
            else:
                print(f"❌ 生成失敗: {response.status_code}")
                print(f"錯誤信息: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ 請求超時，請檢查服務器狀態或增加超時時間")
            return None
        except Exception as e:
            print(f"❌ 生成圖像時發生錯誤: {e}")
            return None

def main():
    """主函數 - 示範如何使用 API"""
    
    print("🚀 Stable Diffusion WebUI API 客戶端")
    print("=" * 50)
    
    # 創建 API 客戶端
    api = SDWebUIAPI()
    
    # 檢查服務器狀態
    if not api.check_server_status():
        print("請先啟動 Stable Diffusion WebUI 服務器")
        return
    
    print("\n" + "=" * 50)
    
    # 獲取可用資源
    api.get_available_models()
    print()
    api.get_available_samplers()
    
    print("\n" + "=" * 50)
    
    # 示範圖像生成
    prompts = [
        {
            "prompt": "a beautiful landscape with mountains and rivers, highly detailed, 4k, photorealistic",
            "negative_prompt": "blurry, low quality, watermark, text",
            "description": "美麗山水風景"
        },
        {
            "prompt": "a cute cat sitting on a wooden table, studio lighting, high quality",
            "negative_prompt": "blurry, low quality, distorted",
            "description": "可愛貓咪"
        },
        {
            "prompt": "cyberpunk city at night, neon lights, futuristic, highly detailed",
            "negative_prompt": "blurry, low quality, daylight",
            "description": "賽博朋克城市"
        }
    ]
    
    for i, example in enumerate(prompts, 1):
        print(f"\n📸 範例 {i}: {example['description']}")
        print("-" * 30)
        
        result = api.text_to_image(
            prompt=example["prompt"],
            negative_prompt=example["negative_prompt"],
            width=512,
            height=512,
            steps=20,
            cfg_scale=7,
            sampler_name="Euler"
        )
        
        if result:
            print("✅ 生成完成!")
        else:
            print("❌ 生成失敗!")
        
        print("-" * 30)

if __name__ == "__main__":
    main()
