#!/usr/bin/env python3
"""
Gradio API 測試工具 - 專門針對 Gradio 界面的 WebUI
"""

import requests
import json
import base64
import os
from datetime import datetime

def test_gradio_api():
    """測試 Gradio API"""
    print("🔍 測試 Gradio API...")
    
    base_url = "http://localhost:7860"
    
    # 首先獲取 API 信息
    try:
        response = requests.get(f"{base_url}/info", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print("✅ 成功獲取 API 信息")
            print(f"   命名空間: {info.get('named_endpoints', {}).keys()}")
            
            # 查找文本到圖像的端點
            named_endpoints = info.get('named_endpoints', {})
            
            # 常見的文本到圖像端點名稱
            txt2img_candidates = []
            for endpoint_name in named_endpoints.keys():
                if any(keyword in endpoint_name.lower() for keyword in ['txt2img', 'text_to_image', 'generate']):
                    txt2img_candidates.append(endpoint_name)
            
            if txt2img_candidates:
                print(f"🎯 找到可能的文本到圖像端點: {txt2img_candidates}")
                return txt2img_candidates[0]  # 返回第一個候選
            else:
                print("⚠️ 沒有找到明顯的文本到圖像端點")
                print("📋 所有可用端點:")
                for name in named_endpoints.keys():
                    print(f"   - {name}")
                return list(named_endpoints.keys())[0] if named_endpoints else None
        else:
            print(f"❌ 無法獲取 API 信息: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ API 信息獲取失敗: {e}")
        return None

def test_gradio_generation(endpoint_name):
    """測試 Gradio 圖像生成"""
    print(f"\n🎨 測試 Gradio 圖像生成: {endpoint_name}")
    
    base_url = "http://localhost:7860"
    
    # Gradio API 的數據格式
    payload = {
        "data": [
            "a beautiful woman in elegant dress",  # prompt
            "",  # negative_prompt
            [],  # styles (可能為空)
            20,  # steps
            "DPM++ 2M Karras",  # sampler
            False,  # restore_faces
            False,  # tiling
            1,  # n_iter
            1,  # batch_size
            7.5,  # cfg_scale
            -1,  # seed
            -1,  # subseed
            0,  # subseed_strength
            0,  # seed_resize_from_h
            0,  # seed_resize_from_w
            False,  # seed_enable_extras
            512,  # height
            512,  # width
            False,  # enable_hr
            0.7,  # denoising_strength
            2,  # hr_scale
            "Latent",  # hr_upscaler
            0,  # hr_second_pass_steps
            0,  # hr_resize_x
            0   # hr_resize_y
        ]
    }
    
    try:
        # 發送請求到 Gradio API
        response = requests.post(
            f"{base_url}/api/{endpoint_name}/",
            json=payload,
            timeout=60
        )
        
        print(f"📡 收到回應，狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Gradio 請求成功")
            
            # Gradio API 返回格式通常是 {"data": [...]}
            if 'data' in result:
                data = result['data']
                print(f"   返回數據項目數: {len(data)}")
                
                # 查找圖像數據
                for i, item in enumerate(data):
                    if isinstance(item, str) and (item.startswith('data:image') or len(item) > 1000):
                        print(f"   找到圖像數據 (項目 {i})")
                        
                        # 保存圖像
                        try:
                            if item.startswith('data:image'):
                                # 如果是 data URL 格式
                                image_data = item.split(',')[1]
                            else:
                                # 如果是純 base64
                                image_data = item
                            
                            # 解碼並保存
                            decoded_data = base64.b64decode(image_data)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_path = f"gradio_test_{timestamp}.png"
                            
                            with open(image_path, "wb") as f:
                                f.write(decoded_data)
                            
                            print(f"✅ 圖像已保存: {image_path}")
                            print(f"   檔案大小: {len(decoded_data)} bytes")
                            return True
                            
                        except Exception as e:
                            print(f"   ❌ 圖像保存失敗: {e}")
                
                print("⚠️ 沒有找到圖像數據")
                print(f"   返回數據類型: {[type(item) for item in data]}")
            else:
                print("❌ 回應中沒有 'data' 字段")
                print(f"   回應內容: {list(result.keys())}")
        else:
            print(f"❌ Gradio 請求失敗: {response.status_code}")
            print(f"   錯誤內容: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Gradio 測試失敗: {e}")
    
    return False

def create_gradio_client():
    """創建 Gradio 客戶端腳本"""
    print("\n📝 創建 Gradio 客戶端腳本...")
    
    gradio_client_code = '''#!/usr/bin/env python3
"""
Gradio WebUI 客戶端 - 專門用於 Gradio 界面的 WebUI
"""

import requests
import json
import base64
import os
from datetime import datetime

class GradioWebUIClient:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url
        self.endpoint_name = None
        self._discover_endpoint()
    
    def _discover_endpoint(self):
        """自動發現文本到圖像端點"""
        try:
            response = requests.get(f"{self.base_url}/info", timeout=10)
            if response.status_code == 200:
                info = response.json()
                named_endpoints = info.get('named_endpoints', {})
                
                # 查找文本到圖像端點
                for name in named_endpoints.keys():
                    if any(keyword in name.lower() for keyword in ['txt2img', 'text_to_image', 'generate']):
                        self.endpoint_name = name
                        print(f"✅ 找到端點: {name}")
                        return
                
                # 如果沒找到，使用第一個
                if named_endpoints:
                    self.endpoint_name = list(named_endpoints.keys())[0]
                    print(f"⚠️ 使用第一個端點: {self.endpoint_name}")
        except Exception as e:
            print(f"❌ 端點發現失敗: {e}")
    
    def generate_image(self, prompt, negative_prompt="", steps=20, width=512, height=512):
        """生成圖像"""
        if not self.endpoint_name:
            print("❌ 沒有可用的端點")
            return None
        
        payload = {
            "data": [
                prompt,
                negative_prompt,
                [],  # styles
                steps,
                "DPM++ 2M Karras",
                False,  # restore_faces
                False,  # tiling
                1,  # n_iter
                1,  # batch_size
                7.5,  # cfg_scale
                -1,  # seed
                -1,  # subseed
                0,  # subseed_strength
                0,  # seed_resize_from_h
                0,  # seed_resize_from_w
                False,  # seed_enable_extras
                height,
                width,
                False,  # enable_hr
                0.7,  # denoising_strength
                2,  # hr_scale
                "Latent",  # hr_upscaler
                0,  # hr_second_pass_steps
                0,  # hr_resize_x
                0   # hr_resize_y
            ]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/{self.endpoint_name}/",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'data' in result:
                    # 查找圖像數據
                    for item in result['data']:
                        if isinstance(item, str) and len(item) > 1000:
                            return item  # 返回 base64 圖像數據
            
            print(f"❌ 生成失敗: {response.status_code}")
            return None
            
        except Exception as e:
            print(f"❌ 生成錯誤: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    client = GradioWebUIClient()
    
    if client.endpoint_name:
        print("🎨 測試圖像生成...")
        image_data = client.generate_image("a beautiful woman in elegant dress")
        
        if image_data:
            # 保存圖像
            try:
                if image_data.startswith('data:image'):
                    decoded_data = base64.b64decode(image_data.split(',')[1])
                else:
                    decoded_data = base64.b64decode(image_data)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"gradio_generated_{timestamp}.png"
                
                with open(image_path, "wb") as f:
                    f.write(decoded_data)
                
                print(f"✅ 圖像已生成並保存: {image_path}")
            except Exception as e:
                print(f"❌ 圖像保存失敗: {e}")
        else:
            print("❌ 圖像生成失敗")
    else:
        print("❌ 無法找到可用的 API 端點")
'''
    
    with open("gradio_webui_client.py", "w", encoding="utf-8") as f:
        f.write(gradio_client_code)
    
    print("✅ Gradio 客戶端腳本已創建: gradio_webui_client.py")

def main():
    print("=" * 50)
    print("  Gradio WebUI API 測試")
    print("=" * 50)
    
    # 1. 測試 API 連接並找到端點
    endpoint = test_gradio_api()
    
    if not endpoint:
        print("❌ 無法找到可用的 API 端點")
        return False
    
    # 2. 測試圖像生成
    success = test_gradio_generation(endpoint)
    
    # 3. 創建客戶端腳本
    create_gradio_client()
    
    if success:
        print(f"\n🎉 Gradio API 測試成功！")
        print(f"使用端點: {endpoint}")
        print("\n下一步:")
        print("1. 使用 gradio_webui_client.py 進行圖像生成")
        print("2. 修改 day2_enhanced_test.py 以使用 Gradio API")
        return True
    else:
        print(f"\n❌ Gradio API 測試失敗")
        return False

if __name__ == "__main__":
    success = main()
    input(f"\n{'✅ 成功' if success else '❌ 失敗'}！按 Enter 鍵結束...")
