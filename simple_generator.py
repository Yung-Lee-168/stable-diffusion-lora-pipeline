#!/usr/bin/env python3
"""
簡單的圖像生成範例
輸入文字描述，生成對應圖像
"""

import requests
import base64
import json
from datetime import datetime
import os

def generate_image(prompt, output_filename=None):
    """
    簡單的圖像生成函數
    
    Args:
        prompt (str): 圖像描述文字
        output_filename (str): 輸出檔案名稱（可選）
    
    Returns:
        bool: 生成是否成功
    """
    
    # API 設定
    api_url = "http://localhost:7860/sdapi/v1/txt2img"
    
    # 生成參數
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, watermark, text, deformed",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg_scale": 7,
        "sampler_name": "Euler",
        "seed": -1,
        "n_iter": 1,
        "batch_size": 1
    }
    
    print(f"🎨 正在生成圖像...")
    print(f"📝 描述: {prompt}")
    
    try:
        # 發送 API 請求
        response = requests.post(api_url, json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            
            # 準備輸出檔案名稱
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"generated_{timestamp}.png"
            
            # 確保輸出目錄存在
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, output_filename)
            
            # 解碼並保存圖像
            image_data = base64.b64decode(result['images'][0])
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            print(f"✅ 圖像生成成功!")
            print(f"💾 已保存至: {filepath}")
            return True
            
        else:
            print(f"❌ API 請求失敗: {response.status_code}")
            print(f"錯誤信息: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 無法連接到 Stable Diffusion WebUI 服務器")
        print("請確認服務器是否已啟動 (http://localhost:7860)")
        return False
    except requests.exceptions.Timeout:
        print("❌ 請求超時，圖像生成可能需要更長時間")
        return False
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        return False

def main():
    """主程式"""
    print("🎨 Stable Diffusion 圖像生成器")
    print("=" * 40)
    
    # 檢查服務器連接
    try:
        response = requests.get("http://localhost:7860/sdapi/v1/options", timeout=5)
        if response.status_code != 200:
            print("❌ 服務器連接失敗")
            return
    except:
        print("❌ 無法連接到服務器，請先啟動 webui-user.bat")
        return
    
    print("✅ 服務器連接正常")
    print()
    
    # 互動式輸入
    while True:
        print("請輸入圖像描述 (輸入 'quit' 退出):")
        user_input = input("📝 > ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 再見!")
            break
        
        if user_input.strip():
            success = generate_image(user_input)
            if success:
                print("🎉 完成! 您可以在 'outputs' 資料夾中找到生成的圖像")
            print("-" * 40)
        else:
            print("請輸入有效的描述文字")

if __name__ == "__main__":
    main()
