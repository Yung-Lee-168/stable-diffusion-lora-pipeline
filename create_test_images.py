#!/usr/bin/env python3
"""
創建測試圖片用於 CLIP 分析
如果沒有現有圖片，這個腳本會創建一些簡單的測試圖片
"""

import os
from PIL import Image, ImageDraw, ImageFont
import random

def create_test_images():
    """創建一些簡單的測試圖片"""
    output_dir = "test_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # 定義一些基本的圖片類型
    image_configs = [
        {
            "name": "formal_dress",
            "description": "Formal Evening Dress",
            "colors": [(25, 25, 112), (75, 0, 130), (72, 61, 139)],  # 深藍、紫色
        },
        {
            "name": "casual_shirt", 
            "description": "Casual T-Shirt",
            "colors": [(70, 130, 180), (100, 149, 237), (135, 206, 235)],  # 天藍色系
        },
        {
            "name": "winter_coat",
            "description": "Winter Coat",
            "colors": [(47, 79, 79), (105, 105, 105), (128, 128, 128)],  # 灰色系
        }
    ]
    
    created_files = []
    
    for i, config in enumerate(image_configs):
        try:
            # 創建 512x768 的圖片 (標準人像尺寸)
            img = Image.new('RGB', (512, 768), color=random.choice(config["colors"]))
            draw = ImageDraw.Draw(img)
            
            # 添加一些基本形狀來模擬服裝
            # 上身區域
            upper_color = random.choice(config["colors"])
            draw.rectangle([100, 200, 412, 400], fill=upper_color)
            
            # 下身區域
            lower_color = random.choice(config["colors"])
            draw.rectangle([120, 400, 392, 650], fill=lower_color)
            
            # 添加文字說明
            try:
                # 嘗試使用系統字體
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                # 如果沒有找到字體，使用默認字體
                font = ImageFont.load_default()
            
            draw.text((50, 50), config["description"], fill=(255, 255, 255), font=font)
            draw.text((50, 700), f"Test Image {i+1}", fill=(255, 255, 255), font=font)
            
            # 保存圖片
            filename = f"{config['name']}_test.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            created_files.append(filepath)
            print(f"✅ 創建測試圖片: {filename}")
            
        except Exception as e:
            print(f"❌ 創建圖片失敗: {e}")
    
    return created_files

if __name__ == "__main__":
    print("🎨 創建測試圖片...")
    files = create_test_images()
    print(f"\n✅ 完成！創建了 {len(files)} 張測試圖片")
    print("這些圖片可用於 CLIP 分析測試")
