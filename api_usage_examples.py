#!/usr/bin/env python3
"""
API 客戶端使用範例
展示如何呼叫 text_to_image_service 函數
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from text_to_image_service import text_to_image_service, StableDiffusionAPI
    import json
    import requests
except ImportError as e:
    print(f"❌ 模組導入錯誤: {e}")
    print("請先安裝必要套件:")
    print("pip install requests pillow")
    sys.exit(1)

def example_1_simple_usage():
    """範例 1: 簡單使用"""
    print("=" * 60)
    print("範例 1: 簡單的文字轉圖片")
    print("=" * 60)
    
    prompt = "a beautiful sunset over the ocean, highly detailed, 4k"
    
    result = text_to_image_service(prompt)
    
    if result["success"]:
        print("✅ 生成成功!")
        print(f"📁 圖片保存至: {result['saved_files'][0]}")
        print(f"⏱️ 生成時間: {result['generation_time']:.2f} 秒")
    else:
        print("❌ 生成失敗:", result["error"])

def example_2_custom_parameters():
    """範例 2: 自定義參數"""
    print("\n" + "=" * 60)
    print("範例 2: 使用自定義參數")
    print("=" * 60)
    
    prompt = "a cute robot cat, cyberpunk style, neon lights"
    
    result = text_to_image_service(
        prompt=prompt,
        negative_prompt="blurry, low quality, deformed",
        width=768,
        height=768,
        steps=30,
        cfg_scale=8,
        sampler_name="DPM++ 2M Karras"
    )
    
    if result["success"]:
        print("✅ 生成成功!")
        print(f"📁 圖片保存至: {result['saved_files'][0]}")
    else:
        print("❌ 生成失敗:", result["error"])

def example_3_batch_generation():
    """範例 3: 批次生成多張圖片"""
    print("\n" + "=" * 60)
    print("範例 3: 批次生成")
    print("=" * 60)
    
    prompts = [
        "a serene mountain landscape at dawn",
        "a futuristic city with flying cars",
        "a magical forest with glowing mushrooms",
        "a steampunk airship in the clouds"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🎨 生成第 {i} 張圖片...")
        print(f"描述: {prompt}")
        
        result = text_to_image_service(
            prompt=prompt,
            output_path=f"batch_generated_{i}.png",
            steps=25
        )
        
        if result["success"]:
            print(f"✅ 完成! 保存至: {result['saved_files'][0]}")
        else:
            print(f"❌ 失敗: {result['error']}")

def example_4_api_class_usage():
    """範例 4: 使用 API 類別進行進階操作"""
    print("\n" + "=" * 60)
    print("範例 4: 進階 API 使用")
    print("=" * 60)
    
    # 創建 API 實例
    api = StableDiffusionAPI()
    
    # 檢查服務器狀態
    if not api.is_server_ready():
        print("❌ 服務器未就緒")
        return
    
    # 獲取可用模型
    models = api.get_models()
    if models:
        print(f"📋 當前模型: {models[0].get('model_name', 'Unknown')}")
    
    # 生成圖片
    prompt = "a portrait of a wise old wizard, detailed, fantasy art"
    
    result = api.generate_image(
        prompt=prompt,
        negative_prompt="blurry, low quality",
        width=512,
        height=768,  # 直向比例
        steps=25,
        cfg_scale=7.5
    )
    
    if result["success"]:
        # 手動保存圖片
        saved_files = api.save_images(result, "custom_output", "wizard_portrait")
        print(f"✅ 圖片保存至: {saved_files[0]}")
        
        # 顯示詳細信息
        print(f"⏱️ 生成時間: {result['generation_time']:.2f} 秒")
        print(f"🖼️ 圖片數量: {len(result['images'])}")
    else:
        print("❌ 生成失敗:", result["error"])

def example_5_integration_example():
    """範例 5: 整合到其他程式的範例"""
    print("\n" + "=" * 60)
    print("範例 5: 程式整合範例")
    print("=" * 60)
    
    # 模擬接收到外部程式的文字訊息
    incoming_messages = [
        "Generate an image of a peaceful garden",
        "Create a picture of a modern office space",
        "Draw a cartoon character of a friendly dragon"
    ]
    
    generated_images = []
    
    for message in incoming_messages:
        print(f"\n📨 接收到訊息: {message}")
        
        # 處理文字並生成圖片
        result = text_to_image_service(message)
        
        if result["success"]:
            image_path = result["saved_files"][0]
            generated_images.append({
                "prompt": message,
                "image_path": image_path,
                "generation_time": result["generation_time"]
            })
            print(f"✅ 圖片已生成: {image_path}")
        else:
            print(f"❌ 生成失敗: {result['error']}")
    
    # 總結
    print(f"\n📊 總結:")
    print(f"   收到訊息: {len(incoming_messages)} 條")
    print(f"   成功生成: {len(generated_images)} 張圖片")
    
    for i, img_info in enumerate(generated_images, 1):
        print(f"   {i}. {img_info['image_path']} ({img_info['generation_time']:.1f}s)")

def main():
    """主函數 - 執行所有範例"""
    print("🎨 Stable Diffusion API 使用範例")
    print("這些範例展示如何在您的程式中整合圖片生成功能")
    
    try:
        # 檢查服務器是否運行
        api = StableDiffusionAPI()
        
        print("🔍 檢查服務器連接...")
        if not api.wait_for_server(timeout=10):
            print("\n❌ 無法連接到 Stable Diffusion WebUI 服務器")
            print("\n請按照以下步驟操作:")
            print("   1. 打開新的命令提示字元視窗")
            print("   2. 執行: webui-user.bat")
            print("   3. 等待看到 'Running on local URL: http://127.0.0.1:7860' 訊息")
            print("   4. 重新運行此範例程式")
            print("\n服務器啟動可能需要幾分鐘時間，請耐心等待...")
            input("\n按 Enter 鍵退出...")
            return
            
        print("✅ 服務器連接成功!")
        
    except Exception as e:
        print(f"❌ 初始化錯誤: {e}")
        print("請檢查 text_to_image_service.py 檔案是否存在")
        input("按 Enter 鍵退出...")
        return
    
    try:
        # 執行範例
        example_1_simple_usage()
        example_2_custom_parameters()
        example_3_batch_generation()
        example_4_api_class_usage()
        example_5_integration_example()
        
        print("\n" + "=" * 60)
        print("🎉 所有範例執行完成!")
        print("📁 生成的圖片保存在 'generated_images' 資料夾中")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ 範例執行已中斷")
    except Exception as e:
        print(f"\n❌ 執行範例時發生錯誤: {e}")

if __name__ == "__main__":
    main()
