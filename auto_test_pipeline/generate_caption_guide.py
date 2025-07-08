#!/usr/bin/env python3
"""
generate_caption_fashionclip.py 使用指南
"""

def main():
    print("🎯 generate_caption_fashionclip.py 設定完成！")
    print("=" * 60)
    
    print("📁 目錄結構:")
    print("  E:\\Yung_Folder\\Project\\stable-diffusion-webui\\auto_test_pipeline\\")
    print("  ├── generate_caption_fashionclip.py  (主腳本)")
    print("  ├── 特徵值.py                        (特徵定義檔案)")
    print("  ├── source_image/                   (來源圖片目錄)")
    print("  │   ├── 253.JPG")
    print("  │   ├── 254.jpg") 
    print("  │   └── ... (共100張圖片)")
    print("  └── lora_train_set/10_test/         (目標目錄)")
    
    print("\n🚀 使用方式:")
    print("1. 確保在 auto_test_pipeline 目錄中:")
    print("   cd E:\\Yung_Folder\\Project\\stable-diffusion-webui\\auto_test_pipeline")
    
    print("\n2. 執行圖片描述生成:")
    print("   python generate_caption_fashionclip.py")
    
    print("\n3. 腳本會自動:")
    print("   ✅ 檢查圖片尺寸 (≤512x512)")
    print("   ✅ 縮放過大的圖片")
    print("   ✅ 複製/移動圖片到 lora_train_set/10_test/")
    print("   ✅ 使用 FashionCLIP 生成描述")
    print("   ✅ 為每張圖片創建對應的 .txt 檔案")
    
    print("\n📊 特徵分類:")
    print("   • Gender (性別): male, female, unisex, androgynous")
    print("   • Age (年齡): child, teenager, young adult, adult, etc.")
    print("   • Season (季節): spring, summer, autumn, winter")
    print("   • Occasion (場合): casual, formal, business, sport, etc.")
    print("   • Upper Body (上身): t-shirt, shirt, jacket, etc.")
    print("   • Lower Body (下身): jeans, trousers, shorts, etc.")
    print("   • Colors (顏色): black, white, red, blue, etc.")
    print("   • Materials (材質): cotton, denim, silk, etc.")
    print("   • Styles (風格): vintage, modern, classic, etc.")
    print("   • Patterns (圖案): solid, striped, floral, etc.")
    print("   • Accessories (配件): hat, scarf, belt, etc.")
    print("   • Footwear (鞋類): sneakers, boots, heels, etc.")
    
    print("\n💡 注意事項:")
    print("   • 腳本必須在 auto_test_pipeline 目錄中執行")
    print("   • 需要安裝 transformers, torch, PIL 等套件")
    print("   • 首次執行會下載 FashionCLIP 模型")
    print("   • 處理100張圖片大約需要10-15分鐘")
    
    print("\n🎉 設定完成，可以開始使用！")

if __name__ == "__main__":
    main()
