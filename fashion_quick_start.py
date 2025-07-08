#!/usr/bin/env python3
"""
Fashion Project Quick Start
快速開始時尚專案的概念驗證
"""

import os
import json
from datetime import datetime

def create_project_structure():
    """創建專案目錄結構"""
    
    directories = [
        "fashion_project",
        "fashion_project/data",
        "fashion_project/data/original_images", 
        "fashion_project/data/generated_images",
        "fashion_project/data/features",
        "fashion_project/models",
        "fashion_project/results",
        "fashion_project/scripts",
        "fashion_project/docs"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 創建目錄: {dir_path}")
    
    return "fashion_project"

def create_sample_config():
    """創建示例配置檔案"""
    
    config = {
        "project_info": {
            "name": "Fashion Style Transfer with Stable Diffusion",
            "version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            "description": "使用 FashionCLIP 分析時尚圖片特徵，訓練 SD 模型生成類似風格圖片"
        },
        "data_config": {
            "original_images_dir": "data/original_images",
            "generated_images_dir": "data/generated_images", 
            "features_dir": "data/features",
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp"],
            "target_image_size": [512, 512]
        },
        "feature_extraction": {
            "model": "FashionCLIP",
            "categories": [
                "gender", "age_group", "top_clothing", "bottom_clothing",
                "style", "color_scheme", "season", "occasion"
            ],
            "confidence_threshold": 0.3
        },
        "sd_generation": {
            "base_model": "v1-5-pruned-emaonly.safetensors",
            "default_params": {
                "width": 512,
                "height": 512,
                "steps": 20,
                "cfg_scale": 7.5,
                "sampler": "Euler"
            }
        },
        "training_config": {
            "method": "LoRA",
            "batch_size": 4,
            "learning_rate": 1e-4,
            "max_train_steps": 1000,
            "validation_split": 0.2
        }
    }
    
    config_file = "fashion_project/config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"⚙️ 創建配置檔案: {config_file}")
    return config_file

def create_readme():
    """創建說明文件"""
    
    readme_content = """# Fashion Style Transfer Project

## 專案概述
使用 FashionCLIP 分析時尚雜誌圖片特徵，訓練 Stable Diffusion 模型生成相似風格的服裝圖片。

## 專案結構
```
fashion_project/
├── data/
│   ├── original_images/     # 原始時尚雜誌圖片
│   ├── generated_images/    # SD 生成的圖片
│   └── features/           # 提取的特徵資料
├── models/                 # 訓練好的模型
├── results/               # 實驗結果和評估
├── scripts/              # 執行腳本
├── docs/                # 文檔
└── config.json          # 配置檔案
```

## 使用步驟

### 1. 資料準備
```bash
# 將時尚雜誌圖片放入 data/original_images/
cp your_fashion_images/* data/original_images/
```

### 2. 特徵提取
```bash
python fashion_feature_extractor.py
```

### 3. 生成訓練資料
```bash
python fashion_sd_trainer.py
```

### 4. 模型訓練 (未來實作)
```bash
python train_fashion_lora.py
```

## 技術架構

1. **特徵提取**: FashionCLIP → 服裝特徵向量
2. **提示詞生成**: 特徵 → SD 提示詞
3. **圖片生成**: SD API → 生成圖片
4. **品質評估**: 原始圖 vs 生成圖
5. **模型微調**: LoRA/Dreambooth

## 依賴套件
- torch
- transformers
- clip-by-openai
- Pillow
- requests
- numpy
- pandas

## 安裝
```bash
pip install torch transformers clip-by-openai pillow requests numpy pandas
```

## 使用範例

```python
from fashion_feature_extractor import FashionFeatureExtractor

# 提取特徵
extractor = FashionFeatureExtractor()
features = extractor.extract_features_from_image("image.jpg")

# 生成圖片
from fashion_sd_trainer import FashionSDTrainer
trainer = FashionSDTrainer()
result = trainer.generate_from_features(features)
```

## 評估指標
- 視覺相似度 (CLIP Score)
- 特徵一致性
- 風格保持度
- 生成品質 (FID)

## 未來改進
- [ ] 更精確的特徵提取
- [ ] 多模型ensemble
- [ ] 實時生成優化
- [ ] 用戶介面開發
"""
    
    readme_file = "fashion_project/README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📖 創建說明文件: {readme_file}")
    return readme_file

def create_sample_script():
    """創建示例執行腳本"""
    
    script_content = '''#!/usr/bin/env python3
"""
Fashion Project Demo Script
時尚專案示例腳本
"""

import os
import sys

# 添加專案路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demo_workflow():
    """示範完整工作流程"""
    
    print("🎨 Fashion Style Transfer Demo")
    print("=" * 50)
    
    # 1. 檢查資料
    original_dir = "data/original_images"
    if not os.listdir(original_dir):
        print(f"⚠️ 請將時尚圖片放入 {original_dir}/")
        return
    
    # 2. 特徵提取
    print("\\n📊 Step 1: 特徵提取")
    try:
        from fashion_feature_extractor import FashionFeatureExtractor
        extractor = FashionFeatureExtractor()
        features = extractor.process_fashion_magazine_dataset(original_dir)
        print("✅ 特徵提取完成")
    except Exception as e:
        print(f"❌ 特徵提取失敗: {e}")
        return
    
    # 3. 圖片生成
    print("\\n🎨 Step 2: 圖片生成")
    try:
        from fashion_sd_trainer import FashionSDTrainer
        trainer = FashionSDTrainer()
        generated = trainer.generate_training_images(max_samples=5)
        print("✅ 圖片生成完成")
    except Exception as e:
        print(f"❌ 圖片生成失敗: {e}")
        return
    
    # 4. 評估
    print("\\n📈 Step 3: 品質評估")
    try:
        evaluation = trainer.evaluate_generation_quality()
        print("✅ 評估完成")
    except Exception as e:
        print(f"❌ 評估失敗: {e}")
    
    print("\\n🎉 Demo 完成!")

if __name__ == "__main__":
    demo_workflow()
'''
    
    script_file = "fashion_project/scripts/demo.py"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"🚀 創建示例腳本: {script_file}")
    return script_file

def main():
    """主函數 - 快速開始"""
    
    print("🚀 Fashion Project Quick Start")
    print("=" * 50)
    
    # 創建專案結構
    project_dir = create_project_structure()
    
    # 創建配置檔案
    config_file = create_sample_config()
    
    # 創建說明文件
    readme_file = create_readme()
    
    # 創建示例腳本
    script_file = create_sample_script()
    
    print("\n🎉 專案初始化完成!")
    print(f"📁 專案目錄: {project_dir}")
    print(f"\n接下來的步驟:")
    print(f"1. 將時尚雜誌圖片放入 {project_dir}/data/original_images/")
    print(f"2. 安裝依賴: pip install torch transformers clip-by-openai")
    print(f"3. 執行示例: python {script_file}")
    print(f"4. 查看說明: {readme_file}")

if __name__ == "__main__":
    main()
