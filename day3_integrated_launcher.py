#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: Integrated Fashion AI Training Launcher
整合式時尚 AI 訓練啟動器

🎯 功能選擇:
1. 提示詞優化訓練 (原始流程)
2. 真正的 SD v1.5 微調訓練
3. 訓練監控和可視化
4. 配置管理和驗證
5. 批次實驗對比

使用說明:
python day3_integrated_launcher.py --mode [prompt|finetune|monitor|config|batch]
"""

import argparse
import os
import sys
import json
from datetime import datetime

def print_banner():
    """打印歡迎橫幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        🎨 Day 3: Fashion AI Training Suite                   ║
║                                                                               ║
║  🎯 提示詞優化 + 真正的模型微調 + 監控可視化 + 配置管理                        ║
║                                                                               ║
║  基於 FashionCLIP 特徵提取的 Stable Diffusion v1.5 智能訓練系統              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_dependencies():
    """檢查依賴項"""
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'diffusers': 'Diffusers',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'cv2': 'OpenCV',
        'sklearn': 'Scikit-learn'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(name)
    
    if missing_packages:
        print("❌ 缺少必需的套件:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n請先安裝必需套件:")
        print("pip install torch torchvision transformers diffusers pillow opencv-python scikit-learn matplotlib seaborn")
        return False
    
    return True

def check_gpu():
    """檢查 GPU 可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU 可用: {gpu_name}")
            print(f"💾 GPU 記憶體: {gpu_memory:.1f} GB")
            
            if gpu_memory < 8:
                print("⚠️  GPU 記憶體較低，建議使用 LoRA 微調")
            
            return True
        else:
            print("⚠️  未檢測到 GPU，將使用 CPU (速度較慢)")
            return False
    except:
        return False

def mode_prompt_optimization():
    """模式1: 提示詞優化訓練"""
    print("\n🎯 啟動提示詞優化訓練流程")
    print("=" * 50)
    
    try:
        # 導入原始的提示詞優化模組
        from day3_fashion_training import FashionTrainingPipeline
        
        # 創建訓練流程
        pipeline = FashionTrainingPipeline()
        
        # 顯示配置選項
        print("📋 可用的提示詞配置:")
        configs = ["minimal_prompt", "high_confidence_only", "detailed_focused", "standard"]
        for i, config in enumerate(configs, 1):
            print(f"   {i}. {config}")
        
        # 用戶選擇
        choice = input(f"\n請選擇配置 (1-{len(configs)}) [默認: 4]: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(configs):
            selected_config = configs[int(choice) - 1]
        else:
            selected_config = "standard"
        
        print(f"✅ 選擇配置: {selected_config}")
        pipeline.set_prompt_config(selected_config)
        
        # 執行訓練
        pipeline.run_training_pipeline()
        
    except ImportError as e:
        print(f"❌ 導入模組失敗: {e}")
        print("請確保 day3_fashion_training.py 存在")
    except Exception as e:
        print(f"❌ 訓練過程出錯: {e}")

def mode_real_finetuning():
    """模式2: 真正的 SD v1.5 微調訓練"""
    print("\n🔧 啟動 Stable Diffusion v1.5 微調訓練")
    print("=" * 50)
    
    try:
        # 導入微調配置管理器
        from day3_finetuning_config import FineTuningConfig, TrainingValidator
        
        config_manager = FineTuningConfig()
        validator = TrainingValidator()
        
        # 顯示可用配置
        print("📋 可用的微調配置:")
        config_manager.list_configs()
        
        # 用戶選擇配置
        config_names = list(config_manager.configs.keys())
        choice = input(f"\n請選擇配置 [{'/'.join(config_names)}] [默認: quick_test]: ").strip()
        
        if choice not in config_names:
            choice = "quick_test"
        
        # 獲取並驗證配置
        config = config_manager.get_config(choice)
        errors, warnings = validator.validate_config(config)
        memory_estimate = validator.estimate_gpu_memory(config)
        
        print(f"\n📊 配置驗證結果:")
        print(f"   選擇配置: {choice}")
        print(f"   GPU 記憶體估算: {memory_estimate:.1f} GB")
        
        if errors:
            print(f"   ❌ 錯誤: {', '.join(errors)}")
            return
        
        if warnings:
            print(f"   ⚠️  警告: {', '.join(warnings)}")
            confirm = input("是否繼續? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ 訓練取消")
                return
        
        # 啟動真正的微調
        try:
            from day3_real_finetuning import FashionSDFineTuner
            
            print(f"\n🚀 開始微調訓練...")
            finetuner = FashionSDFineTuner()
            finetuner.config.update(config)
            finetuner.train()
            
        except ImportError as e:
            print(f"❌ 導入微調模組失敗: {e}")
            print("請確保 day3_real_finetuning.py 存在")
        
    except Exception as e:
        print(f"❌ 微調過程出錯: {e}")
        import traceback
        traceback.print_exc()

def mode_training_monitor():
    """模式3: 訓練監控和可視化"""
    print("\n🔍 啟動訓練監控和可視化")
    print("=" * 50)
    
    try:
        from day3_training_monitor import TrainingMonitor, ValidationImageAnalyzer
        
        # 創建監控器
        monitor = TrainingMonitor()
        analyzer = ValidationImageAnalyzer()
        
        print("📊 可用的監控功能:")
        print("   1. 實時訓練監控")
        print("   2. 生成訓練圖表")
        print("   3. 驗證圖片分析")
        print("   4. 訓練摘要報告")
        
        choice = input("\n請選擇功能 (1-4) [默認: 2]: ").strip()
        
        if choice == "1":
            print("⚠️  實時監控需要在訓練過程中運行")
            print("請在訓練時使用 --mode monitor 參數")
            
        elif choice == "3":
            print("\n🖼️  分析驗證圖片...")
            analyzer.analyze_generated_images()
            
        elif choice == "4":
            print("\n📄 生成訓練摘要...")
            monitor.save_training_summary()
            
        else:  # 默認選擇 2
            print("\n📈 生成訓練圖表...")
            plot_path = monitor.generate_training_plots()
            if plot_path:
                print(f"✅ 圖表已生成: {plot_path}")
            else:
                print("⚠️  沒有找到訓練數據")
    
    except ImportError as e:
        print(f"❌ 導入監控模組失敗: {e}")
        print("請確保 day3_training_monitor.py 存在")
    except Exception as e:
        print(f"❌ 監控過程出錯: {e}")

def mode_config_management():
    """模式4: 配置管理和驗證"""
    print("\n⚙️  配置管理和驗證")
    print("=" * 50)
    
    try:
        from day3_finetuning_config import FineTuningConfig, TrainingValidator
        
        config_manager = FineTuningConfig()
        validator = TrainingValidator()
        
        print("🔧 配置管理功能:")
        print("   1. 查看所有配置")
        print("   2. 驗證配置參數")
        print("   3. 創建自定義配置")
        print("   4. GPU 記憶體估算")
        
        choice = input("\n請選擇功能 (1-4) [默認: 1]: ").strip()
        
        if choice == "2":
            config_name = input("請輸入要驗證的配置名稱: ").strip()
            config = config_manager.get_config(config_name)
            errors, warnings = validator.validate_config(config)
            memory = validator.estimate_gpu_memory(config)
            
            print(f"\n📊 配置驗證結果:")
            print(f"   配置: {config_name}")
            print(f"   GPU 記憶體估算: {memory:.1f} GB")
            
            if errors:
                print(f"   ❌ 錯誤: {', '.join(errors)}")
            if warnings:
                print(f"   ⚠️  警告: {', '.join(warnings)}")
            if not errors and not warnings:
                print("   ✅ 配置有效")
        
        elif choice == "3":
            print("\n🔧 創建自定義配置...")
            base_config = input("基礎配置 [standard]: ").strip() or "standard"
            
            custom_config = config_manager.create_custom_config(
                base_config=base_config,
                learning_rate=float(input("學習率 [1e-4]: ") or "1e-4"),
                num_epochs=int(input("訓練輪數 [50]: ") or "50"),
                lora_rank=int(input("LoRA Rank [8]: ") or "8")
            )
            
            # 驗證自定義配置
            errors, warnings = validator.validate_config(custom_config)
            memory = validator.estimate_gpu_memory(custom_config)
            
            print(f"\n📊 自定義配置:")
            print(f"   學習率: {custom_config['learning_rate']}")
            print(f"   訓練輪數: {custom_config['num_epochs']}")
            print(f"   LoRA Rank: {custom_config['lora_rank']}")
            print(f"   GPU 記憶體估算: {memory:.1f} GB")
            
            if errors:
                print(f"   ❌ 錯誤: {', '.join(errors)}")
            if warnings:
                print(f"   ⚠️  警告: {', '.join(warnings)}")
        
        elif choice == "4":
            print("\n💾 GPU 記憶體估算:")
            for config_name in config_manager.configs.keys():
                config = config_manager.get_config(config_name)
                memory = validator.estimate_gpu_memory(config)
                print(f"   {config_name}: {memory:.1f} GB")
        
        else:  # 默認選擇 1
            config_manager.list_configs()
    
    except ImportError as e:
        print(f"❌ 導入配置模組失敗: {e}")
    except Exception as e:
        print(f"❌ 配置管理出錯: {e}")

def mode_batch_experiments():
    """模式5: 批次實驗對比"""
    print("\n🧪 批次實驗對比")
    print("=" * 50)
    
    try:
        # 嘗試導入批次優化模組
        from day3_batch_optimization import BatchOptimizer
        from prompt_optimization_test import PromptOptimizationTester
        
        print("🔬 批次實驗功能:")
        print("   1. 提示詞策略對比")
        print("   2. 微調配置對比")
        print("   3. 大規模批次測試")
        
        choice = input("\n請選擇功能 (1-3) [默認: 1]: ").strip()
        
        if choice == "2":
            print("\n⚠️  微調配置對比需要大量時間和 GPU 資源")
            confirm = input("是否繼續? (y/N): ").strip().lower()
            if confirm == 'y':
                # 批次微調對比
                configs_to_test = ["quick_test", "standard"]
                print(f"🧪 測試配置: {configs_to_test}")
                # 這裡可以實現批次微調對比邏輯
        
        elif choice == "3":
            batch_optimizer = BatchOptimizer()
            batch_optimizer.run_batch_optimization()
        
        else:  # 默認選擇 1
            print("\n🎯 提示詞策略對比...")
            tester = PromptOptimizationTester()
            tester.run_all_tests()
    
    except ImportError as e:
        print(f"❌ 導入批次實驗模組失敗: {e}")
        print("請確保相關模組存在")
    except Exception as e:
        print(f"❌ 批次實驗出錯: {e}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="Day 3: Fashion AI Training Launcher")
    parser.add_argument(
        "--mode", 
        choices=["prompt", "finetune", "monitor", "config", "batch"],
        default="prompt",
        help="選擇運行模式"
    )
    parser.add_argument(
        "--config",
        help="指定配置文件或配置名稱"
    )
    parser.add_argument(
        "--gpu-check",
        action="store_true",
        help="檢查 GPU 狀態"
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="列出所有可用模式"
    )
    
    args = parser.parse_args()
    
    # 打印橫幅
    print_banner()
    
    # 檢查 GPU
    if args.gpu_check:
        check_gpu()
        return
    
    # 列出模式
    if args.list_modes:
        print("📋 可用的運行模式:")
        print("   prompt    - 提示詞優化訓練 (原始流程)")
        print("   finetune  - 真正的 SD v1.5 微調訓練")
        print("   monitor   - 訓練監控和可視化")
        print("   config    - 配置管理和驗證")
        print("   batch     - 批次實驗對比")
        return
    
    # 檢查依賴項
    if not check_dependencies():
        return
    
    # 檢查源目錄
    if not os.path.exists("day1_results"):
        print("⚠️  警告: 找不到 day1_results 目錄")
        print("請確保有訓練用的圖片資料")
    
    # 根據模式執行相應功能
    try:
        if args.mode == "prompt":
            mode_prompt_optimization()
        elif args.mode == "finetune":
            mode_real_finetuning()
        elif args.mode == "monitor":
            mode_training_monitor()
        elif args.mode == "config":
            mode_config_management()
        elif args.mode == "batch":
            mode_batch_experiments()
        else:
            print(f"❌ 未知模式: {args.mode}")
            
    except KeyboardInterrupt:
        print("\n⏹️  程序被用戶中斷")
    except Exception as e:
        print(f"\n❌ 程序執行出錯: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
