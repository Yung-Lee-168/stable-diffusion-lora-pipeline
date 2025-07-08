#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fashion AI Complete Package - 主程式啟動器
整合所有功能模組的統一入口程式

功能：
- 系統初始化和檢查
- 模型載入和配置
- Web 介面啟動
- API 服務管理
- 命令列界面
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加專案路徑
sys.path.append(str(Path(__file__).parent))

from core.fashion_analyzer import FashionTrainingPipeline
from core.webui_connector import ColabEnvironmentSetup
from core.config_manager import FineTuningConfig
from utils.system_check import SystemTester
from utils.report_generator import TrainingMonitor

class FashionAIMain:
    """Fashion AI 主程式管理器"""
    
    def __init__(self):
        self.setup_logging()
        self.config = self.load_config()
        self.components = {}
        
    def setup_logging(self):
        """設置日誌系統"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fashion_ai.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self):
        """載入配置檔案"""
        config_path = Path(__file__).parent / 'config' / 'default_config.yaml'
        if config_path.exists():
            # 載入 YAML 配置
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 返回默認配置
            return {
                'webui_url': 'http://localhost:7860',
                'web_port': 8080,
                'batch_size': 1,
                'max_image_size': 512
            }
    
    def check_system(self):
        """檢查系統狀態"""
        self.logger.info("🔍 檢查系統狀態...")
        
        # 檢查 GPU
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self.logger.warning("⚠️ 沒有可用的 GPU，將使用 CPU 模式")
        
        # 檢查 WebUI 連接
        try:
            import requests
            response = requests.get(f"{self.config['webui_url']}/sdapi/v1/options", timeout=5)
            if response.status_code == 200:
                self.logger.info("✅ WebUI API 連接正常")
            else:
                self.logger.error(f"❌ WebUI API 連接失敗: {response.status_code}")
        except Exception as e:
            self.logger.error(f"❌ WebUI API 連接失敗: {e}")
            
    def init_components(self):
        """初始化各個組件"""
        self.logger.info("🚀 初始化組件...")
        
        try:
            # 初始化時尚分析器
            self.components['analyzer'] = FashionTrainingPipeline()
            self.logger.info("✅ 時尚分析器初始化完成")
            
            # 初始化配置管理器
            self.components['config_manager'] = FineTuningConfig()
            self.logger.info("✅ 配置管理器初始化完成")
            
            # 初始化系統檢查器
            self.components['system_checker'] = SystemTester()
            self.logger.info("✅ 系統檢查器初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 組件初始化失敗: {e}")
            raise
    
    def start_web_interface(self):
        """啟動 Web 介面"""
        self.logger.info("🌐 啟動 Web 介面...")
        
        try:
            # 導入 Web 應用
            from fashion_web_ui import create_app
            app = create_app(self.components, self.config)
            
            # 啟動 Flask 應用
            app.run(
                host='0.0.0.0',
                port=self.config['web_port'],
                debug=False
            )
            
        except Exception as e:
            self.logger.error(f"❌ Web 介面啟動失敗: {e}")
            raise
    
    def start_api_only(self):
        """只啟動 API 服務"""
        self.logger.info("🔧 啟動 API 服務...")
        
        try:
            from web.api.main import create_api_app
            app = create_api_app(self.components, self.config)
            
            app.run(
                host='0.0.0.0',
                port=self.config['web_port'],
                debug=False
            )
            
        except Exception as e:
            self.logger.error(f"❌ API 服務啟動失敗: {e}")
            raise
    
    def run_batch_processing(self, input_dir, output_dir):
        """執行批次處理"""
        self.logger.info(f"📦 開始批次處理: {input_dir} -> {output_dir}")
        
        try:
            analyzer = self.components['analyzer']
            
            # 獲取圖片檔案
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(Path(input_dir).glob(f"*{ext}"))
            
            self.logger.info(f"找到 {len(image_files)} 個圖片檔案")
            
            # 處理每個圖片
            for i, image_path in enumerate(image_files):
                self.logger.info(f"處理 {i+1}/{len(image_files)}: {image_path.name}")
                
                # 分析圖片
                result = analyzer.analyze_image(str(image_path))
                
                # 保存結果
                output_file = Path(output_dir) / f"{image_path.stem}_analysis.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.logger.info("✅ 批次處理完成")
            
        except Exception as e:
            self.logger.error(f"❌ 批次處理失敗: {e}")
            raise
    
    def run_interactive_mode(self):
        """執行互動模式"""
        print("\n🎯 Fashion AI Complete Package - 互動模式")
        print("=" * 50)
        
        while True:
            print("\n可用功能:")
            print("1. 圖片分析")
            print("2. 圖片生成")
            print("3. 批次處理")
            print("4. 系統狀態")
            print("5. 配置設定")
            print("0. 退出")
            
            choice = input("\n請選擇功能 (0-5): ").strip()
            
            try:
                if choice == '1':
                    self.interactive_analyze()
                elif choice == '2':
                    self.interactive_generate()
                elif choice == '3':
                    self.interactive_batch()
                elif choice == '4':
                    self.check_system()
                elif choice == '5':
                    self.interactive_config()
                elif choice == '0':
                    print("👋 再見！")
                    break
                else:
                    print("❌ 無效的選擇，請重新輸入")
                    
            except KeyboardInterrupt:
                print("\n👋 再見！")
                break
            except Exception as e:
                print(f"❌ 執行失敗: {e}")
    
    def interactive_analyze(self):
        """互動式圖片分析"""
        image_path = input("請輸入圖片路徑: ").strip()
        
        if not os.path.exists(image_path):
            print("❌ 圖片檔案不存在")
            return
        
        print("🔍 分析中...")
        analyzer = self.components['analyzer']
        result = analyzer.analyze_image(image_path)
        
        print("\n📊 分析結果:")
        print(f"類別: {result.get('category', 'Unknown')}")
        print(f"風格: {result.get('style', 'Unknown')}")
        print(f"顏色: {result.get('colors', [])}")
        print(f"置信度: {result.get('confidence', 0.0):.2f}")
    
    def interactive_generate(self):
        """互動式圖片生成"""
        prompt = input("請輸入提示詞: ").strip()
        
        if not prompt:
            print("❌ 提示詞不能為空")
            return
        
        print("🎨 生成中...")
        analyzer = self.components['analyzer']
        result = analyzer.generate_image(prompt)
        
        if result:
            print(f"✅ 圖片已生成: {result.get('image_path', 'Unknown')}")
        else:
            print("❌ 圖片生成失敗")
    
    def interactive_batch(self):
        """互動式批次處理"""
        input_dir = input("請輸入輸入目錄: ").strip()
        output_dir = input("請輸入輸出目錄: ").strip()
        
        if not os.path.exists(input_dir):
            print("❌ 輸入目錄不存在")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        self.run_batch_processing(input_dir, output_dir)
    
    def interactive_config(self):
        """互動式配置設定"""
        print("\n當前配置:")
        for key, value in self.config.items():
            print(f"{key}: {value}")
        
        print("\n輸入新的配置值 (直接按 Enter 跳過):")
        
        for key in self.config:
            new_value = input(f"{key} ({self.config[key]}): ").strip()
            if new_value:
                try:
                    # 嘗試轉換類型
                    if isinstance(self.config[key], int):
                        self.config[key] = int(new_value)
                    elif isinstance(self.config[key], float):
                        self.config[key] = float(new_value)
                    else:
                        self.config[key] = new_value
                except ValueError:
                    print(f"❌ 無效的值: {new_value}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Fashion AI Complete Package')
    parser.add_argument('--mode', choices=['web', 'api', 'batch', 'interactive'], 
                       default='interactive', help='運行模式')
    parser.add_argument('--port', type=int, default=8080, help='Web 服務端口')
    parser.add_argument('--webui-url', default='http://localhost:7860', 
                       help='WebUI API URL')
    parser.add_argument('--input-dir', help='批次處理輸入目錄')
    parser.add_argument('--output-dir', help='批次處理輸出目錄')
    
    args = parser.parse_args()
    
    # 初始化主程式
    fashion_ai = FashionAIMain()
    
    # 更新配置
    fashion_ai.config['web_port'] = args.port
    fashion_ai.config['webui_url'] = args.webui_url
    
    # 檢查系統
    fashion_ai.check_system()
    
    # 初始化組件
    fashion_ai.init_components()
    
    # 根據模式執行
    try:
        if args.mode == 'web':
            fashion_ai.start_web_interface()
        elif args.mode == 'api':
            fashion_ai.start_api_only()
        elif args.mode == 'batch':
            if not args.input_dir or not args.output_dir:
                print("❌ 批次模式需要指定 --input-dir 和 --output-dir")
                sys.exit(1)
            fashion_ai.run_batch_processing(args.input_dir, args.output_dir)
        elif args.mode == 'interactive':
            fashion_ai.run_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 程式已停止")
    except Exception as e:
        print(f"❌ 程式執行失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
