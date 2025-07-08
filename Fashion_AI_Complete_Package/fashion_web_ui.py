#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fashion AI Complete Package - Web 介面主程式
基於 Flask 的 Web 應用程式

功能：
- 圖片上傳和預覽
- 即時分析和生成
- 批次處理管理
- 結果展示和下載
- API 端點服務
"""

import os
import sys
from pathlib import Path
import json
import base64
from datetime import datetime
from typing import Dict, Any, Optional

# Flask 相關
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

# 添加專案路徑
sys.path.append(str(Path(__file__).parent))

from core.fashion_analyzer import FashionTrainingPipeline
from core.config_manager import FineTuningConfig
from utils.system_check import SystemTester

class FashionAIWebApp:
    """Fashion AI Web 應用程式"""
    
    def __init__(self, components: Dict[str, Any], config: Dict[str, Any]):
        self.components = components
        self.config = config
        self.app = Flask(__name__)
        
        # 設置上傳配置
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
        self.app.config['UPLOAD_FOLDER'] = 'data/input'
        self.app.config['OUTPUT_FOLDER'] = 'data/output'
        
        # 確保目錄存在
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(self.app.config['OUTPUT_FOLDER'], exist_ok=True)
        
        # 設置路由
        self.setup_routes()
        
        # 允許的檔案類型
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        
    def allowed_file(self, filename: str) -> bool:
        """檢查檔案類型是否允許"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def setup_routes(self):
        """設置 Flask 路由"""
        
        @self.app.route('/')
        def index():
            """首頁"""
            return render_template('index.html', config=self.config)
        
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            """上傳檔案"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': '沒有檔案'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': '沒有選擇檔案'}), 400
                
                if file and self.allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}_{filename}"
                    filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    return jsonify({
                        'success': True,
                        'filename': filename,
                        'filepath': filepath,
                        'url': url_for('uploaded_file', filename=filename)
                    })
                
                return jsonify({'error': '不支援的檔案類型'}), 400
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/uploads/<filename>')
        def uploaded_file(filename):
            """提供上傳的檔案"""
            return send_file(
                os.path.join(self.app.config['UPLOAD_FOLDER'], filename),
                as_attachment=False
            )
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_image():
            """分析圖片"""
            try:
                data = request.get_json()
                if not data or 'filepath' not in data:
                    return jsonify({'error': '缺少檔案路徑'}), 400
                
                filepath = data['filepath']
                if not os.path.exists(filepath):
                    return jsonify({'error': '檔案不存在'}), 404
                
                # 使用時尚分析器
                analyzer = self.components.get('analyzer')
                if not analyzer:
                    return jsonify({'error': '分析器未初始化'}), 500
                
                # 執行分析
                result = analyzer.analyze_image(filepath)
                
                # 保存結果
                output_filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                output_path = os.path.join(self.app.config['OUTPUT_FOLDER'], output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                return jsonify({
                    'success': True,
                    'result': result,
                    'output_file': output_filename
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/generate', methods=['POST'])
        def generate_image():
            """生成圖片"""
            try:
                data = request.get_json()
                if not data or 'prompt' not in data:
                    return jsonify({'error': '缺少提示詞'}), 400
                
                prompt = data['prompt']
                options = data.get('options', {})
                
                # 使用時尚分析器的生成功能
                analyzer = self.components.get('analyzer')
                if not analyzer:
                    return jsonify({'error': '分析器未初始化'}), 500
                
                # 執行生成
                result = analyzer.generate_image(prompt, options)
                
                if result and 'image_path' in result:
                    # 移動生成的圖片到輸出目錄
                    output_filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    output_path = os.path.join(self.app.config['OUTPUT_FOLDER'], output_filename)
                    
                    # 複製檔案
                    import shutil
                    shutil.copy2(result['image_path'], output_path)
                    
                    return jsonify({
                        'success': True,
                        'image_url': url_for('output_file', filename=output_filename),
                        'image_path': output_path,
                        'generation_info': result
                    })
                else:
                    return jsonify({'error': '圖片生成失敗'}), 500
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/outputs/<filename>')
        def output_file(filename):
            """提供輸出檔案"""
            return send_file(
                os.path.join(self.app.config['OUTPUT_FOLDER'], filename),
                as_attachment=False
            )
        
        @self.app.route('/download/<filename>')
        def download_file(filename):
            """下載檔案"""
            return send_file(
                os.path.join(self.app.config['OUTPUT_FOLDER'], filename),
                as_attachment=True
            )
        
        @self.app.route('/batch', methods=['POST'])
        def batch_process():
            """批次處理"""
            try:
                data = request.get_json()
                if not data or 'files' not in data:
                    return jsonify({'error': '缺少檔案列表'}), 400
                
                files = data['files']
                options = data.get('options', {})
                
                # 執行批次處理
                results = []
                analyzer = self.components.get('analyzer')
                
                for file_path in files:
                    if os.path.exists(file_path):
                        try:
                            result = analyzer.analyze_image(file_path)
                            results.append({
                                'file': file_path,
                                'success': True,
                                'result': result
                            })
                        except Exception as e:
                            results.append({
                                'file': file_path,
                                'success': False,
                                'error': str(e)
                            })
                    else:
                        results.append({
                            'file': file_path,
                            'success': False,
                            'error': '檔案不存在'
                        })
                
                # 保存批次結果
                batch_filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                batch_path = os.path.join(self.app.config['OUTPUT_FOLDER'], batch_filename)
                with open(batch_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                return jsonify({
                    'success': True,
                    'results': results,
                    'batch_file': batch_filename,
                    'download_url': url_for('download_file', filename=batch_filename)
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/status')
        def system_status():
            """系統狀態"""
            try:
                # 檢查各組件狀態
                status = {
                    'analyzer': self.components.get('analyzer') is not None,
                    'config_manager': self.components.get('config_manager') is not None,
                    'system_checker': self.components.get('system_checker') is not None,
                    'webui_api': self.check_webui_api(),
                    'gpu_available': self.check_gpu(),
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify(status)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/config')
        def get_config():
            """獲取配置"""
            return jsonify(self.config)
        
        @self.app.route('/config', methods=['POST'])
        def update_config():
            """更新配置"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': '缺少配置數據'}), 400
                
                # 更新配置
                self.config.update(data)
                
                # 保存配置到檔案
                config_path = Path(__file__).parent / 'config' / 'default_config.yaml'
                if config_path.exists():
                    import yaml
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(self.config, f, default_flow_style=False)
                
                return jsonify({'success': True, 'config': self.config})
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # API 端點 (RESTful)
        @self.app.route('/api/v1/analyze', methods=['POST'])
        def api_analyze():
            """API: 分析圖片"""
            return analyze_image()
        
        @self.app.route('/api/v1/generate', methods=['POST'])
        def api_generate():
            """API: 生成圖片"""
            return generate_image()
        
        @self.app.route('/api/v1/batch', methods=['POST'])
        def api_batch():
            """API: 批次處理"""
            return batch_process()
        
        @self.app.route('/api/v1/status', methods=['GET'])
        def api_status():
            """API: 系統狀態"""
            return system_status()
    
    def check_webui_api(self) -> bool:
        """檢查 WebUI API 狀態"""
        try:
            import requests
            response = requests.get(f"{self.config['webui_url']}/sdapi/v1/options", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_gpu(self) -> bool:
        """檢查 GPU 狀態"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """啟動 Web 應用程式"""
        self.app.run(host=host, port=port, debug=debug)


def create_app(components: Dict[str, Any], config: Dict[str, Any]) -> Flask:
    """創建 Flask 應用程式"""
    web_app = FashionAIWebApp(components, config)
    return web_app.app


def main():
    """主函數 - 獨立執行 Web 應用程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fashion AI Web Interface')
    parser.add_argument('--port', type=int, default=8080, help='Web 服務端口')
    parser.add_argument('--host', default='0.0.0.0', help='監聽地址')
    parser.add_argument('--debug', action='store_true', help='啟用調試模式')
    args = parser.parse_args()
    
    # 基本配置
    config = {
        'webui_url': 'http://localhost:7860',
        'web_port': args.port,
        'batch_size': 1,
        'max_image_size': 512
    }
    
    # 初始化組件
    components = {
        'analyzer': FashionTrainingPipeline(),
        'config_manager': FineTuningConfig(),
        'system_checker': SystemTester()
    }
    
    # 創建和啟動應用程式
    web_app = FashionAIWebApp(components, config)
    web_app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
