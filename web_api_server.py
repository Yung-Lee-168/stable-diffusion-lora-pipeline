#!/usr/bin/env python3
"""
Web API 服務器 - 提供 HTTP 接口
其他程式可以透過 HTTP POST 請求來生成圖片

啟動方式：python web_api_server.py
使用方式：POST http://localhost:8000/generate
"""

from flask import Flask, request, jsonify, send_file
import base64
import io
import os
from datetime import datetime
from text_to_image_service import text_to_image_service

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    return jsonify({
        "status": "healthy",
        "service": "Stable Diffusion Text-to-Image API",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/generate', methods=['POST'])
def generate_image_api():
    """
    圖片生成 API 端點
    
    請求格式：
    {
        "prompt": "圖片描述文字",
        "negative_prompt": "負向描述（可選）",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg_scale": 7,
        "return_base64": true
    }
    
    回應格式：
    {
        "success": true,
        "images": ["base64_encoded_image"],
        "saved_files": ["path/to/saved/image.png"],
        "generation_time": 15.23,
        "parameters": {...}
    }
    """
    
    try:
        # 解析請求數據
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required field 'prompt'"
            }), 400
        
        prompt = data['prompt']
        
        # 提取可選參數
        generation_params = {}
        optional_fields = [
            'negative_prompt', 'width', 'height', 'steps', 
            'cfg_scale', 'sampler_name', 'seed'
        ]
        
        for field in optional_fields:
            if field in data:
                generation_params[field] = data[field]
        
        # 生成圖片
        result = text_to_image_service(prompt, **generation_params)
        
        # 處理返回格式
        return_base64 = data.get('return_base64', True)
        
        if result["success"] and not return_base64:
            # 如果不需要 base64，移除圖片數據以減少傳輸量
            result.pop('images', None)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/generate_and_download', methods=['POST'])
def generate_and_download():
    """
    生成圖片並直接下載
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required field 'prompt'"
            }), 400
        
        prompt = data['prompt']
        
        # 生成圖片
        result = text_to_image_service(prompt)
        
        if result["success"] and result.get("images"):
            # 解碼第一張圖片
            image_data = base64.b64decode(result["images"][0])
            
            # 創建內存檔案
            img_io = io.BytesIO(image_data)
            img_io.seek(0)
            
            # 生成檔案名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}.png"
            
            return send_file(
                img_io,
                mimetype='image/png',
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({
                "success": False,
                "error": result.get("error", "Generation failed")
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/', methods=['GET'])
def index():
    """API 使用說明"""
    
    usage_info = """
    <h1>Stable Diffusion Text-to-Image API</h1>
    
    <h2>可用端點：</h2>
    
    <h3>POST /generate</h3>
    <p>生成圖片並返回 base64 編碼</p>
    <pre>
請求：
{
    "prompt": "a beautiful sunset over the ocean",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg_scale": 7
}

回應：
{
    "success": true,
    "images": ["iVBORw0KGgoAAAANS..."],
    "saved_files": ["generated_images/generated_20240703_143052_1.png"],
    "generation_time": 15.23
}
    </pre>
    
    <h3>POST /generate_and_download</h3>
    <p>生成圖片並直接下載檔案</p>
    
    <h3>GET /health</h3>
    <p>檢查服務狀態</p>
    
    <h2>使用範例 (Python):</h2>
    <pre>
import requests

# 生成圖片
response = requests.post('http://localhost:8000/generate', json={
    'prompt': 'a cute cat sitting on a table'
})

result = response.json()
if result['success']:
    print(f"圖片已保存至: {result['saved_files'][0]}")
    </pre>
    
    <h2>使用範例 (curl):</h2>
    <pre>
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful landscape"}'
    </pre>
    """
    
    return usage_info

if __name__ == '__main__':
    print("🚀 啟動 Stable Diffusion Web API 服務器")
    print("📡 服務地址: http://localhost:8000")
    print("📖 API 文檔: http://localhost:8000")
    print("🏥 健康檢查: http://localhost:8000/health")
    print("=" * 50)
    
    # 檢查 Flask 是否安裝
    try:
        import flask
        print("✅ Flask 已安裝")
    except ImportError:
        print("❌ 請先安裝 Flask: pip install flask")
        exit(1)
    
    app.run(host='0.0.0.0', port=8000, debug=True)
