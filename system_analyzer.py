#!/usr/bin/env python3
"""
系統硬體分析工具 - 推薦最適合的 FashionCLIP 模型
"""

import psutil
import platform
import subprocess
import json
from datetime import datetime

class SystemAnalyzer:
    def __init__(self):
        self.results = {}
        
    def check_system_specs(self):
        """檢查系統基本規格"""
        print("🔍 檢查系統規格...")
        
        # CPU 資訊
        self.results['cpu'] = {
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 'Unknown',
            'usage': psutil.cpu_percent(interval=1)
        }
        
        # 記憶體資訊
        memory = psutil.virtual_memory()
        self.results['memory'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'usage_percent': memory.percent
        }
        
        # 系統資訊
        self.results['system'] = {
            'os': platform.system(),
            'version': platform.version(),
            'architecture': platform.architecture()[0]
        }
        
        print(f"✅ CPU: {self.results['cpu']['cores']}核心/{self.results['cpu']['threads']}執行緒")
        print(f"✅ 記憶體: {self.results['memory']['total_gb']} GB")
        print(f"✅ 系統: {self.results['system']['os']} {self.results['system']['architecture']}")
    
    def check_gpu_info(self):
        """檢查 GPU 資訊"""
        print("\n🎮 檢查 GPU 資訊...")
        
        gpu_info = {
            'has_cuda': False,
            'gpu_count': 0,
            'gpu_memory': 0,
            'gpu_name': 'Unknown'
        }
        
        try:
            # 檢查是否有 CUDA
            import torch
            if torch.cuda.is_available():
                gpu_info['has_cuda'] = True
                gpu_info['gpu_count'] = torch.cuda.device_count()
                gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
                gpu_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"✅ CUDA GPU: {gpu_info['gpu_name']}")
                print(f"✅ GPU 記憶體: {gpu_info['gpu_memory']:.1f} GB")
            else:
                print("⚠️ 未偵測到 CUDA GPU，將使用 CPU")
        except ImportError:
            print("⚠️ PyTorch 未安裝，無法檢查 GPU")
        
        self.results['gpu'] = gpu_info
    
    def check_available_models(self):
        """檢查可用的 FashionCLIP 模型"""
        print("\n📋 分析可用的 FashionCLIP 模型...")
        
        models = {
            'lightweight': {
                'name': 'openai/clip-vit-base-patch32',
                'size_mb': 600,
                'memory_requirement_gb': 2,
                'description': '標準 CLIP - 輕量級，適合大多數電腦',
                'fashion_optimized': False
            },
            'fashion_specialized': {
                'name': 'patrickjohncyh/fashion-clip',
                'size_mb': 1200,
                'memory_requirement_gb': 3,
                'description': '專業時尚 CLIP - 針對服飾優化',
                'fashion_optimized': True
            },
            'large_model': {
                'name': 'openai/clip-vit-large-patch14',
                'size_mb': 1800,
                'memory_requirement_gb': 4,
                'description': '大型 CLIP - 最佳效果但需要更多資源',
                'fashion_optimized': False
            }
        }
        
        self.results['available_models'] = models
        
        for model_type, info in models.items():
            print(f"   {model_type}: {info['name']}")
            print(f"     大小: {info['size_mb']} MB")
            print(f"     記憶體需求: {info['memory_requirement_gb']} GB")
    
    def recommend_model(self):
        """基於系統規格推薦最適合的模型"""
        print("\n🎯 模型推薦分析...")
        
        memory_gb = self.results['memory']['total_gb']
        has_gpu = self.results['gpu']['has_cuda']
        gpu_memory = self.results['gpu']['gpu_memory']
        
        recommendations = []
        
        # 基於記憶體推薦
        if memory_gb >= 8:
            if has_gpu and gpu_memory >= 4:
                recommendations.append({
                    'model': 'fashion_specialized',
                    'reason': '充足的系統記憶體和 GPU，推薦專業時尚模型',
                    'priority': 1
                })
                recommendations.append({
                    'model': 'large_model',
                    'reason': '高階配置可嘗試大型模型',
                    'priority': 2
                })
            else:
                recommendations.append({
                    'model': 'fashion_specialized',
                    'reason': '充足記憶體，推薦專業時尚模型（CPU 版本）',
                    'priority': 1
                })
        elif memory_gb >= 4:
            recommendations.append({
                'model': 'lightweight',
                'reason': '中等記憶體配置，推薦輕量級模型',
                'priority': 1
            })
            if has_gpu:
                recommendations.append({
                    'model': 'fashion_specialized',
                    'reason': '有 GPU 加速，可嘗試專業模型',
                    'priority': 2
                })
        else:
            recommendations.append({
                'model': 'lightweight',
                'reason': '記憶體較少，僅推薦輕量級模型',
                'priority': 1
            })
        
        # 排序推薦
        recommendations.sort(key=lambda x: x['priority'])
        self.results['recommendations'] = recommendations
        
        print("推薦結果：")
        for i, rec in enumerate(recommendations, 1):
            model_info = self.results['available_models'][rec['model']]
            print(f"   {i}. {model_info['name']}")
            print(f"      原因: {rec['reason']}")
            print(f"      時尚優化: {'是' if model_info['fashion_optimized'] else '否'}")
    
    def generate_optimal_config(self):
        """生成最佳配置建議"""
        print("\n⚙️ 生成最佳配置...")
        
        top_recommendation = self.results['recommendations'][0]
        model_key = top_recommendation['model']
        model_info = self.results['available_models'][model_key]
        
        config = {
            'recommended_model': model_info['name'],
            'use_gpu': self.results['gpu']['has_cuda'],
            'batch_size': 1 if self.results['memory']['total_gb'] < 8 else 2,
            'precision': 'float16' if self.results['gpu']['has_cuda'] else 'float32',
            'cache_dir': './models_cache'
        }
        
        self.results['optimal_config'] = config
        
        print("最佳配置:")
        for key, value in config.items():
            print(f"   {key}: {value}")
    
    def save_analysis_report(self):
        """保存分析報告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_analysis': self.results,
            'summary': {
                'recommended_model': self.results['recommendations'][0]['model'],
                'model_name': self.results['available_models'][self.results['recommendations'][0]['model']]['name'],
                'reason': self.results['recommendations'][0]['reason']
            }
        }
        
        with open('system_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 分析報告已保存: system_analysis_report.json")
        return report
    
    def run_analysis(self):
        """執行完整分析"""
        print("=" * 60)
        print("系統硬體分析 & FashionCLIP 模型推薦")
        print("=" * 60)
        
        self.check_system_specs()
        self.check_gpu_info()
        self.check_available_models()
        self.recommend_model()
        self.generate_optimal_config()
        
        report = self.save_analysis_report()
        
        print("\n" + "=" * 60)
        print("🎯 最終推薦")
        print("=" * 60)
        print(f"推薦模型: {report['summary']['model_name']}")
        print(f"推薦原因: {report['summary']['reason']}")
        
        return report

if __name__ == "__main__":
    analyzer = SystemAnalyzer()
    analyzer.run_analysis()
