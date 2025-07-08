#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fashion AI - 提示詞生成器
基於 FashionCLIP 分析結果生成高品質的 Stable Diffusion 提示詞

功能：
- 基於時尚特徵生成提示詞
- 多種生成策略和風格
- 提示詞優化和調整
- 負面提示詞生成
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class PromptStyle(Enum):
    """提示詞風格枚舉"""
    MINIMAL = "minimal"
    DETAILED = "detailed"
    ARTISTIC = "artistic"
    COMMERCIAL = "commercial"
    RUNWAY = "runway"
    STREET = "street"
    VINTAGE = "vintage"
    MODERN = "modern"

@dataclass
class PromptConfig:
    """提示詞配置"""
    style: PromptStyle = PromptStyle.DETAILED
    max_length: int = 200
    include_quality_tags: bool = True
    include_camera_settings: bool = False
    include_lighting: bool = True
    include_composition: bool = True
    confidence_threshold: float = 0.3
    max_features: int = 8

class FashionPromptGenerator:
    """時尚提示詞生成器"""
    
    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()
        self.load_templates()
        self.load_vocabularies()
        
    def load_templates(self):
        """載入提示詞模板"""
        self.templates = {
            PromptStyle.MINIMAL: "{category}, {style}, {colors}",
            PromptStyle.DETAILED: "{category}, {style}, {colors}, {materials}, {details}, {quality}",
            PromptStyle.ARTISTIC: "{category}, {style}, {colors}, {artistic_elements}, {composition}, {lighting}",
            PromptStyle.COMMERCIAL: "{category}, {style}, {colors}, {commercial_elements}, {quality}, {camera}",
            PromptStyle.RUNWAY: "{category}, {style}, {colors}, {runway_elements}, {lighting}, {composition}",
            PromptStyle.STREET: "{category}, {style}, {colors}, {street_elements}, {casual_elements}",
            PromptStyle.VINTAGE: "{category}, {style}, {colors}, {vintage_elements}, {retro_elements}",
            PromptStyle.MODERN: "{category}, {style}, {colors}, {modern_elements}, {contemporary_elements}"
        }
        
    def load_vocabularies(self):
        """載入詞彙表"""
        self.vocabularies = {
            # 服裝類別映射
            'category_mapping': {
                'dress': ['dress', 'gown', 'frock'],
                'top': ['top', 'blouse', 'shirt', 'sweater'],
                'bottom': ['pants', 'trousers', 'skirt', 'shorts'],
                'outerwear': ['jacket', 'coat', 'blazer', 'cardigan'],
                'footwear': ['shoes', 'boots', 'sneakers', 'sandals'],
                'accessories': ['bag', 'jewelry', 'hat', 'scarf']
            },
            
            # 風格描述
            'style_descriptors': {
                'elegant': ['elegant', 'sophisticated', 'refined', 'graceful'],
                'casual': ['casual', 'relaxed', 'comfortable', 'laid-back'],
                'formal': ['formal', 'professional', 'business', 'official'],
                'sporty': ['sporty', 'athletic', 'active', 'fitness'],
                'bohemian': ['bohemian', 'boho', 'free-spirited', 'artistic'],
                'vintage': ['vintage', 'retro', 'classic', 'timeless'],
                'modern': ['modern', 'contemporary', 'current', 'trendy']
            },
            
            # 顏色描述
            'color_descriptors': {
                'red': ['red', 'crimson', 'scarlet', 'burgundy'],
                'blue': ['blue', 'navy', 'azure', 'cobalt'],
                'green': ['green', 'emerald', 'olive', 'forest'],
                'yellow': ['yellow', 'golden', 'amber', 'lemon'],
                'purple': ['purple', 'violet', 'lavender', 'plum'],
                'pink': ['pink', 'rose', 'coral', 'blush'],
                'black': ['black', 'jet', 'ebony', 'charcoal'],
                'white': ['white', 'ivory', 'cream', 'pearl'],
                'gray': ['gray', 'silver', 'slate', 'ash'],
                'brown': ['brown', 'tan', 'caramel', 'chocolate']
            },
            
            # 材質描述
            'material_descriptors': {
                'cotton': ['cotton', 'soft cotton', 'organic cotton'],
                'silk': ['silk', 'smooth silk', 'luxurious silk'],
                'wool': ['wool', 'fine wool', 'merino wool'],
                'leather': ['leather', 'genuine leather', 'smooth leather'],
                'denim': ['denim', 'blue denim', 'distressed denim'],
                'lace': ['lace', 'delicate lace', 'floral lace'],
                'satin': ['satin', 'smooth satin', 'lustrous satin'],
                'chiffon': ['chiffon', 'flowing chiffon', 'sheer chiffon']
            },
            
            # 品質標籤
            'quality_tags': [
                'high quality', 'professional photography', 'detailed', 'sharp focus',
                'realistic', 'photorealistic', 'ultra-detailed', 'high resolution',
                'studio lighting', 'perfect lighting', 'soft lighting', 'natural lighting'
            ],
            
            # 相機設置
            'camera_settings': [
                'shot on Canon EOS R5', 'shot on Sony A7R IV', 'shot on Nikon D850',
                'professional camera', 'DSLR', 'mirrorless camera',
                'portrait lens', '85mm lens', '50mm lens', 'shallow depth of field',
                'bokeh background', 'sharp focus', 'perfect exposure'
            ],
            
            # 燈光設置
            'lighting_options': [
                'natural lighting', 'studio lighting', 'soft lighting', 'dramatic lighting',
                'golden hour', 'rim lighting', 'backlighting', 'diffused lighting',
                'window light', 'professional lighting', 'perfect lighting'
            ],
            
            # 構圖選項
            'composition_options': [
                'centered composition', 'rule of thirds', 'portrait composition',
                'full body shot', 'half body shot', 'close-up', 'medium shot',
                'fashion photography', 'editorial photography', 'commercial photography'
            ],
            
            # 負面提示詞
            'negative_prompts': [
                'blurry', 'low quality', 'bad anatomy', 'deformed', 'distorted',
                'oversaturated', 'undersaturated', 'bad lighting', 'harsh shadows',
                'overexposed', 'underexposed', 'noise', 'artifacts', 'compression',
                'watermark', 'text', 'logo', 'signature', 'username'
            ]
        }
    
    def generate_prompt(self, analysis_result: Dict[str, Any], 
                       custom_style: Optional[PromptStyle] = None) -> Dict[str, str]:
        """
        基於分析結果生成提示詞
        
        Args:
            analysis_result: FashionCLIP 分析結果
            custom_style: 自定義風格（可選）
            
        Returns:
            包含正面和負面提示詞的字典
        """
        style = custom_style or self.config.style
        
        # 提取核心特徵
        features = self._extract_features(analysis_result)
        
        # 生成提示詞組件
        components = self._generate_components(features, style)
        
        # 組裝提示詞
        positive_prompt = self._assemble_prompt(components, style)
        negative_prompt = self._generate_negative_prompt()
        
        # 優化提示詞
        positive_prompt = self._optimize_prompt(positive_prompt)
        
        return {
            'positive_prompt': positive_prompt,
            'negative_prompt': negative_prompt,
            'style': style.value,
            'features_used': features,
            'components': components
        }
    
    def _extract_features(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """從分析結果中提取特徵"""
        features = {}
        
        # 提取基本類別
        if 'category' in analysis_result:
            features['category'] = analysis_result['category']
        
        # 提取風格
        if 'style' in analysis_result:
            features['style'] = analysis_result['style']
        
        # 提取顏色
        if 'colors' in analysis_result:
            features['colors'] = analysis_result['colors']
        
        # 提取材質
        if 'materials' in analysis_result:
            features['materials'] = analysis_result['materials']
        
        # 提取詳細特徵
        if 'detailed_features' in analysis_result:
            features['details'] = analysis_result['detailed_features']
        
        # 提取置信度
        if 'confidence' in analysis_result:
            features['confidence'] = analysis_result['confidence']
        
        return features
    
    def _generate_components(self, features: Dict[str, Any], 
                           style: PromptStyle) -> Dict[str, str]:
        """生成提示詞組件"""
        components = {}
        
        # 類別組件
        if 'category' in features:
            components['category'] = self._get_category_description(features['category'])
        
        # 風格組件
        if 'style' in features:
            components['style'] = self._get_style_description(features['style'])
        
        # 顏色組件
        if 'colors' in features:
            components['colors'] = self._get_color_description(features['colors'])
        
        # 材質組件
        if 'materials' in features:
            components['materials'] = self._get_material_description(features['materials'])
        
        # 詳細特徵組件
        if 'details' in features:
            components['details'] = self._get_detail_description(features['details'])
        
        # 風格特定組件
        components.update(self._get_style_specific_components(style))
        
        return components
    
    def _get_category_description(self, category: str) -> str:
        """獲取類別描述"""
        category_map = self.vocabularies['category_mapping']
        if category.lower() in category_map:
            return category_map[category.lower()][0]
        return category
    
    def _get_style_description(self, style: str) -> str:
        """獲取風格描述"""
        style_map = self.vocabularies['style_descriptors']
        if style.lower() in style_map:
            return style_map[style.lower()][0]
        return style
    
    def _get_color_description(self, colors: List[str]) -> str:
        """獲取顏色描述"""
        if not colors:
            return ""
        
        color_map = self.vocabularies['color_descriptors']
        descriptions = []
        
        for color in colors[:3]:  # 最多取前3個顏色
            if color.lower() in color_map:
                descriptions.append(color_map[color.lower()][0])
            else:
                descriptions.append(color)
        
        return ", ".join(descriptions)
    
    def _get_material_description(self, materials: List[str]) -> str:
        """獲取材質描述"""
        if not materials:
            return ""
        
        material_map = self.vocabularies['material_descriptors']
        descriptions = []
        
        for material in materials[:2]:  # 最多取前2個材質
            if material.lower() in material_map:
                descriptions.append(material_map[material.lower()][0])
            else:
                descriptions.append(material)
        
        return ", ".join(descriptions)
    
    def _get_detail_description(self, details: List[str]) -> str:
        """獲取詳細特徵描述"""
        if not details:
            return ""
        
        # 過濾和限制詳細特徵
        filtered_details = []
        for detail in details:
            if len(detail) > 2 and detail.lower() not in ['the', 'and', 'or', 'but']:
                filtered_details.append(detail)
        
        return ", ".join(filtered_details[:self.config.max_features])
    
    def _get_style_specific_components(self, style: PromptStyle) -> Dict[str, str]:
        """獲取風格特定組件"""
        components = {}
        
        if self.config.include_quality_tags:
            components['quality'] = ", ".join(self.vocabularies['quality_tags'][:2])
        
        if self.config.include_camera_settings:
            components['camera'] = ", ".join(self.vocabularies['camera_settings'][:2])
        
        if self.config.include_lighting:
            components['lighting'] = ", ".join(self.vocabularies['lighting_options'][:2])
        
        if self.config.include_composition:
            components['composition'] = ", ".join(self.vocabularies['composition_options'][:2])
        
        # 風格特定元素
        if style == PromptStyle.ARTISTIC:
            components['artistic_elements'] = "artistic photography, creative composition"
        elif style == PromptStyle.COMMERCIAL:
            components['commercial_elements'] = "commercial photography, product photography"
        elif style == PromptStyle.RUNWAY:
            components['runway_elements'] = "runway fashion, fashion show, catwalk"
        elif style == PromptStyle.STREET:
            components['street_elements'] = "street fashion, urban style"
            components['casual_elements'] = "everyday wear, casual style"
        elif style == PromptStyle.VINTAGE:
            components['vintage_elements'] = "vintage fashion, retro style"
            components['retro_elements'] = "classic design, timeless style"
        elif style == PromptStyle.MODERN:
            components['modern_elements'] = "modern fashion, contemporary style"
            components['contemporary_elements'] = "current trend, innovative design"
        
        return components
    
    def _assemble_prompt(self, components: Dict[str, str], 
                        style: PromptStyle) -> str:
        """組裝提示詞"""
        template = self.templates[style]
        
        # 替換模板中的占位符
        prompt = template
        for key, value in components.items():
            if value:
                prompt = prompt.replace(f"{{{key}}}", value)
        
        # 移除空的占位符
        prompt = re.sub(r'\{[^}]+\}', '', prompt)
        
        # 清理多餘的標點符號
        prompt = re.sub(r',\s*,', ',', prompt)
        prompt = re.sub(r',\s*$', '', prompt)
        prompt = re.sub(r'^\s*,', '', prompt)
        prompt = re.sub(r'\s+', ' ', prompt)
        
        return prompt.strip()
    
    def _generate_negative_prompt(self) -> str:
        """生成負面提示詞"""
        negative_prompts = self.vocabularies['negative_prompts']
        return ", ".join(negative_prompts[:10])  # 取前10個負面提示詞
    
    def _optimize_prompt(self, prompt: str) -> str:
        """優化提示詞"""
        # 移除重複的詞彙
        words = prompt.split(', ')
        unique_words = []
        seen = set()
        
        for word in words:
            word = word.strip()
            if word and word.lower() not in seen:
                unique_words.append(word)
                seen.add(word.lower())
        
        optimized_prompt = ', '.join(unique_words)
        
        # 長度限制
        if len(optimized_prompt) > self.config.max_length:
            words = optimized_prompt.split(', ')
            truncated_words = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 2 <= self.config.max_length:
                    truncated_words.append(word)
                    current_length += len(word) + 2
                else:
                    break
            
            optimized_prompt = ', '.join(truncated_words)
        
        return optimized_prompt
    
    def generate_multiple_prompts(self, analysis_result: Dict[str, Any],
                                 styles: List[PromptStyle] = None) -> Dict[str, Dict[str, str]]:
        """生成多種風格的提示詞"""
        if styles is None:
            styles = [PromptStyle.MINIMAL, PromptStyle.DETAILED, PromptStyle.ARTISTIC]
        
        prompts = {}
        for style in styles:
            prompts[style.value] = self.generate_prompt(analysis_result, style)
        
        return prompts
    
    def enhance_prompt(self, base_prompt: str, 
                      enhancements: List[str] = None) -> str:
        """增強現有提示詞"""
        if enhancements is None:
            enhancements = ['high quality', 'detailed', 'professional photography']
        
        enhanced_prompt = base_prompt
        for enhancement in enhancements:
            if enhancement not in enhanced_prompt:
                enhanced_prompt += f", {enhancement}"
        
        return self._optimize_prompt(enhanced_prompt)


# 使用範例
if __name__ == "__main__":
    # 模擬分析結果
    analysis_result = {
        'category': 'dress',
        'style': 'elegant',
        'colors': ['red', 'black'],
        'materials': ['silk', 'lace'],
        'detailed_features': ['long sleeves', 'v-neck', 'floor length'],
        'confidence': 0.85
    }
    
    # 創建提示詞生成器
    generator = FashionPromptGenerator()
    
    # 生成單個提示詞
    result = generator.generate_prompt(analysis_result, PromptStyle.DETAILED)
    print("詳細風格提示詞:")
    print(f"正面: {result['positive_prompt']}")
    print(f"負面: {result['negative_prompt']}")
    print()
    
    # 生成多種風格提示詞
    multiple_results = generator.generate_multiple_prompts(analysis_result)
    print("多種風格提示詞:")
    for style, prompt_data in multiple_results.items():
        print(f"{style}: {prompt_data['positive_prompt']}")
    print()
    
    # 增強現有提示詞
    base_prompt = "red dress, elegant style"
    enhanced = generator.enhance_prompt(base_prompt)
    print(f"增強後的提示詞: {enhanced}")
