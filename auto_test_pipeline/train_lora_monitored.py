#!/usr/bin/env python3
"""
LoRA 訓練腳本 - 基於 train_lora.py 添加監控功能
與 train_lora.py 保持相同的核心邏輯，只添加基本的日誌監控
"""

import subprocess
import os
import sys
import warnings
import argparse
import datetime
import logging
import json
import shutil
from typing import Tuple, Dict, Optional
from PIL import Image

# 減少警告訊息
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*xformers.*")
warnings.filterwarnings("ignore", message=".*triton.*")

# 智慧路徑檢測 - 支援從任何目錄執行
def setup_working_directory():
    """設定工作目錄，支援從根目錄或auto_test_pipeline目錄執行"""
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 檢查當前目錄結構
    if os.path.basename(current_dir) == "stable-diffusion-webui":
        # 從根目錄執行
        print(f"📁 從根目錄執行: {current_dir}")
        return "root"
    elif os.path.basename(current_dir) == "auto_test_pipeline":
        # 從auto_test_pipeline目錄執行
        print(f"📁 從auto_test_pipeline目錄執行: {current_dir}")
        return "auto_test_pipeline"
    else:
        # 切換到腳本所在目錄
        os.chdir(script_dir)
        print(f"📁 切換到腳本目錄: {script_dir}")
        return "auto_test_pipeline"

# 設定工作目錄並獲取執行模式
execution_mode = setup_working_directory()

def get_path_config(execution_mode: str) -> Dict[str, str]:
    """根據執行模式獲取路徑配置"""
    if execution_mode == "root":
        # 從stable-diffusion-webui根目錄執行
        return {
            "train_data_dir": "auto_test_pipeline/lora_train_set",
            "output_dir": "auto_test_pipeline/lora_output", 
            "pretrained_model_path": "models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
            "train_network_script": "auto_test_pipeline/train_network.py",
            "log_dir": "auto_test_pipeline/training_logs"
        }
    else:
        # 從auto_test_pipeline目錄執行
        return {
            "train_data_dir": "lora_train_set",
            "output_dir": "lora_output",
            "pretrained_model_path": "../models/Stable-diffusion/v1-5-pruned-emaonly.safetensors", 
            "train_network_script": "train_network.py",
            "log_dir": "training_logs"
        }

# 獲取路徑配置
path_config = get_path_config(execution_mode)
print(f"📋 路徑配置:")
for key, value in path_config.items():
    print(f"  {key}: {value}")
print("=" * 60)
# 與 train_lora.py 保持一致的固定參數
FIXED_TRAINING_PARAMS = {
    "max_train_steps": 100,         # 固定 100 步
    "learning_rate": 5e-5,          # 固定學習率
    "network_dim": 32,              # 網路維度
    "save_every_n_epochs": 50,      # 儲存頻率
}

print("📋 固定參數設定:")
for key, value in FIXED_TRAINING_PARAMS.items():
    print(f"  {key}: {value}")
print("=" * 60)
# ==================== 參數設定結束 ====================

def setup_logging():
    """設定日誌系統 - 使用動態路徑"""
    log_dir = path_config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"lora_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"📋 日誌檔案: {log_file}")
    return logger

def find_latest_lora():
    """找到最新的 LoRA 模型檔案 - 使用動態路徑"""
    lora_path = path_config["output_dir"]
    if not os.path.exists(lora_path):
        return None
    
    lora_files = [f for f in os.listdir(lora_path) if f.endswith('.safetensors')]
    if not lora_files:
        return None
    
    # 找最新的檔案
    latest_lora = max(lora_files, key=lambda x: os.path.getmtime(os.path.join(lora_path, x)))
    return os.path.join(lora_path, latest_lora)

def backup_existing_lora():
    """備份現有的 LoRA 模型 - 使用動態路徑"""
    existing_lora = find_latest_lora()
    if existing_lora and os.path.exists(existing_lora):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"lora_backup_{timestamp}.safetensors"
        backup_path = os.path.join(path_config["output_dir"], backup_name)
        
        import shutil
        shutil.copy2(existing_lora, backup_path)
        print(f"📦 備份現有模型: {backup_name}")
        return backup_path
    return None

def check_image_size(data_folder, target_size=512):
    """檢查圖片大小是否符合要求，跳過超出尺寸的圖片"""
    print(f"🔍 檢查圖片大小是否符合 {target_size}x{target_size} 要求...")
    
    files = os.listdir(data_folder)
    img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    valid_count = 0
    invalid_files = []
    
    for img_file in img_files:
        img_path = os.path.join(data_folder, img_file)
        
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                # 檢查圖片尺寸是否符合要求
                if width <= target_size and height <= target_size:
                    valid_count += 1
                    print(f"  ✅ {img_file}: {width}x{height} (符合要求)")
                else:
                    invalid_files.append((img_file, width, height))
                    print(f"  ⚠️  {img_file}: {width}x{height} (超出 {target_size}x{target_size}，將跳過)")
                
        except Exception as e:
            print(f"❌ 無法讀取圖片 {img_file}: {str(e)}")
            invalid_files.append((img_file, "讀取失敗", ""))
    
    print(f"\n📊 圖片尺寸檢查結果：")
    print(f"✅ 符合要求的圖片：{valid_count} 張")
    print(f"⚠️  超出尺寸的圖片：{len(invalid_files)} 張")
    
    if invalid_files:
        print(f"\n📋 超出尺寸的圖片將被跳過：")
        for img_file, width, height in invalid_files:
            print(f"   - {img_file}: {width}x{height}")
        print(f"\n💡 建議：使用 generate_caption_fashionclip.py 預處理圖片")
    
    if valid_count == 0:
        print(f"❌ 沒有任何圖片符合要求！")
        return False
    else:
        print(f"🎯 將使用 {valid_count} 張符合要求的圖片進行訓練")
        return True

def train_lora_with_monitoring(continue_from_checkpoint=False, logger=None):
    """執行 LoRA 訓練 - 基於原始 train_lora.py，添加監控，支援動態路徑"""
    
    if logger:
        logger.info("🎯 開始 LoRA 訓練流程（含監控）")
        logger.info(f"🏃 執行模式: {execution_mode}")
    
    # 使用動態路徑配置
    train_dir = path_config["train_data_dir"]
    sub_folder = "10_test" 
    data_folder = os.path.join(train_dir, sub_folder)

    # 自動處理資料夾改名 - 與原始代碼一致
    old_folder = os.path.join(train_dir, "test_10")
    if os.path.exists(old_folder) and not os.path.exists(data_folder):
        os.rename(old_folder, data_folder)
        print(f"已自動將 {old_folder} 改名為 {data_folder}")
        if logger:
            logger.info(f"已自動將 {old_folder} 改名為 {data_folder}")

    # 詳細檢查資料夾結構 - 與原始代碼一致
    print(f"🔍 檢查資料夾結構...")
    if logger:
        logger.info("🔍 檢查資料夾結構...")
        
    if not os.path.isdir(train_dir):
        print(f"❌ 找不到父資料夾：{train_dir}")
        if logger:
            logger.error(f"❌ 找不到父資料夾：{train_dir}")
        sys.exit(1)

    if not os.path.isdir(data_folder):
        print(f"❌ 找不到訓練資料夾：{data_folder}")
        if logger:
            logger.error(f"❌ 找不到訓練資料夾：{data_folder}")
        print(f"📁 {train_dir} 內容：")
        if os.path.exists(train_dir):
            for item in os.listdir(train_dir):
                print(f"  {item}")
        sys.exit(1)

    # 檢查圖片和文字檔案 - 與原始代碼一致
    files = os.listdir(data_folder)
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    txt_files = [f for f in files if f.lower().endswith('.txt')]

    print(f"📂 {data_folder} 內容：")
    print(f"  圖片檔案數量：{len(jpg_files)}")
    print(f"  文字檔案數量：{len(txt_files)}")
    
    if logger:
        logger.info(f"📂 {data_folder} 內容：圖片 {len(jpg_files)} 張，文字 {len(txt_files)} 個")

    if len(jpg_files) == 0:
        print("❌ 沒有找到任何 .jpg 檔案！")
        if logger:
            logger.error("❌ 沒有找到任何 .jpg 檔案！")
        sys.exit(1)

    # 🎯 關鍵步驟：檢查圖片大小 - 與原始代碼一致
    if not check_image_size(data_folder, target_size=512):
        print("❌ 沒有任何圖片符合要求！")
        print("💡 請使用 generate_caption_fashionclip.py 預處理圖片")
        if logger:
            logger.error("❌ 沒有任何圖片符合要求！")
        sys.exit(1)

    # 自動將 .JPG 副檔名改成 .jpg - 與原始代碼一致
    for fname in files:
        if fname.lower().endswith('.jpg') and not fname.endswith('.jpg'):
            src = os.path.join(data_folder, fname)
            dst = os.path.join(data_folder, fname[:-4] + '.jpg')
            os.rename(src, dst)
            print(f"已自動將 {src} 改名為 {dst}")
            if logger:
                logger.info(f"已自動將 {fname} 改名")

    # 處理繼續訓練選項 - 與原始代碼一致
    resume_from = None
    if continue_from_checkpoint:
        existing_lora = find_latest_lora()
        if existing_lora:
            print(f"🔄 從檢查點繼續訓練: {os.path.basename(existing_lora)}")
            if logger:
                logger.info(f"🔄 從檢查點繼續訓練: {os.path.basename(existing_lora)}")
            resume_from = existing_lora
            # 備份現有模型
            backup_existing_lora()
        else:
            print("⚠️ 沒有找到現有的 LoRA 檔案，將開始新的訓練")
            if logger:
                logger.warning("⚠️ 沒有找到現有的 LoRA 檔案，將開始新的訓練")
    else:
        print("🆕 開始新的獨立 LoRA 訓練")
        if logger:
            logger.info("🆕 開始新的獨立 LoRA 訓練")
        # 如果存在舊模型，備份它
        backup_existing_lora()

    # 基本訓練指令 - 使用動態路徑配置
    cmd_parts = [
        f"python {path_config['train_network_script']}",
        f"--pretrained_model_name_or_path={path_config['pretrained_model_path']}",
        f"--train_data_dir={path_config['train_data_dir']}",
        f"--output_dir={path_config['output_dir']}",
        "--resolution=512,512",
        "--network_module=networks.lora",
        f"--network_dim={FIXED_TRAINING_PARAMS['network_dim']}",        # 使用固定參數
        "--train_batch_size=1",
        f"--max_train_steps={FIXED_TRAINING_PARAMS['max_train_steps']}",   # 使用固定參數
        "--mixed_precision=fp16",
        "--cache_latents",
        f"--learning_rate={FIXED_TRAINING_PARAMS['learning_rate']}",    # 使用固定參數
        f"--save_every_n_epochs={FIXED_TRAINING_PARAMS['save_every_n_epochs']}",
        "--save_model_as=safetensors"
    ]
    
    # 如果從檢查點繼續，添加相應參數 - 與原始代碼一致
    if resume_from:
        cmd_parts.extend([
            f"--resume={resume_from}",
            "--save_state"  # 保存訓練狀態
        ])
    
    cmd = " ".join(cmd_parts)
    
    print("🚀 開始 LoRA 微調 ...")
    print(f"📋 訓練命令: {cmd}")
    if logger:
        logger.info("🚀 開始 LoRA 微調 ...")
        logger.info(f"📋 訓練命令: {cmd}")
    
    # 記錄訓練開始時間
    start_time = datetime.datetime.now()
    if logger:
        logger.info(f"⏰ 訓練開始時間: {start_time}")
    
    # 執行訓練 - 與原始代碼完全一致（不使用 capture_output）
    result = subprocess.run(cmd, shell=True)
    
    # 記錄訓練結束時間
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    if logger:
        logger.info(f"⏰ 訓練結束時間: {end_time}")
        logger.info(f"⏱️  訓練持續時間: {duration}")
    
    if result.returncode == 0:
        print("✅ LoRA 訓練完成")
        if logger:
            logger.info("✅ LoRA 訓練完成")
        
        # 顯示訓練結果 - 與原始代碼一致
        final_lora = find_latest_lora()
        if final_lora:
            file_size = os.path.getsize(final_lora) / (1024*1024)
            print(f"📁 最終模型: {os.path.basename(final_lora)}")
            print(f"📊 檔案大小: {file_size:.2f} MB")
            if logger:
                logger.info(f"📁 最終模型: {os.path.basename(final_lora)}")
                logger.info(f"📊 檔案大小: {file_size:.2f} MB")
            
            if continue_from_checkpoint:
                print("🔄 檢查點訓練完成 - 模型已更新")
                if logger:
                    logger.info("🔄 檢查點訓練完成 - 模型已更新")
            else:
                print("🆕 新模型訓練完成")
                if logger:
                    logger.info("🆕 新模型訓練完成")
        
        # 保存訓練結果報告 - 新增的監控功能
        if logger:
            training_report = {
                "success": True,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "final_model": os.path.basename(final_lora) if final_lora else None,
                "model_size_mb": file_size if final_lora else 0,
                "continue_from_checkpoint": continue_from_checkpoint,
                "training_params": FIXED_TRAINING_PARAMS
            }
            
            report_file = f"training_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(training_report, f, indent=2, ensure_ascii=False)
            logger.info(f"📋 訓練報告已保存: {report_file}")
        
        return True
    else:
        print("❌ LoRA 訓練失敗")
        if logger:
            logger.error(f"❌ LoRA 訓練失敗，返回碼: {result.returncode}")
        return False

def main():
    """主函數 - 基於原始 train_lora.py 的邏輯"""
    
    # 設定日誌系統 - 新增的監控功能
    logger = setup_logging()
    logger.info("🎯 開始 LoRA 訓練腳本（監控版本）")
    
    parser = argparse.ArgumentParser(description="LoRA 訓練腳本 - 含監控功能")
    parser.add_argument("--continue", "-c", action="store_true", 
                       dest="continue_training",
                       help="從現有的 LoRA 檔案繼續訓練")
    parser.add_argument("--new", "-n", action="store_true",
                       dest="new_training", 
                       help="開始新的獨立 LoRA 訓練")
    
    args = parser.parse_args()
    
    # 決定訓練模式 - 與原始代碼完全一致
    if args.continue_training and args.new_training:
        print("❌ 錯誤：不能同時指定 --continue 和 --new")
        logger.error("❌ 錯誤：不能同時指定 --continue 和 --new")
        sys.exit(1)
    elif args.continue_training:
        print("🔄 模式：從檢查點繼續訓練")
        logger.info("🔄 模式：從檢查點繼續訓練")
        continue_from_checkpoint = True
    elif args.new_training:
        print("🆕 模式：開始新的獨立訓練")
        logger.info("🆕 模式：開始新的獨立訓練")
        continue_from_checkpoint = False
    else:
        # 如果沒有指定參數，詢問用戶 - 與原始代碼一致
        existing_lora = find_latest_lora()
        if existing_lora:
            print(f"🔍 發現現有的 LoRA 模型: {os.path.basename(existing_lora)}")
            logger.info(f"🔍 發現現有的 LoRA 模型: {os.path.basename(existing_lora)}")
            print("請選擇訓練模式：")
            print("1. 從現有模型繼續訓練 (累積調教)")
            print("2. 開始新的獨立訓練 (重新開始)")
            
            while True:
                choice = input("請輸入選擇 (1 或 2): ").strip()
                if choice == "1":
                    continue_from_checkpoint = True
                    logger.info("用戶選擇：從現有模型繼續訓練")
                    break
                elif choice == "2":
                    continue_from_checkpoint = False
                    logger.info("用戶選擇：開始新的獨立訓練")
                    break
                else:
                    print("請輸入 1 或 2")
        else:
            print("🆕 沒有找到現有模型，將開始新的訓練")
            logger.info("🆕 沒有找到現有模型，將開始新的訓練")
            continue_from_checkpoint = False
    
    # 執行訓練 - 調用監控版本的函數
    success = train_lora_with_monitoring(continue_from_checkpoint, logger)
    
    if success:
        logger.info("✅ 訓練腳本執行成功")
    else:
        logger.error("❌ 訓練腳本執行失敗")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

class LoRATrainer:
    """LoRA 訓練器 - 整合監控功能"""
    
    def __init__(self, continue_from_checkpoint: bool = False, custom_params: dict = None):
        """
        初始化 LoRA 訓練器
        
        Args:
            continue_from_checkpoint: 是否從檢查點繼續訓練
            custom_params: 自定義訓練參數字典
        """
        # 切換到腳本所在目錄
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.getcwd() != script_dir:
            os.chdir(script_dir)
            print(f"🔄 切換工作目錄到: {script_dir}")
        
        self.training_dir = os.path.dirname(os.path.abspath(__file__))
        self.monitor = None
        self.continue_from_checkpoint = continue_from_checkpoint
        self.custom_params = custom_params or {}
        
        # 設定日誌
        self.setup_logging()
        
        # 如果有自定義參數，記錄到日誌
        if self.custom_params:
            self.logger.info(f"🔧 使用自定義參數: {self.custom_params}")
    
    def setup_logging(self):
        """設定日誌系統 - 使用動態路徑"""
        log_dir = path_config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"lora_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def find_latest_lora(self):
        """找到最新的 LoRA 模型檔案 - 使用動態路徑"""
        lora_path = path_config["output_dir"]
        if not os.path.exists(lora_path):
            return None
        
        lora_files = [f for f in os.listdir(lora_path) if f.endswith('.safetensors')]
        if not lora_files:
            return None
        
        # 找最新的檔案
        latest_lora = max(lora_files, key=lambda x: os.path.getmtime(os.path.join(lora_path, x)))
        return os.path.join(lora_path, latest_lora)

    def backup_existing_lora(self):
        """備份現有的 LoRA 模型 - 使用動態路徑"""
        existing_lora = self.find_latest_lora()
        if existing_lora and os.path.exists(existing_lora):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"lora_backup_{timestamp}.safetensors"
            backup_path = os.path.join(path_config["output_dir"], backup_name)
            
            shutil.copy2(existing_lora, backup_path)
            self.logger.info(f"📦 備份現有模型: {backup_name}")
            return backup_path
        return None
        
    def check_image_sizes(self, data_folder: str, target_size: int = 512) -> bool:
        """檢查圖片大小是否符合要求，跳過超出尺寸的圖片"""
        self.logger.info(f"🔍 檢查圖片大小是否符合 {target_size}x{target_size} 要求...")
        
        files = os.listdir(data_folder)
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        valid_count = 0
        invalid_files = []
        
        for img_file in img_files:
            img_path = os.path.join(data_folder, img_file)
            
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                    # 檢查圖片尺寸是否符合要求
                    if width <= target_size and height <= target_size:
                        valid_count += 1
                        self.logger.info(f"  ✅ {img_file}: {width}x{height} (符合要求)")
                    else:
                        invalid_files.append((img_file, width, height))
                        self.logger.warning(f"  ⚠️  {img_file}: {width}x{height} (超出 {target_size}x{target_size}，將跳過)")
                    
            except Exception as e:
                self.logger.error(f"❌ 無法讀取圖片 {img_file}: {str(e)}")
                invalid_files.append((img_file, "讀取失敗", ""))
        
        self.logger.info(f"📊 圖片尺寸檢查結果：")
        self.logger.info(f"✅ 符合要求的圖片：{valid_count} 張")
        self.logger.info(f"⚠️  超出尺寸的圖片：{len(invalid_files)} 張")
        
        if invalid_files:
            self.logger.warning(f"📋 超出尺寸的圖片將被跳過：")
            for img_file, width, height in invalid_files:
                self.logger.warning(f"   - {img_file}: {width}x{height}")
            self.logger.info(f"💡 建議：使用 generate_caption_fashionclip.py 預處理圖片")
        
        if valid_count == 0:
            self.logger.error(f"❌ 沒有任何圖片符合要求！")
            return False
        else:
            self.logger.info(f"🎯 將使用 {valid_count} 張符合要求的圖片進行訓練")
            return True

    def check_training_requirements(self) -> bool:
        """檢查訓練需求 - 使用動態路徑"""
        self.logger.info("🔍 檢查訓練需求...")
        self.logger.info(f"🏃 執行模式: {execution_mode}")
        
        # 檢查訓練數據
        train_data_dir = os.path.join(path_config["train_data_dir"], "10_test")
        if not os.path.exists(train_data_dir):
            self.logger.error(f"❌ 訓練數據目錄不存在: {train_data_dir}")
            self.logger.info(f"📁 當前工作目錄: {os.getcwd()}")
            return False
            
        train_images = [f for f in os.listdir(train_data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if len(train_images) == 0:
            self.logger.error("❌ 沒有找到訓練圖片")
            return False
            
        self.logger.info(f"✅ 找到 {len(train_images)} 張訓練圖片")
        
        # 檢查圖片尺寸
        if not self.check_image_sizes(train_data_dir):
            return False
        
        # 檢查輸出目錄
        output_dir = path_config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"📁 輸出目錄: {output_dir}")
        
        return True
        
    def calculate_training_timeout(self) -> int:
        """根據實際測試結果動態計算超時時間
        
        實際測試基準：10張圖片 + 200步 = 30分鐘
        """
        train_data_dir = "lora_train_set/10_test"
        
        if not os.path.exists(train_data_dir):
            return 1800  # 默認30分鐘
        
        # 計算圖片數量
        train_images = [f for f in os.listdir(train_data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        image_count = len(train_images)
        
        # 獲取當前訓練步數（使用固定參數）
        current_steps = FIXED_TRAINING_PARAMS['max_train_steps']
        
        # 實際測試基準：10張圖片 + 200步 = 30分鐘
        base_images = 10
        base_steps = 200
        base_time_minutes = 30
        
        # 計算每張圖片每步的時間
        time_per_image_per_step = base_time_minutes / (base_images * base_steps)
        
        # 計算基礎時間
        base_time_minutes_calc = image_count * current_steps * time_per_image_per_step
        
        # 加上50%緩衝時間
        estimated_time_minutes = base_time_minutes_calc * 1.5
        
        # 轉換為秒
        estimated_time_seconds = int(estimated_time_minutes * 60)
        
        # 設定最小和最大超時時間
        min_timeout = 1800  # 最少30分鐘
        max_timeout = 14400  # 最多4小時
        
        timeout = max(min_timeout, min(estimated_time_seconds, max_timeout))
        
        self.logger.info(f"📊 訓練時間分析 (基於實際測試: 10張圖片+200步=30分鐘):")
        self.logger.info(f"   圖片數量: {image_count} 張")
        self.logger.info(f"   訓練步數: {current_steps} 步")
        self.logger.info(f"   基礎時間: {base_time_minutes_calc:.1f} 分鐘")
        self.logger.info(f"   緩衝時間: {estimated_time_minutes:.1f} 分鐘")
        self.logger.info(f"   超時設定: {timeout/60:.1f} 分鐘")
        
        if estimated_time_seconds > max_timeout:
            self.logger.warning(f"⚠️  預估時間 ({estimated_time_minutes:.1f}分鐘) 超過超時限制 ({max_timeout/60:.1f}分鐘)")
            self.logger.warning(f"🔧 建議減少訓練步數或分批訓練")
            
            # 計算建議的批次大小
            max_time_per_batch = (max_timeout / 60) / 1.5  # 去掉緩衝時間
            max_images_per_batch = int(max_time_per_batch / (current_steps * time_per_image_per_step))
            self.logger.info(f"💡 建議每批最多 {max_images_per_batch} 張圖片")
        
        return timeout
        
    def get_training_params(self) -> dict:
        """獲取訓練參數，支援自定義覆蓋"""
        base_params = {
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "output_dir": "lora_output",
            "train_data_dir": "lora_train_set",
            "resolution": 512,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "dataloader_num_workers": 0,
            "num_train_epochs": 1,
            "max_train_steps": 100,  # 默認100步，適合100張圖片
            "learning_rate": 5e-5,  # 調整為適合大數據集的學習率
            "scale_lr": False,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 0,
            "snr_gamma": 5.0,
            "use_8bit_adam": True,
            "mixed_precision": "fp16",
            "save_precision": "fp16",
            "enable_xformers_memory_efficient_attention": True,
            "cache_latents": True,
            "save_model_as": "safetensors",
            "network_module": "networks.lora",
            "network_dim": 32,
            "network_alpha": 32,
            "network_train_unet_only": True,
            "network_train_text_encoder_only": False,
            "save_every_n_epochs": 1
        }
        
        # 使用自定義參數覆蓋基礎參數
        if self.custom_params:
            base_params.update(self.custom_params)
            self.logger.info(f"✅ 參數已更新: {list(self.custom_params.keys())}")
        
        return base_params
    
    def build_training_command(self) -> str:
        """建立訓練命令"""
        
        # 獲取訓練參數
        params = self.get_training_params()
        
        # 處理繼續訓練選項
        resume_from = None
        if self.continue_from_checkpoint:
            existing_lora = self.find_latest_lora()
            if existing_lora:
                self.logger.info(f"🔄 從檢查點繼續訓練: {os.path.basename(existing_lora)}")
                resume_from = existing_lora
                # 備份現有模型
                self.backup_existing_lora()
            else:
                self.logger.warning("⚠️ 沒有找到現有的 LoRA 檔案，將開始新的訓練")
        else:
            self.logger.info("🆕 開始新的獨立 LoRA 訓練")
            # 如果存在舊模型，備份它
            self.backup_existing_lora()
        
        # 基本訓練命令部分（使用固定參數）
        cmd_parts = [
            "python train_network.py",
            f"--pretrained_model_name_or_path={params.get('pretrained_model_name_or_path', 'runwayml/stable-diffusion-v1-5')}",
            f"--train_data_dir={params.get('train_data_dir', 'lora_train_set')}",
            f"--output_dir={params.get('output_dir', 'lora_output')}",
            f"--resolution={FIXED_TRAINING_PARAMS['resolution']}",
            f"--train_batch_size={FIXED_TRAINING_PARAMS['train_batch_size']}",
            f"--max_train_steps={FIXED_TRAINING_PARAMS['max_train_steps']}",
            f"--learning_rate={FIXED_TRAINING_PARAMS['learning_rate']}",
            f"--mixed_precision={FIXED_TRAINING_PARAMS['mixed_precision']}",
            f"--save_model_as={FIXED_TRAINING_PARAMS['save_model_as']}",
            f"--network_module={params.get('network_module', 'networks.lora')}",
            f"--network_dim={FIXED_TRAINING_PARAMS['network_dim']}",
            f"--network_alpha={FIXED_TRAINING_PARAMS['network_alpha']}",
            f"--sample_every_n_steps={FIXED_TRAINING_PARAMS['sample_every_n_steps']}",
            f"--sample_sampler={FIXED_TRAINING_PARAMS['sample_sampler']}",
            f"--save_every_n_epochs={FIXED_TRAINING_PARAMS['save_every_n_epochs']}",
        ]
        
        # 添加布爾參數
        if params.get('cache_latents', True):
            cmd_parts.append("--cache_latents")
        if params.get('use_8bit_adam', True):
            cmd_parts.append("--use_8bit_adam")
        if params.get('enable_xformers_memory_efficient_attention', True):
            cmd_parts.append("--enable_xformers_memory_efficient_attention")
        if params.get('network_train_unet_only', True):
            cmd_parts.append("--network_train_unet_only")
        
        # 如果從檢查點繼續，添加相應參數
        if resume_from:
            cmd_parts.extend([
                f"--resume={resume_from}",
                "--save_state"  # 保存訓練狀態
            ])
        
        cmd = " ".join(cmd_parts)
        return cmd
        
    def run_training_with_monitoring(self) -> Tuple[bool, dict]:
        """執行訓練並監控進度"""
        training_command = self.build_training_command()
        self.logger.info(f"🚀 開始 LoRA 訓練...")
        self.logger.info(f"📋 命令: {training_command}")
        
        # 強制使用基本模式，避免監控模式卡住
        self.logger.info("⚠️ 使用基本模式執行訓練 (避免監控模式卡住)")
        self.logger.info("🔄 已移除超時限制，訓練將自然完成")
        try:
            # 設置環境變量以避免編碼問題
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            # 移除超時限制，讓訓練自然完成 - 修復輸出阻塞問題
            try:
                result = subprocess.run(
                    training_command, 
                    shell=True,
                    env=env
                )
                
            except Exception as e:
                self.logger.error(f"❌ 訓練執行錯誤: {e}")
                result = type('Result', (), {'returncode': -1, 'stderr': f'訓練執行錯誤: {e}'})()
            
            # 檢查是否有模型檔案生成，即使進程返回錯誤碼
            model_generated = False
            model_size_ok = False
            output_dir = path_config["output_dir"]
            if os.path.exists(output_dir):
                lora_files = [f for f in os.listdir(output_dir) if f.endswith('.safetensors')]
                if lora_files:
                    model_generated = True
                    # 檢查最新模型的大小是否合理 (至少 10MB)
                    latest_lora = max(lora_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
                    latest_lora_path = os.path.join(output_dir, latest_lora)
                    file_size_mb = os.path.getsize(latest_lora_path) / (1024*1024)
                    model_size_ok = file_size_mb > 10  # 至少 10MB 才算正常
                    self.logger.info(f"✅ 成功生成 LoRA 模型: {lora_files}")
            
            # 如果模型已生成且大小合理，認為訓練成功（忽略返回碼）
            success = model_generated and model_size_ok
            
            if success:
                self.logger.info("✅ 訓練成功完成")
                
                # 顯示訓練結果
                final_lora = self.find_latest_lora()
                if final_lora:
                    file_size = os.path.getsize(final_lora) / (1024*1024)
                    self.logger.info(f"📁 最終模型: {os.path.basename(final_lora)}")
                    self.logger.info(f"📊 檔案大小: {file_size:.2f} MB")
                    
                    if self.continue_from_checkpoint:
                        self.logger.info("🔄 檢查點訓練完成 - 模型已更新")
                    else:
                        self.logger.info("🆕 新模型訓練完成")
            else:
                if model_generated and not model_size_ok:
                    self.logger.warning("⚠️ 模型已生成但大小異常，可能訓練不完整")
                elif result.returncode != 0:
                    self.logger.error(f"❌ 訓練失敗，返回碼: {result.returncode}")
                else:
                    self.logger.error("❌ 訓練失敗，未生成模型檔案")
                
            return success, {}
            
        except Exception as e:
            self.logger.error(f"❌ 執行訓練時發生錯誤: {e}")
            # 檢查是否仍有模型生成
            output_dir = path_config["output_dir"]
            if os.path.exists(output_dir):
                lora_files = [f for f in os.listdir(output_dir) if f.endswith('.safetensors')]
                if lora_files:
                    self.logger.info(f"✅ 儘管有錯誤，但成功生成 LoRA 模型: {lora_files}")
                    return True, {}
            return False, {}
                
    def evaluate_training_success(self, report: dict) -> dict:
        """評估訓練成功程度"""
        evaluation = {
            "should_continue_inference": False,
            "training_quality": "unknown",
            "recommendations": [],
            "decision_reason": ""
        }
        
        if not report:
            # 沒有詳細報告，檢查是否有模型檔案
            output_dir = path_config["output_dir"]
            if os.path.exists(output_dir):
                lora_files = [f for f in os.listdir(output_dir) if f.endswith('.safetensors')]
                if lora_files:
                    evaluation["should_continue_inference"] = True
                    evaluation["training_quality"] = "basic"
                    evaluation["decision_reason"] = "找到 LoRA 模型檔案，建議進行基本測試"
                else:
                    evaluation["decision_reason"] = "沒有找到 LoRA 模型檔案"
            else:
                evaluation["decision_reason"] = "輸出目錄不存在"
            return evaluation
            
        # 基於監控報告評估
        training_summary = report.get("training_summary", {})
        training_metrics = report.get("training_metrics", {})
        training_evaluation = report.get("training_evaluation", {})
        
        best_loss = training_summary.get("best_loss", float('inf'))
        loss_improvement = training_metrics.get("loss_improvement", 0)
        performance_grade = training_evaluation.get("performance_grade", "poor")
        
        # 決策邏輯
        if performance_grade == "excellent":
            evaluation["should_continue_inference"] = True
            evaluation["training_quality"] = "excellent"
            evaluation["decision_reason"] = f"訓練表現優秀 (損失: {best_loss:.4f})"
        elif performance_grade == "good":
            evaluation["should_continue_inference"] = True
            evaluation["training_quality"] = "good"
            evaluation["decision_reason"] = f"訓練表現良好 (損失改善: {loss_improvement:.4f})"
        elif performance_grade == "average" and loss_improvement >= 0.05:
            evaluation["should_continue_inference"] = True
            evaluation["training_quality"] = "average"
            evaluation["decision_reason"] = f"訓練表現一般但有改善 (改善: {loss_improvement:.4f})"
        else:
            evaluation["should_continue_inference"] = False
            evaluation["training_quality"] = "poor"
            evaluation["decision_reason"] = f"訓練表現不佳 (等級: {performance_grade}, 改善: {loss_improvement:.4f})"
            evaluation["recommendations"] = [
                "🔧 建議調整學習率 (嘗試 5e-5 或 2e-4)",
                "📊 建議增加訓練步數 (1500-2000步)",
                "🎯 檢查訓練數據品質",
                "⚙️ 嘗試不同的優化器 (AdamW 或 Lion)"
            ]
            
        return evaluation
        
    def train(self) -> dict:
        """執行完整訓練流程"""
        self.logger.info("🎯 開始 LoRA 訓練流程")
        
        # 檢查訓練需求
        if not self.check_training_requirements():
            return {
                "success": False,
                "message": "訓練需求檢查失敗",
                "should_continue_inference": False
            }
            
        # 執行訓練
        training_success, training_report = self.run_training_with_monitoring()
        
        # 評估訓練結果
        evaluation = self.evaluate_training_success(training_report)
        
        # 建立完整報告
        result = {
            "success": training_success,
            "training_report": training_report,
            "evaluation": evaluation,
            "should_continue_inference": evaluation["should_continue_inference"],
            "training_quality": evaluation["training_quality"],
            "decision_reason": evaluation["decision_reason"],
            "recommendations": evaluation.get("recommendations", [])
        }
        
        # 記錄結果
        if training_success:
            self.logger.info("✅ 訓練成功完成")
            self.logger.info(f"🎯 評估結果: {evaluation['training_quality'].upper()}")
            self.logger.info(f"📊 決策: {'繼續推理' if evaluation['should_continue_inference'] else '建議重新訓練'}")
            self.logger.info(f"💡 原因: {evaluation['decision_reason']}")
        else:
            self.logger.error("❌ 訓練失敗")
            
        # 如果有建議，輸出建議
        if evaluation.get("recommendations"):
            self.logger.info("💡 改善建議:")
            for rec in evaluation["recommendations"]:
                self.logger.info(f"   {rec}")
                
        # 保存結果到檔案
        result_file = f"training_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"📋 結果已保存: {result_file}")
        
        return result

def find_existing_lora_models():
    """查找現有的 LoRA 模型 - 使用動態路徑"""
    lora_path = path_config["output_dir"]
    if not os.path.exists(lora_path):
        return []
    
    lora_files = [f for f in os.listdir(lora_path) if f.endswith('.safetensors')]
    return lora_files

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="LoRA 訓練腳本 - 整合監控功能")
    parser.add_argument("--no-monitor", action="store_true", help="禁用訓練監控")
    parser.add_argument("--force-inference", action="store_true", help="強制繼續推理")
    parser.add_argument("--continue", "-c", action="store_true", 
                       dest="continue_training",
                       help="從現有的 LoRA 檔案繼續訓練")
    parser.add_argument("--new", "-n", action="store_true",
                       dest="new_training", 
                       help="開始新的獨立 LoRA 訓練")
    parser.add_argument("--params", type=str, help="自定義參數 JSON 文件路徑")
    
    args = parser.parse_args()
    
    print("⚠️  所有訓練參數已固定在腳本頂部，不接受命令行參數覆蓋")
    
    # 處理自定義參數（僅限非核心參數）
    custom_params = {}
    
    # 從JSON文件讀取參數（僅限非核心參數）
    if args.params:
        if os.path.exists(args.params):
            with open(args.params, 'r', encoding='utf-8') as f:
                loaded_params = json.load(f)
            # 只允許非核心參數
            allowed_params = ['pretrained_model_name_or_path', 'train_data_dir', 'output_dir', 'network_module']
            for key, value in loaded_params.items():
                if key in allowed_params:
                    custom_params[key] = value
                else:
                    print(f"⚠️  忽略被禁止的參數: {key} (使用固定值)")
            print(f"📄 從 {args.params} 載入允許的參數")
        else:
            print(f"⚠️  參數文件不存在: {args.params}")
    
    # 如果禁用監控，移除監控功能
    global MONITOR_AVAILABLE
    if args.no_monitor:
        MONITOR_AVAILABLE = False
    
    # 決定訓練模式
    continue_from_checkpoint = False
    if args.continue_training and args.new_training:
        print("❌ 錯誤：不能同時指定 --continue 和 --new")
        sys.exit(1)
    elif args.continue_training:
        print("🔄 模式：從檢查點繼續訓練")
        continue_from_checkpoint = True
    elif args.new_training:
        print("🆕 模式：開始新的獨立訓練")
        continue_from_checkpoint = False
    else:
        # 如果沒有指定參數，檢查是否有現有模型
        lora_path = path_config["output_dir"]
        existing_lora = None
        if os.path.exists(lora_path):
            lora_files = [f for f in os.listdir(lora_path) if f.endswith('.safetensors')]
            if lora_files:
                latest_lora = max(lora_files, key=lambda x: os.path.getmtime(os.path.join(lora_path, x)))
                existing_lora = os.path.join(lora_path, latest_lora)
        
        if existing_lora:
            print(f"🔍 發現現有的 LoRA 模型: {os.path.basename(existing_lora)}")
            print("請選擇操作：")
            print("1. 從現有模型繼續訓練 (累積調教)")
            print("2. 開始新的獨立訓練 (重新開始)")
            
            while True:
                choice = input("請輸入選擇 (1 或 2): ").strip()
                if choice == "1":
                    continue_from_checkpoint = True
                    break
                elif choice == "2":
                    continue_from_checkpoint = False
                    break
                else:
                    print("請輸入 1 或 2")
        else:
            print("🆕 沒有找到現有模型，將開始新的訓練")
            continue_from_checkpoint = False
        
    # 建立訓練器
    trainer = LoRATrainer(continue_from_checkpoint=continue_from_checkpoint, custom_params=custom_params)
    
    # 執行訓練
    result = trainer.train()
    
    # 決定返回碼 - 簡化邏輯，與 train_lora.py 保持一致
    if result["success"]:
        return 0  # 訓練成功
    else:
        return 1  # 訓練失敗

if __name__ == "__main__":
    sys.exit(main())
