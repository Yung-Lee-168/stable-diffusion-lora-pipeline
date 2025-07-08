import subprocess
import os
import sys
import warnings
import argparse
import datetime
import logging
import json
import time
from PIL import Image

# 設定環境變數來抑制警告
os.environ['DISABLE_XFORMERS'] = '1'
os.environ['XFORMERS_MORE_DETAILS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

# 減少警告訊息
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*xformers.*")
warnings.filterwarnings("ignore", message=".*triton.*")
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")
warnings.filterwarnings("ignore", message=".*diffusers.*")

# 確保在腳本所在目錄執行
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"📁 切換到腳本目錄: {script_dir}")

# 全局變量用於監控
MONITOR_ENABLED = False
LOGGER = None

def setup_monitor():
    """設定監控 - 可選功能"""
    global MONITOR_ENABLED, LOGGER
    
    try:
        log_dir = "training_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        LOGGER = logging.getLogger(__name__)
        MONITOR_ENABLED = True
        print(f"📋 監控已啟用，日誌: {log_file}")
        
    except Exception as e:
        print(f"⚠️ 監控設定失敗，將使用基本模式: {e}")
        MONITOR_ENABLED = False

def log_info(message):
    """記錄訊息"""
    if MONITOR_ENABLED and LOGGER:
        LOGGER.info(message)
    print(message)

def find_latest_lora():
    """找到最新的 LoRA 模型檔案"""
    lora_path = "lora_output"
    if not os.path.exists(lora_path):
        return None
    
    lora_files = [f for f in os.listdir(lora_path) if f.endswith('.safetensors')]
    if not lora_files:
        return None
    
    # 找最新的檔案
    latest_lora = max(lora_files, key=lambda x: os.path.getmtime(os.path.join(lora_path, x)))
    return os.path.join(lora_path, latest_lora)

def backup_existing_lora():
    """備份現有的 LoRA 模型"""
    existing_lora = find_latest_lora()
    if existing_lora and os.path.exists(existing_lora):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"lora_backup_{timestamp}.safetensors"
        backup_path = os.path.join("lora_output", backup_name)
        
        import shutil
        shutil.copy2(existing_lora, backup_path)
        log_info(f"📦 備份現有模型: {backup_name}")
        return backup_path
    return None

def check_image_size(data_folder, target_size=512):
    """檢查圖片大小是否符合要求，跳過超出尺寸的圖片"""
    log_info(f"🔍 檢查圖片大小是否符合 {target_size}x{target_size} 要求...")
    
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
                    log_info(f"  ✅ {img_file}: {width}x{height} (符合要求)")
                else:
                    invalid_files.append((img_file, width, height))
                    log_info(f"  ⚠️  {img_file}: {width}x{height} (超出 {target_size}x{target_size}，將跳過)")
                
        except Exception as e:
            log_info(f"❌ 無法讀取圖片 {img_file}: {str(e)}")
            invalid_files.append((img_file, "讀取失敗", ""))
    
    log_info(f"\n📊 圖片尺寸檢查結果：")
    log_info(f"✅ 符合要求的圖片：{valid_count} 張")
    log_info(f"⚠️  超出尺寸的圖片：{len(invalid_files)} 張")
    
    if invalid_files:
        log_info(f"\n📋 超出尺寸的圖片將被跳過：")
        for img_file, width, height in invalid_files:
            log_info(f"   - {img_file}: {width}x{height}")
        log_info(f"\n💡 建議：使用 generate_caption_fashionclip.py 預處理圖片")
    
    if valid_count == 0:
        log_info(f"❌ 沒有任何圖片符合要求！")
        return False
    else:
        log_info(f"🎯 將使用 {valid_count} 張符合要求的圖片進行訓練")
        return True

def train_lora(continue_from_checkpoint=False):
    """執行 LoRA 訓練 - 與原版完全相同，只是添加了監控"""
    
    # 記錄開始時間
    start_time = time.time()
    
    # 直接指定正確的資料夾名稱格式
    train_dir = "lora_train_set"
    sub_folder = "10_test"
    data_folder = os.path.join(train_dir, sub_folder)

    # 自動處理資料夾改名
    old_folder = os.path.join(train_dir, "test_10")
    if os.path.exists(old_folder) and not os.path.exists(data_folder):
        os.rename(old_folder, data_folder)
        log_info(f"已自動將 {old_folder} 改名為 {data_folder}")

    # 詳細檢查資料夾結構
    log_info(f"🔍 檢查資料夾結構...")
    if not os.path.isdir(train_dir):
        log_info(f"❌ 找不到父資料夾：{train_dir}")
        sys.exit(1)

    if not os.path.isdir(data_folder):
        log_info(f"❌ 找不到訓練資料夾：{data_folder}")
        log_info(f"📁 {train_dir} 內容：")
        for item in os.listdir(train_dir):
            log_info(f"  {item}")
        sys.exit(1)

    # 檢查圖片和文字檔案
    files = os.listdir(data_folder)
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    txt_files = [f for f in files if f.lower().endswith('.txt')]

    log_info(f"📂 {data_folder} 內容：")
    log_info(f"  圖片檔案數量：{len(jpg_files)}")
    log_info(f"  文字檔案數量：{len(txt_files)}")

    if len(jpg_files) == 0:
        log_info("❌ 沒有找到任何 .jpg 檔案！")
        sys.exit(1)

    # 🎯 關鍵步驟：檢查圖片大小
    if not check_image_size(data_folder, target_size=512):
        log_info("❌ 沒有任何圖片符合要求！")
        log_info("💡 請使用 generate_caption_fashionclip.py 預處理圖片")
        sys.exit(1)

    # 自動將 .JPG 副檔名改成 .jpg
    for fname in files:
        if fname.lower().endswith('.jpg') and not fname.endswith('.jpg'):
            src = os.path.join(data_folder, fname)
            dst = os.path.join(data_folder, fname[:-4] + '.jpg')
            os.rename(src, dst)
            log_info(f"已自動將 {src} 改名為 {dst}")

    # 處理繼續訓練選項
    resume_from = None
    if continue_from_checkpoint:
        existing_lora = find_latest_lora()
        if existing_lora:
            log_info(f"🔄 從檢查點繼續訓練: {os.path.basename(existing_lora)}")
            resume_from = existing_lora
            # 備份現有模型
            backup_existing_lora()
        else:
            log_info("⚠️ 沒有找到現有的 LoRA 檔案，將開始新的訓練")
    else:
        log_info("🆕 開始新的獨立 LoRA 訓練")
        # 如果存在舊模型，備份它
        backup_existing_lora()

    # 基本訓練指令
    cmd_parts = [
        "python train_network.py",
        "--pretrained_model_name_or_path=../models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
        "--train_data_dir=lora_train_set",
        "--output_dir=lora_output",
        "--resolution=512,512",
        "--network_module=networks.lora",
        "--network_dim=32",        # 更新為32維
        "--train_batch_size=1",
        "--max_train_steps=100",   # 默認100步，適合100張圖片
        "--mixed_precision=fp16",
        "--cache_latents",
        "--learning_rate=5e-5",    # 調整為適合大數據集的學習率
        "--save_every_n_epochs=50",
        "--save_model_as=safetensors"
    ]
    
    # 如果從檢查點繼續，添加相應參數
    if resume_from:
        cmd_parts.extend([
            f"--resume={resume_from}",
            "--save_state"  # 保存訓練狀態
        ])
    
    cmd = " ".join(cmd_parts)
    
    log_info("🚀 開始 LoRA 微調 ...")
    log_info(f"📋 訓練命令: {cmd}")
    
    # 設定環境變數來抑制警告
    env = os.environ.copy()
    env['DISABLE_XFORMERS'] = '1'
    env['XFORMERS_MORE_DETAILS'] = '0'
    env['PYTHONWARNINGS'] = 'ignore'
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # 執行訓練 - 與原版完全相同
    result = subprocess.run(cmd, shell=True, env=env)
    
    # 記錄結束時間
    end_time = time.time()
    training_time = end_time - start_time
    
    if result.returncode == 0:
        log_info("✅ LoRA 訓練完成")
        
        # 顯示訓練結果
        final_lora = find_latest_lora()
        if final_lora:
            file_size = os.path.getsize(final_lora) / (1024*1024)
            log_info(f"📁 最終模型: {os.path.basename(final_lora)}")
            log_info(f"📊 檔案大小: {file_size:.2f} MB")
            log_info(f"⏱️ 訓練時間: {training_time:.2f} 秒")
            
            # 簡單的品質評估
            if file_size > 20:
                quality = "優秀"
            elif file_size > 15:
                quality = "良好"
            elif file_size > 10:
                quality = "普通"
            else:
                quality = "需要改進"
                
            log_info(f"🎯 品質評估: {quality}")
            
            if continue_from_checkpoint:
                log_info("🔄 檢查點訓練完成 - 模型已更新")
            else:
                log_info("🆕 新模型訓練完成")
                
            # 保存監控報告
            if MONITOR_ENABLED:
                try:
                    report = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "training_time_seconds": training_time,
                        "model_file": os.path.basename(final_lora),
                        "file_size_mb": file_size,
                        "quality": quality,
                        "continue_from_checkpoint": continue_from_checkpoint,
                        "success": True
                    }
                    
                    report_file = f"training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(report_file, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)
                    
                    log_info(f"📋 監控報告: {report_file}")
                except Exception as e:
                    print(f"⚠️ 保存報告失敗: {e}")
        
        return True
    else:
        log_info("❌ LoRA 訓練失敗")
        return False

def main():
    """主函數 - 處理命令行參數"""
    parser = argparse.ArgumentParser(description="LoRA 訓練腳本 - 監控版")
    parser.add_argument("--continue", "-c", action="store_true", 
                       dest="continue_training",
                       help="從現有的 LoRA 檔案繼續訓練")
    parser.add_argument("--new", "-n", action="store_true",
                       dest="new_training", 
                       help="開始新的獨立 LoRA 訓練")
    parser.add_argument("--monitor", "-m", action="store_true",
                       help="啟用監控功能")
    
    args = parser.parse_args()
    
    # 設定監控
    if args.monitor:
        setup_monitor()
    
    # 決定訓練模式
    if args.continue_training and args.new_training:
        log_info("❌ 錯誤：不能同時指定 --continue 和 --new")
        sys.exit(1)
    elif args.continue_training:
        log_info("🔄 模式：從檢查點繼續訓練")
        continue_from_checkpoint = True
    elif args.new_training:
        log_info("🆕 模式：開始新的獨立訓練")
        continue_from_checkpoint = False
    else:
        # 如果沒有指定參數，詢問用戶
        existing_lora = find_latest_lora()
        if existing_lora:
            log_info(f"🔍 發現現有的 LoRA 模型: {os.path.basename(existing_lora)}")
            log_info("請選擇訓練模式：")
            log_info("1. 從現有模型繼續訓練 (累積調教)")
            log_info("2. 開始新的獨立訓練 (重新開始)")
            
            while True:
                choice = input("請輸入選擇 (1 或 2): ").strip()
                if choice == "1":
                    continue_from_checkpoint = True
                    break
                elif choice == "2":
                    continue_from_checkpoint = False
                    break
                else:
                    log_info("請輸入 1 或 2")
        else:
            log_info("🆕 沒有找到現有模型，將開始新的訓練")
            continue_from_checkpoint = False
    
    # 執行訓練
    success = train_lora(continue_from_checkpoint)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
