import subprocess
import os
import sys
import warnings
import argparse
import datetime
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

def find_latest_state_dir():
    """找到最新的訓練狀態目錄"""
    lora_path = "lora_output"
    if not os.path.exists(lora_path):
        return None
    
    print(f"🔍 檢查 {lora_path} 目錄內容...")
    
    # 尋找所有狀態目錄（不是 .safetensors 檔案的目錄）
    state_dirs = []
    all_items = []
    
    for item in os.listdir(lora_path):
        item_path = os.path.join(lora_path, item)
        all_items.append(f"  {'[DIR]' if os.path.isdir(item_path) else '[FILE]'} {item}")
        
        if os.path.isdir(item_path) and not item.endswith('.safetensors'):
            state_dirs.append(item_path)
            print(f"  📁 發現狀態目錄: {item}")
    
    # 顯示所有項目
    print("📂 lora_output 目錄內容:")
    for item in all_items:
        print(item)
    
    if not state_dirs:
        print("❌ 沒有找到任何狀態目錄")
        return None
    
    # 找最新的目錄
    latest_state_dir = max(state_dirs, key=os.path.getmtime)
    print(f"✅ 最新狀態目錄: {os.path.basename(latest_state_dir)}")
    return latest_state_dir

def cleanup_old_states(keep_recent=2):
    """清理舊的狀態目錄，只保留最近的幾個"""
    lora_path = "lora_output"
    if not os.path.exists(lora_path):
        return
    
    # 尋找所有狀態目錄
    state_dirs = []
    for item in os.listdir(lora_path):
        item_path = os.path.join(lora_path, item)
        if os.path.isdir(item_path) and not item.endswith('.safetensors'):
            state_dirs.append(item_path)
    
    if len(state_dirs) <= keep_recent:
        return
    
    # 按修改時間排序，刪除舊的
    state_dirs.sort(key=os.path.getmtime, reverse=True)
    old_dirs = state_dirs[keep_recent:]
    
    for old_dir in old_dirs:
        try:
            import shutil
            shutil.rmtree(old_dir)
            print(f"🗑️ 清理舊狀態目錄: {os.path.basename(old_dir)}")
        except Exception as e:
            print(f"⚠️ 無法清理 {old_dir}: {e}")

def train_lora(continue_from_checkpoint=False):
    """執行 LoRA 訓練"""
    
    # 直接指定正確的資料夾名稱格式
    train_dir = "lora_train_set"
    sub_folder = "10_test"
    data_folder = os.path.join(train_dir, sub_folder)

    # 自動處理資料夾改名
    old_folder = os.path.join(train_dir, "test_10")
    if os.path.exists(old_folder) and not os.path.exists(data_folder):
        os.rename(old_folder, data_folder)
        print(f"已自動將 {old_folder} 改名為 {data_folder}")

    # 詳細檢查資料夾結構
    print(f"🔍 檢查資料夾結構...")
    if not os.path.isdir(train_dir):
        print(f"❌ 找不到父資料夾：{train_dir}")
        sys.exit(1)

    if not os.path.isdir(data_folder):
        print(f"❌ 找不到訓練資料夾：{data_folder}")
        print(f"📁 {train_dir} 內容：")
        for item in os.listdir(train_dir):
            print(f"  {item}")
        sys.exit(1)

    # 檢查圖片和文字檔案
    files = os.listdir(data_folder)
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    txt_files = [f for f in files if f.lower().endswith('.txt')]

    print(f"📂 {data_folder} 內容：")
    print(f"  圖片檔案數量：{len(jpg_files)}")
    print(f"  文字檔案數量：{len(txt_files)}")

    if len(jpg_files) == 0:
        print("❌ 沒有找到任何 .jpg 檔案！")
        sys.exit(1)

    # 🎯 關鍵步驟：檢查圖片大小
    if not check_image_size(data_folder, target_size=512):
        print("❌ 沒有任何圖片符合要求！")
        print("💡 請使用 generate_caption_fashionclip.py 預處理圖片")
        sys.exit(1)

    # 自動將 .JPG 副檔名改成 .jpg
    for fname in files:
        if fname.lower().endswith('.jpg') and not fname.endswith('.jpg'):
            src = os.path.join(data_folder, fname)
            dst = os.path.join(data_folder, fname[:-4] + '.jpg')
            os.rename(src, dst)
            print(f"已自動將 {src} 改名為 {dst}")

    # 處理繼續訓練選項
    resume_from = None
    if continue_from_checkpoint:
        # 先查找狀態目錄
        state_dir = find_latest_state_dir()
        existing_lora = find_latest_lora()
        
        if state_dir:
            print(f"🔄 找到訓練狀態目錄: {os.path.basename(state_dir)}")
            resume_from = state_dir
            # 備份現有模型
            backup_existing_lora()
        elif existing_lora:
            print(f"⚠️ 找到 LoRA 檔案但無狀態目錄: {os.path.basename(existing_lora)}")
            print("💡 將使用現有 LoRA 作為基礎繼續訓練")
            # 使用現有 LoRA 檔案作為起點
            resume_from = existing_lora
            backup_existing_lora()
        else:
            print("⚠️ 沒有找到現有的 LoRA 檔案或狀態，將開始新的訓練")
    else:
        print("🆕 開始新的獨立 LoRA 訓練")
        # 如果存在舊模型，備份它
        backup_existing_lora()
        # 清理舊的狀態目錄
        cleanup_old_states()

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
        "--save_model_as=safetensors",
        "--save_state"             # 總是保存狀態以便將來繼續訓練
    ]
    
    # 如果從檢查點繼續，添加相應參數
    if resume_from:
        if resume_from.endswith('.safetensors'):
            # 從 LoRA 檔案繼續訓練，使用 network_weights 參數
            cmd_parts.append(f"--network_weights={resume_from}")
            print(f"🔄 將從 LoRA 檔案繼續訓練: {os.path.basename(resume_from)}")
        else:
            # 從狀態目錄繼續訓練，使用 resume 參數
            cmd_parts.append(f"--resume={resume_from}")
            print(f"🔄 將從狀態目錄繼續訓練: {os.path.basename(resume_from)}")
    else:
        print("🆕 開始全新訓練")
    
    cmd = " ".join(cmd_parts)
    
    print("🚀 開始 LoRA 微調 ...")
    print(f"📋 訓練命令: {cmd}")
    
    # 設定環境變數來抑制警告 - 更完整的設定
    env = os.environ.copy()
    env['DISABLE_XFORMERS'] = '1'
    env['XFORMERS_MORE_DETAILS'] = '0'
    env['PYTHONWARNINGS'] = 'ignore'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['CUDA_LAUNCH_BLOCKING'] = '0'
    env['TRANSFORMERS_VERBOSITY'] = 'error'
    env['DIFFUSERS_VERBOSITY'] = 'error'
    env['TRITON_DISABLE'] = '1'
    env['NO_TRITON'] = '1'
    
    # 直接執行命令，不使用過濾器避免編碼問題
    print("🚀 正在執行訓練...")
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode == 0:
        print("✅ LoRA 訓練完成")
        
        # 顯示訓練結果
        final_lora = find_latest_lora()
        if final_lora:
            file_size = os.path.getsize(final_lora) / (1024*1024)
            print(f"📁 最終模型: {os.path.basename(final_lora)}")
            print(f"📊 檔案大小: {file_size:.2f} MB")
            
            if continue_from_checkpoint:
                print("🔄 檢查點訓練完成 - 模型已更新")
            else:
                print("🆕 新模型訓練完成")
        
        return True
    else:
        print("❌ LoRA 訓練失敗")
        return False

def main():
    """主函數 - 處理命令行參數"""
    parser = argparse.ArgumentParser(description="LoRA 訓練腳本")
    parser.add_argument("--continue", "-c", action="store_true", 
                       dest="continue_training",
                       help="從現有的 LoRA 檔案繼續訓練")
    parser.add_argument("--new", "-n", action="store_true",
                       dest="new_training", 
                       help="開始新的獨立 LoRA 訓練")
    
    args = parser.parse_args()
    
    # 決定訓練模式
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
        # 如果沒有指定參數，詢問用戶
        existing_lora = find_latest_lora()
        if existing_lora:
            print(f"🔍 發現現有的 LoRA 模型: {os.path.basename(existing_lora)}")
            print("請選擇訓練模式：")
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
    
    # 執行訓練
    success = train_lora(continue_from_checkpoint)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()