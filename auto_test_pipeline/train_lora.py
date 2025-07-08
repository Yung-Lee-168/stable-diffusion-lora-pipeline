import subprocess
import os
import sys
import warnings
import argparse
import datetime
import json
from PIL import Image

# Set environment variables to suppress warnings
os.environ['DISABLE_XFORMERS'] = '1'
os.environ['XFORMERS_MORE_DETAILS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Reduce warning messages
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*xformers.*")
warnings.filterwarnings("ignore", message=".*triton.*")
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")
warnings.filterwarnings("ignore", message=".*diffusers.*")

# Ensure execution in script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"📁 Switched to script directory: {script_dir}")

def check_dependencies():
    """Check and report required and optional dependencies"""
    required_deps = ['torch', 'PIL', 'numpy']
    optional_deps = ['tensorboard', 'matplotlib']
    
    missing_required = []
    missing_optional = []
    
    for dep in required_deps:
        try:
            __import__(dep)
            print(f"✅ Required dependency installed: {dep}")
        except ImportError:
            missing_required.append(dep)
            print(f"❌ Missing required dependency: {dep}")
    
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"✅ Optional dependency installed: {dep}")
        except ImportError:
            missing_optional.append(dep)
            print(f"⚠️ Missing optional dependency: {dep} (will use alternative features)")
    
    if missing_required:
        print(f"\n❌ Cannot continue because of missing required dependencies: {missing_required}")
        print(f"💡 You can install them with: pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n💡 Optional features explanation:")
        if 'tensorboard' in missing_optional:
            print(f"   - tensorboard: will use built-in loss tracking instead of TensorBoard")
        if 'matplotlib' in missing_optional:
            print(f"   - matplotlib: will skip PNG chart generation, only generate JSON reports")
        print(f"   You can install optional dependencies with: pip install {' '.join(missing_optional)}")
    
    return True

def find_latest_lora():
    """Find the latest LoRA model file"""
    # 🔧 FIX: Search in lora_output directory
    lora_path = r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\lora_output"
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
        # 🔧 FIX: 備份也放在 lora_output 目錄
        backup_path = os.path.join(r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\lora_output", backup_name)
        
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
    # 🔧 FIX: 改為從 lora_output 目錄查找狀態
    lora_path = r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\lora_output"
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
    # 🔧 FIX: 改為從 lora_output 目錄清理
    lora_path = r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\lora_output"
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

def train_lora(continue_from_checkpoint=False, custom_steps=None):
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
    current_step = 0
    
    if continue_from_checkpoint:
        # 先查找狀態目錄
        state_dir = find_latest_state_dir()
        existing_lora = find_latest_lora()
        
        if state_dir:
            print(f"🔄 找到訓練狀態目錄: {os.path.basename(state_dir)}")
            current_step = get_current_training_step(state_dir)
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
    
    # 🎯 智能計算最大訓練步數
    if custom_steps is not None:
        # 使用用戶指定的步數
        additional_steps = custom_steps
        print(f"📊 使用用戶指定步數: {custom_steps}")
    else:
        # 交互式詢問步數
        if continue_from_checkpoint:
            print(f"\n🔢 請設定要繼續訓練的步數:")
            print(f"   當前已完成: {current_step} 步")
            default_steps = 100
        else:
            print(f"\n🔢 請設定新訓練的總步數:")
            default_steps = 100
        
        while True:
            try:
                user_input = input(f"請輸入步數 (默認 {default_steps}): ").strip()
                if user_input == "":
                    additional_steps = default_steps
                    break
                else:
                    additional_steps = int(user_input)
                    if additional_steps > 0:
                        break
                    else:
                        print("❌ 步數必須大於0，請重新輸入")
            except ValueError:
                print("❌ 請輸入有效的數字")
    
    max_train_steps = calculate_smart_max_steps(current_step, additional_steps=additional_steps)

    # 基本訓練指令 - 使用當前Python解釋器
    python_executable = sys.executable  # 獲取當前Python解釋器路徑
    print(f"🐍 使用Python解釋器: {python_executable}")
    
    # 🔧 FIX: 確保正確的輸出目錄存在
    # LoRA模型和狀態輸出到 lora_output
    # 報告和日誌輸出到 training_logs
    lora_output_dir = r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\lora_output"
    training_logs_dir = r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\training_logs"
    logs_dir = os.path.join(training_logs_dir, "logs")
    
    os.makedirs(lora_output_dir, exist_ok=True)
    os.makedirs(training_logs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"📁 LoRA模型輸出目錄: {lora_output_dir}")
    print(f"📁 報告和日誌目錄: {training_logs_dir}")
    
    cmd_parts = [
        f'"{python_executable}" train_network.py',  # 使用當前Python環境
        "--pretrained_model_name_or_path=../models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
        "--train_data_dir=lora_train_set",
        f"--output_dir={lora_output_dir}",  # 🔧 FIX: LoRA模型輸出到 lora_output
        f"--logging_dir={logs_dir}",       # 🔧 FIX: 日誌輸出到 training_logs/logs
        "--resolution=512,512",
        "--network_module=networks.lora",
        "--network_dim=32",        # 更新為32維
        "--train_batch_size=1",
        f"--max_train_steps={max_train_steps}",   # 智能調整的最大訓練步數
        "--mixed_precision=fp16",
        "--cache_latents",
        "--learning_rate=5e-5",    # 調整為適合大數據集的學習率
        "--save_every_n_epochs=50",
        "--save_model_as=safetensors",
        "--save_state",            # 總是保存狀態以便將來繼續訓練
        "--log_with=tensorboard",  # 📊 使用TensorBoard記錄訓練過程（可選）
        "--gradient_accumulation_steps=1",  # 🎯 明確設定累積步數，確保步數控制精確
        "--save_precision=fp16",   # 🔧 FIX: 確保保存精度一致
        "--log_tracker_name=lora_training",  # 🔧 FIX: 設定追蹤器名稱
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
    env['PYTORCH_DISABLE_XFORMERS'] = '1'  # 🔧 FIX: 額外的 xformers 抑制
    env['FORCE_XFORMERS'] = '0'           # 🔧 FIX: 強制禁用 xformers
    env['XFORMERS_DISABLED'] = '1'        # 🔧 FIX: 明確禁用 xformers
    
    # 直接執行命令，使用內建監控
    print("🚀 正在執行訓練...")
    
    # 🔧 FIX: 使用內建loss監控替代純TensorBoard依賴
    success = monitor_training_process(cmd, env, training_logs_dir)
    
    if success:
        print("✅ LoRA 訓練完成")
        
        # 🎯 詳細顯示所有訓練輸出文件
        print("\n" + "="*60)
        print("📁 LoRA 訓練完成後的輸出文件詳細說明")
        print("="*60)
        
        # 1. 主要LoRA模型文件
        final_lora = find_latest_lora()
        if final_lora:
            file_size = os.path.getsize(final_lora) / (1024*1024)
            print(f"\n🎯 主要LoRA模型文件:")
            print(f"   📄 文件名: {os.path.basename(final_lora)}")
            print(f"   📂 位置: {os.path.abspath(final_lora)}")
            print(f"   📊 大小: {file_size:.2f} MB")
            print(f"   💡 說明: 這是訓練完成的LoRA權重文件，可直接在WebUI中使用")
        
        # 2. 訓練狀態目錄
        state_dir = find_latest_state_dir()
        if state_dir:
            print(f"\n🔄 訓練狀態目錄:")
            print(f"   📂 位置: {os.path.abspath(state_dir)}")
            print(f"   💡 說明: 包含完整的訓練狀態，可用於繼續訓練")
            
            # 列出狀態目錄內容
            if os.path.exists(state_dir):
                state_files = os.listdir(state_dir)
                print(f"   📋 包含文件: {', '.join(state_files[:5])}")
                if len(state_files) > 5:
                    print(f"   　　　　　　（還有 {len(state_files)-5} 個其他文件...）")
        
        # 3. 訓練日誌
        log_dir = r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\training_logs\logs"
        if os.path.exists(log_dir):
            print(f"\n📊 訓練日誌:")
            print(f"   📂 位置: {os.path.abspath(log_dir)}")
            print(f"   💡 說明: 包含每步的loss記錄和TensorBoard日誌")
            
            # 檢查TensorBoard事件文件
            tb_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
            if tb_files:
                print(f"   📈 TensorBoard文件: {len(tb_files)} 個")
                print(f"   🎯 查看方法: 在training_logs/logs目錄執行 'tensorboard --logdir .'")
        
        # 4. 備份文件
        training_logs_dir = r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\training_logs"
        backup_files = [f for f in os.listdir(training_logs_dir) if f.startswith('lora_backup_') and f.endswith('.safetensors')]
        if backup_files:
            print(f"\n🗄️ 備份文件:")
            print(f"   📂 位置: {os.path.abspath(training_logs_dir)}")
            print(f"   📄 備份文件: {len(backup_files)} 個")
            print(f"   💡 說明: 訓練前的舊模型備份")
        
        # 5. 輸出文件使用說明
        print(f"\n🎯 如何使用這些文件:")
        print(f"   1️⃣ 主模型文件（{os.path.basename(final_lora) if final_lora else 'last.safetensors'}）")
        print(f"      → 複製到 WebUI 的 models/Lora/ 目錄")
        print(f"      → 在 WebUI 中可直接選擇使用")
        print(f"   2️⃣ 狀態目錄")
        print(f"      → 用於繼續訓練：python train_lora.py --continue")
        print(f"   3️⃣ TensorBoard日誌")
        print(f"      → 查看訓練曲線：cd training_logs/logs && tensorboard --logdir .")
        
        print("="*60)
        
        # 🎯 生成loss訓練報告 - 優先使用內建日誌
        print(f"\n📊 正在生成訓練報告...")
        
        # 首先嘗試從內建日誌生成報告
        builtin_log_file = os.path.join(training_logs_dir, "training_loss_log.txt")
        if os.path.exists(builtin_log_file):
            print(f"✅ 使用內建loss日誌生成報告")
            report_success = generate_loss_report_from_log(builtin_log_file, training_logs_dir)
        else:
            # 如果內建日誌不存在，嘗試使用TensorBoard日誌
            print(f"⚠️ 內建日誌不存在，嘗試使用TensorBoard日誌")
            report_success = generate_loss_report(
                log_dir=r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\training_logs\logs", 
                output_dir=r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\training_logs"
            )
        
        if report_success:
            print(f"✅ 訓練報告生成完成")
            print(f"   📄 JSON報告: lora_training_report_*.json")
            print(f"   📈 PNG圖表: lora_training_curves_*.png")
        else:
            print(f"⚠️ 報告生成失敗，但訓練已完成")
        
        if continue_from_checkpoint:
            print("🔄 檢查點訓練完成 - 模型已更新")
        else:
            print("🆕 新模型訓練完成")
        
        return True
    else:
        print("❌ LoRA 訓練失敗")
        return False

def get_current_training_step(state_dir):
    """從訓練狀態中獲取當前步數"""
    try:
        import json
        state_file = os.path.join(state_dir, "train_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                return state.get('current_step', 0)
    except Exception as e:
        print(f"⚠️ 無法讀取訓練狀態: {e}")
    return 0

def calculate_smart_max_steps(current_step, additional_steps=100):
    """智能計算最大訓練步數"""
    if current_step == 0:
        # 新訓練，使用默認步數
        return additional_steps
    else:
        # 繼續訓練，在當前步數基礎上增加
        new_max_steps = current_step + additional_steps
        print(f"📊 當前已完成步數: {current_step}")
        print(f"📊 計劃增加步數: {additional_steps}")
        print(f"📊 新的最大步數: {new_max_steps}")
        return new_max_steps

def generate_loss_report(log_dir=r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\training_logs\logs", 
                        output_dir=r"E:\Yung_Folder\Project\stable-diffusion-webui\auto_test_pipeline\training_logs"):
    """
    Generate loss report in both JSON and PNG formats with English-only content.
    
    Features:
    - All JSON keys, values, and descriptions are in English
    - PNG chart titles, labels, and annotations are in English  
    - Metrics are organized by type (loss_data, learning_rate_data, other_metrics)
    - Clean JSON keys with normalized naming (no special characters)
    - Comprehensive metadata including descriptions and statistics
    - Professional chart formatting with main title and grid
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_dir: Directory to save JSON and PNG reports
        
    Returns:
        bool: True if successful, False if failed
    """
    print(f"\n📊 正在生成loss訓練報告...")
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Check log directory
        if not os.path.exists(log_dir):
            print(f"❌ 日誌目錄不存在: {log_dir}")
            return False
        
        # Load TensorBoard data
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        # Get all available scalar tags
        scalar_tags = ea.Tags()['scalars']
        print(f"📋 找到數據標籤: {scalar_tags}")
        
        # Prepare report data with English-only content
        report_data = {
            "training_info": {
                "timestamp": datetime.datetime.now().isoformat(),
                "log_directory": os.path.abspath(log_dir),
                "total_metrics": len(scalar_tags),
                "description": "LoRA Training Report - Generated automatically after training completion"
            },
            "loss_data": {},
            "learning_rate_data": {},
            "other_metrics": {}
        }
        
        # Extract all loss-related data
        loss_tags = [tag for tag in scalar_tags if 'loss' in tag.lower()]
        lr_tags = [tag for tag in scalar_tags if 'lr' in tag.lower()]
        other_tags = [tag for tag in scalar_tags if 'loss' not in tag.lower() and 'lr' not in tag.lower()]
        
        print(f"📈 Loss指標: {len(loss_tags)} 個")
        print(f"📉 學習率指標: {len(lr_tags)} 個") 
        print(f"📊 其他指標: {len(other_tags)} 個")
        
        # Process loss data with English keys
        for tag in loss_tags:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            
            # Clean tag name for JSON key
            clean_tag = tag.replace('/', '_').replace(' ', '_').replace('-', '_').lower()
            
            report_data["loss_data"][clean_tag] = {
                "metric_name": tag,
                "steps": steps,
                "values": values,
                "total_points": len(steps),
                "min_value": min(values) if values else 0,
                "max_value": max(values) if values else 0,
                "final_value": values[-1] if values else 0,
                "step_range": [min(steps), max(steps)] if steps else [0, 0],
                "description": f"Loss metric tracking for {tag}"
            }
        
        # Process learning rate data with English keys
        for tag in lr_tags:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            
            # Clean tag name for JSON key
            clean_tag = tag.replace('/', '_').replace(' ', '_').replace('-', '_').lower()
            
            report_data["learning_rate_data"][clean_tag] = {
                "metric_name": tag,
                "steps": steps,
                "values": values,
                "total_points": len(steps),
                "initial_lr": values[0] if values else 0,
                "final_lr": values[-1] if values else 0,
                "description": f"Learning rate schedule for {tag}"
            }
        
        # Process other metrics with English keys
        for tag in other_tags:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            
            # Clean tag name for JSON key
            clean_tag = tag.replace('/', '_').replace(' ', '_').replace('-', '_').lower()
            
            report_data["other_metrics"][clean_tag] = {
                "metric_name": tag,
                "steps": steps,
                "values": values,
                "total_points": len(steps),
                "description": f"Training metric for {tag}"
            }
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save JSON report with English content
        json_filename = f"lora_training_report_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON報告已保存: {json_filename}")
        
        # 2. Generate PNG chart with English labels
        png_filename = f"lora_training_curves_{timestamp}.png"
        png_path = os.path.join(output_dir, png_filename)
        
        # Set English font (no Chinese fonts)
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Calculate number of subplots
        total_plots = len(loss_tags) + (1 if lr_tags else 0)
        if total_plots == 0:
            print(f"⚠️ 沒有找到可繪製的數據")
            return True
        
        # Create subplots
        if total_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(12, 6))
            axes = [axes]
        elif total_plots <= 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        elif total_plots <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        else:
            rows = (total_plots + 2) // 3
            fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))
            axes = axes.flatten() if rows > 1 else axes
        
        plot_idx = 0
        
        # Plot loss curves with English labels
        for tag in loss_tags:
            if plot_idx >= len(axes):
                break
                
            clean_tag = tag.replace('/', '_').replace(' ', '_').replace('-', '_').lower()
            data = report_data["loss_data"][clean_tag]
            
            axes[plot_idx].plot(data["steps"], data["values"], 'b-', linewidth=2, label=tag)
            axes[plot_idx].set_title(f'Loss Curve: {tag}', fontsize=12, fontweight='bold')
            axes[plot_idx].set_xlabel('Training Steps')
            axes[plot_idx].set_ylabel('Loss Value')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].legend()
            
            # Add statistics in English
            final_loss = data["final_value"]
            min_loss = data["min_value"]
            axes[plot_idx].text(0.02, 0.98, f'Final: {final_loss:.6f}\nMin: {min_loss:.6f}', 
                               transform=axes[plot_idx].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            plot_idx += 1
        
        # Plot learning rate curves with English labels (if available)
        if lr_tags and plot_idx < len(axes):
            # Plot all learning rate curves on the same chart
            for tag in lr_tags:
                clean_tag = tag.replace('/', '_').replace(' ', '_').replace('-', '_').lower()
                data = report_data["learning_rate_data"][clean_tag]
                axes[plot_idx].plot(data["steps"], data["values"], linewidth=2, label=tag)
            
            axes[plot_idx].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            axes[plot_idx].set_xlabel('Training Steps')
            axes[plot_idx].set_ylabel('Learning Rate')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].legend()
            axes[plot_idx].set_yscale('log')  # Use logarithmic scale
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        # Add main title
        fig.suptitle('LoRA Training Progress Report', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ PNG圖表已保存: {png_filename}")
        
        # 3. Generate summary statistics with English content
        print(f"\n📊 訓練統計摘要:")
        
        if report_data['loss_data']:
            max_steps = max([max(data['steps']) for data in report_data['loss_data'].values()])
            print(f"   總訓練步數: {max_steps}")
            
            for key, data in report_data["loss_data"].items():
                metric_name = data['metric_name']
                print(f"   {metric_name}:")
                print(f"     最終值: {data['final_value']:.6f}")
                print(f"     最小值: {data['min_value']:.6f}")
                print(f"     數據點: {data['total_points']} 個")
        else:
            print(f"   總訓練步數: 0")
        
        return True
        
    except ImportError:
        print(f"⚠️ 缺少必要的庫，無法生成圖表")
        print(f"💡 請安裝: pip install tensorboard matplotlib")
        return False
    except Exception as e:
        print(f"❌ 生成報告時出錯: {e}")
        return False

def create_loss_tracker(output_dir):
    """創建內建的loss追蹤器，不依賴TensorBoard"""
    tracker_file = os.path.join(output_dir, "training_loss_log.txt")
    
    # 創建loss追蹤日誌文件
    with open(tracker_file, 'w', encoding='utf-8') as f:
        f.write("# LoRA Training Loss Log\n")
        f.write("# Format: step,epoch,loss,learning_rate,timestamp\n")
        f.write("step,epoch,loss,learning_rate,timestamp\n")
    
    return tracker_file

def monitor_training_process(cmd, env, output_dir):
    """監控訓練過程並記錄loss數據"""
    import subprocess
    import re
    import time
    
    # 創建loss追蹤器
    loss_tracker_file = create_loss_tracker(output_dir)
    
    print("🚀 正在執行訓練並監控loss...")
    
    # 創建進程來監控輸出
    process = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding='utf-8',  # 🔧 FIX: 明確指定 UTF-8 編碼
        errors='replace',  # 🔧 FIX: 遇到無法解碼的字符時替換而非報錯
        bufsize=1
    )
    
    # 監控輸出並提取loss數據 - 支援多種主流格式
    loss_patterns = [
        # 標準格式: Step: 100, Loss: 0.5
        re.compile(r'(?:step|Step):\s*(\d+).*?(?:loss|Loss):\s*([\d\.e\-\+]+)', re.IGNORECASE),
        # 括號格式: [100/1000] loss: 0.5
        re.compile(r'\[(\d+)/\d+\].*?(?:loss|Loss):\s*([\d\.e\-\+]+)', re.IGNORECASE),
        # 簡短格式: 100 loss 0.5
        re.compile(r'(\d+)\s+(?:loss|Loss)\s+([\d\.e\-\+]+)', re.IGNORECASE),
        # 進度條格式: Step 100/1000 Loss: 0.5
        re.compile(r'(?:step|Step)\s+(\d+)/\d+.*?(?:loss|Loss):\s*([\d\.e\-\+]+)', re.IGNORECASE),
        # 斜線格式: 100/1000 loss=0.5
        re.compile(r'(\d+)/\d+.*?(?:loss|Loss)=\s*([\d\.e\-\+]+)', re.IGNORECASE),
        # 時間戳格式: [2024-01-01 10:00:00] Step: 100 Loss: 0.5
        re.compile(r'\[.*?\].*?(?:step|Step):\s*(\d+).*?(?:loss|Loss):\s*([\d\.e\-\+]+)', re.IGNORECASE)
    ]
    epoch_pattern = re.compile(r'(?:epoch|Epoch):\s*(\d+)', re.IGNORECASE)
    lr_pattern = re.compile(r'(?:lr|learning.rate):\s*([\d\.e\-\+]+)', re.IGNORECASE)
    
    current_epoch = 0
    current_lr = "unknown"
    
    try:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            
            if output:
                line = output.strip()
                print(line)  # 顯示原始輸出
                
                # 檢查是否包含epoch信息
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    current_epoch = epoch_match.group(1)
                
                # 檢查是否包含學習率信息
                lr_match = lr_pattern.search(line)
                if lr_match:
                    current_lr = lr_match.group(1)
                
                # 檢查是否包含loss信息 - 嘗試所有格式
                loss_match = None
                step = None
                loss = None
                for pattern in loss_patterns:
                    loss_match = pattern.search(line)
                    if loss_match:
                        step = loss_match.group(1)
                        loss = loss_match.group(2)
                        break
                
                if loss_match:
                    timestamp = datetime.datetime.now().isoformat()
                    
                    # 記錄到loss追蹤文件
                    with open(loss_tracker_file, 'a', encoding='utf-8') as f:
                        f.write(f"{step},{current_epoch},{loss},{current_lr},{timestamp}\n")
                    
                    print(f"📊 記錄Loss: Step {step}, Loss {loss}")
        
        return_code = process.poll()
        return return_code == 0
        
    except KeyboardInterrupt:
        print("⚠️ 訓練被用戶中斷")
        process.terminate()
        return False
    except Exception as e:
        print(f"❌ 監控訓練過程時出錯: {e}")
        return False

def generate_loss_report_from_log(log_file, output_dir):
    """從內建日誌文件生成loss報告"""
    print(f"\n📊 從內建日誌生成訓練報告...")
    
    try:
        if not os.path.exists(log_file):
            print(f"❌ 找不到訓練日誌文件: {log_file}")
            return False
        
        # 讀取loss數據
        steps = []
        epochs = []
        losses = []
        learning_rates = []
        timestamps = []
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 跳過標題行
        data_lines = [line for line in lines if not line.startswith('#') and line.strip() and 'step,epoch' not in line]
        
        for line in data_lines:
            try:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    step = int(parts[0])
                    epoch = parts[1]
                    loss = float(parts[2])
                    lr = parts[3]
                    timestamp = parts[4]
                    
                    steps.append(step)
                    epochs.append(epoch)
                    losses.append(loss)
                    learning_rates.append(lr)
                    timestamps.append(timestamp)
            except (ValueError, IndexError) as e:
                print(f"⚠️ 跳過無效行: {line.strip()}")
                continue
        
        if not steps:
            print(f"❌ 沒有找到有效的loss數據")
            return False
        
        print(f"✅ 成功讀取 {len(steps)} 個訓練步驟的數據")
        
        # 生成時間戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 生成JSON報告
        report_data = {
            "training_info": {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_steps": len(steps),
                "final_step": max(steps) if steps else 0,
                "final_loss": losses[-1] if losses else 0,
                "best_loss": min(losses) if losses else 0,
                "description": "LoRA Training Report - Generated from built-in loss tracking"
            },
            "loss_data": {
                "training_loss": {
                    "metric_name": "training_loss",
                    "steps": steps,
                    "values": losses,
                    "total_points": len(steps),
                    "min_value": min(losses) if losses else 0,
                    "max_value": max(losses) if losses else 0,
                    "final_value": losses[-1] if losses else 0,
                    "step_range": [min(steps), max(steps)] if steps else [0, 0],
                    "description": "Training loss tracked during LoRA training"
                }
            },
            "raw_data": {
                "steps": steps,
                "epochs": epochs,
                "losses": losses,
                "learning_rates": learning_rates,
                "timestamps": timestamps
            }
        }
        
        json_filename = f"lora_training_report_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON報告已保存: {json_filename}")
        
        # 2. 生成PNG圖表
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # 設定字體
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 創建圖表
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # 繪製loss曲線
            ax.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
            ax.set_title('LoRA Training Loss Curve', fontsize=16, fontweight='bold')
            ax.set_xlabel('Training Steps', fontsize=12)
            ax.set_ylabel('Loss Value', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加統計信息
            final_loss = losses[-1] if losses else 0
            min_loss = min(losses) if losses else 0
            max_loss = max(losses) if losses else 0
            
            stats_text = f'Final Loss: {final_loss:.6f}\nMin Loss: {min_loss:.6f}\nMax Loss: {max_loss:.6f}\nTotal Steps: {len(steps)}'
            ax.text(0.02, 0.98, stats_text, 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            png_filename = f"lora_training_curves_{timestamp}.png"
            png_path = os.path.join(output_dir, png_filename)
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ PNG圖表已保存: {png_filename}")
            
        except ImportError:
            print(f"⚠️ matplotlib未安裝，跳過PNG圖表生成")
        except Exception as e:
            print(f"⚠️ PNG圖表生成失敗: {e}")
        
        # 3. 生成統計摘要
        print(f"\n📊 訓練統計摘要:")
        print(f"   總訓練步數: {len(steps)}")
        print(f"   最終Loss: {losses[-1]:.6f}")
        print(f"   最佳Loss: {min(losses):.6f}")
        print(f"   最差Loss: {max(losses):.6f}")
        print(f"   Loss改善: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%" if len(losses) > 1 else "N/A")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成報告時出錯: {e}")
        return False

# 檢查依賴項的函數
def main():
    """主函數 - 處理命令行參數"""
    # 🔧 FIX: 開始時就檢查依賴項
    print("🔍 檢查系統依賴...")
    if not check_dependencies():
        print("❌ 依賴檢查失敗，無法繼續")
        sys.exit(1)
    print("✅ 依賴檢查通過")
    
    parser = argparse.ArgumentParser(description="LoRA 訓練腳本")
    parser.add_argument("--continue", "-c", action="store_true", 
                       dest="continue_training",
                       help="從現有的 LoRA 檔案繼續訓練")
    parser.add_argument("--new", "-n", action="store_true",
                       dest="new_training", 
                       help="開始新的獨立 LoRA 訓練")
    parser.add_argument("--steps", "-s", type=int,
                       help="指定訓練步數 (跳過交互式詢問)")
    
    args = parser.parse_args()
    
    # 🔧 FIX: 增強模式決定邏輯
    print("\n🔍 檢查現有模型和狀態...")
    existing_lora = find_latest_lora()
    existing_state = find_latest_state_dir()
    
    if existing_lora:
        print(f"📄 發現 LoRA 模型: {os.path.basename(existing_lora)}")
    else:
        print("❌ 沒有發現現有 LoRA 模型")
        
    if existing_state:
        print(f"📁 發現訓練狀態: {os.path.basename(existing_state)}")
    else:
        print("❌ 沒有發現訓練狀態")
    
    # 決定訓練模式
    if args.continue_training and args.new_training:
        print("❌ 錯誤：不能同時指定 --continue 和 --new")
        sys.exit(1)
    elif args.continue_training:
        if not existing_lora and not existing_state:
            print("❌ 錯誤：指定繼續訓練但沒有找到現有模型或狀態")
            sys.exit(1)
        print("🔄 模式：從檢查點繼續訓練")
        continue_from_checkpoint = True
    elif args.new_training:
        print("🆕 模式：開始新的獨立訓練")
        continue_from_checkpoint = False
    else:
        # 🔧 FIX: 改進交互式選擇邏輯
        if existing_lora or existing_state:
            print(f"\n請選擇訓練模式：")
            print("1. 從現有模型繼續訓練 (累積調教)")
            print("2. 開始新的獨立訓練 (重新開始)")
            
            while True:
                choice = input("請輸入選擇 (1 或 2): ").strip()
                if choice == "1":
                    continue_from_checkpoint = True
                    print("🔄 已選擇：繼續訓練模式")
                    break
                elif choice == "2":
                    continue_from_checkpoint = False
                    print("🆕 已選擇：新訓練模式")
                    break
                else:
                    print("❌ 請輸入 1 或 2")
        else:
            print("🆕 沒有找到現有模型，將開始新的訓練")
            continue_from_checkpoint = False
    
    # 🔧 FIX: 驗證步數參數
    if args.steps is not None:
        if args.steps <= 0:
            print(f"❌ 錯誤：步數必須大於0，您輸入的是 {args.steps}")
            sys.exit(1)
        print(f"📊 將使用指定步數: {args.steps}")
    
    # 執行訓練
    success = train_lora(continue_from_checkpoint, custom_steps=args.steps)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()