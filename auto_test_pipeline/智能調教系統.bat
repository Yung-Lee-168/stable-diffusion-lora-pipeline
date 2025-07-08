@echo off
chcp 65001
echo.
echo ========================================================
echo 🎯 LoRA 智能調教系統 - 整合訓練監控
echo ========================================================
echo.

:menu
echo 請選擇操作模式：
echo.
echo 1. 🚀 智能訓練模式 (自動監控 + 決策)
echo 2. 📊 訓練 + 推理 + 分析 (完整流程)
echo 3. 🔍 僅訓練監控
echo 4. 🎨 僅推理測試
echo 5. 📈 僅結果分析
echo 6. 📋 查看最新技術指標
echo 7. 🔧 高級設定
echo 8. 📖 說明文檔
echo 9. 退出
echo.

set /p choice=請輸入選項 (1-9): 

if "%choice%"=="1" goto smart_training
if "%choice%"=="2" goto full_pipeline
if "%choice%"=="3" goto monitor_only
if "%choice%"=="4" goto inference_only
if "%choice%"=="5" goto analysis_only
if "%choice%"=="6" goto view_metrics
if "%choice%"=="7" goto advanced_settings
if "%choice%"=="8" goto documentation
if "%choice%"=="9" goto exit

echo 無效選項，請重新選擇
pause
goto menu

:smart_training
echo.
echo 🚀 智能訓練模式
echo ================
echo 此模式將：
echo 1. 監控訓練過程中的技術指標
echo 2. 基於訓練表現自動決定是否繼續推理
echo 3. 產生詳細的訓練報告和建議
echo.

echo 🔍 開始智能訓練...
python train_lora_monitored.py

if errorlevel 2 (
    echo.
    echo ❌ 訓練失敗
    echo 請檢查日誌檔案並調整參數
    pause
    goto menu
) else if errorlevel 1 (
    echo.
    echo ⚠️ 訓練完成但表現不佳
    echo 建議調整參數後重新訓練
    echo.
    echo 是否要查看訓練建議？ (y/n)
    set /p view_suggestions=
    if /i "%view_suggestions%"=="y" (
        echo 📋 正在分析訓練結果...
        python analyze_results.py
    )
    pause
    goto menu
) else (
    echo.
    echo ✅ 訓練成功！系統建議繼續推理測試
    echo.
    echo 是否要立即進行推理測試？ (y/n)
    set /p do_inference=
    if /i "%do_inference%"=="y" (
        echo 🎨 開始推理測試...
        python infer_lora.py
        
        echo 📊 開始結果分析...
        python analyze_results.py
        
        echo.
        echo ✅ 完整流程執行完畢
        echo 📋 請查看 test_results 目錄中的報告
    )
    pause
    goto menu
)

:full_pipeline
echo.
echo 📊 完整訓練流程
echo ================
echo 執行順序：訓練 → 推理 → 分析
echo.

echo 🚀 階段 1/3: LoRA 訓練
python train_lora_monitored.py

if errorlevel 2 (
    echo ❌ 訓練階段失敗，流程中止
    pause
    goto menu
)

echo.
echo 🎨 階段 2/3: 推理測試
python infer_lora.py

echo.
echo 📊 階段 3/3: 結果分析
python analyze_results.py

echo.
echo ✅ 完整流程執行完畢
echo 📋 請查看 test_results 目錄中的詳細報告
pause
goto menu

:monitor_only
echo.
echo 🔍 訓練監控模式
echo ================
echo.

echo 請輸入要監控的訓練命令 (或按 Enter 使用預設):
set /p training_cmd=
if "%training_cmd%"=="" set training_cmd=python train_network.py --config train_config.toml

echo.
echo 📈 開始監控訓練: %training_cmd%
python training_progress_monitor.py --training-command "%training_cmd%"

pause
goto menu

:inference_only
echo.
echo 🎨 推理測試模式
echo ================
echo.

echo 🔍 檢查 LoRA 模型...
if not exist "lora_output\*.safetensors" (
    echo ❌ 沒有找到 LoRA 模型檔案
    echo 請先執行訓練步驟
    pause
    goto menu
)

echo ✅ 找到 LoRA 模型，開始推理測試...
python infer_lora.py

echo.
echo 是否要立即分析結果？ (y/n)
set /p analyze_now=
if /i "%analyze_now%"=="y" (
    python analyze_results.py
)

pause
goto menu

:analysis_only
echo.
echo 📊 結果分析模式
echo ================
echo.

echo 🔍 檢查測試結果...
if not exist "test_images\*.png" (
    echo ❌ 沒有找到測試圖片
    echo 請先執行推理測試
    pause
    goto menu
)

echo ✅ 找到測試圖片，開始分析...
python analyze_results.py

echo.
echo 📋 分析完成，結果已保存到 test_results 目錄
pause
goto menu

:view_metrics
echo.
echo 📋 技術指標查看
echo ================
echo.

echo 🔍 載入技術指標查看器...
python quick_metrics_viewer.py --latest

echo.
echo 是否要查看歷史趨勢？ (y/n)
set /p view_history=
if /i "%view_history%"=="y" (
    python quick_metrics_viewer.py --history
)

pause
goto menu

:advanced_settings
echo.
echo 🔧 高級設定
echo ============
echo.
echo 1. 編輯訓練配置 (train_config.toml)
echo 2. 調整監控參數
echo 3. 設定預警閾值
echo 4. 返回主選單
echo.

set /p adv_choice=請選擇 (1-4): 

if "%adv_choice%"=="1" (
    if exist "train_config.toml" (
        notepad train_config.toml
    ) else (
        echo 📝 創建預設配置檔案...
        echo # LoRA 訓練配置 > train_config.toml
        echo pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5" >> train_config.toml
        echo train_data_dir = "lora_train_set" >> train_config.toml
        echo output_dir = "lora_output" >> train_config.toml
        echo network_dim = 32 >> train_config.toml
        echo network_alpha = 32 >> train_config.toml
        echo learning_rate = 1e-4 >> train_config.toml
        echo max_train_steps = 1000 >> train_config.toml
        notepad train_config.toml
    )
) else if "%adv_choice%"=="2" (
    echo 📊 當前監控參數：
    echo - 檢查點間隔: 50 步
    echo - 早停耐心值: 200 步
    echo - 目標損失: 0.1
    echo.
    echo 如需修改，請編輯 training_progress_monitor.py
    pause
) else if "%adv_choice%"=="3" (
    echo ⚠️ 當前預警閾值：
    echo - 損失突增閾值: 2.0
    echo - 最小損失改善: 0.01
    echo - 最大學習率衰減: 0.1
    echo.
    echo 如需修改，請編輯 training_progress_monitor.py
    pause
) else if "%adv_choice%"=="4" (
    goto menu
)

goto advanced_settings

:documentation
echo.
echo 📖 說明文檔
echo ============
echo.

echo 1. 技術指標追蹤指南
echo 2. README - 系統總覽
echo 3. 返回主選單
echo.

set /p doc_choice=請選擇 (1-3): 

if "%doc_choice%"=="1" (
    if exist "技術指標追蹤指南.md" (
        start notepad "技術指標追蹤指南.md"
    ) else (
        echo ❌ 找不到技術指標追蹤指南
    )
) else if "%doc_choice%"=="2" (
    if exist "README.md" (
        start notepad "README.md"
    ) else (
        echo ❌ 找不到 README 檔案
    )
) else if "%doc_choice%"=="3" (
    goto menu
)

goto documentation

:exit
echo.
echo 👋 感謝使用 LoRA 智能調教系統！
echo.
echo 📊 系統特色回顧：
echo   • 即時訓練監控與技術指標追蹤
echo   • 智能決策：基於訓練表現自動決定後續步驟
echo   • 完整評估：三基準點 + LoRA 專業指標
echo   • 自動化報告：HTML/JSON/PNG 多格式輸出
echo.
pause
exit
