@echo off
echo ====================================
echo      LoRA 調優完整流程啟動器
echo ====================================
echo.

REM 設定基礎路徑
set BASE_DIR=%~dp0
set PIPELINE_DIR=%BASE_DIR%auto_test_pipeline

echo 🔍 檢查環境...
if not exist "%PIPELINE_DIR%" (
    echo ❌ 找不到 auto_test_pipeline 目錄
    pause
    exit /b 1
)

echo 📁 工作目錄：%BASE_DIR%
echo 📂 流程目錄：%PIPELINE_DIR%
echo.

REM 選擇執行模式
echo 請選擇執行模式：
echo 1. 完整自動化流程（推薦）
echo 2. 單次訓練測試
echo 3. 分析現有結果
echo 4. 參數優化建議
echo 5. 監控儀表板
echo 6. 自訂配置
echo.

set /p choice="請輸入選擇 (1-6): "

if "%choice%"=="1" goto full_pipeline
if "%choice%"=="2" goto single_test
if "%choice%"=="3" goto analyze_only
if "%choice%"=="4" goto optimize_only
if "%choice%"=="5" goto monitor_dashboard
if "%choice%"=="6" goto custom_config

echo ❌ 無效選擇，請重新執行腳本
pause
exit /b 1

:full_pipeline
echo.
echo 🚀 執行完整自動化流程
echo ====================================
echo 此模式將執行：
echo 1. 多輪 LoRA 訓練
echo 2. 自動推理測試
echo 3. 結果分析與評估
echo 4. 參數自動優化
echo 5. 生成完整報告
echo.

set /p max_iter="請輸入最大迭代次數 (預設5): "
if "%max_iter%"=="" set max_iter=5

set /p target_score="請輸入目標整體分數 (預設0.7): "
if "%target_score%"=="" set target_score=0.7

echo.
echo 📊 配置資訊：
echo    最大迭代次數：%max_iter%
echo    目標整體分數：%target_score%
echo.

echo 🔄 開始執行...
python "%PIPELINE_DIR%\lora_optimization_pipeline.py" --max_iterations %max_iter% --target_overall %target_score%

goto end

:single_test
echo.
echo 🎯 執行單次訓練測試
echo ====================================
echo 此模式將執行：
echo 1. 單次 LoRA 訓練
echo 2. 推理測試
echo 3. 結果分析
echo.

echo 🔄 開始訓練...
python "%PIPELINE_DIR%\train_lora.py"

if errorlevel 1 (
    echo ❌ 訓練失敗
    goto end
)

echo 🎨 開始推理...
python "%PIPELINE_DIR%\infer_lora.py"

if errorlevel 1 (
    echo ❌ 推理失敗
    goto end
)

echo 📊 開始分析...
python "%PIPELINE_DIR%\analyze_results.py"

goto end

:analyze_only
echo.
echo 📊 分析現有結果
echo ====================================
echo 此模式將分析現有的訓練結果並生成報告
echo.

python "%PIPELINE_DIR%\analyze_results.py"

goto end

:optimize_only
echo.
echo 🔧 生成參數優化建議
echo ====================================
echo 此模式將基於現有結果生成下一輪訓練的參數建議
echo.

python "%PIPELINE_DIR%\lora_tuning_optimizer.py"

goto end

:monitor_dashboard
echo.
echo 📈 監控儀表板
echo ====================================
echo 請選擇監控模式：
echo 1. 即時監控
echo 2. 生成報告
echo 3. 生成儀表板
echo.

set /p monitor_choice="請輸入選擇 (1-3): "

if "%monitor_choice%"=="1" (
    echo 🔍 開始即時監控（按 Ctrl+C 停止）...
    python "%PIPELINE_DIR%\lora_tuning_monitor.py" --mode monitor
) else if "%monitor_choice%"=="2" (
    echo 📋 生成監控報告...
    python "%PIPELINE_DIR%\lora_tuning_monitor.py" --mode report
) else if "%monitor_choice%"=="3" (
    echo 📊 生成監控儀表板...
    python "%PIPELINE_DIR%\lora_tuning_monitor.py" --mode dashboard
) else (
    echo ❌ 無效選擇
)

goto end

:custom_config
echo.
echo ⚙️ 自訂配置
echo ====================================
echo 請輸入自訂參數（直接按 Enter 使用預設值）：
echo.

set /p learning_rate="學習率 (預設0.0005): "
if "%learning_rate%"=="" set learning_rate=0.0005

set /p steps="訓練步數 (預設100): "
if "%steps%"=="" set steps=100

set /p resolution="解析度 (預設512x512): "
if "%resolution%"=="" set resolution=512x512

set /p batch_size="批次大小 (預設1): "
if "%batch_size%"=="" set batch_size=1

echo.
echo 📊 自訂配置：
echo    學習率：%learning_rate%
echo    訓練步數：%steps%
echo    解析度：%resolution%
echo    批次大小：%batch_size%
echo.

echo 🔄 開始自訂訓練...
python "%PIPELINE_DIR%\train_lora.py" --learning_rate %learning_rate% --max_train_steps %steps% --resolution %resolution% --train_batch_size %batch_size%

if errorlevel 1 (
    echo ❌ 訓練失敗
    goto end
)

echo 🎨 開始推理...
python "%PIPELINE_DIR%\infer_lora.py"

if errorlevel 1 (
    echo ❌ 推理失敗
    goto end
)

echo 📊 開始分析...
python "%PIPELINE_DIR%\analyze_results.py"

goto end

:end
echo.
echo ====================================
echo 🎉 執行完成
echo ====================================
echo.

REM 檢查結果目錄
if exist "%PIPELINE_DIR%\test_results" (
    echo 📁 結果目錄：%PIPELINE_DIR%\test_results
    echo 📋 可查看以下文件：
    echo    - training_report_*.html ^(HTML 報告^)
    echo    - training_report_*.json ^(JSON 數據^)
    echo    - training_charts_*.png ^(圖表^)
    echo    - optimization_report_*.json ^(優化報告^)
    echo    - final_summary_*.md ^(總結報告^)
) else (
    echo ⚠️ 未找到結果目錄
)

echo.
echo 💡 提示：
echo    - 可以重複執行此腳本進行多輪調優
echo    - 建議查看 HTML 報告了解詳細結果
echo    - 使用監控模式可以追蹤調優進展
echo.

pause
