@echo off
echo ===============================================
echo LoRA訓練快速開始 (Conda環境支持)
echo ===============================================

echo 🔍 檢查Conda環境...
python conda_environment_checker.py

echo.
echo 1. 檢查修復狀態...
python verify_training_stop_fix.py

echo.
echo 2. 確認性能指標一致性...
python auto_test_pipeline\performance_metrics_final_confirmation.py

echo.
echo 3. 開始LoRA訓練...
python day3_fashion_training.py

echo.
echo 4. 分析訓練結果...
python auto_test_pipeline\analyze_results.py

echo.
echo ===============================================
echo 訓練完成！檢查結果輸出。
echo ===============================================
pause
