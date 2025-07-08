@echo off
chcp 65001 >nul
title 3-Day Test
echo ============================================================
echo               3-Day Feasibility Test
echo ============================================================
echo.

cd /d "e:\Yung_Folder\Project\stable-diffusion-webui"

echo Checking WebUI connection...
python -c "import requests; r=requests.get('http://localhost:7860/sdapi/v1/memory',timeout=5); print('WebUI API OK')" 2>nul
if errorlevel 1 (
    echo ERROR: WebUI API not accessible
    echo.
    echo Please:
    echo   1. Double-click START_WEBUI.bat
    echo   2. Wait for WebUI to fully start
    echo   3. Run this test again
    echo.
    pause
    exit /b 1
)

echo WebUI API connection OK
echo.

:menu
echo ============================================================
echo                     Test Options
echo ============================================================
echo 1. Day 1 Test - Basic Functions
echo 2. Day 2 Test - Advanced Functions  
echo 3. Day 3 Test - Results Evaluation
echo 4. Full 3-Day Test (Auto run all)
echo 5. View Results
echo 0. Exit
echo ============================================================

set /p choice="Please choose (0-5): "

if "%choice%"=="1" goto day1
if "%choice%"=="2" goto day2  
if "%choice%"=="3" goto day3
if "%choice%"=="4" goto full_test
if "%choice%"=="5" goto view_results
if "%choice%"=="0" goto exit

echo Invalid choice, please try again
goto menu

:day1
echo.
echo Day 1 Test: Basic Functions
echo ============================================================
python day1_basic_test.py
echo.
pause
goto menu

:day2
echo.
echo Day 2 Test: Advanced Functions
echo ============================================================
python day2_advanced_test.py
echo.
pause
goto menu

:day3
echo.
echo Day 3 Test: Results Evaluation
echo ============================================================
python day3_evaluation.py
echo.
pause
goto menu

:full_test
echo.
echo Running Full 3-Day Test
echo ============================================================
echo.
echo Day 1: Basic Functions
python day1_basic_test.py
if errorlevel 1 (
    echo Day 1 test failed
    pause
    goto menu
)
echo.

echo Day 2: Advanced Functions
python day2_advanced_test.py
if errorlevel 1 (
    echo Day 2 test failed
    pause
    goto menu
)
echo.

echo Day 3: Results Evaluation
python day3_evaluation.py
echo.
echo 3-Day Test Complete!
pause
goto menu

:view_results
echo.
echo Test Results Overview
echo ============================================================
if exist "day1_results\day1_report.json" (
    echo Day 1 results: EXISTS
) else (
    echo Day 1 results: NOT FOUND
)

if exist "day2_results\day2_report.json" (
    echo Day 2 results: EXISTS
) else (
    echo Day 2 results: NOT FOUND
)

if exist "day3_evaluation\final_feasibility_report.json" (
    echo Final report: EXISTS
    echo.
    echo Showing summary...
    python -c "import json; data=json.load(open('day3_evaluation/final_feasibility_report.json', encoding='utf-8')); print(f'Feasibility: {data[\"feasibility_assessment\"][\"feasibility_level\"]}'); print(f'Success Rate: {data[\"technical_analysis\"][\"overall_success_rate\"]*100:.1f}%%')" 2>nul
) else (
    echo Final report: NOT FOUND
)
echo.
pause
goto menu

:exit
echo.
echo Thank you for using the 3-Day Test!
echo Check these folders for detailed results:
echo   day1_results/
echo   day2_results/  
echo   day3_evaluation/
pause
exit /b 0
