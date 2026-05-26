@echo off
setlocal enabledelayedexpansion
chcp 932 > nul
cd /d "%~dp0"

echo ========================================
echo HGNN 実行環境セットアップ [v0.0.2]
echo ========================================
echo.

echo =======================================================================
echo 【重要】事前準備の確認
echo 本システムのセットアップおよび動作には、CPU・GPU を問わず
echo 「Visual Studio C++ Build Tools」が Windows にインストールされている必要があります。
echo.
echo まだインストールされていない場合は、下記 URL からダウンロードして
echo 「C++ によるデスクトップ開発」にチェックを入れてインストールしてください。
echo https://visualstudio.microsoft.com/ja/visual-cpp-build-tools/
echo =======================================================================
echo.
set /p SETUP_CONFIRM="続行するには Enter キーを押してください（中断する場合は Ctrl+C）..."

set "ZIP_URL=https://sourceforge.net/projects/winpython/files/WinPython_3.12/3.12.10.1/Winpython64-3.12.10.1dot.zip/download"
set "ZIP_FILE=Winpython64-3.12.10.1dot.zip"
set "TEMP_DIR=temp_winpython"

if exist "python" (
    echo [情報] python フォルダが既に存在します。ダウンロードをスキップします。
    goto :install_libs
)

if exist "%ZIP_FILE%" (
    echo [情報] %ZIP_FILE% が既に存在します。ダウンロードをスキップします。
    goto :extract
)

echo [1/9] WinPython をダウンロード中...
curl -L -o "%ZIP_FILE%" "%ZIP_URL%"
if %errorlevel% neq 0 (
    echo [エラー] WinPython のダウンロードに失敗しました。
    pause
    exit /b 1
)

:extract
echo [2/9] WinPython を展開中...
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
mkdir "%TEMP_DIR%"
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%TEMP_DIR%'"
if %errorlevel% neq 0 (
    echo [エラー] ZIP ファイルの展開に失敗しました。
    pause
    exit /b 1
)

echo [3/9] Python バイナリをコピー中...
set "PYTHON_SRC="

for /d %%A in ("%TEMP_DIR%\WPy64-*") do (
    if exist "%%A\python\python.exe" (
        set "PYTHON_SRC=%%A\python"
    )
)

if "%PYTHON_SRC%"=="" (
    echo [エラー] python フォルダが見つかりませんでした。
    echo %TEMP_DIR% の内容:
    dir "%TEMP_DIR%" /s /b /ad
    pause
    exit /b 1
)

echo 検出されたパス: %PYTHON_SRC%
xcopy /e /i /h /y "%PYTHON_SRC%" "python"
if %errorlevel% neq 0 (
    echo [エラー] python フォルダのコピーに失敗しました。
    pause
    exit /b 1
)

if not exist "python\python.exe" (
    echo [エラー] コピー後に python.exe が見つかりませんでした。
    pause
    exit /b 1
)

echo [4/9] 一時ファイルをクリーンアップ中...
rmdir /s /q "%TEMP_DIR%"
del "%ZIP_FILE%"

:install_libs
echo [5/9] ハードウェア構成の選択と基本パッケージのインストール...
rem Windows環境でのファイルロック競合によるフリーズを防ぐため、pip自体の自己アップデートはスキップします。
rem WinPython搭載の標準pipで全ての要件を問題なくインストール可能です。

echo.
echo ========================================
echo ハードウェア構成の選択 [CPU / GPU]
echo ========================================
echo AI 処理のパフォーマンスを最適化するため、ハードウェア環境を選択してください。
echo.
echo 【注意】本システムは [1] CPU のみを推奨【動作保証】としています。
echo GPU環境は環境によって動作しない可能性があり、動作確認は RTX 2060 6GB のみです。
echo Intel Iris や AMD Radeon [Vulkan/DirectML] は実機検証を行っておらず未サポートです。
echo.
echo ※文字認識 (NDLOCR-Lite) および埋め込み (Embedding) は常に CPU で動作します。
echo   GPUを選択した場合、対話AI (LLM) のみがGPUで高速化されます。
echo.
echo [1] CPU のみ [推奨・最も安定しています]
echo [2] NVIDIA GPU [CUDA アクセラレーション - RTX 2060等検証済み]
echo [3] Intel / AMD GPU [DirectML / Vulkan アクセラレーション - 未検証・サポート外]
echo.
set /p HW_CHOICE="選択肢を入力してください [1, 2, または 3]: "

set "GPU_BACKEND=cpu"
set "ONNX_LIB=onnxruntime==1.23.2"
set "CMAKE_ARGS="

if "!HW_CHOICE!"=="2" (
    echo.
    echo [警告] NVIDIA GPU が選択されました。
    echo 本システムで検証済みのグラフィックボードは【RTX 2060 6GB】のみであり、
    echo お使いの環境によっては動作しない、またはコンパイルエラーとなる場合があります。
    echo これには CUDA Toolkit がシステムにインストールされている必要があります。
    set /p CONFIRM="GPU 用のインストールを続行しますか？ [Y/N]: "
    if /I not "!CONFIRM!"=="Y" (
        echo インストールをキャンセルしました.
        pause
        exit /b 1
    )
    set "GPU_BACKEND=cuda"
    set "ONNX_LIB=onnxruntime==1.23.2"
    set "CMAKE_ARGS=-DGGML_CUDA=ON"
) else if "!HW_CHOICE!"=="3" (
    echo.
    echo [警告] Intel / AMD GPU が選択されました。
    echo 【注意】Intel Iris や AMD Radeon などの環境は実機検証を行っておらず、未サポートです。
    echo 環境によってはインストールが失敗したり動作しない場合があるため、あらかじめご了承ください。
    set /p CONFIRM="GPU 用のインストールを続行しますか？ [Y/N]: "
    if /I not "!CONFIRM!"=="Y" (
        echo インストールをキャンセルしました。
        pause
        exit /b 1
    )
    set "GPU_BACKEND=directml"
    set "ONNX_LIB=onnxruntime==1.23.2"
    set "CMAKE_ARGS=-DGGML_VULKAN=ON"
)

echo.
set "PREV_BACKEND="
if exist "scripts\config_ai.txt" (
    for /f "tokens=2 delims==" %%A in ('findstr "gpu_backend" scripts\config_ai.txt 2^>nul') do (
        set "PREV_BACKEND=%%A"
    )
)

echo 基本の依存パッケージをインストール中...
findstr /v "ndlocr-lite onnxruntime llama-cpp-python" requirements.txt > requirements_filtered.txt
python\python.exe -m pip install --no-warn-script-location -r requirements_filtered.txt
del requirements_filtered.txt
if !errorlevel! neq 0 (
    echo [エラー] 基本パッケージのインストールに失敗しました。
    pause
    exit /b 1
)

echo.
echo ハードウェア構成を保存中...
if not exist "scripts" mkdir scripts
echo gpu_backend=!GPU_BACKEND!>> scripts\config_ai.txt

echo.
echo [6/9] ONNX Runtime (CPU) のインストール...
echo ONNX Runtime [!ONNX_LIB!] をインストール中...
python\python.exe -m pip install --no-warn-script-location !ONNX_LIB!
if !errorlevel! neq 0 (
    echo [エラー] !ONNX_LIB! のインストールに失敗しました。
    pause
    exit /b 1
)

echo.
echo [7/9] llama-cpp-python のコンパイルとインストール...
set "SKIP_LLAMA_BUILD=0"
if "!PREV_BACKEND!"=="!GPU_BACKEND!" (
    python\python.exe -m pip show llama-cpp-python > nul 2>&1
    if !errorlevel! equ 0 (
        set "SKIP_LLAMA_BUILD=1"
    )
)

if "!SKIP_LLAMA_BUILD!"=="1" (
    echo [情報] 現在の構成 [!GPU_BACKEND!] 用の llama-cpp-python は既にインストールされています。
    echo ビルドとインストールをスキップします。
) else (
    echo llama-cpp-python をビルド・インストール中... [環境によって10分～30分程度かかる場合があります]
    rem Windows の 260文字パス制限によるファイル展開・コンパイル失敗を回避するため、一時フォルダをワークスペース直下の短いパスに一時的に変更します。
    setlocal
    set "TEMP_PIP_DIR=%~dp0temp_pip"
    if not exist "!TEMP_PIP_DIR!" mkdir "!TEMP_PIP_DIR!"
    set "TEMP=!TEMP_PIP_DIR!"
    set "TMP=!TEMP_PIP_DIR!"
    
    python\python.exe -m pip install --no-warn-script-location --upgrade --force-reinstall --no-deps llama-cpp-python==0.3.20 --no-cache-dir
    set "PIP_ERR=!errorlevel!"
    
    rmdir /s /q "!TEMP_PIP_DIR!" 2>nul
    
    if !PIP_ERR! neq 0 (
        echo [エラー] llama-cpp-python のコンパイルとインストールに失敗しました。
        endlocal
        pause
        exit /b 1
    )
    endlocal
)

echo.
echo [8/9] ndlocr-lite をセットアップ中...
python\python.exe -m pip show ndlocr-lite > nul 2>&1
if %errorlevel% equ 0 (
    echo [情報] ndlocr-lite は既にインストールされています。スキップします。
    goto :create_folders
)

if exist "ndlocr-lite-master" rmdir /s /q "ndlocr-lite-master"
if exist "ndlocr-lite.zip" del "ndlocr-lite.zip"
echo ndlocr-lite をダウンロード中...
curl -L -o "ndlocr-lite.zip" "https://github.com/ndl-lab/ndlocr-lite/archive/refs/heads/master.zip"
if %errorlevel% neq 0 (
    echo [エラー] ndlocr-lite のダウンロードに失敗しました。
    pause
    exit /b 1
)

echo ndlocr-lite を展開中...
powershell -Command "Expand-Archive -Path 'ndlocr-lite.zip' -DestinationPath '.'"
if %errorlevel% neq 0 (
    echo [エラー] ndlocr-lite の展開に失敗しました。
    del "ndlocr-lite.zip"
    pause
    exit /b 1
)

python\python.exe -m pip install --no-warn-script-location --no-deps ./ndlocr-lite-master
if %errorlevel% neq 0 (
    echo [エラー] ndlocr-lite のインストールに失敗しました。
    rmdir /s /q "ndlocr-lite-master"
    del "ndlocr-lite.zip"
    pause
    exit /b 1
)

rmdir /s /q "ndlocr-lite-master"
del "ndlocr-lite.zip"

:create_folders
echo.
echo [9/9] 作業用フォルダの作成と動作確認...
echo 作業用フォルダを作成中...
echo ========================================
for %%F in (models db kb) do (
    if not exist "%%F" (
        mkdir "%%F"
        echo [作成完了] %%F
    ) else (
        echo [既に存在] %%F
    )
)

echo.
echo ========================================
echo 動作確認
echo ========================================
python\python.exe -m ocr --help > nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] ndlocr-lite の動作確認に失敗しました。
) else (
    echo [成功] ndlocr-lite の準備が完了しました！
)

echo.
echo セットアップが正常に完了しました！
echo.
pause
