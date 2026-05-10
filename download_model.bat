@echo off
setlocal
rem UTF-8 characters might be garbled in CMD if not handled carefully.
rem We use chcp 65001 but avoid putting Japanese on the same lines as logic.

chcp 65001 > nul
cd /d "%~dp0"

echo ========================================
echo Model Download Task
echo Target: LFM2.5-1.2B-JP-Q8_0.gguf
echo ========================================

if not exist "models" (
    echo Creating models directory...
    mkdir models
)

set "URL=https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-GGUF/resolve/main/LFM2.5-1.2B-JP-Q8_0.gguf"
set "DEST=models\LFM2.5-1.2B-JP-Q8_0.gguf"

if exist "%DEST%" (
    echo [SKIP] Model already exists at %DEST%
    goto :END
)

echo.
echo Downloading model... (This will take time)
echo URL: %URL%
echo.

curl.exe -L "%URL%" -o "%DEST%"

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Download completed.
    echo Saved to: %DEST%
) else (
    echo.
    echo [ERROR] Download failed. Please check your internet connection.
)

:END
echo.
pause
