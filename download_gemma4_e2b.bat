@echo off
setlocal
rem UTF-8 characters might be garbled in CMD if not handled carefully.
rem We use chcp 65001 but avoid putting Japanese on the same lines as logic.

chcp 65001 > nul
cd /d "%~dp0"

echo ========================================
echo Model Download Task
echo Target: gemma-4-E2B-it-Q4_K_M.gguf [approx. 3.08 GB]
echo ========================================

if not exist "models" (
    echo Creating models directory...
    mkdir models
)

set "URL=https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
set "DEST=models\gemma-4-E2B-it-Q4_K_M.gguf"

if exist "%DEST%" (
    echo [SKIP] Model already exists at %DEST%
    goto :END
)

echo.
echo Downloading model... [This will take time, size is 3.08 GB]
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
