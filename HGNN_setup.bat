@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul
cd /d "%~dp0"

echo ========================================
echo HGNN Environment Setup
echo ========================================
echo.

set "ZIP_URL=https://sourceforge.net/projects/winpython/files/WinPython_3.12/3.12.10.1/Winpython64-3.12.10.1dot.zip/download"
set "ZIP_FILE=Winpython64-3.12.10.1dot.zip"
set "TEMP_DIR=temp_winpython"

if exist "python" (
    echo [INFO] python folder already exists. Skipping download.
    goto :install_libs
)

if exist "%ZIP_FILE%" (
    echo [INFO] %ZIP_FILE% already exists. Skipping download.
    goto :extract
)

echo [1/5] Downloading WinPython...
curl -L -o "%ZIP_FILE%" "%ZIP_URL%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download WinPython.
    pause
    exit /b 1
)

:extract
echo [2/5] Extracting WinPython...
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
mkdir "%TEMP_DIR%"
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%TEMP_DIR%'"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to extract zip file.
    pause
    exit /b 1
)

echo [3/5] Copying python binaries...
set "PYTHON_SRC="

for /d %%A in ("%TEMP_DIR%\WPy64-*") do (
    if exist "%%A\python\python.exe" (
        set "PYTHON_SRC=%%A\python"
    )
)

if "%PYTHON_SRC%"=="" (
    echo [ERROR] Could not locate python folder.
    echo Contents of %TEMP_DIR%:
    dir "%TEMP_DIR%" /s /b /ad
    pause
    exit /b 1
)

echo Found: %PYTHON_SRC%
xcopy /e /i /h /y "%PYTHON_SRC%" "python"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to copy python folder.
    pause
    exit /b 1
)

if not exist "python\python.exe" (
    echo [ERROR] python.exe not found after copy.
    pause
    exit /b 1
)

echo [4/5] Cleaning up...
rmdir /s /q "%TEMP_DIR%"
del "%ZIP_FILE%"

:install_libs
echo [5/5] Installing requirements...
python\python.exe -m pip install --no-warn-script-location --upgrade pip
findstr /v "ndlocr-lite" requirements.txt > requirements_filtered.txt
python\python.exe -m pip install --no-warn-script-location -r requirements_filtered.txt
del requirements_filtered.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo Setting up ndlocr-lite...
python\python.exe -m pip show ndlocr-lite > nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] ndlocr-lite already installed. Skipping.
    goto :create_folders
)

if exist "ndlocr-lite" rmdir /s /q "ndlocr-lite"
git clone https://github.com/ndl-lab/ndlocr-lite
if %errorlevel% neq 0 (
    echo [ERROR] Failed to clone ndlocr-lite.
    pause
    exit /b 1
)

python\python.exe -m pip install --no-warn-script-location ./ndlocr-lite
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install ndlocr-lite.
    rmdir /s /q "ndlocr-lite"
    pause
    exit /b 1
)

rmdir /s /q "ndlocr-lite"

:create_folders
echo.
echo ========================================
echo Creating work folders...
echo ========================================
for %%F in (models db kb) do (
    if not exist "%%F" (
        mkdir "%%F"
        echo [CREATED] %%F
    ) else (
        echo [EXISTS] %%F
    )
)

echo.
echo ========================================
echo Verification
echo ========================================
python\python.exe -m ocr --help > nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] ndlocr-lite verification failed.
) else (
    echo [SUCCESS] ndlocr-lite is ready!
)

echo.
echo Setup Finished Successfully!
echo.
pause