@echo off
echo Starting AR 3D Placement App...
echo Ensure you have installed requirements: pip install -r requirements.txt
echo.

python src/main.py %*

if %errorlevel% neq 0 (
    echo.
    echo Script failed with error code %errorlevel%.
    echo Make sure 'python' is in your PATH.
    pause
) else (
    echo.
    echo Script finished successfully.
    pause
)
