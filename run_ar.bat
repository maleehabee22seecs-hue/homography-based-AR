
@echo off
echo Starting AR 3D Placement App...
echo Using Python at: C:\Users\Maleeha\AppData\Local\Programs\Python\Python312\python.exe

"C:\Users\Maleeha\AppData\Local\Programs\Python\Python312\python.exe" src/main.py

if %errorlevel% neq 0 (
    echo.
    echo Script failed with error code %errorlevel%.
    pause
) else (
    echo.
    echo Script finished successfully.
    pause
)
