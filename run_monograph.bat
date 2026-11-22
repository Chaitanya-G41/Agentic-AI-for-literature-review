@echo off
echo ===================================================
echo   Starting Monograph Literature Review System
echo ===================================================
echo.

cd /d "%~dp0"

echo [1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ...
start "Monograph Backend (FastAPI)" cmd /k "python server.py"

echo [2/2] Launching Monograph Frontend on http://localhost:5174 ...
cd frontend_v2
start "Monograph Frontend (Vite)" cmd /k "npm run dev"

echo.
echo ===================================================
echo   System is launching!
echo   Open your browser at: http://localhost:5174
echo ===================================================
echo.
pause
