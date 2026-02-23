@echo off
echo STARTING > C:\repositories\cpp\MPS_DAWN\test_output.txt
echo %DATE% %TIME% >> C:\repositories\cpp\MPS_DAWN\test_output.txt
cd /d C:\repositories\cpp\MPS_DAWN\build\bin\x64\Debug
start "" /B mps_dawn.exe >> C:\repositories\cpp\MPS_DAWN\test_output.txt 2>&1
set PID=
timeout /t 8 /nobreak > nul
echo TIMEOUT_REACHED >> C:\repositories\cpp\MPS_DAWN\test_output.txt
echo %DATE% %TIME% >> C:\repositories\cpp\MPS_DAWN\test_output.txt
taskkill /IM mps_dawn.exe /F >> C:\repositories\cpp\MPS_DAWN\test_output.txt 2>&1
echo EXIT=%ERRORLEVEL% >> C:\repositories\cpp\MPS_DAWN\test_output.txt
