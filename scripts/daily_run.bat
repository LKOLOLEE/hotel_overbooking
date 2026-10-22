@echo off
rem Codzienne uruchomienie raportu - wywolywane przez Harmonogram zadan Windows.
rem Dodatkowe argumenty przekazywane sa do daily_run.py, np.:
rem   daily_run.bat --publish-dir "\\serwer\www\raport"
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
set PYTHONUTF8=1
call ".venv\Scripts\activate.bat"
echo ===== %date% %time% ===== >> "logs\daily_run.log"
python daily_run.py --replay %* >> "logs\daily_run.log" 2>&1
endlocal
