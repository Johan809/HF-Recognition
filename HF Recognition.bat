@ECHO OFF
cd /d "%~dp0"
python "./gui/start.py"
IF ERRORLEVEL 1 PAUSE
