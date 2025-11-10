@echo off
REM Ensure we're in the project root (folder that contains src, app, config.yaml)
setlocal
set PROJ=%~dp0
cd /d "%PROJ%"
set PYTHONPATH=%PROJ%\src;%PYTHONPATH%
streamlit run app\app.py
endlocal
