@echo off
rem Launch the oblivion2vmf Qt GUI (PySide6 + pyvista 3D collision editor).
rem Double-click this file, or run it from anywhere.
setlocal
cd /d "%~dp0"
set "PYTHONPATH=src;%PYTHONPATH%"

rem Prefer the py launcher if present, else plain python.
where py >nul 2>nul && (set "PY=py") || (set "PY=python")

%PY% -m oblivion2vmf.qtgui
if errorlevel 1 (
    echo.
    echo [oblivion2vmf] GUI exited with an error ^(code %errorlevel%^).
    echo If it says a module is missing, install the GUI deps:
    echo     %PY% -m pip install PySide6 pyvista pyvistaqt numpy
    echo.
    pause
)
endlocal
