@echo off
setlocal

:: Variables
set PYINSTALLER=pyinstaller
set SCRIPT=src\gui.py
set DISTDIR=executable
set WORKDIR=%DISTDIR%\build
set SPECDIR=%DISTDIR%\spec

:: Paths
for /f "tokens=*" %%i in ('cd') do set PROJECT_ROOT=%%i
set ENV_PATH=%PROJECT_ROOT%\.env\Lib\site-packages
set GDAL_DATA_PATH=%ENV_PATH%\rasterio\gdal_data

:: Common flags
set COMMON_FLAGS= ^
  --windowed ^
  --paths "%ENV_PATH%" ^
  --hidden-import PIL._tkinter_finder ^
  --additional-hooks-dir=hook ^
  --add-data "%GDAL_DATA_PATH%;gdal_data" ^
  --distpath %DISTDIR% ^
  --workpath %WORKDIR% ^
  --specpath %SPECDIR%

:: Build
%PYINSTALLER% %COMMON_FLAGS% %SCRIPT%

:: Postbuild
echo Copying icon\ folder to %DISTDIR%\gui\
if not exist "%DISTDIR%\gui\icon" mkdir "%DISTDIR%\gui\icon"
xcopy /E /I /Y icon "%DISTDIR%\gui\icon\"

endlocal