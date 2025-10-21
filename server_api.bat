@echo off
REM เก็บ path ปัจจุบัน
set "current_dir=%CD%"

REM เรียก conda
call C:\Users\User\anaconda3\Scripts\activate.bat py10_f5tts

REM เปลี่ยน directory ไปที่ folder ของ server_api.py (ถ้าจำเป็น)
cd /d "%current_dir%"

REM รัน Python script
python src\f5_tts\server_api.py

pause
