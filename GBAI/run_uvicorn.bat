@echo off

cd /d C:\Users\Administrator\PycharmProjects\GBAI

powershell -WindowStyle Hidden -Command "& '.venv\Scripts\activate'; uvicorn main:app --host 0.0.0.0 --port 8000"