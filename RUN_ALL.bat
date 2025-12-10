@echo off
chcp 65001 >nul
echo ====================================================================
echo          СИСТЕМА ДЕТЕКЦИИ МАСОК - ПОЛНЫЙ ЗАПУСК
echo ====================================================================
echo.

REM Активация виртуального окружения
echo [0/7] Активация виртуального окружения...
call venv\Scripts\activate
if errorlevel 1 (
    echo ОШИБКА: Виртуальное окружение не найдено!
    echo Создайте его командой: python -m venv venv
    pause
    exit /b 1
)
echo ✓ Окружение активировано
echo.

REM Проверка структуры данных
echo [1/7] Проверка структуры данных...
python scripts/check_data_structure.py
if errorlevel 1 (
    echo.
    echo ОШИБКА: Неправильная структура данных!
    pause
    exit /b 1
)
echo.
pause

REM Проверка зависимостей
echo [2/7] Проверка зависимостей...
python scripts/check_dependencies.py
if errorlevel 1 (
    echo.
    echo ОШИБКА: Не все зависимости установлены!
    echo Установите их: pip install -r requirements.txt
    pause
    exit /b 1
)
echo.
pause

REM Загрузка ресурсов
echo [3/7] Загрузка дополнительных ресурсов...
python scripts/download_resources.py
if errorlevel 1 (
    echo.
    echo ОШИБКА при загрузке ресурсов!
    pause
    exit /b 1
)
echo.
pause

REM Анализ данных
echo [4/7] Анализ данных...
python scripts/01_analyze_data.py
if errorlevel 1 (
    echo.
    echo ОШИБКА при анализе данных!
    pause
    exit /b 1
)
echo.
pause

REM Обучение моделей
echo [5/7] Обучение моделей (это займет время!)...
python scripts/02_Train_models.py
if errorlevel 1 (
    echo.
    echo ОШИБКА при обучении моделей!
    pause
    exit /b 1
)
echo.
pause

REM Оценка моделей
echo [6/7] Оценка моделей...
python scripts/03_evaluate_models.py
if errorlevel 1 (
    echo.
    echo ОШИБКА при оценке моделей!
    pause
    exit /b 1
)
echo.

echo ====================================================================
echo                    ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ!
echo ====================================================================
echo.
echo [7/7] Запуск веб-приложения...
cd web_app
streamlit run app.py

pause