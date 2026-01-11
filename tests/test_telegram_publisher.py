#!/usr/bin/env python3
"""
Тесты для модуля telegram_publisher.py
"""

import sys
import os
from pathlib import Path
import tempfile
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from telegram_publisher import (
    convert_notebook_to_markdown,
    split_message,
    publish_notebook_to_telegram
)


def create_test_notebook(filepath: Path) -> None:
    """Создает тестовый notebook для тестирования."""
    nb = new_notebook()
    
    # Добавляем markdown ячейку
    nb.cells.append(new_markdown_cell("# Тестовый заголовок\n\nЭто тестовое объяснение."))
    
    # Добавляем code ячейку
    nb.cells.append(new_code_cell("print('Hello, World!')"))
    
    # Добавляем еще одну markdown ячейку
    nb.cells.append(new_markdown_cell("## Заключение\n\nЭто заключение."))
    
    # Сохраняем notebook
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)


def test_convert_notebook_to_markdown():
    """Тест конвертации notebook в markdown."""
    print("Тест 1: Конвертация notebook в markdown")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Создаем тестовый notebook
        notebook_path = Path(tmpdir) / "test.ipynb"
        create_test_notebook(notebook_path)
        
        try:
            # Конвертируем в markdown
            markdown_content = convert_notebook_to_markdown(notebook_path)
            
            # Проверяем, что контент не пустой
            if not markdown_content:
                print("  ✗ Контент пустой")
                return False
            
            # Проверяем, что контент содержит ожидаемый текст
            if "Тестовый заголовок" not in markdown_content:
                print("  ✗ Контент не содержит ожидаемый текст")
                return False
            
            print(f"  ✓ Notebook успешно конвертирован ({len(markdown_content)} символов)")
            return True
            
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            return False


def test_convert_nonexistent_file():
    """Тест конвертации несуществующего файла."""
    print("\nТест 2: Обработка несуществующего файла")
    
    try:
        convert_notebook_to_markdown("/nonexistent/file.ipynb")
        print("  ✗ Исключение не было вызвано")
        return False
    except FileNotFoundError:
        print("  ✓ FileNotFoundError корректно вызвано")
        return True
    except Exception as e:
        print(f"  ✗ Неожиданное исключение: {e}")
        return False


def test_convert_wrong_extension():
    """Тест конвертации файла с неправильным расширением."""
    print("\nТест 3: Обработка файла с неправильным расширением")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Создаем файл с неправильным расширением
        wrong_file = Path(tmpdir) / "test.txt"
        wrong_file.write_text("Some content")
        
        try:
            convert_notebook_to_markdown(wrong_file)
            print("  ✗ Исключение не было вызвано")
            return False
        except ValueError:
            print("  ✓ ValueError корректно вызвано")
            return True
        except Exception as e:
            print(f"  ✗ Неожиданное исключение: {e}")
            return False


def test_split_message():
    """Тест разбиения длинного сообщения на части."""
    print("\nТест 4: Разбиение длинного сообщения")
    
    # Короткое сообщение
    short_text = "Это короткое сообщение"
    parts = split_message(short_text)
    
    if len(parts) != 1:
        print(f"  ✗ Короткое сообщение должно остаться в одной части, получено: {len(parts)}")
        return False
    
    print(f"  ✓ Короткое сообщение: {len(parts)} часть")
    
    # Длинное сообщение
    long_text = "Параграф 1.\n\n" + ("Очень длинный текст. " * 500) + "\n\nПараграф 2."
    parts = split_message(long_text, max_length=1000)
    
    if len(parts) <= 1:
        print(f"  ✗ Длинное сообщение должно быть разбито, получено: {len(parts)} частей")
        return False
    
    # Проверяем, что каждая часть не превышает максимальную длину
    for i, part in enumerate(parts):
        if len(part) > 1000:
            print(f"  ✗ Часть {i+1} превышает максимальную длину: {len(part)} символов")
            return False
    
    print(f"  ✓ Длинное сообщение разбито на {len(parts)} частей")
    return True


def test_publish_without_credentials():
    """Тест публикации без учетных данных."""
    print("\nТест 5: Публикация без учетных данных")
    
    # Сохраняем текущие переменные окружения
    old_token = os.getenv('TELEGRAM_BOT_TOKEN')
    old_channel = os.getenv('TELEGRAM_CHANNEL_ID')
    
    # Удаляем переменные окружения
    if 'TELEGRAM_BOT_TOKEN' in os.environ:
        del os.environ['TELEGRAM_BOT_TOKEN']
    if 'TELEGRAM_CHANNEL_ID' in os.environ:
        del os.environ['TELEGRAM_CHANNEL_ID']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Создаем тестовый notebook
        notebook_path = Path(tmpdir) / "test.ipynb"
        create_test_notebook(notebook_path)
        
        # Пытаемся опубликовать без учетных данных
        result = publish_notebook_to_telegram(notebook_path)
        
        # Восстанавливаем переменные окружения
        if old_token:
            os.environ['TELEGRAM_BOT_TOKEN'] = old_token
        if old_channel:
            os.environ['TELEGRAM_CHANNEL_ID'] = old_channel
        
        if result:
            print("  ✗ Публикация должна была завершиться неудачей без учетных данных")
            return False
        
        print("  ✓ Публикация корректно завершилась неудачей без учетных данных")
        return True


def test_markdown_output_format():
    """Тест проверки формата markdown на выходе."""
    print("\nТест 6: Проверка формата markdown")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Создаем notebook с различными типами ячеек
        notebook_path = Path(tmpdir) / "format_test.ipynb"
        nb = new_notebook()
        
        # Markdown с заголовками
        nb.cells.append(new_markdown_cell("# Заголовок 1\n\n## Заголовок 2\n\nТекст параграфа."))
        
        # Code ячейка
        nb.cells.append(new_code_cell("def hello():\n    print('Hello')"))
        
        # Еще markdown
        nb.cells.append(new_markdown_cell("**Жирный текст** и *курсив*."))
        
        # Сохраняем
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        try:
            markdown_content = convert_notebook_to_markdown(notebook_path)
            
            # Проверяем наличие ключевых элементов
            if "# Заголовок 1" not in markdown_content:
                print("  ✗ Заголовки не сохранились")
                return False
            
            if "def hello():" not in markdown_content:
                print("  ✗ Код не сохранился")
                return False
            
            if "**Жирный текст**" not in markdown_content and "Жирный текст" not in markdown_content:
                print("  ✗ Форматирование не сохранилось")
                return False
            
            print("  ✓ Формат markdown сохранен корректно")
            return True
            
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            return False


def main():
    """Запуск всех тестов."""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ TELEGRAM_PUBLISHER")
    print("=" * 80)
    
    tests = [
        test_convert_notebook_to_markdown,
        test_convert_nonexistent_file,
        test_convert_wrong_extension,
        test_split_message,
        test_publish_without_credentials,
        test_markdown_output_format,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"РЕЗУЛЬТАТЫ: {passed}/{total} тестов пройдено")
    
    if all(results):
        print("✓ Все тесты пройдены успешно!")
        print("\nМодуль готов к использованию.")
        print("\nПример использования:")
        print("  1. Установите зависимости: pip install -r requirements.txt")
        print("  2. Создайте Telegram бота через @BotFather")
        print("  3. Добавьте бота в ваш канал как администратора")
        print("  4. Установите переменные окружения в .env:")
        print("     TELEGRAM_BOT_TOKEN=ваш_токен")
        print("     TELEGRAM_CHANNEL_ID=@ваш_канал")
        print("  5. Используйте:")
        print("     python telegram_publisher.py outputs/your_notebook.ipynb")
        return 0
    else:
        print("✗ Некоторые тесты не прошли")
        return 1


if __name__ == "__main__":
    sys.exit(main())
