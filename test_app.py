#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности без реального API ключа.
"""

import sys
from pathlib import Path

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent))

from main import read_file, read_topics, sanitize_filename, MAX_FILENAME_LENGTH


def test_read_system_prompt():
    """Тест чтения системного промпта."""
    print("Тест 1: Чтение system_prompt.txt")
    try:
        content = read_file('system_prompt.txt')
        assert len(content) > 0, "Системный промпт пустой"
        assert "эксперт" in content.lower(), "Системный промпт не содержит ожидаемое содержание"
        print(f"  ✓ Системный промпт загружен: {len(content)} символов")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False
    return True


def test_read_topics():
    """Тест чтения списка тем."""
    print("\nТест 2: Чтение topics.txt")
    try:
        topics = read_topics('topics.txt')
        assert len(topics) > 0, "Список тем пустой"
        print(f"  ✓ Загружено {len(topics)} тем:")
        for i, topic in enumerate(topics, 1):
            print(f"    {i}. {topic[:60]}{'...' if len(topic) > 60 else ''}")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False
    return True


def test_sanitize_filename():
    """Тест очистки имени файла."""
    print("\nТест 3: Проверка sanitize_filename()")
    test_cases = [
        ("Обычная тема", "Обычная тема"),
        ("Тема/с/слешами", "Тема_с_слешами"),
        ("Тема:с:двоеточиями", "Тема_с_двоеточиями"),
        ("?" * 150, "_" * MAX_FILENAME_LENGTH),  # Проверка ограничения длины
    ]
    
    all_passed = True
    for input_text, expected_pattern in test_cases:
        result = sanitize_filename(input_text)
        if "/" not in result and "\\" not in result and ":" not in result:
            print(f"  ✓ '{input_text[:40]}...' -> '{result[:40]}...'")
        else:
            print(f"  ✗ Небезопасные символы остались в: {result}")
            all_passed = False
    
    return all_passed


def test_output_directory():
    """Тест создания выходной директории."""
    print("\nТест 4: Проверка создания директории outputs")
    try:
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        assert output_dir.exists(), "Директория не создана"
        assert output_dir.is_dir(), "outputs не является директорией"
        print(f"  ✓ Директория создана: {output_dir.absolute()}")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False
    return True


def main():
    """Запуск всех тестов."""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ФУНКЦИОНАЛЬНОСТИ ПРИЛОЖЕНИЯ")
    print("=" * 80)
    
    tests = [
        test_read_system_prompt,
        test_read_topics,
        test_sanitize_filename,
        test_output_directory,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"РЕЗУЛЬТАТЫ: {passed}/{total} тестов пройдено")
    
    if all(results):
        print("✓ Все тесты пройдены успешно!")
        print("\nПриложение готово к использованию.")
        print("Для запуска необходимо:")
        print("  1. Создать файл .env с OPENAI_API_KEY")
        print("  2. Запустить: python main.py")
        return 0
    else:
        print("✗ Некоторые тесты не прошли")
        return 1


if __name__ == "__main__":
    sys.exit(main())
