#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности без реального API ключа.
"""

import sys
from pathlib import Path

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import read_file, read_topics


def test_read_system_prompt():
    """Тест чтения системного промпта."""
    print("Тест 1: Чтение system_prompt.txt")
    try:
        content = read_file('system_prompt.txt')
        assert len(content) > 0, "Системный промпт пустой"
        assert "блога" in content.lower(), "Системный промпт не содержит ожидаемое содержание"
        print(f"  ✓ Системный промпт загружен: {len(content)} символов")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False
    return True


def test_read_topics():
    """Тест чтения списка тем в новом формате."""
    print("\nТест 2: Чтение topics.txt")
    try:
        topics = read_topics('topics.txt')
        assert len(topics) > 0, "Список тем пустой"
        assert all(isinstance(t, dict) for t in topics), "Темы должны быть словарями"
        assert all('code' in t and 'detailed_query' in t and 'image_query' in t for t in topics), \
            "Каждая тема должна содержать поля: code, detailed_query, image_query"
        
        print(f"  ✓ Загружено {len(topics)} тем:")
        for i, topic in enumerate(topics[:3], 1):  # Показываем первые 3 темы
            print(f"    {i}. Код: {topic['code']}, Запрос: {topic['detailed_query'][:50]}...")
        if len(topics) > 3:
            print(f"    ... и еще {len(topics) - 3} тем")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False
    return True


def test_output_directory():
    """Тест создания выходной директории."""
    print("\nТест 3: Проверка создания директории outputs")
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
