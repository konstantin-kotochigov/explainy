#!/usr/bin/env python3
"""
Тест для проверки парсинга комментариев и пустых строк в topics.txt.
"""

import sys
import tempfile
from pathlib import Path

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import read_topics


def test_topics_with_comments_and_empty_lines():
    """Тест чтения topics.txt с комментариями и пустыми строками."""
    print("Тест: Парсинг topics.txt с комментариями и пустыми строками")
    
    # Создаем временный файл с тестовыми данными
    test_content = """# Это комментарий в начале файла
# Еще один комментарий
topic1;Описание темы 1;Image query 1

# Комментарий посередине
topic2;Описание темы 2;Image query 2

topic3;Описание темы 3;Image query 3
# Комментарий в конце
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        topics = read_topics(temp_file)
        
        # Проверяем, что загружены только 3 темы (без комментариев и пустых строк)
        assert len(topics) == 3, f"Ожидалось 3 темы, получено {len(topics)}"
        
        # Проверяем содержимое тем
        assert topics[0]['code'] == 'topic1', "Неверный код первой темы"
        assert topics[0]['detailed_query'] == 'Описание темы 1', "Неверное описание первой темы"
        assert topics[0]['image_query'] == 'Image query 1', "Неверный image_query первой темы"
        
        assert topics[1]['code'] == 'topic2', "Неверный код второй темы"
        assert topics[2]['code'] == 'topic3', "Неверный код третьей темы"
        
        print(f"  ✓ Загружено {len(topics)} тем (комментарии и пустые строки пропущены)")
        for i, topic in enumerate(topics, 1):
            print(f"    {i}. Код: {topic['code']}, Запрос: {topic['detailed_query']}")
        return True
        
    except AssertionError as e:
        print(f"  ✗ Ошибка проверки: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False
    finally:
        # Удаляем временный файл
        Path(temp_file).unlink(missing_ok=True)


def test_topics_only_comments():
    """Тест чтения topics.txt с только комментариями."""
    print("\nТест: Парсинг topics.txt с только комментариями")
    
    test_content = """# Только комментарии
# Еще комментарий

# И еще один
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        topics = read_topics(temp_file)
        
        # Проверяем, что список пустой
        assert len(topics) == 0, f"Ожидался пустой список, получено {len(topics)} тем"
        
        print(f"  ✓ Загружено {len(topics)} тем (все комментарии пропущены)")
        return True
        
    except AssertionError as e:
        print(f"  ✗ Ошибка проверки: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False
    finally:
        Path(temp_file).unlink(missing_ok=True)


def test_topics_various_comment_styles():
    """Тест различных стилей комментариев."""
    print("\nТест: Различные стили комментариев")
    
    test_content = """#Комментарий без пробела
# Комментарий с пробелом
#    Комментарий с табами
topic1;Описание 1;Query 1
## Двойной решетка
topic2;Описание 2;Query 2
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        topics = read_topics(temp_file)
        
        # Проверяем, что загружены только 2 темы
        assert len(topics) == 2, f"Ожидалось 2 темы, получено {len(topics)}"
        assert topics[0]['code'] == 'topic1', "Неверный код первой темы"
        assert topics[1]['code'] == 'topic2', "Неверный код второй темы"
        
        print(f"  ✓ Загружено {len(topics)} тем (все варианты комментариев пропущены)")
        return True
        
    except AssertionError as e:
        print(f"  ✗ Ошибка проверки: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False
    finally:
        Path(temp_file).unlink(missing_ok=True)


def main():
    """Запуск всех тестов."""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ПАРСИНГА КОММЕНТАРИЕВ В TOPICS.TXT")
    print("=" * 80)
    
    tests = [
        test_topics_with_comments_and_empty_lines,
        test_topics_only_comments,
        test_topics_various_comment_styles,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"РЕЗУЛЬТАТЫ: {passed}/{total} тестов пройдено")
    
    if all(results):
        print("✓ Все тесты пройдены успешно!")
        return 0
    else:
        print("✗ Некоторые тесты не прошли")
        return 1


if __name__ == "__main__":
    sys.exit(main())
