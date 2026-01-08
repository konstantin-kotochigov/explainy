#!/usr/bin/env python3
"""
Тест для проверки функциональности критики и улучшения notebooks.
"""

import sys
from pathlib import Path
import tempfile
import nbformat

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import parse_notebook, enhance_notebook, save_explanation, save_complete_notebook


def test_parse_notebook():
    """Тест парсинга Jupyter Notebook."""
    print("Тест: Парсинг Jupyter Notebook")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Создаем тестовый notebook
        code = "test"
        explanation = "# Заголовок\n\nПервый параграф.\n\n## Подзаголовок\n\nВторой параграф."
        
        filepath = save_explanation(output_dir, code, explanation)
        if not filepath:
            print("  ✗ Не удалось создать notebook")
            return False
        
        # Парсим notebook
        parsed = parse_notebook(filepath)
        
        if not parsed:
            print("  ✗ Не удалось распарсить notebook")
            return False
        
        print("  ✓ Notebook успешно распарсен")
        
        # Проверяем структуру
        if 'notebook' not in parsed or 'content' not in parsed or 'cell_count' not in parsed:
            print("  ✗ Неправильная структура результата парсинга")
            return False
        
        print("  ✓ Структура результата корректна")
        
        # Проверяем содержимое
        if parsed['content'] != explanation:
            print("  ✗ Содержимое не совпадает")
            print(f"  Ожидалось: {explanation[:50]}...")
            print(f"  Получено: {parsed['content'][:50]}...")
            return False
        
        print("  ✓ Содержимое корректно извлечено")
        
        # Проверяем количество ячеек
        if parsed['cell_count'] != 1:
            print(f"  ✗ Ожидалось 1 ячейка, получено {parsed['cell_count']}")
            return False
        
        print("  ✓ Количество ячеек корректно")
    
    return True


def test_enhance_notebook_with_critique_only():
    """Тест улучшения notebook только с критикой."""
    print("\nТест: Улучшение notebook только с критикой")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Создаем тестовый notebook
        code = "test"
        explanation = "# Тестовое объяснение"
        
        filepath = save_explanation(output_dir, code, explanation)
        if not filepath:
            print("  ✗ Не удалось создать notebook")
            return False
        
        # Добавляем только критику
        critique = "Это хорошее объяснение, но можно добавить больше примеров."
        result = enhance_notebook(filepath, critique, None)
        
        if not result:
            print("  ✗ Не удалось улучшить notebook")
            return False
        
        # Проверяем результат
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Должно быть 2 ячейки: оригинал + критика
        if len(nb.cells) != 2:
            print(f"  ✗ Ожидалось 2 ячейки, получено {len(nb.cells)}")
            return False
        
        if "Критический анализ" not in nb.cells[1].source:
            print("  ✗ Критика не добавлена корректно")
            return False
        
        print("  ✓ Критика добавлена успешно")
    
    return True


def test_enhance_notebook_with_code_only():
    """Тест улучшения notebook только с кодом."""
    print("\nТест: Улучшение notebook только с кодом")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Создаем тестовый notebook
        code = "test"
        explanation = "# Тестовое объяснение"
        
        filepath = save_explanation(output_dir, code, explanation)
        if not filepath:
            print("  ✗ Не удалось создать notebook")
            return False
        
        # Добавляем только код
        code_example = "print('Hello, World!')\n# Это пример кода"
        result = enhance_notebook(filepath, None, code_example)
        
        if not result:
            print("  ✗ Не удалось улучшить notebook")
            return False
        
        # Проверяем результат
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Должно быть 3 ячейки: оригинал + заголовок кода + код
        if len(nb.cells) != 3:
            print(f"  ✗ Ожидалось 3 ячейки, получено {len(nb.cells)}")
            return False
        
        if nb.cells[2].cell_type != 'code':
            print("  ✗ Последняя ячейка должна быть кодом")
            return False
        
        if "Hello, World!" not in nb.cells[2].source:
            print("  ✗ Код не добавлен корректно")
            return False
        
        print("  ✓ Код добавлен успешно")
    
    return True


def test_enhance_notebook_with_both():
    """Тест улучшения notebook с критикой и кодом."""
    print("\nТест: Улучшение notebook с критикой и кодом")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Создаем тестовый notebook
        code = "test"
        explanation = "# Тестовое объяснение"
        
        filepath = save_explanation(output_dir, code, explanation)
        if not filepath:
            print("  ✗ Не удалось создать notebook")
            return False
        
        # Добавляем и критику, и код
        critique = "Хорошее объяснение с некоторыми улучшениями."
        code_example = "# Пример использования\nimport numpy as np\nprint(np.__version__)"
        result = enhance_notebook(filepath, critique, code_example)
        
        if not result:
            print("  ✗ Не удалось улучшить notebook")
            return False
        
        # Проверяем результат
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Должно быть 4 ячейки: оригинал + критика + заголовок кода + код
        if len(nb.cells) != 4:
            print(f"  ✗ Ожидалось 4 ячейки, получено {len(nb.cells)}")
            return False
        
        # Проверяем наличие критики
        if "Критический анализ" not in nb.cells[1].source:
            print("  ✗ Критика не найдена")
            return False
        
        # Проверяем наличие кода
        if nb.cells[3].cell_type != 'code':
            print("  ✗ Последняя ячейка должна быть кодом")
            return False
        
        print("  ✓ Критика и код добавлены успешно")
    
    return True


def test_save_complete_notebook():
    """Тест сохранения полного notebook за один раз (оптимизированный workflow)."""
    print("\nТест: Сохранение полного notebook за один раз")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Создаем полный notebook с объяснением, критикой и кодом
        code = "test"
        explanation = "# Тестовое объяснение\n\nЭто основное содержимое."
        critique = "Хорошее объяснение с некоторыми улучшениями."
        code_example = "# Пример использования\nimport numpy as np\nprint(np.__version__)"
        
        filepath = save_complete_notebook(output_dir, code, explanation, critique, code_example)
        
        if not filepath:
            print("  ✗ Не удалось создать notebook")
            return False
        
        # Проверяем результат
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Должно быть 4 ячейки: объяснение + критика + заголовок кода + код
        if len(nb.cells) != 4:
            print(f"  ✗ Ожидалось 4 ячейки, получено {len(nb.cells)}")
            return False
        
        # Проверяем наличие объяснения
        if "Тестовое объяснение" not in nb.cells[0].source:
            print("  ✗ Объяснение не найдено")
            return False
        
        # Проверяем наличие критики
        if "Критический анализ" not in nb.cells[1].source:
            print("  ✗ Критика не найдена")
            return False
        
        # Проверяем наличие кода
        if nb.cells[3].cell_type != 'code':
            print("  ✗ Последняя ячейка должна быть кодом")
            return False
        
        print("  ✓ Полный notebook создан за один раз успешно")
    
    return True


def test_save_complete_notebook_without_critique():
    """Тест сохранения notebook без критики (только объяснение)."""
    print("\nТест: Сохранение notebook без критики")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Создаем notebook только с объяснением
        code = "test"
        explanation = "# Тестовое объяснение\n\nЭто основное содержимое."
        
        filepath = save_complete_notebook(output_dir, code, explanation)
        
        if not filepath:
            print("  ✗ Не удалось создать notebook")
            return False
        
        # Проверяем результат
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Должна быть 1 ячейка: только объяснение
        if len(nb.cells) != 1:
            print(f"  ✗ Ожидалось 1 ячейка, получено {len(nb.cells)}")
            return False
        
        # Проверяем наличие объяснения
        if "Тестовое объяснение" not in nb.cells[0].source:
            print("  ✗ Объяснение не найдено")
            return False
        
        print("  ✓ Notebook без критики создан успешно")
    
    return True


def main():
    """Запуск всех тестов."""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ КРИТИКИ И УЛУЧШЕНИЯ NOTEBOOKS")
    print("=" * 80)
    
    tests = [
        test_parse_notebook,
        test_enhance_notebook_with_critique_only,
        test_enhance_notebook_with_code_only,
        test_enhance_notebook_with_both,
        test_save_complete_notebook,
        test_save_complete_notebook_without_critique,
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
