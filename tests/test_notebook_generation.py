#!/usr/bin/env python3
"""
Тест для проверки генерации Jupyter Notebooks.
"""

import sys
from pathlib import Path
import tempfile
import nbformat

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import save_explanation


def test_notebook_generation():
    """Тест генерации Jupyter Notebook."""
    print("Тест: Генерация Jupyter Notebook")
    
    # Создаем временную директорию для тестов
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Тестовые данные
        code = "dpr"
        explanation = "# DPR (2020)\n\nDeep Passage Retrieval - это метод для поиска релевантных документов."
        
        # Генерируем notebook
        filepath = save_explanation(output_dir, code, explanation)
        
        # Проверяем, что файл создан с капитализированным именем
        from main import capitalize_first_letter
        expected_filename = f"{capitalize_first_letter(code)}.ipynb"
        expected_path = output_dir / expected_filename
        
        if not expected_path.exists():
            print(f"  ✗ Файл не создан: {expected_path}")
            return False
        
        print(f"  ✓ Файл создан: {expected_filename}")
        
        # Проверяем содержимое notebook
        try:
            with open(expected_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Проверяем, что есть хотя бы одна ячейка
            if len(nb.cells) == 0:
                print(f"  ✗ Notebook не содержит ячеек")
                return False
            
            print(f"  ✓ Notebook содержит {len(nb.cells)} ячейку(и)")
            
            # Проверяем, что первая ячейка - Markdown
            first_cell = nb.cells[0]
            if first_cell.cell_type != 'markdown':
                print(f"  ✗ Первая ячейка не Markdown, а {first_cell.cell_type}")
                return False
            
            print(f"  ✓ Первая ячейка - Markdown")
            
            # Проверяем, что содержимое совпадает
            if first_cell.source != explanation:
                print(f"  ✗ Содержимое ячейки не совпадает с ожидаемым")
                print(f"  Ожидалось: {explanation[:50]}...")
                print(f"  Получено: {first_cell.source[:50]}...")
                return False
            
            print(f"  ✓ Содержимое ячейки совпадает с ожидаемым")
            
        except Exception as e:
            print(f"  ✗ Ошибка при чтении notebook: {e}")
            return False
    
    return True


def test_notebook_enhancement():
    """Тест улучшения Jupyter Notebook с критикой и кодом."""
    print("\nТест: Улучшение Jupyter Notebook")
    
    from main import enhance_notebook
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Создаем тестовый notebook
        code = "test_topic"
        explanation = "# Тестовое объяснение\n\nЭто тестовое содержимое."
        
        filepath = save_explanation(output_dir, code, explanation)
        if not filepath:
            print("  ✗ Не удалось создать начальный notebook")
            return False
        
        # Улучшаем notebook
        critique = "## Критика\nЭто хорошее объяснение, но можно добавить больше деталей."
        code_example = "# Пример кода\nprint('Hello, World!')"
        
        result = enhance_notebook(filepath, critique, code_example)
        if not result:
            print("  ✗ Не удалось улучшить notebook")
            return False
        
        # Проверяем результат
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Должно быть: 1 исходная ячейка + 1 критика + 1 заголовок кода + 1 код = 4 ячейки
            if len(nb.cells) != 4:
                print(f"  ✗ Ожидалось 4 ячейки, получено {len(nb.cells)}")
                return False
            
            print(f"  ✓ Notebook содержит {len(nb.cells)} ячейки")
            
            # Проверяем типы ячеек
            if nb.cells[0].cell_type != 'markdown':
                print("  ✗ Первая ячейка должна быть markdown")
                return False
            if nb.cells[1].cell_type != 'markdown':
                print("  ✗ Вторая ячейка (критика) должна быть markdown")
                return False
            if nb.cells[2].cell_type != 'markdown':
                print("  ✗ Третья ячейка (заголовок кода) должна быть markdown")
                return False
            if nb.cells[3].cell_type != 'code':
                print("  ✗ Четвертая ячейка должна быть code")
                return False
            
            print("  ✓ Типы ячеек корректны")
            
            # Проверяем содержимое
            if "Критический анализ" not in nb.cells[1].source:
                print("  ✗ Ячейка критики не содержит ожидаемый заголовок")
                return False
            
            if "Пример кода" not in nb.cells[2].source:
                print("  ✗ Заголовок кода не найден")
                return False
            
            print("  ✓ Содержимое улучшений добавлено корректно")
            
        except Exception as e:
            print(f"  ✗ Ошибка при проверке улучшенного notebook: {e}")
            return False
    
    return True


def test_filename_generation():
    """Тест генерации имени файла для notebooks."""
    print("\nТест: Генерация имени файла для notebooks")
    
    from main import capitalize_first_letter
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Тестируем с простым кодом
        code = "simple_code"
        explanation = "# Тестовое объяснение"
        
        filepath = save_explanation(output_dir, code, explanation)
        
        # Проверяем, что файл создан с правильным именем (капитализированным)
        expected_filename = f"{capitalize_first_letter(code)}.ipynb"
        expected_path = output_dir / expected_filename
        
        if not expected_path.exists():
            print(f"  ✗ Файл не создан с ожидаемым именем")
            return False
        
        print(f"  ✓ Файл создан с правильным именем: {expected_filename}")
    
    return True


def main():
    """Запуск всех тестов."""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ JUPYTER NOTEBOOKS")
    print("=" * 80)
    
    tests = [
        test_notebook_generation,
        test_notebook_enhancement,
        test_filename_generation,
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
