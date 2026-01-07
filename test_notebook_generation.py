#!/usr/bin/env python3
"""
Тест для проверки генерации Jupyter Notebooks.
"""

import sys
from pathlib import Path
import tempfile
import nbformat

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent))

from main import save_explanation, sanitize_filename


def test_notebook_generation():
    """Тест генерации Jupyter Notebook."""
    print("Тест: Генерация Jupyter Notebook")
    
    # Создаем временную директорию для тестов
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Тестовые данные
        topic = "Методы информационного поиска / метод DPR"
        explanation = "# DPR (2020)\n\nDeep Passage Retrieval - это метод для поиска релевантных документов."
        index = 1
        
        # Генерируем notebook
        save_explanation(output_dir, topic, explanation, index)
        
        # Проверяем, что файл создан
        expected_filename = f"{index:02d}_{sanitize_filename(topic)}.ipynb"
        filepath = output_dir / expected_filename
        
        if not filepath.exists():
            print(f"  ✗ Файл не создан: {filepath}")
            return False
        
        print(f"  ✓ Файл создан: {expected_filename}")
        
        # Проверяем содержимое notebook
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
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


def test_filename_sanitization_in_notebooks():
    """Тест очистки имени файла для notebooks."""
    print("\nТест: Очистка имени файла для notebooks")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Тестируем тему с небезопасными символами
        topic = "Тема/с/слешами:и:двоеточиями?"
        explanation = "# Тестовое объяснение"
        index = 1
        
        save_explanation(output_dir, topic, explanation, index)
        
        # Проверяем, что файл создан с безопасным именем
        expected_filename = f"{index:02d}_Тема_с_слешами_и_двоеточиями_.ipynb"
        filepath = output_dir / expected_filename
        
        if not filepath.exists():
            print(f"  ✗ Файл не создан с безопасным именем")
            # Показываем, какие файлы были созданы
            files = list(output_dir.glob("*.ipynb"))
            if files:
                print(f"  Созданные файлы: {[f.name for f in files]}")
            return False
        
        print(f"  ✓ Файл создан с безопасным именем: {expected_filename}")
    
    return True


def main():
    """Запуск всех тестов."""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ JUPYTER NOTEBOOKS")
    print("=" * 80)
    
    tests = [
        test_notebook_generation,
        test_filename_sanitization_in_notebooks,
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
