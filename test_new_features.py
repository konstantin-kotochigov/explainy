#!/usr/bin/env python3
"""
Тесты для новой функциональности: прогресс и Jupyter Notebook.
"""

import sys
import json
from pathlib import Path
import nbformat

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent))

from main import (
    load_progress, 
    save_progress, 
    create_notebook, 
    add_topic_to_notebook,
    save_notebook
)


def test_progress_tracking():
    """Тест системы отслеживания прогресса."""
    print("Тест 1: Система отслеживания прогресса")
    
    # Удаляем существующий файл прогресса если есть
    progress_file = Path('progress.json')
    if progress_file.exists():
        progress_file.unlink()
    
    # Тест загрузки прогресса (должен создать новый)
    progress = load_progress()
    assert 'processed_topics' in progress, "Ключ 'processed_topics' не найден"
    assert 'last_index' in progress, "Ключ 'last_index' не найден"
    assert progress['processed_topics'] == [], "Начальный список должен быть пустым"
    assert progress['last_index'] == 0, "Начальный индекс должен быть 0"
    print("  ✓ Загрузка пустого прогресса работает")
    
    # Тест сохранения прогресса
    progress['processed_topics'] = ['Тема 1', 'Тема 2']
    progress['last_index'] = 2
    save_progress(progress)
    
    assert progress_file.exists(), "Файл прогресса не создан"
    print("  ✓ Сохранение прогресса работает")
    
    # Тест загрузки сохраненного прогресса
    loaded_progress = load_progress()
    assert loaded_progress['processed_topics'] == ['Тема 1', 'Тема 2'], "Темы не загружены"
    assert loaded_progress['last_index'] == 2, "Индекс не загружен"
    print("  ✓ Загрузка сохраненного прогресса работает")
    
    # Очистка
    progress_file.unlink()
    
    return True


def test_notebook_creation():
    """Тест создания и работы с Jupyter Notebook."""
    print("\nТест 2: Создание Jupyter Notebook")
    
    # Тест создания нового notebook
    notebook = create_notebook()
    assert isinstance(notebook, nbformat.NotebookNode), "Notebook не создан"
    assert hasattr(notebook, 'cells'), "Notebook не имеет атрибута 'cells'"
    print("  ✓ Создание нового notebook работает")
    
    # Тест добавления ячейки
    topic = "Тестовая тема"
    explanation = "Это тестовое объяснение темы."
    add_topic_to_notebook(notebook, topic, explanation)
    
    assert len(notebook.cells) == 1, "Ячейка не добавлена"
    assert notebook.cells[0].cell_type == 'markdown', "Ячейка не markdown типа"
    assert topic in notebook.cells[0].source, "Тема не найдена в ячейке"
    assert explanation in notebook.cells[0].source, "Объяснение не найдено в ячейке"
    print("  ✓ Добавление ячейки в notebook работает")
    
    # Тест добавления второй ячейки
    topic2 = "Вторая тема"
    explanation2 = "Второе объяснение."
    add_topic_to_notebook(notebook, topic2, explanation2)
    
    assert len(notebook.cells) == 2, "Вторая ячейка не добавлена"
    print("  ✓ Добавление нескольких ячеек работает")
    
    # Тест сохранения notebook
    test_output_dir = Path('outputs')
    test_output_dir.mkdir(exist_ok=True)
    test_notebook_path = test_output_dir / 'test_notebook.ipynb'
    
    save_notebook(notebook, test_notebook_path)
    assert test_notebook_path.exists(), "Notebook не сохранен"
    print("  ✓ Сохранение notebook работает")
    
    # Тест чтения сохраненного notebook
    with open(test_notebook_path, 'r', encoding='utf-8') as f:
        loaded_notebook = nbformat.read(f, as_version=4)
    
    assert len(loaded_notebook.cells) == 2, "Ячейки не загружены"
    assert topic in loaded_notebook.cells[0].source, "Первая тема не загружена"
    assert topic2 in loaded_notebook.cells[1].source, "Вторая тема не загружена"
    print("  ✓ Загрузка сохраненного notebook работает")
    
    # Очистка
    test_notebook_path.unlink()
    
    return True


def main():
    """Запуск всех тестов."""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ НОВОЙ ФУНКЦИОНАЛЬНОСТИ")
    print("=" * 80)
    
    tests = [
        test_progress_tracking,
        test_notebook_creation,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"РЕЗУЛЬТАТЫ: {passed}/{total} тестов пройдено")
    
    if all(results):
        print("✓ Все тесты пройдены успешно!")
        print("\nНовая функциональность работает корректно:")
        print("  1. Отслеживание прогресса в progress.json")
        print("  2. Генерация Jupyter Notebook вместо текстовых файлов")
        return 0
    else:
        print("✗ Некоторые тесты не прошли")
        return 1


if __name__ == "__main__":
    sys.exit(main())
