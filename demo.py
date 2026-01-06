#!/usr/bin/env python3
"""
Демонстрация работы приложения без реального API.
Показывает процесс обработки тем с использованием прогресса и notebook.
"""

import sys
from pathlib import Path
import nbformat

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent))

from main import (
    read_topics,
    load_progress,
    save_progress,
    create_notebook,
    add_topic_to_notebook,
    save_notebook,
)


def demo_workflow():
    """Демонстрация полного рабочего процесса."""
    print("=" * 80)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ ПРИЛОЖЕНИЯ")
    print("=" * 80)
    
    # 1. Читаем темы
    print("\n1. Чтение тем из topics.txt...")
    topics = read_topics('topics.txt')
    print(f"   Загружено {len(topics)} тем")
    print(f"   Первые 3 темы:")
    for i, topic in enumerate(topics[:3], 1):
        print(f"   {i}. {topic[:70]}...")
    
    # 2. Загружаем прогресс
    print("\n2. Загрузка прогресса...")
    progress = load_progress()
    processed = set(progress.get('processed_topics', []))
    print(f"   Уже обработано тем: {len(processed)}")
    
    # 3. Определяем темы для обработки
    topics_to_process = [t for t in topics if t not in processed]
    print(f"   Осталось обработать: {len(topics_to_process)} тем")
    
    # 4. Создаем или загружаем notebook
    print("\n3. Подготовка Jupyter Notebook...")
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    notebook_path = output_dir / 'explanations.ipynb'
    
    if notebook_path.exists():
        print(f"   Загружаем существующий notebook")
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        print(f"   Текущее количество ячеек: {len(notebook.cells)}")
    else:
        print(f"   Создаем новый notebook")
        notebook = create_notebook()
        print(f"   Notebook создан (ячеек: 0)")
    
    # 5. Симуляция добавления тем (без реального API)
    print("\n4. Симуляция обработки тем...")
    print("   (В реальности здесь вызывается API для генерации объяснений)")
    
    # Добавляем несколько демонстрационных тем
    demo_topics = topics_to_process[:2] if topics_to_process else topics[:2]
    
    for i, topic in enumerate(demo_topics, 1):
        print(f"\n   [{i}/{len(demo_topics)}] Обработка: {topic[:60]}...")
        
        # Симулируем объяснение
        explanation = f"Это демонстрационное объяснение темы.\n\n**Важно:** В реальном приложении здесь будет подробное объяснение, сгенерированное LLM моделью."
        
        # Добавляем в notebook
        add_topic_to_notebook(notebook, topic, explanation)
        print(f"      ✓ Добавлено в notebook")
        
        # Сохраняем notebook
        save_notebook(notebook, notebook_path)
        
        # Обновляем прогресс
        processed.add(topic)
        progress['processed_topics'] = list(processed)
        progress['last_index'] = len(processed)
        save_progress(progress)
        print(f"      ✓ Прогресс сохранен")
    
    # 6. Итоги
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)
    print(f"\n✓ Notebook сохранен: {notebook_path}")
    print(f"✓ Количество ячеек в notebook: {len(notebook.cells)}")
    print(f"✓ Обработано тем: {len(processed)}")
    print(f"✓ Прогресс сохранен в: progress.json")
    
    print("\n" + "-" * 80)
    print("Для запуска с реальным API:")
    print("  1. Создайте файл .env с вашим API ключом")
    print("  2. Запустите: python main.py")
    print("  3. При прерывании запустите снова - работа продолжится с места остановки")
    print("-" * 80)


if __name__ == "__main__":
    try:
        demo_workflow()
    except Exception as e:
        print(f"\nОшибка: {e}")
        sys.exit(1)
