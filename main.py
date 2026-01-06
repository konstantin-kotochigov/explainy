#!/usr/bin/env python3
"""
Генератор объяснений сложных технических тем с использованием LLM.

Приложение:
1. Читает список тем из файла topics.txt
2. Читает системный промпт из файла system_prompt.txt
3. Генерирует объяснение для каждой темы используя OpenAI API
4. Сохраняет объяснения в Jupyter Notebook
5. Отслеживает прогресс и может продолжить с места остановки
"""

import os
import sys
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell

# Константы
MAX_FILENAME_LENGTH = 100
PROGRESS_FILE = 'progress.json'


def read_file(filepath: str) -> str:
    """Читает содержимое файла."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Ошибка: файл {filepath} не найден")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла {filepath}: {e}")
        sys.exit(1)


def read_topics(filepath: str) -> list[str]:
    """Читает список тем из файла, возвращает список непустых строк."""
    content = read_file(filepath)
    topics = [line.strip() for line in content.split('\n') if line.strip()]
    return topics


def load_progress() -> dict:
    """Загружает прогресс обработки из файла."""
    try:
        if Path(PROGRESS_FILE).exists():
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'processed_topics': [], 'last_index': 0}
    except Exception as e:
        print(f"Предупреждение: не удалось загрузить прогресс: {e}")
        return {'processed_topics': [], 'last_index': 0}


def save_progress(progress: dict):
    """Сохраняет прогресс обработки в файл."""
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Предупреждение: не удалось сохранить прогресс: {e}")


def generate_explanation(client: OpenAI, system_prompt: str, topic: str) -> str:
    """Генерирует объяснение темы используя OpenAI API."""
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Объясни следующую тему: {topic}"}
            ],
            # service_tier="flex"
        )
        if response.choices and len(response.choices) > 0:
            print("total tokens processed = {}".format(response.usage.total_tokens))
            return response.choices[0].message.content
        return None
    except Exception as e:
        print(f"Ошибка при генерации объяснения для темы '{topic}': {e}")
        return None


def sanitize_filename(topic: str) -> str:
    """Создает безопасное имя файла из темы."""
    # Заменяем небезопасные символы на подчеркивания
    safe_name = topic.replace('/', '_').replace('\\', '_').replace(':', '_')
    safe_name = safe_name.replace('?', '_').replace('*', '_').replace('"', '_')
    safe_name = safe_name.replace('<', '_').replace('>', '_').replace('|', '_')
    # Ограничиваем длину имени файла
    if len(safe_name) > MAX_FILENAME_LENGTH:
        safe_name = safe_name[:MAX_FILENAME_LENGTH]
    return safe_name


def create_notebook() -> nbformat.NotebookNode:
    """Создает новый Jupyter Notebook."""
    return new_notebook()


def add_topic_to_notebook(notebook: nbformat.NotebookNode, topic: str, explanation: str):
    """Добавляет тему и объяснение как markdown ячейку в notebook."""
    # Форматируем содержимое ячейки
    cell_content = f"# {topic}\n\n{explanation}"
    
    # Создаем markdown ячейку и добавляем в notebook
    cell = new_markdown_cell(cell_content)
    notebook.cells.append(cell)


def save_notebook(notebook: nbformat.NotebookNode, filepath: Path):
    """Сохраняет notebook в файл."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        print(f"✓ Notebook сохранен: {filepath}")
    except Exception as e:
        print(f"Ошибка при сохранении notebook {filepath}: {e}")


def main():
    """Основная функция приложения."""
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    # Проверяем наличие API ключа
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Ошибка: не установлена переменная окружения OPENAI_API_KEY")
        print("Создайте файл .env и добавьте в него: OPENAI_API_KEY=ваш_ключ")
        sys.exit(1)
    
    # Инициализируем клиент OpenAI
    client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/", timeout=900.0)
    
    # Читаем системный промпт
    print("Читаем системный промпт...")
    system_prompt = read_file('system_prompt.txt')
    print(f"✓ Системный промпт загружен ({len(system_prompt)} символов)")
    
    # Читаем список тем
    print("\nЧитаем список тем...")
    topics = read_topics('topics.txt')
    print(f"✓ Загружено тем: {len(topics)}")
    
    # Загружаем прогресс
    print("\nЗагружаем прогресс...")
    progress = load_progress()
    processed_topics = set(progress.get('processed_topics', []))
    print(f"✓ Ранее обработано тем: {len(processed_topics)}")
    
    # Создаем или загружаем notebook
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    notebook_path = output_dir / 'explanations.ipynb'
    
    if notebook_path.exists():
        print(f"✓ Загружаем существующий notebook: {notebook_path}")
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
    else:
        print(f"✓ Создаем новый notebook: {notebook_path}")
        notebook = create_notebook()
    
    # Генерируем объяснения для каждой темы
    topics_to_process = [t for t in topics if t not in processed_topics]
    
    if not topics_to_process:
        print("\n✓ Все темы уже обработаны!")
        print(f"✓ Результаты сохранены в: {notebook_path}")
        return
    
    print(f"\nГенерация объяснений для {len(topics_to_process)} новых тем:")
    print("-" * 80)
    
    for i, topic in enumerate(topics_to_process, start=1):
        print(f"\n[{i}/{len(topics_to_process)}] Генерируем объяснение для: {topic}")
        
        explanation = generate_explanation(client, system_prompt, topic)
        
        if explanation:
            # Добавляем в notebook
            add_topic_to_notebook(notebook, topic, explanation)
            
            # Сохраняем notebook
            save_notebook(notebook, notebook_path)
            
            # Обновляем прогресс
            processed_topics.add(topic)
            progress['processed_topics'] = list(processed_topics)
            progress['last_index'] = len(processed_topics)
            save_progress(progress)
            
            print(f"✓ Тема добавлена в notebook и прогресс сохранен")
        else:
            print(f"✗ Не удалось сгенерировать объяснение для темы: {topic}")
    
    print("\n" + "=" * 80)
    print(f"✓ Завершено! Обработано тем: {len(processed_topics)}")
    print(f"✓ Результаты сохранены в: {notebook_path}")


if __name__ == "__main__":
    main()
