#!/usr/bin/env python3
"""
Генератор объяснений сложных технических тем с использованием LLM.

Приложение:
1. Читает список тем из файла topics.txt
2. Читает системный промпт из файла system_prompt.txt
3. Генерирует объяснение для каждой темы используя OpenAI API
4. Сохраняет каждое объяснение в отдельный Jupyter Notebook (.ipynb)
"""

import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell
import requests

# Константы
MAX_FILENAME_LENGTH = 100


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


def download_images(topic: str) -> str | None:
    """
    Загружает изображения для темы используя Google Custom Search API.
    
    Args:
        topic: Тема для поиска изображений
        
    Returns:
        Путь к директории с загруженными изображениями или None в случае ошибки
    """
    # Получаем API ключ и ID поискового движка из переменных окружения
    api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
    search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    
    if not api_key or not search_engine_id:
        print("  ⚠ Google Custom Search API не настроен (пропущены GOOGLE_SEARCH_API_KEY или GOOGLE_SEARCH_ENGINE_ID)")
        return None
    
    try:
        # Формируем запрос к Google Custom Search API
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': search_engine_id,
            'q': f"{topic} imagesize:large",
            'searchType': 'image',
            'num': 10  # Максимум 10 изображений за запрос
        }
        
        print(f"  Поиск изображений для темы: {topic}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Проверяем наличие результатов
        if 'items' not in data or len(data['items']) == 0:
            print("  ⚠ Изображения не найдены")
            return None
        
        # Создаем директорию для изображений
        safe_topic_name = sanitize_filename(topic)
        img_dir = Path('img') / safe_topic_name
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем изображения
        downloaded_count = 0
        for i, item in enumerate(data['items'], start=1):
            try:
                img_url = item['link']
                img_response = requests.get(img_url, timeout=10, stream=True)
                img_response.raise_for_status()
                
                # Определяем расширение файла из URL или Content-Type
                file_ext = '.png'  # По умолчанию
                if '.' in img_url.split('/')[-1]:
                    url_ext = '.' + img_url.split('.')[-1].split('?')[0].lower()
                    # Проверяем, что это известное расширение изображения
                    if url_ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                        file_ext = url_ext
                
                # Сохраняем изображение
                img_path = img_dir / f"img{i}{file_ext}"
                with open(img_path, 'wb') as f:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded_count += 1
            except Exception as e:
                print(f"  ⚠ Не удалось загрузить изображение {i}: {e}")
                continue
        
        if downloaded_count > 0:
            print(f"  ✓ Загружено изображений: {downloaded_count} в {img_dir}")
            return str(img_dir)
        else:
            print("  ⚠ Не удалось загрузить ни одного изображения")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"  ⚠ Ошибка при запросе к Google Custom Search API: {e}")
        return None
    except Exception as e:
        print(f"  ⚠ Неожиданная ошибка при загрузке изображений: {e}")
        return None


def save_explanation(output_dir: Path, topic: str, explanation: str, index: int):
    """Сохраняет объяснение в Jupyter Notebook."""
    filename = f"{index:02d}_{sanitize_filename(topic)}.ipynb"
    filepath = output_dir / filename
    
    try:
        # Создаем новый notebook
        nb = new_notebook()
        
        # Добавляем одну Markdown ячейку с объяснением
        nb.cells.append(new_markdown_cell(explanation))
        
        # Сохраняем notebook
        with open(filepath, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"✓ Сохранено: {filepath}")
    except Exception as e:
        print(f"Ошибка при сохранении файла {filepath}: {e}")


def main():
    """Основная функция приложения."""
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    # Проверяем наличие API ключа
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Ошибка: не установлена переменная окружения GOOGLE_API_KEY")
        print("Создайте файл .env и добавьте в него: GOOGLE_API_KEY=ваш_ключ")
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
    
    # Создаем директорию для выходных файлов
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Директория для сохранения: {output_dir}")
    
    # Генерируем объяснения для каждой темы
    print(f"\nГенерация объяснений для {len(topics)} тем:")
    print("-" * 80)
    
    for i, topic in enumerate(topics, start=1):
        print(f"\n[{i}/{len(topics)}] Генерируем объяснение для: {topic}")
        
        # Загружаем изображения для темы
        # TODO: В будущем можно использовать img_dir для встраивания изображений в notebook
        img_dir = download_images(topic)
        
        explanation = generate_explanation(client, system_prompt, topic)
        
        if explanation:
            save_explanation(output_dir, topic, explanation, i)
        else:
            print(f"✗ Не удалось сгенерировать объяснение для темы: {topic}")
    
    print("\n" + "=" * 80)
    print(f"✓ Завершено! Обработано тем: {len(topics)}")
    print(f"✓ Результаты сохранены в директории: {output_dir}")


if __name__ == "__main__":
    main()
