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
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import requests


# Configuration Constants
# API timeouts (in seconds)
GEMINI_API_TIMEOUT = 900.0  # 15 minutes for primary LLM generation
OPENAI_API_TIMEOUT = 120.0  # 2 minutes for secondary LLM (critique and code)
GOOGLE_SEARCH_TIMEOUT = 30  # 30 seconds for Google Custom Search API
IMAGE_DOWNLOAD_TIMEOUT = 10  # 10 seconds per image download

# Image search settings
MAX_IMAGES_PER_QUERY = 5  # Maximum number of images to download per topic

# LLM model names
PRIMARY_MODEL = "gemini-2.5-flash"  # Google Gemini model for main content
SECONDARY_MODEL = "gpt-4o-mini"  # OpenAI model for critique and code generation

# LLM generation parameters
CRITIQUE_TEMPERATURE = 0.2  # Temperature for critique generation
CODE_TEMPERATURE = 0.2  # Temperature for code example generation


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


def read_topics(filepath: str) -> list[dict]:
    """
    Читает список тем из файла в новом формате.
    
    Формат: code;detailed_query;image_query
    
    Returns:
        Список словарей с ключами: code, detailed_query, image_query
    """
    content = read_file(filepath)
    topics = []
    for line_num, line in enumerate(content.split('\n'), start=1):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(';')
        if len(parts) != 3:
            print(f"Предупреждение: неверный формат строки {line_num} (ожидается 3 поля, получено {len(parts)}): {line[:60]}...")
            continue
        
        topics.append({
            'code': parts[0].strip(),
            'detailed_query': parts[1].strip(),
            'image_query': parts[2].strip()
        })
    
    return topics


def generate_explanation(client: OpenAI, system_prompt: str, topic: str) -> str:
    """Генерирует объяснение темы используя OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
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


def download_images(code: str, image_query: str) -> str | None:
    """
    Загружает изображения используя Google Custom Search API.
    
    Args:
        code: Кодовое имя темы для создания директории
        image_query: Поисковый запрос на английском для API поиска изображений
        
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
            'q': image_query,
            'searchType': 'image',
            'num': MAX_IMAGES_PER_QUERY  # Максимум изображений за запрос
        }
        
        print(f"  Поиск изображений по запросу: {image_query}")
        response = requests.get(url, params=params, timeout=GOOGLE_SEARCH_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        # Проверяем наличие результатов
        if 'items' not in data or len(data['items']) == 0:
            print("  ⚠ Изображения не найдены")
            return None
        
        # Создаем директорию для изображений по кодовому имени внутри outputs/img
        img_dir = Path('outputs') / 'img' / code
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем изображения
        downloaded_count = 0
        for i, item in enumerate(data['items'], start=1):
            try:
                img_url = item['link']
                img_response = requests.get(img_url, timeout=IMAGE_DOWNLOAD_TIMEOUT, stream=True)
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


def save_explanation(output_dir: Path, code: str, explanation: str):
    """Сохраняет объяснение в Jupyter Notebook."""
    filename = f"{code}.ipynb"
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
        return filepath
    except Exception as e:
        print(f"Ошибка при сохранении файла {filepath}: {e}")
        return None


def parse_notebook(filepath: Path) -> dict:
    """
    Парсит Jupyter Notebook и возвращает структурированные данные.
    
    Args:
        filepath: Путь к файлу notebook
        
    Returns:
        Словарь с содержимым notebook
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Собираем весь текст из markdown ячеек
        content = []
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                content.append(cell.source)
        
        return {
            'notebook': nb,
            'content': '\n\n'.join(content),
            'cell_count': len(nb.cells)
        }
    except Exception as e:
        print(f"Ошибка при парсинге notebook {filepath}: {e}")
        return None


def generate_critique(client: OpenAI, critic_system_prompt: str, content: str, topic: str) -> str | None:
    """
    Генерирует критику содержимого используя OpenAI API.
    
    Args:
        client: OpenAI клиент для вторичного LLM
        critic_system_prompt: Системный промпт для критика
        content: Содержимое для критики
        topic: Тема объяснения
        
    Returns:
        Текст критики или None в случае ошибки
    """
    critique_prompt = f"""Проанализируй следующее объяснение темы "{topic}". Будь конкретным и конструктивным. Формат ответа - Markdown.
Содержимое для анализа:
{content}"""
    
    try:
        response = client.chat.completions.create(
            model=SECONDARY_MODEL,
            messages=[
                {"role": "system", "content": critic_system_prompt},
                {"role": "user", "content": critique_prompt}
            ],
            temperature=CRITIQUE_TEMPERATURE
        )
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        return None
    except Exception as e:
        print(f"  ⚠ Ошибка при генерации критики: {e}")
        return None


def generate_code_example(client: OpenAI, content: str, topic: str) -> str | None:
    """
    Генерирует Python код-пример используя OpenAI API.
    
    Args:
        client: OpenAI клиент для вторичного LLM
        content: Содержимое объяснения
        topic: Тема объяснения
        
    Returns:
        Python код-пример или None в случае ошибки
    """
    code_prompt = f"""На основе следующего объяснения темы "{topic}", создай один или несколько примеров на Python, которые проиллюстрировали бы основные концепции, описанные в документе.
Если есть готовые реализации бибилотек на базе описываемой модель ии метод, отлично - покажи, как их применять. Если понятно, проще реализовать метод самому, ok, напиши нативный код.
Важно, чтобы код иллюстрировал не абстрактную общую концепцию, а именно специфику данной темы - чтобы было видно отличия метода от его альтренатив.
Не нужно пписать production-level код, достаточно пары простых игрушечных примеров.
НЕ забудь добавить комментарии, объясняющие ключевые моменты.
Верни ТОЛЬКО Python код с комментариями. Формат - Markdown.

Содержимое объяснения:
{content}"""
    
    try:
        response = client.chat.completions.create(
            model=SECONDARY_MODEL,
            messages=[
                {"role": "system", "content": "Ты эксперт Python программист, специализирующийся на AI/ML и Computer Science."},
                {"role": "user", "content": code_prompt}
            ],
            temperature=CODE_TEMPERATURE
        )
        if response.choices and len(response.choices) > 0:
            code = response.choices[0].message.content
            # Убираем markdown форматирование кода если есть
            PYTHON_FENCE = "```python"
            CODE_FENCE = "```"
            
            if code.startswith(PYTHON_FENCE):
                code = code[len(PYTHON_FENCE):].strip()
            elif code.startswith(CODE_FENCE):
                code = code[len(CODE_FENCE):].strip()
            if code.endswith(CODE_FENCE):
                code = code[:-len(CODE_FENCE)].strip()
            return code.strip()
        return None
    except Exception as e:
        print(f"  ⚠ Ошибка при генерации кода: {e}")
        return None


def enhance_notebook(filepath: Path, critique: str, code_example: str) -> bool:
    """
    Добавляет критику и код-пример в существующий notebook.
    
    Args:
        filepath: Путь к файлу notebook
        critique: Текст критики (может быть None или пустой строкой)
        code_example: Python код-пример (может быть None или пустой строкой)
        
    Returns:
        True если успешно, False в случае ошибки
    """
    # Если нет содержимого для добавления, возвращаем успех без изменений
    has_critique = critique and critique.strip()
    has_code = code_example and code_example.strip()
    
    if not has_critique and not has_code:
        return True
    
    try:
        # Читаем существующий notebook
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Добавляем ячейку с критикой
        if has_critique:
            critique_cell = new_markdown_cell(f"## 📝 Критический анализ\n\n{critique}")
            nb.cells.append(critique_cell)
        
        # Добавляем ячейку с кодом
        if has_code:
            code_header = new_markdown_cell("## 💻 Пример кода\n\nИллюстративный Python пример, демонстрирующий основные концепции:")
            nb.cells.append(code_header)
            code_cell = new_code_cell(code_example)
            nb.cells.append(code_cell)
        
        # Сохраняем обновленный notebook
        with open(filepath, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        return True
    except Exception as e:
        print(f"  ⚠ Ошибка при улучшении notebook: {e}")
        return False


def main():
    """Основная функция приложения."""
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    # Проверяем наличие API ключей
    gemini_api_key = os.getenv('GOOGLE_API_KEY')
    if not gemini_api_key:
        print("Ошибка: не установлена переменная окружения GOOGLE_API_KEY")
        print("Создайте файл .env и добавьте в него: GOOGLE_API_KEY=ваш_ключ")
        sys.exit(1)
    
    # Проверяем наличие OpenAI API ключа для критики
    openai_api_key = os.getenv('OPENAI_API_KEY')
    use_critique = bool(openai_api_key)
    
    if not use_critique:
        print("⚠ OPENAI_API_KEY не установлен - критика и генерация кода будут пропущены")
        print("  Для активации: добавьте OPENAI_API_KEY в файл .env")
    
    # Инициализируем клиенты
    gemini_client = OpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/", timeout=GEMINI_API_TIMEOUT)
    openai_client = OpenAI(api_key=openai_api_key, timeout=OPENAI_API_TIMEOUT) if use_critique else None
    
    # Читаем системный промпт
    print("Читаем системный промпт...")
    system_prompt = read_file('system_prompt.txt')
    print(f"✓ Системный промпт загружен ({len(system_prompt)} символов)")
    
    # Читаем системный промпт для критика (только если включена критика)
    critic_system_prompt = None
    if use_critique:
        print("Читаем системный промпт для критика...")
        critic_system_prompt = read_file('critic_system_prompt.txt')
        print(f"✓ Системный промпт критика загружен ({len(critic_system_prompt)} символов)")
    
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
    
    for i, topic_data in enumerate(topics, start=1):
        code = topic_data['code']
        detailed_query = topic_data['detailed_query']
        image_query = topic_data['image_query']
        
        print(f"\n[{i}/{len(topics)}] Генерируем объяснение для: {detailed_query}")
        
        # Загружаем изображения для темы, используя кодовое имя и запрос на английском
        img_dir = download_images(code, image_query)
        
        explanation = generate_explanation(gemini_client, system_prompt, detailed_query)
        
        if explanation:
            filepath = save_explanation(output_dir, code, explanation)
            
            # Если сохранение успешно и включена критика, добавляем улучшения
            if filepath and use_critique:
                print(f"  Генерируем критику и код-примеры...")
                
                # Парсим созданный notebook
                parsed = parse_notebook(filepath)
                if parsed:
                    # Генерируем критику
                    critique = generate_critique(openai_client, critic_system_prompt, parsed['content'], detailed_query)
                    if critique:
                        print(f"  ✓ Критика сгенерирована")
                    
                    # Генерируем код-пример
                    code_example = generate_code_example(openai_client, parsed['content'], detailed_query)
                    if code_example:
                        print(f"  ✓ Код-пример сгенерирован")
                    
                    # Добавляем в notebook
                    if critique or code_example:
                        if enhance_notebook(filepath, critique, code_example):
                            print(f"  ✓ Notebook улучшен с критикой и кодом")
        else:
            print(f"✗ Не удалось сгенерировать объяснение для темы: {detailed_query}")
    
    print("\n" + "=" * 80)
    print(f"✓ Завершено! Обработано тем: {len(topics)}")
    print(f"✓ Результаты сохранены в директории: {output_dir}")
    if use_critique:
        print(f"✓ Notebooks улучшены критикой и примерами кода")


if __name__ == "__main__":
    main()
