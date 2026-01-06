#!/usr/bin/env python3
"""
Генератор объяснений сложных технических тем с использованием LLM.

Приложение:
1. Читает список тем из файла topics.txt
2. Читает системный промпт из файла system_prompt.txt
3. Генерирует объяснение для каждой темы используя OpenAI API
4. Сохраняет каждое объяснение в отдельный файл
"""

import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

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


def save_explanation(output_dir: Path, topic: str, explanation: str, index: int):
    """Сохраняет объяснение в файл."""
    filename = f"{index:02d}_{sanitize_filename(topic)}.txt"
    filepath = output_dir / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Тема: {topic}\n")
            f.write("=" * 80 + "\n\n")
            f.write(explanation)
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
    
    # Создаем директорию для выходных файлов
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Директория для сохранения: {output_dir}")
    
    # Генерируем объяснения для каждой темы
    print(f"\nГенерация объяснений для {len(topics)} тем:")
    print("-" * 80)
    
    for i, topic in enumerate(topics, start=1):
        print(f"\n[{i}/{len(topics)}] Генерируем объяснение для: {topic}")
        
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
