#!/usr/bin/env python3
"""
Тест для проверки функции загрузки изображений.
"""

import sys
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch
import os

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import download_images, capitalize_first_letter


def test_download_images_without_api_keys():
    """Тест функции download_images без API ключей."""
    print("Тест 1: download_images без API ключей")
    
    # Убеждаемся, что API ключи не установлены
    old_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
    old_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    
    if old_api_key:
        del os.environ['GOOGLE_SEARCH_API_KEY']
    if old_engine_id:
        del os.environ['GOOGLE_SEARCH_ENGINE_ID']
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = download_images("test_code", "test query", output_dir)
            
            if result is None:
                print("  ✓ Функция корректно возвращает None при отсутствии API ключей")
                return True
            else:
                print(f"  ✗ Ожидался None, получено: {result}")
                return False
    finally:
        # Восстанавливаем старые значения
        if old_api_key:
            os.environ['GOOGLE_SEARCH_API_KEY'] = old_api_key
        if old_engine_id:
            os.environ['GOOGLE_SEARCH_ENGINE_ID'] = old_engine_id


def test_download_images_with_mock_api():
    """Тест функции download_images с мок API."""
    print("\nТест 2: download_images с мок Google API")
    
    # Устанавливаем тестовые API ключи
    os.environ['GOOGLE_SEARCH_API_KEY'] = 'test_key'
    os.environ['GOOGLE_SEARCH_ENGINE_ID'] = 'test_engine_id'
    
    try:
        # Создаем временную директорию для теста
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                # Мокаем requests.get для имитации успешного ответа API
                with patch('main.requests.get') as mock_get:
                    # Мокаем первый запрос к Search API
                    mock_search_response = Mock()
                    mock_search_response.json.return_value = {
                        'items': [
                            {'link': 'https://example.com/img1.jpg'},
                            {'link': 'https://example.com/img2.jpg'},
                        ]
                    }
                    mock_search_response.raise_for_status = Mock()
                    
                    # Мокаем запросы к изображениям
                    mock_img_response = Mock()
                    mock_img_response.iter_content = lambda chunk_size: [b'fake_image_data']
                    mock_img_response.raise_for_status = Mock()
                    
                    # Настраиваем mock_get для возврата разных ответов
                    def side_effect(*args, **kwargs):
                        if 'customsearch' in args[0]:
                            return mock_search_response
                        else:
                            return mock_img_response
                    
                    mock_get.side_effect = side_effect
                    
                    # Вызываем функцию
                    output_dir = Path(tmpdir)
                    result = download_images("ml", "machine learning", output_dir)
                    
                    # Проверяем результат
                    if result is None:
                        print("  ✗ Функция вернула None вместо пути к директории")
                        return False
                    
                    result_path = Path(result)
                    if not result_path.exists():
                        print(f"  ✗ Директория не создана: {result}")
                        return False
                    
                    print(f"  ✓ Директория создана: {result}")
                    
                    # Проверяем наличие файлов изображений (любое расширение)
                    img_files = list(result_path.glob("img*"))
                    if len(img_files) == 0:
                        print("  ✗ Файлы изображений не созданы")
                        return False
                    
                    print(f"  ✓ Создано файлов изображений: {len(img_files)}")
                    
                    return True
                    
            finally:
                os.chdir(old_cwd)
                
    finally:
        # Удаляем тестовые переменные окружения
        if 'GOOGLE_SEARCH_API_KEY' in os.environ:
            del os.environ['GOOGLE_SEARCH_API_KEY']
        if 'GOOGLE_SEARCH_ENGINE_ID' in os.environ:
            del os.environ['GOOGLE_SEARCH_ENGINE_ID']


def test_download_images_directory_structure():
    """Тест структуры директорий для изображений."""
    print("\nТест 3: Проверка структуры директорий")
    
    # Устанавливаем тестовые API ключи
    os.environ['GOOGLE_SEARCH_API_KEY'] = 'test_key'
    os.environ['GOOGLE_SEARCH_ENGINE_ID'] = 'test_engine_id'
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                with patch('main.requests.get') as mock_get:
                    # Мокаем ответ API
                    mock_search_response = Mock()
                    mock_search_response.json.return_value = {
                        'items': [{'link': 'https://example.com/img1.jpg'}]
                    }
                    mock_search_response.raise_for_status = Mock()
                    
                    mock_img_response = Mock()
                    mock_img_response.iter_content = lambda chunk_size: [b'test']
                    mock_img_response.raise_for_status = Mock()
                    
                    def side_effect(*args, **kwargs):
                        if 'customsearch' in args[0]:
                            return mock_search_response
                        else:
                            return mock_img_response
                    
                    mock_get.side_effect = side_effect
                    
                    # Тестируем с кодовым именем
                    code = "test_code"
                    image_query = "information retrieval diagram"
                    output_dir = Path(tmpdir)
                    result = download_images(code, image_query, output_dir)
                    
                    if result is None:
                        print("  ✗ Функция вернула None")
                        return False
                    
                    # Ожидаем капитализированные названия: 'Img' вместо 'img' и 'Test_code' вместо 'test_code'
                    expected_path = output_dir / 'Img' / capitalize_first_letter(code)
                    
                    if str(expected_path) != result:
                        print(f"  ✗ Неверный путь. Ожидалось: {expected_path}, получено: {result}")
                        return False
                    
                    if not expected_path.exists():
                        print(f"  ✗ Директория не создана: {expected_path}")
                        return False
                    
                    print(f"  ✓ Директория создана с правильным именем: {expected_path}")
                    
                    return True
                    
            finally:
                os.chdir(old_cwd)
                
    finally:
        if 'GOOGLE_SEARCH_API_KEY' in os.environ:
            del os.environ['GOOGLE_SEARCH_API_KEY']
        if 'GOOGLE_SEARCH_ENGINE_ID' in os.environ:
            del os.environ['GOOGLE_SEARCH_ENGINE_ID']


def main():
    """Запуск всех тестов."""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ФУНКЦИИ ЗАГРУЗКИ ИЗОБРАЖЕНИЙ")
    print("=" * 80)
    
    tests = [
        test_download_images_without_api_keys,
        test_download_images_with_mock_api,
        test_download_images_directory_structure,
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
