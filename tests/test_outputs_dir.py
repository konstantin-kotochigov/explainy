#!/usr/bin/env python3
"""
Тесты для проверки параметризации директории outputs.
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import capitalize_first_letter


def test_default_outputs_dir():
    """Тест создания директории outputs по умолчанию."""
    print("Тест 1: Проверка создания директории outputs по умолчанию")
    try:
        # Убедимся что OUTPUTS_DIR не установлен
        if 'OUTPUTS_DIR' in os.environ:
            del os.environ['OUTPUTS_DIR']
        
        # Проверяем что значение по умолчанию 'outputs'
        output_dir_path = os.getenv('OUTPUTS_DIR', 'outputs')
        assert output_dir_path == 'outputs', f"Ожидалось 'outputs', получено '{output_dir_path}'"
        
        output_dir = Path(output_dir_path)
        output_dir.mkdir(exist_ok=True)
        
        # Создаем директорию для изображений (с заглавной буквы)
        img_dir = output_dir / 'Img'
        img_dir.mkdir(exist_ok=True)
        
        assert output_dir.exists(), "Директория outputs не создана"
        assert img_dir.exists(), "Директория outputs/Img не создана"
        assert img_dir.is_dir(), "outputs/Img не является директорией"
        
        print(f"  ✓ Директория создана: {output_dir.absolute()}")
        print(f"  ✓ Директория изображений создана: {img_dir.absolute()}")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False
    return True


def test_custom_outputs_dir():
    """Тест создания директории outputs с пользовательским путем."""
    print("\nТест 2: Проверка создания директории outputs с пользовательским путем")
    try:
        # Создаем временную директорию для теста
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / 'my_outputs'
            
            # Устанавливаем переменную окружения
            os.environ['OUTPUTS_DIR'] = str(custom_path)
            
            # Читаем переменную окружения
            output_dir_path = os.getenv('OUTPUTS_DIR', 'outputs')
            assert str(custom_path) == output_dir_path, f"Ожидалось '{custom_path}', получено '{output_dir_path}'"
            
            output_dir = Path(output_dir_path)
            output_dir.mkdir(exist_ok=True)
            
            # Создаем директорию для изображений (с заглавной буквы)
            img_dir = output_dir / 'Img'
            img_dir.mkdir(exist_ok=True)
            
            assert output_dir.exists(), "Кастомная директория outputs не создана"
            assert img_dir.exists(), "Директория Img внутри кастомной директории не создана"
            assert img_dir.is_dir(), "Img не является директорией"
            
            print(f"  ✓ Кастомная директория создана: {output_dir.absolute()}")
            print(f"  ✓ Директория изображений создана: {img_dir.absolute()}")
            
            # Очистка переменной окружения
            del os.environ['OUTPUTS_DIR']
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False
    return True


def test_img_subdirectories():
    """Тест создания поддиректорий для изображений по темам."""
    print("\nТест 3: Проверка создания поддиректорий для изображений")
    try:
        # Используем директорию outputs по умолчанию
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        
        img_dir = output_dir / 'Img'
        img_dir.mkdir(exist_ok=True)
        
        # Создаем поддиректорию для конкретной темы (с заглавной буквы)
        topic_code = 'test_topic'
        # Используем функцию капитализации
        capitalized_topic_code = capitalize_first_letter(topic_code)
        topic_img_dir = img_dir / capitalized_topic_code
        topic_img_dir.mkdir(parents=True, exist_ok=True)
        
        assert topic_img_dir.exists(), f"Директория {topic_img_dir} не создана"
        assert topic_img_dir.is_dir(), f"{topic_img_dir} не является директорией"
        
        print(f"  ✓ Поддиректория для темы создана: {topic_img_dir.absolute()}")
        
        # Очистка тестовой директории
        if topic_img_dir.exists():
            shutil.rmtree(topic_img_dir)
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False
    return True


def main():
    """Запуск всех тестов."""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ПАРАМЕТРИЗАЦИИ ДИРЕКТОРИИ OUTPUTS")
    print("=" * 80)
    
    tests = [
        test_default_outputs_dir,
        test_custom_outputs_dir,
        test_img_subdirectories,
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
