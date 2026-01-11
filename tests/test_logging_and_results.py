#!/usr/bin/env python3
"""
Тесты для проверки функциональности логирования и сохранения результатов.
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    save_results, load_results, update_result,
    log_processing
)


def test_save_and_load_results():
    """Тест сохранения и загрузки результатов."""
    print("=" * 80)
    print("ТЕСТ 1: Сохранение и загрузка результатов")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results_file = Path(tmpdir) / 'results.json'
        
        # Создаем тестовые данные
        test_data = {
            'prf': {
                'model': 'gemini-3-preview',
                'status': 'success',
                'last_updated': datetime.now().isoformat()
            },
            'dpr': {
                'model': 'gemini-3-preview',
                'status': 'failed',
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # Сохраняем результаты
        print("\n[1/3] Сохранение результатов...")
        success = save_results(results_file, test_data)
        if not success:
            print("  ✗ ОШИБКА: не удалось сохранить результаты")
            return False
        print("  ✓ Результаты сохранены")
        
        # Проверяем, что файл создан
        if not results_file.exists():
            print("  ✗ ОШИБКА: файл результатов не создан")
            return False
        print(f"  ✓ Файл создан: {results_file}")
        
        # Загружаем результаты обратно
        print("\n[2/3] Загрузка результатов...")
        loaded_data = load_results(results_file)
        if not loaded_data:
            print("  ✗ ОШИБКА: не удалось загрузить результаты")
            return False
        print(f"  ✓ Результаты загружены: {len(loaded_data)} записей")
        
        # Проверяем содержимое
        print("\n[3/3] Проверка содержимого...")
        if 'prf' not in loaded_data or 'dpr' not in loaded_data:
            print("  ✗ ОШИБКА: не все записи загружены")
            return False
        
        if loaded_data['prf']['status'] != 'success':
            print("  ✗ ОШИБКА: неверный статус для prf")
            return False
        
        if loaded_data['dpr']['status'] != 'failed':
            print("  ✗ ОШИБКА: неверный статус для dpr")
            return False
        
        print("  ✓ Все данные корректны")
        print(f"\n  Содержимое файла результатов:")
        for code, info in loaded_data.items():
            print(f"    • {code}: {info['model']}, {info['status']}, {info['last_updated'][:19]}")
        
        print("\n✓ ТЕСТ 1 ПРОЙДЕН")
        return True


def test_update_result():
    """Тест обновления результатов."""
    print("\n" + "=" * 80)
    print("ТЕСТ 2: Обновление результатов")
    print("=" * 80)
    
    # Создаем пустой словарь результатов
    results_data = {}
    
    print("\n[1/3] Добавление первой записи...")
    update_result(results_data, 'colbert', 'gemini-3-preview', 'success')
    if 'colbert' not in results_data:
        print("  ✗ ОШИБКА: запись не добавлена")
        return False
    print(f"  ✓ Запись добавлена: {results_data['colbert']}")
    
    print("\n[2/3] Добавление второй записи...")
    update_result(results_data, 'rag', 'gemini-3-preview', 'failed')
    if 'rag' not in results_data:
        print("  ✗ ОШИБКА: запись не добавлена")
        return False
    print(f"  ✓ Запись добавлена: {results_data['rag']}")
    
    print("\n[3/3] Обновление существующей записи...")
    old_time = results_data['colbert']['last_updated']
    import time
    time.sleep(0.1)  # Небольшая задержка для изменения времени
    update_result(results_data, 'colbert', 'gpt-4o', 'failed')
    new_time = results_data['colbert']['last_updated']
    
    if results_data['colbert']['model'] != 'gpt-4o':
        print("  ✗ ОШИБКА: модель не обновлена")
        return False
    
    if results_data['colbert']['status'] != 'failed':
        print("  ✗ ОШИБКА: статус не обновлен")
        return False
    
    if old_time == new_time:
        print("  ✗ ОШИБКА: время не обновлено")
        return False
    
    print(f"  ✓ Запись обновлена: {results_data['colbert']}")
    print(f"    Старое время: {old_time[:19]}")
    print(f"    Новое время:  {new_time[:19]}")
    
    print("\n✓ ТЕСТ 2 ПРОЙДЕН")
    return True


def test_log_processing():
    """Тест логирования обработки."""
    print("\n" + "=" * 80)
    print("ТЕСТ 3: Логирование обработки")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / 'processing.log'
        
        # Записываем несколько логов
        print("\n[1/3] Запись логов...")
        entries = [
            ('Pseudo-Relevance Feedback', 'gemini-3-preview', 1234, 'success'),
            ('Deep Passage Retrieval', 'gemini-3-preview', 2345, 'success'),
            ('ColBERT', 'gemini-3-preview', 0, 'failed'),
        ]
        
        for topic, model, tokens, status in entries:
            success = log_processing(log_file, topic, model, tokens, status)
            if not success:
                print(f"  ✗ ОШИБКА: не удалось записать лог для {topic}")
                return False
        
        print(f"  ✓ Записано {len(entries)} логов")
        
        # Проверяем, что файл создан
        print("\n[2/3] Проверка создания файла...")
        if not log_file.exists():
            print("  ✗ ОШИБКА: файл логов не создан")
            return False
        print(f"  ✓ Файл создан: {log_file}")
        
        # Читаем и проверяем содержимое
        print("\n[3/3] Проверка содержимого логов...")
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) != len(entries):
            print(f"  ✗ ОШИБКА: ожидалось {len(entries)} строк, получено {len(lines)}")
            return False
        
        print(f"  ✓ Количество строк корректно: {len(lines)}")
        
        # Проверяем формат каждой строки
        print("\n  Содержимое лог-файла:")
        for i, line in enumerate(lines, 1):
            parts = line.strip().split('\t')
            if len(parts) != 5:
                print(f"  ✗ ОШИБКА: строка {i} имеет неверный формат (ожидается 5 полей)")
                return False
            
            timestamp, topic, model, tokens, status = parts
            print(f"    {i}. {timestamp[:19]} | {topic[:30]:30} | {model:20} | {tokens:6} tokens | {status}")
            
            # Проверяем, что timestamp - валидная ISO дата
            try:
                datetime.fromisoformat(timestamp)
            except ValueError:
                print(f"  ✗ ОШИБКА: неверный формат временной метки в строке {i}")
                return False
        
        print("\n  ✓ Все записи корректны")
        
        # Проверяем режим append
        print("\n[4/4] Проверка режима append...")
        log_processing(log_file, 'Test Topic', 'test-model', 999, 'success')
        
        with open(log_file, 'r', encoding='utf-8') as f:
            new_lines = f.readlines()
        
        if len(new_lines) != len(lines) + 1:
            print(f"  ✗ ОШИБКА: режим append не работает")
            return False
        
        print(f"  ✓ Режим append работает корректно")
        
        print("\n✓ ТЕСТ 3 ПРОЙДЕН")
        return True


def test_load_nonexistent_file():
    """Тест загрузки несуществующего файла результатов."""
    print("\n" + "=" * 80)
    print("ТЕСТ 4: Загрузка несуществующего файла")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results_file = Path(tmpdir) / 'nonexistent.json'
        
        print("\n[1/1] Загрузка несуществующего файла...")
        loaded_data = load_results(results_file)
        
        if loaded_data != {}:
            print(f"  ✗ ОШИБКА: ожидался пустой словарь, получено: {loaded_data}")
            return False
        
        print("  ✓ Возвращен пустой словарь (ожидаемое поведение)")
        
        print("\n✓ ТЕСТ 4 ПРОЙДЕН")
        return True


def test_status_validation():
    """Тест валидации статусов."""
    print("\n" + "=" * 80)
    print("ТЕСТ 5: Валидация статусов")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / 'test.log'
        results_data = {}
        
        print("\n[1/3] Валидация корректных статусов...")
        success1 = log_processing(log_file, 'Test', 'model', 100, 'success')
        success2 = log_processing(log_file, 'Test', 'model', 100, 'failed')
        
        if not success1 or not success2:
            print("  ✗ ОШИБКА: корректные статусы отклонены")
            return False
        print("  ✓ Корректные статусы приняты")
        
        print("\n[2/3] Валидация некорректного статуса в log_processing...")
        success3 = log_processing(log_file, 'Test', 'model', 100, 'invalid_status')
        
        if success3:
            print("  ✗ ОШИБКА: некорректный статус принят")
            return False
        print("  ✓ Некорректный статус отклонен")
        
        print("\n[3/3] Валидация некорректного статуса в update_result...")
        initial_count = len(results_data)
        update_result(results_data, 'test', 'model', 'invalid_status')
        
        if len(results_data) != initial_count:
            print("  ✗ ОШИБКА: некорректный статус добавлен в результаты")
            return False
        print("  ✓ Некорректный статус не добавлен в результаты")
        
        print("\n✓ ТЕСТ 5 ПРОЙДЕН")
        return True


def main():
    """Запуск всех тестов."""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ФУНКЦИОНАЛЬНОСТИ ЛОГИРОВАНИЯ И РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    tests = [
        test_save_and_load_results,
        test_update_result,
        test_log_processing,
        test_load_nonexistent_file,
        test_status_validation,
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
