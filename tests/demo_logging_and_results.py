#!/usr/bin/env python3
"""
Демонстрационный скрипт для проверки логирования и сохранения результатов
в реалистичном сценарии (без реальных API вызовов).
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
    log_processing, save_complete_notebook
)


def simulate_topic_processing():
    """
    Симулирует обработку нескольких тем с логированием и сохранением результатов.
    """
    print("=" * 80)
    print("ДЕМОНСТРАЦИЯ: Обработка тем с логированием и сохранением результатов")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        results_filepath = output_dir / 'results.json'
        log_filepath = output_dir / 'processing.log'
        
        # Загружаем существующие результаты (будут пустыми)
        results_data = load_results(results_filepath)
        
        # Симулируем обработку нескольких тем
        topics = [
            {'code': 'prf', 'query': 'Pseudo-Relevance Feedback', 'tokens': 1234, 'success': True},
            {'code': 'dpr', 'query': 'Deep Passage Retrieval', 'tokens': 2345, 'success': True},
            {'code': 'colbert', 'query': 'ColBERT', 'tokens': 0, 'success': False},
            {'code': 'rag', 'query': 'Retrieval Augmented Generation', 'tokens': 3456, 'success': True},
        ]
        
        print(f"\nОбработка {len(topics)} тем...")
        print("-" * 80)
        
        for i, topic in enumerate(topics, 1):
            code = topic['code']
            query = topic['query']
            tokens = topic['tokens']
            success = topic['success']
            
            print(f"\n[{i}/{len(topics)}] Обработка: {query}")
            
            if success:
                # Симуляция успешной обработки
                explanation = f"# {query}\n\nЭто тестовое объяснение для темы {query}."
                filepath = save_complete_notebook(output_dir, code, explanation, None, None)
                
                if filepath:
                    print(f"  ✓ Notebook сохранен: {code}.ipynb")
                    
                    # Обновляем результаты и логируем
                    update_result(results_data, code, 'gemini-3-preview', 'success')
                    log_processing(log_filepath, query, 'gemini-3-preview', tokens, 'success')
                    print(f"  ✓ Результат сохранен и залогирован")
            else:
                # Симуляция неудачной обработки
                print(f"  ✗ Не удалось сгенерировать объяснение")
                
                # Обновляем результаты и логируем неудачу
                update_result(results_data, code, 'gemini-3-preview', 'failed')
                log_processing(log_filepath, query, 'gemini-3-preview', tokens, 'failed')
                print(f"  ✓ Неудача залогирована")
        
        # Сохраняем обновленные результаты
        print("\n" + "-" * 80)
        print("Сохранение результатов...")
        save_results(results_filepath, results_data)
        print(f"✓ Результаты сохранены в: {results_filepath}")
        
        # Показываем содержимое файла результатов
        print("\n" + "=" * 80)
        print("СОДЕРЖИМОЕ ФАЙЛА РЕЗУЛЬТАТОВ (results.json)")
        print("=" * 80)
        
        with open(results_filepath, 'r', encoding='utf-8') as f:
            results_json = json.load(f)
        
        print(json.dumps(results_json, indent=2, ensure_ascii=False))
        
        # Показываем содержимое лог-файла
        print("\n" + "=" * 80)
        print("СОДЕРЖИМОЕ ЛОГ-ФАЙЛА (processing.log)")
        print("=" * 80)
        
        with open(log_filepath, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        print("Timestamp\t\t\tТема\t\t\t\tМодель\t\t\tТокены\tСтатус")
        print("-" * 80)
        print(log_content)
        
        # Статистика
        print("=" * 80)
        print("СТАТИСТИКА")
        print("=" * 80)
        
        success_count = sum(1 for r in results_json.values() if r['status'] == 'success')
        failed_count = sum(1 for r in results_json.values() if r['status'] == 'failed')
        total_tokens = sum(topic['tokens'] for topic in topics if topic['success'])
        
        print(f"Успешно обработано: {success_count}")
        print(f"Неудач: {failed_count}")
        print(f"Всего токенов использовано: {total_tokens}")
        
        print("\n" + "=" * 80)
        print("✓ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО")
        print("=" * 80)
        
        print("\nФайлы создаются в директории outputs/:")
        print(f"  • results.json - результаты обработки (режим rewrite)")
        print(f"  • processing.log - лог обработки (режим append)")
        print(f"\nФормат results.json:")
        print(f"  {{\n    \"topic_code\": {{")
        print(f"      \"model\": \"llm-model-name\",")
        print(f"      \"status\": \"success|failed\",")
        print(f"      \"last_updated\": \"ISO-timestamp\"")
        print(f"    }}\n  }}")
        print(f"\nФормат processing.log (табулированные поля):")
        print(f"  timestamp\\ttopic\\tmodel\\ttokens\\tstatus")


if __name__ == "__main__":
    simulate_topic_processing()
