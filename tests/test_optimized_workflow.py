#!/usr/bin/env python3
"""
Тест для проверки оптимизированного workflow без реальных API вызовов.
Этот тест симулирует полный процесс генерации notebook с критикой и кодом.
"""

import sys
from pathlib import Path
import tempfile
import nbformat

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import save_complete_notebook


def test_optimized_workflow():
    """
    Тест оптимизированного workflow:
    1. Генерируем объяснение (мок-данные)
    2. Генерируем критику (мок-данные)
    3. Генерируем код (мок-данные)
    4. Сохраняем все за один раз
    """
    print("=" * 80)
    print("ТЕСТ ОПТИМИЗИРОВАННОГО WORKFLOW")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Шаг 1: Симуляция генерации объяснения (Primary LLM)
        print("\n[1/4] Симуляция генерации объяснения с Gemini...")
        code = "test_topic"
        explanation = """# Тестовая тема: Dense Passage Retrieval (DPR)

## Что это такое?

DPR - это метод информационного поиска, который использует нейронные сети
для кодирования запросов и документов в векторное пространство.

## Как это работает?

1. Обучение двух энкодеров (query и passage)
2. Кодирование всех документов заранее
3. Быстрый поиск ближайших соседей

## Преимущества

- Семантический поиск - понимание смысла
- Высокая точность
- Быстрый инференс"""
        print("  ✓ Объяснение сгенерировано")
        
        # Шаг 2: Симуляция генерации критики (Secondary LLM)
        print("\n[2/4] Симуляция генерации критики с OpenAI...")
        critique = """### Сильные стороны

✅ Четкая структура с подзаголовками
✅ Описаны основные концепции
✅ Упомянуты практические преимущества

### Возможные улучшения

⚠️ Можно добавить больше деталей о процессе обучения
⚠️ Стоит упомянуть ограничения метода
⚠️ Полезно добавить сравнение с альтернативами"""
        print("  ✓ Критика сгенерирована")
        
        # Шаг 3: Симуляция генерации кода (Secondary LLM)
        print("\n[3/4] Симуляция генерации кода с OpenAI...")
        code_example = """# Пример использования DPR с библиотекой transformers
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
import torch

# Инициализация энкодеров и токенизаторов
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# Кодирование документов
documents = ["Python - это язык программирования", "JavaScript используется для веба"]
context_inputs = context_tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
context_embeddings = context_encoder(**context_inputs).pooler_output

# Кодирование запроса
query = "Что такое Python?"
question_inputs = question_tokenizer(query, return_tensors='pt')
question_embedding = question_encoder(**question_inputs).pooler_output

# Вычисление сходства
similarities = torch.matmul(question_embedding, context_embeddings.T)
print(f"Наиболее релевантный документ: {documents[similarities.argmax()]}")"""
        print("  ✓ Код-пример сгенерирован")
        
        # Шаг 4: Сохранение полного notebook за один раз (оптимизация!)
        print("\n[4/4] Сохранение полного notebook за один раз...")
        filepath = save_complete_notebook(output_dir, code, explanation, critique, code_example)
        
        if not filepath:
            print("  ✗ ОШИБКА: не удалось создать notebook")
            return False
        
        # Проверка структуры созданного notebook
        print("\n" + "=" * 80)
        print("ПРОВЕРКА РЕЗУЛЬТАТА")
        print("=" * 80)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        print(f"\nВсего ячеек в notebook: {len(nb.cells)}")
        
        # Проверяем структуру
        expected_cells = 4  # объяснение + критика + заголовок кода + код
        if len(nb.cells) != expected_cells:
            print(f"  ✗ ОШИБКА: ожидалось {expected_cells} ячеек, получено {len(nb.cells)}")
            return False
        
        print("\nСтруктура ячеек:")
        for i, cell in enumerate(nb.cells, 1):
            cell_type = cell.cell_type.upper()
            preview = cell.source[:60].replace('\n', ' ')
            print(f"  [{i}] {cell_type:8} - {preview}...")
        
        # Проверяем наличие ключевого контента
        checks = [
            ("Объяснение", "Dense Passage Retrieval" in nb.cells[0].source),
            ("Критика", "Критический анализ" in nb.cells[1].source),
            ("Заголовок кода", "Пример кода" in nb.cells[2].source),
            ("Код", nb.cells[3].cell_type == 'code' and 'transformers' in nb.cells[3].source),
        ]
        
        print("\nПроверка содержимого:")
        all_ok = True
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}")
            all_ok = all_ok and check_result
        
        if not all_ok:
            print("\n✗ Некоторые проверки не прошли")
            return False
        
        print("\n" + "=" * 80)
        print("✓ ОПТИМИЗИРОВАННЫЙ WORKFLOW РАБОТАЕТ КОРРЕКТНО!")
        print("=" * 80)
        print("\nПреимущества нового подхода:")
        print("  • Одна операция записи вместо двух")
        print("  • Нет промежуточного чтения файла")
        print("  • Нет риска ошибок кодировки при чтении")
        print("  • Более чистый и понятный код")
        
        return True


if __name__ == "__main__":
    success = test_optimized_workflow()
    sys.exit(0 if success else 1)
