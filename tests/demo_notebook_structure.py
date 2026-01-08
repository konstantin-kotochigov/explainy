#!/usr/bin/env python3
"""
Демонстрационный скрипт для проверки работы двух LLM без реальных API вызовов.
Создает пример notebook с мок-данными, чтобы показать структуру результата.
"""

import sys
from pathlib import Path
import tempfile
import nbformat

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import save_explanation, enhance_notebook


def demonstrate_notebook_structure():
    """Демонстрирует структуру сгенерированного notebook."""
    print("=" * 80)
    print("ДЕМОНСТРАЦИЯ СТРУКТУРЫ NOTEBOOK С ДВУМЯ LLM")
    print("=" * 80)
    
    # Создаем временную директорию
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Шаг 1: Primary LLM (Gemini) генерирует объяснение
        print("\n[1/3] Primary LLM (Gemini) генерирует объяснение...")
        code = "demo_topic"
        explanation = """# DPR (2020)

**Dense Passage Retrieval** - это подход к информационному поиску, использующий нейронные сети для кодирования запросов и документов в векторное пространство.

## Архитектура

DPR использует two-tower архитектуру с двумя BERT энкодерами:
- **Query Encoder**: кодирует пользовательский запрос
- **Passage Encoder**: кодирует текстовые документы

## Алгоритм работы

1. Обучение энкодеров на парах (запрос, релевантный документ)
2. Кодирование всех документов в базе
3. Быстрый поиск ближайших соседей в векторном пространстве

## Преимущества

- Семантический поиск (понимание смысла, а не только ключевых слов)
- Высокая точность поиска
- Быстрый инференс после предварительной индексации"""
        
        filepath = save_explanation(output_dir, code, explanation)
        print(f"  ✓ Создан базовый notebook: {filepath.name}")
        
        # Шаг 2: Secondary LLM (OpenAI) генерирует критику
        print("\n[2/3] Secondary LLM (OpenAI) генерирует критический анализ...")
        critique = """### Сильные стороны

1. ✅ Четко описана архитектура two-tower с разделением энкодеров
2. ✅ Приведен алгоритм работы по шагам
3. ✅ Указаны ключевые преимущества подхода

### Области для улучшения

1. 📝 Добавить информацию о датасетах для обучения (например, MS MARCO, Natural Questions)
2. 📝 Указать конкретные метрики качества (Recall@20, MRR)
3. 📝 Упомянуть сравнение с BM25 и другими baseline методами
4. 📝 Добавить информацию о размере моделей и требованиях к ресурсам

### Рекомендации

Объяснение хорошо структурировано и понятно. Для полноты картины рекомендуется добавить количественные результаты экспериментов и сравнение с предыдущими подходами."""
        
        print("  ✓ Критика сгенерирована")
        
        # Шаг 3: Secondary LLM (OpenAI) генерирует код-пример
        print("\n[3/3] Secondary LLM (OpenAI) генерирует Python код-пример...")
        code_example = """# Пример использования DPR с библиотекой Hugging Face

from transformers import DPRQuestionEncoder, DPRContextEncoder
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch
import numpy as np

# Инициализация энкодеров
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Запрос пользователя
query = "Что такое квантовая запутанность?"

# База документов
documents = [
    "Квантовая запутанность - это физическое явление, при котором квантовые состояния двух объектов оказываются взаимозависимыми",
    "Машинное обучение использует статистические методы для обучения компьютерных систем",
    "Квантовые компьютеры используют кубиты для выполнения вычислений"
]

# Кодирование запроса
query_input = question_tokenizer(query, return_tensors='pt')
query_embedding = question_encoder(**query_input).pooler_output

# Кодирование документов
doc_embeddings = []
for doc in documents:
    doc_input = context_tokenizer(doc, return_tensors='pt', padding=True, truncation=True)
    doc_embedding = context_encoder(**doc_input).pooler_output
    doc_embeddings.append(doc_embedding)

doc_embeddings = torch.cat(doc_embeddings)

# Вычисление схожести (dot product)
similarities = torch.matmul(query_embedding, doc_embeddings.T)

# Поиск наиболее релевантного документа
top_doc_idx = torch.argmax(similarities).item()
print(f"Наиболее релевантный документ: {documents[top_doc_idx]}")
print(f"Similarity score: {similarities[0][top_doc_idx]:.4f}")"""
        
        print("  ✓ Код-пример сгенерирован")
        
        # Добавляем критику и код в notebook
        print("\n[4/4] Интеграция улучшений в notebook...")
        enhance_notebook(filepath, critique, code_example)
        print("  ✓ Notebook улучшен")
        
        # Показываем структуру финального notebook
        print("\n" + "=" * 80)
        print("СТРУКТУРА ФИНАЛЬНОГО NOTEBOOK")
        print("=" * 80)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        print(f"\nВсего ячеек: {len(nb.cells)}")
        print("\nСодержимое:")
        
        for i, cell in enumerate(nb.cells, 1):
            print(f"\n[Ячейка {i}] Тип: {cell.cell_type.upper()}")
            
            if cell.cell_type == 'markdown':
                # Показываем первые 100 символов
                preview = cell.source[:100].replace('\n', ' ')
                print(f"  Превью: {preview}...")
            else:
                # Для кода показываем первые строки
                lines = cell.source.split('\n')
                print(f"  Строк кода: {len(lines)}")
                print(f"  Первая строка: {lines[0] if lines else 'пусто'}")
        
        print("\n" + "=" * 80)
        print("✓ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
        print("=" * 80)
        
        print("\nОписание структуры:")
        print("  1. Markdown: Основное объяснение от Gemini")
        print("  2. Markdown: 📝 Критический анализ от OpenAI")
        print("  3. Markdown: 💻 Заголовок примера кода")
        print("  4. Code: Python код-пример от OpenAI")
        
        return True


def main():
    """Запуск демонстрации."""
    try:
        result = demonstrate_notebook_structure()
        return 0 if result else 1
    except Exception as e:
        print(f"\n✗ Ошибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
