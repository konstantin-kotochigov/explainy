# explainy
Приложение для генерации текстовых объяснений сложных технических тем в формате блог-поста с автоматической критикой и примерами кода

## Описание

Это Python приложение использует два LLM для генерации высококачественных объяснений сложных технических тем:

**Основной процесс:**
1. Читает список тем из файла `topics.txt`
2. Читает промпты из папки `prompts/`
3. Для каждой темы загружает изображения из Google Custom Search API (опционально)
4. Генерирует объяснение используя **Google Gemini** (primary LLM)
5. Анализирует сгенерированное объяснение используя **OpenAI GPT** (secondary LLM) для:
   - Критического анализа содержимого
   - Генерации иллюстративных Python примеров кода
6. Сохраняет полное объяснение с критикой и кодом в Jupyter Notebook (`.ipynb`) в директории `outputs/`

## Архитектура двух LLM

### Primary LLM (Google Gemini)
- **Роль**: Генерация основного содержимого объяснений
- **Модель**: `gemini-2.5-flash`
- **Входные данные**: Промпты из `prompts/main_system_prompt.txt` и `prompts/main_user_prompt.txt`
- **Выходные данные**: Полное техническое объяснение в формате Markdown

### Secondary LLM (OpenAI GPT)
- **Роль**: Критика и улучшение сгенерированного контента
- **Модель**: `gpt-4o-mini`
- **Функции**:
  1. **Критический анализ**: Анализирует сгенерированное объяснение и предоставляет конструктивную критику (промпты из `prompts/critic_system_prompt.txt` и `prompts/critic_user_prompt.txt`)
  2. **Генерация кода**: Создает Python примеры, демонстрирующие объясненные концепции (промпты из `prompts/code_generation_system_prompt.txt` и `prompts/code_generation_prompt.txt`)

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/konstantin-kotochigov/explainy.git
cd explainy
```

2. Создайте виртуальное окружение и активируйте его:
```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте файл `.env` на основе `.env.example`:
```bash
cp .env.example .env
```

5. Добавьте необходимые API ключи в файл `.env`:
```
# Обязательно: Google Gemini API для генерации объяснений
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Обязательно: OpenAI API для критики и генерации кода
OPENAI_API_KEY=sk-your-actual-api-key-here

# Опционально: Директория для сохранения результатов (по умолчанию: outputs)
OUTPUTS_DIR=outputs
```

6. (Опционально) Для загрузки изображений добавьте Google Custom Search API credentials в файл `.env`:
```
GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
```

Для получения Google Custom Search API credentials:
- API ключ: https://developers.google.com/custom-search/v1/introduction
- Создайте поисковый движок: https://programmablesearchengine.google.com/

## Использование

1. Отредактируйте файл `topics.txt` и добавьте темы, которые хотите объяснить
   
   Формат: `код;детальный_запрос;запрос_для_изображений`
   
   Пример:
   ```
   dpr;Методы информационного поиска / метод Deep Passage Retrieval (DPR);information retrieval Deep Passage Retrieval DPR diagram
   ```

2. При необходимости отредактируйте промпты в папке `prompts/` для изменения стиля генерации (см. `prompts/README.md` для деталей)

3. Запустите приложение:
```bash
python main.py
```

4. Результаты будут сохранены в директории `outputs/` в виде отдельных Jupyter Notebooks (`.ipynb`) для каждой темы
   
   Вы можете указать другую директорию для сохранения результатов через переменную окружения `OUTPUTS_DIR` в файле `.env`:
   ```
   OUTPUTS_DIR=/path/to/custom/directory
   ```

   Каждый notebook будет содержать:
   - **Основное объяснение** (сгенерировано Google Gemini)
   - **Критический анализ** (сгенерирован OpenAI GPT) - если настроен OPENAI_API_KEY
   - **Python примеры кода** (сгенерированы OpenAI GPT) - если настроен OPENAI_API_KEY

5. (Опционально) Если настроен Google Custom Search API, изображения будут автоматически загружены в директорию `outputs/img/<код_темы>/` (или `${OUTPUTS_DIR}/img/<код_темы>/` если настроена переменная `OUTPUTS_DIR`)

6. После завершения обработки в директории `outputs/` будут созданы дополнительные файлы для мониторинга:
   - **`results.json`** - файл с результатами обработки каждой темы (обновляется при каждом запуске)
   - **`processing.log`** - лог-файл с историей обработки (дополняется при каждом запуске)

### Файлы мониторинга

#### results.json
Содержит информацию о результатах обработки каждой темы в формате JSON:

```json
{
  "prf": {
    "model": "gemini-3-preview",
    "status": "success",
    "last_updated": "2026-01-11T15:13:52.192667"
  },
  "dpr": {
    "model": "gemini-3-preview",
    "status": "failed",
    "last_updated": "2026-01-11T15:13:52.193282"
  }
}
```

**Поля:**
- `model` - название LLM модели, использованной для обработки
- `status` - статус обработки (`success` или `failed`)
- `last_updated` - время последнего обновления в формате ISO 8601

**Режим работы:** Файл перезаписывается при каждом запуске приложения, сохраняя актуальное состояние всех обработанных тем.

#### processing.log
Содержит детальный лог обработки каждой темы с информацией о токенах:

```
2026-01-11T15:13:52.192680	Pseudo-Relevance Feedback	gemini-3-preview	1234	success
2026-01-11T15:13:52.193290	Deep Passage Retrieval	gemini-3-preview	2345	success
2026-01-11T15:13:52.193377	ColBERT	gemini-3-preview	0	failed
```

**Поля (разделенные табуляцией):**
1. Временная метка (timestamp) в формате ISO 8601
2. Название темы
3. Использованная модель LLM
4. Количество использованных токенов
5. Статус обработки (`success` или `failed`)

**Режим работы:** Новые записи добавляются в конец файла при каждом запуске, сохраняя полную историю обработки.

## Структура notebook

Каждый сгенерированный Jupyter Notebook включает:

1. **Markdown ячейка**: Полное техническое объяснение темы
2. **Markdown ячейка**: Критический анализ (📝) с предложениями по улучшению
3. **Markdown ячейка**: Заголовок примера кода (💻)
4. **Code ячейка**: Иллюстративный Python код, демонстрирующий концепции

## Структура проекта

```
explainy/
├── main.py                         # Основной скрипт приложения с двумя LLM
├── telegram_publisher.py           # Модуль для публикации notebooks в Telegram
├── topics.txt                      # Список тем для объяснения
├── prompts/                        # Папка с промптами для LLM
│   ├── README.md                   # Документация по промптам
│   ├── main_system_prompt.txt      # Системный промпт для основной модели
│   ├── main_user_prompt.txt        # Шаблон запроса к основной модели
│   ├── critic_system_prompt.txt    # Системный промпт для модели-критика
│   ├── critic_user_prompt.txt      # Шаблон запроса к модели-критику
│   ├── code_generation_system_prompt.txt  # Системный промпт для генерации кода
│   └── code_generation_prompt.txt  # Шаблон запроса для генерации кода
├── requirements.txt                # Python зависимости
├── .env.example                    # Пример файла с переменными окружения
├── .gitignore                      # Файлы для игнорирования Git
├── tests/                          # Тесты
│   ├── test_app.py                 # Тесты основной функциональности
│   ├── test_notebook_generation.py # Тесты генерации Jupyter Notebooks
│   ├── test_critique_enhancement.py # Тесты критики и улучшения notebooks
│   ├── test_image_download.py      # Тесты загрузки изображений
│   ├── test_outputs_dir.py         # Тесты параметризации директории outputs
│   ├── test_logging_and_results.py # Тесты логирования и сохранения результатов
│   ├── test_telegram_publisher.py  # Тесты модуля публикации в Telegram
│   └── demo_logging_and_results.py # Демонстрация логирования и результатов
├── outputs/                        # Директория с сгенерированными объяснениями (создается автоматически)
│   ├── img/                        # Директория с загруженными изображениями
│   ├── results.json                # Файл с результатами обработки тем
│   └── processing.log              # Лог-файл обработки
└── README.md                       # Этот файл
```

## Конфигурация

### Публикация в Telegram

Приложение включает модуль `telegram_publisher.py` для публикации сгенерированных Jupyter Notebooks в Telegram канал.

**Возможности:**
- Конвертация `.ipynb` файлов в Markdown формат
- Автоматическая отправка контента в Telegram канал через Bot API
- Разбиение длинных сообщений на части (Telegram лимит: 4096 символов)
- Обработка ошибок и детальное логирование

**Настройка:**

1. Создайте Telegram бота через [@BotFather](https://t.me/BotFather):
   - Отправьте `/newbot` и следуйте инструкциям
   - Сохраните полученный токен бота

2. Создайте Telegram канал или используйте существующий:
   - Добавьте вашего бота в канал как администратора
   - Получите ID канала (например, `@channel_name` или числовой ID `-1001234567890`)

3. Добавьте учетные данные в файл `.env`:
   ```bash
   TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   TELEGRAM_CHANNEL_ID=@your_channel_name
   ```

**Использование:**

```bash
# Публикация одного notebook
python telegram_publisher.py outputs/dpr.ipynb

# Или через Python код
from telegram_publisher import publish_notebook_to_telegram

# Используя переменные окружения из .env
publish_notebook_to_telegram('outputs/dpr.ipynb')

# Или с явным указанием параметров
publish_notebook_to_telegram(
    'outputs/dpr.ipynb',
    bot_token='your_bot_token',
    channel_id='@your_channel'
)
```

**Примечание**: Модуль автоматически обрабатывает:
- Конвертацию markdown и code ячеек в читаемый формат
- Разбиение длинных сообщений на части
- Повторные попытки отправки при ошибках парсинга форматирования

**Безопасность**: `nbconvert` имеет известную уязвимость (uncontrolled search path) на Windows. Патч пока не доступен. Рекомендуется запускать в контролируемых окружениях. Уязвимость не влияет на Linux/macOS.

### Файл topics.txt
Содержит список тем для объяснения в формате: `code;detailed_query;image_query`

**Формат:**
- `code`: Короткий код темы (используется для имени файла)
- `detailed_query`: Полный запрос для генерации объяснения
- `image_query`: Запрос на английском для поиска изображений

**Пример:**
```
dpr;Методы информационного поиска / метод Deep Passage Retrieval (DPR);information retrieval Deep Passage Retrieval DPR diagram
rag;Методы информационного поиска / метод RAG (2020);information retrieval RAG Retrieval Augmented Generation
colbert;Методы информационного поиска / метод ColBERT (2020);information retrieval ColBERT architecture diagram
```

### Параметризация директории выходных файлов

По умолчанию все сгенерированные файлы (notebooks и изображения) сохраняются в директории `outputs/`:
- `outputs/` - Jupyter notebooks с объяснениями
- `outputs/img/<код_темы>/` - загруженные изображения для каждой темы

Вы можете изменить директорию для сохранения результатов через переменную окружения `OUTPUTS_DIR` в файле `.env`:

```bash
# В файле .env
OUTPUTS_DIR=/path/to/custom/directory
```

При запуске программа автоматически создаст:
- Указанную директорию (если она не существует)
- Поддиректорию `img/` внутри для хранения изображений

**Пример использования:**
```bash
# Сохранить результаты в домашней директории
OUTPUTS_DIR=~/my-explanations

# Сохранить результаты на внешнем диске
OUTPUTS_DIR=/mnt/external/outputs
```

### Настройка промптов

Все промпты хранятся в папке `prompts/`. Подробное описание структуры и назначения каждого промпта см. в `prompts/README.md`.

**Основная модель (Gemini):**
- `prompts/main_system_prompt.txt` - определяет стиль и формат генерации объяснений
- `prompts/main_user_prompt.txt` - шаблон запроса для генерации объяснения

**Модель-критик и генерация кода (OpenAI GPT):**
- `prompts/critic_system_prompt.txt` - требования к критическому анализу
- `prompts/critic_user_prompt.txt` - шаблон запроса критики
- `prompts/code_generation_system_prompt.txt` - роль модели при генерации кода
- `prompts/code_generation_prompt.txt` - инструкции для создания примеров кода

При изменении промптов учитывайте параметры-плейсхолдеры (`{topic}`, `{content}`), которые используются в коде.

### Настройка критики и генерации кода

Secondary LLM (OpenAI GPT) автоматически настроен для:

**Критики:**
- Анализ полноты и точности технической информации
- Оценка ясности изложения и структуры
- Проверка наличия конкретных примеров
- Предложения по улучшению

**Генерации кода:**
- Иллюстративные примеры (не production-level)
- Комментарии, объясняющие ключевые моменты
- Простые и понятные примеры
- Демонстрация основных этапов (для ML моделей)
- Типовые случаи использования (для библиотек)

### Загрузка изображений (опционально)

Приложение может автоматически загружать изображения для каждой темы используя Google Custom Search API:

1. **Настройка**: Добавьте `GOOGLE_SEARCH_API_KEY` и `GOOGLE_SEARCH_ENGINE_ID` в файл `.env`
2. **Работа**: При обработке каждой темы приложение:
   - Выполняет поиск изображений с фильтром `imagesize:large` для получения качественных изображений
   - Создает подкаталог в `${OUTPUTS_DIR}/img/<код_темы>/` (по умолчанию `outputs/img/<код_темы>/`)
   - Загружает найденные изображения и сохраняет их как `img1.jpg`, `img2.png` и т.д. (с сохранением оригинального формата)
   - Возвращает путь к директории с изображениями
3. **Обработка ошибок**: Если API не настроен или произошла ошибка, приложение продолжит работу без загрузки изображений

**Примечание**: Google Custom Search API имеет бесплатную квоту в 100 запросов в день. Каждая тема использует 1 запрос.

## Требования

- Python 3.10+
- Google Gemini API ключ (для генерации объяснений)
- OpenAI API ключ (для критики и генерации кода)
- (Опционально) Google Custom Search API ключ и Search Engine ID для загрузки изображений
- Интернет соединение для доступа к API

## Лицензия

MIT
