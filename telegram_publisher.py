#!/usr/bin/env python3
"""
Модуль для публикации Jupyter Notebooks в Telegram канал.

Функциональность:
1. Конвертация .ipynb файлов в Markdown
2. Отправка Markdown контента в Telegram канал через Bot API
3. Обработка ошибок и логирование
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional, List
import asyncio
import logging
from datetime import datetime

try:
    from nbconvert import MarkdownExporter
except ImportError:
    print("Ошибка: необходимо установить nbconvert")
    print("Выполните: pip install nbconvert>=7.0.0")
    sys.exit(1)

try:
    from telegram import Bot
    from telegram.error import TelegramError
    from telegram.constants import ParseMode
except ImportError:
    print("Ошибка: необходимо установить python-telegram-bot")
    print("Выполните: pip install python-telegram-bot>=20.0")
    sys.exit(1)


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Константы
MAX_MESSAGE_LENGTH = 4096  # Максимальная длина сообщения в Telegram


def convert_notebook_to_markdown(notebook_path: Union[str, Path]) -> Optional[str]:
    """
    Конвертирует Jupyter Notebook в Markdown текст.
    
    Args:
        notebook_path: Путь к файлу .ipynb
        
    Returns:
        Строка с Markdown контентом или None в случае ошибки
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если файл не является корректным notebook
    """
    notebook_path = Path(notebook_path)
    
    # Проверяем существование файла
    if not notebook_path.exists():
        error_msg = f"Файл не найден: {notebook_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Проверяем расширение файла
    if notebook_path.suffix != '.ipynb':
        error_msg = f"Файл должен иметь расширение .ipynb: {notebook_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Создаем экспортер
        exporter = MarkdownExporter()
        
        # Конвертируем notebook в markdown
        (body, resources) = exporter.from_filename(str(notebook_path))
        
        logger.info(f"Notebook успешно конвертирован: {notebook_path}")
        logger.debug(f"Размер markdown контента: {len(body)} символов")
        
        return body
        
    except Exception as e:
        error_msg = f"Ошибка при конвертации notebook {notebook_path}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> List[str]:
    """
    Разбивает длинный текст на части для отправки в Telegram.
    
    Args:
        text: Текст для разбивки
        max_length: Максимальная длина одной части
        
    Returns:
        Список частей текста
    """
    if len(text) <= max_length:
        return [text]
    
    parts = []
    current_part = ""
    
    # Разбиваем по параграфам (двойной перенос строки)
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        # Если параграф слишком длинный, разбиваем по строкам
        if len(paragraph) > max_length:
            lines = paragraph.split('\n')
            for line in lines:
                # Если строка слишком длинная, разбиваем по словам
                if len(line) > max_length:
                    words = line.split(' ')
                    for word in words:
                        if len(current_part) + len(word) + 1 > max_length:
                            if current_part:
                                parts.append(current_part)
                                current_part = ""
                            # Слово слишком длинное, разбиваем его на части
                            while len(word) > max_length:
                                parts.append(word[:max_length])
                                word = word[max_length:]
                            current_part = word
                        else:
                            current_part += (' ' if current_part else '') + word
                else:
                    if len(current_part) + len(line) + 1 > max_length:
                        parts.append(current_part)
                        current_part = line
                    else:
                        current_part += ('\n' if current_part else '') + line
        else:
            if len(current_part) + len(paragraph) + 2 > max_length:
                parts.append(current_part)
                current_part = paragraph
            else:
                current_part += ('\n\n' if current_part else '') + paragraph
    
    if current_part:
        parts.append(current_part)
    
    return parts


async def send_to_telegram_async(
    markdown_content: str,
    bot_token: str,
    channel_id: str,
    parse_mode: str = ParseMode.MARKDOWN_V2
) -> bool:
    """
    Асинхронно отправляет Markdown контент в Telegram канал.
    
    Args:
        markdown_content: Markdown текст для отправки
        bot_token: Токен Telegram бота
        channel_id: ID Telegram канала (например, '@channel_name' или '-1001234567890')
        parse_mode: Режим парсинга (по умолчанию MARKDOWN_V2)
        
    Returns:
        True если успешно отправлено, False в случае ошибки
    """
    if not bot_token:
        logger.error("Токен бота не указан")
        return False
    
    if not channel_id:
        logger.error("ID канала не указан")
        return False
    
    try:
        # Создаем бота
        bot = Bot(token=bot_token)
        
        # Проверяем корректность токена
        bot_info = await bot.get_me()
        logger.info(f"Подключение к боту: @{bot_info.username}")
        
        # Разбиваем контент на части, если он слишком длинный
        parts = split_message(markdown_content)
        logger.info(f"Сообщение разбито на {len(parts)} частей")
        
        # Отправляем каждую часть
        for i, part in enumerate(parts, 1):
            try:
                message_to_send = part
                # Добавляем заголовок для составных сообщений (без форматирования, чтобы избежать проблем с экранированием)
                if len(parts) > 1:
                    header = f"[Часть {i}/{len(parts)}]\n\n"
                    message_to_send = header + part
                
                await bot.send_message(
                    chat_id=channel_id,
                    text=message_to_send,
                    parse_mode=parse_mode
                )
                logger.info(f"Часть {i}/{len(parts)} успешно отправлена")
                
                # Небольшая задержка между сообщениями
                if i < len(parts):
                    await asyncio.sleep(1)
                    
            except TelegramError as e:
                logger.error(f"Ошибка при отправке части {i}/{len(parts)}: {e}")
                # Если ошибка парсинга, пробуем отправить без форматирования
                # Проверяем тип ошибки через атрибуты исключения
                error_message = str(e).lower()
                is_parse_error = ("can't parse" in error_message or 
                                "parse" in error_message or 
                                "markdown" in error_message or
                                "entities" in error_message)
                
                if is_parse_error:
                    logger.info("Повторная попытка отправки без форматирования")
                    try:
                        # Убираем заголовок и отправляем только основной контент
                        await bot.send_message(
                            chat_id=channel_id,
                            text=part,
                            parse_mode=None
                        )
                        logger.info(f"Часть {i}/{len(parts)} отправлена без форматирования")
                    except TelegramError as e2:
                        logger.error(f"Не удалось отправить часть {i}/{len(parts)}: {e2}")
                        return False
                else:
                    return False
        
        logger.info("Все части успешно отправлены")
        return True
        
    except TelegramError as e:
        logger.error(f"Ошибка Telegram API: {e}")
        return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка при отправке в Telegram: {e}")
        return False


def send_to_telegram(
    markdown_content: str,
    bot_token: Optional[str] = None,
    channel_id: Optional[str] = None,
    parse_mode: str = ParseMode.MARKDOWN_V2
) -> bool:
    """
    Отправляет Markdown контент в Telegram канал (синхронная обертка).
    
    Args:
        markdown_content: Markdown текст для отправки
        bot_token: Токен Telegram бота (если None, берется из TELEGRAM_BOT_TOKEN)
        channel_id: ID Telegram канала (если None, берется из TELEGRAM_CHANNEL_ID)
        parse_mode: Режим парсинга (по умолчанию MARKDOWN_V2)
        
    Returns:
        True если успешно отправлено, False в случае ошибки
    """
    # Получаем параметры из переменных окружения, если не указаны
    if bot_token is None:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if channel_id is None:
        channel_id = os.getenv('TELEGRAM_CHANNEL_ID')
    
    if not bot_token:
        logger.error("Токен бота не найден. Укажите TELEGRAM_BOT_TOKEN в переменных окружения")
        return False
    
    if not channel_id:
        logger.error("ID канала не найден. Укажите TELEGRAM_CHANNEL_ID в переменных окружения")
        return False
    
    # Запускаем асинхронную функцию в новом event loop
    return asyncio.run(send_to_telegram_async(markdown_content, bot_token, channel_id, parse_mode))


def publish_notebook_to_telegram(
    notebook_path: Union[str, Path],
    bot_token: Optional[str] = None,
    channel_id: Optional[str] = None,
    parse_mode: str = ParseMode.MARKDOWN_V2
) -> bool:
    """
    Конвертирует Jupyter Notebook в Markdown и публикует в Telegram канал.
    
    Args:
        notebook_path: Путь к файлу .ipynb
        bot_token: Токен Telegram бота (если None, берется из TELEGRAM_BOT_TOKEN)
        channel_id: ID Telegram канала (если None, берется из TELEGRAM_CHANNEL_ID)
        parse_mode: Режим парсинга (по умолчанию MARKDOWN_V2)
        
    Returns:
        True если успешно опубликовано, False в случае ошибки
        
    Example:
        >>> # Публикация с использованием переменных окружения
        >>> publish_notebook_to_telegram('outputs/dpr.ipynb')
        
        >>> # Публикация с явным указанием параметров
        >>> publish_notebook_to_telegram(
        ...     'outputs/dpr.ipynb',
        ...     bot_token='1234567890:ABCdefGHIjklMNOpqrsTUVwxyz',
        ...     channel_id='@my_channel'
        ... )
    """
    logger.info(f"Начало публикации notebook: {notebook_path}")
    
    try:
        # Конвертируем notebook в markdown
        markdown_content = convert_notebook_to_markdown(notebook_path)
        
        if not markdown_content:
            logger.error("Не удалось конвертировать notebook в markdown")
            return False
        
        # Отправляем в Telegram
        success = send_to_telegram(markdown_content, bot_token, channel_id, parse_mode)
        
        if success:
            logger.info(f"Notebook успешно опубликован: {notebook_path}")
        else:
            logger.error(f"Не удалось опубликовать notebook: {notebook_path}")
        
        return success
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Ошибка при обработке notebook: {e}")
        return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка при публикации: {e}")
        return False


def main():
    """
    Пример использования модуля из командной строки.
    
    Usage:
        python telegram_publisher.py <path_to_notebook.ipynb>
    """
    if len(sys.argv) < 2:
        print("Использование: python telegram_publisher.py <path_to_notebook.ipynb>")
        print("\nПеред использованием установите переменные окружения:")
        print("  TELEGRAM_BOT_TOKEN - токен вашего Telegram бота")
        print("  TELEGRAM_CHANNEL_ID - ID вашего Telegram канала (например, @channel_name)")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    
    # Проверяем наличие необходимых переменных окружения
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    channel_id = os.getenv('TELEGRAM_CHANNEL_ID')
    
    if not bot_token:
        print("Ошибка: переменная окружения TELEGRAM_BOT_TOKEN не установлена")
        print("Установите ее в файле .env или через командную строку:")
        print("  export TELEGRAM_BOT_TOKEN='ваш_токен_бота'")
        sys.exit(1)
    
    if not channel_id:
        print("Ошибка: переменная окружения TELEGRAM_CHANNEL_ID не установлена")
        print("Установите ее в файле .env или через командную строку:")
        print("  export TELEGRAM_CHANNEL_ID='@ваш_канал'")
        sys.exit(1)
    
    # Публикуем notebook
    print(f"Публикация notebook: {notebook_path}")
    success = publish_notebook_to_telegram(notebook_path)
    
    if success:
        print("✓ Notebook успешно опубликован в Telegram!")
        sys.exit(0)
    else:
        print("✗ Не удалось опубликовать notebook")
        sys.exit(1)


if __name__ == "__main__":
    main()
