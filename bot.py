import os
import sqlite3
import time
import datetime
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from groq import Groq
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

# Импортируем информацию о моделях и обработчик голосовых сообщений
from model_info import (
    AVAILABLE_MODELS, get_model_info, get_all_models_info,
    WHISPER_MODEL_INFO, MAX_VOICE_DURATION
)
from voice_handler import process_voice_message, WHISPER_MODEL


class DatabaseManager:
    def __init__(self, db_file="bot_data.db"):
        self.db_file = db_file
        self._init_db()

    def _init_db(self):
        """Инициализирует базу данных"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Таблица для настроек чатов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_settings (
            chat_id INTEGER PRIMARY KEY,
            model TEXT,
            temperature REAL,
            max_tokens INTEGER,
            active INTEGER,
            system_prompt TEXT,
            updated_at TEXT
        )
        ''')

        # Таблица для истории сообщений
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY (chat_id) REFERENCES chat_settings (chat_id)
        )
        ''')

        conn.commit()
        conn.close()

    def get_chat_settings(self, chat_id):
        """Получает настройки чата из БД или возвращает значения по умолчанию"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM chat_settings WHERE chat_id = ?', (chat_id,))
        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                "model": result[1],
                "temperature": result[2],
                "max_tokens": result[3],
                "active": bool(result[4]),
                "system_prompt": result[5]
            }
        else:
            # Значения по умолчанию
            default_settings = {
                "model": "deepseek-r1-distill-llama-70b",
                "temperature": 0.7,
                "max_tokens": 3000,
                "active": True,
                "system_prompt": DEFAULT_SYSTEM_PROMPT
            }
            self.save_chat_settings(chat_id, default_settings)
            return default_settings

    def save_chat_settings(self, chat_id, settings):
        """Сохраняет настройки чата в БД"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        now = datetime.datetime.now().isoformat()

        cursor.execute('''
        INSERT OR REPLACE INTO chat_settings 
        (chat_id, model, temperature, max_tokens, active, system_prompt, updated_at) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            chat_id,
            settings["model"],
            settings["temperature"],
            settings["max_tokens"],
            int(settings["active"]),
            settings["system_prompt"],
            now
        ))

        conn.commit()
        conn.close()

    def get_message_history(self, chat_id, limit=50):
        """Получает историю сообщений для чата"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT role, content FROM message_history 
        WHERE chat_id = ? 
        ORDER BY created_at ASC 
        LIMIT ?
        ''', (chat_id, limit))

        results = cursor.fetchall()
        conn.close()

        return [{"role": role, "content": content} for role, content in results]

    def add_message(self, chat_id, role, content):
        """Добавляет сообщение в историю"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        now = datetime.datetime.now().isoformat()

        cursor.execute('''
        INSERT INTO message_history (chat_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        ''', (chat_id, role, content, now))

        conn.commit()
        conn.close()

    def clear_chat_history(self, chat_id):
        """Очищает историю сообщений для чата"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM message_history WHERE chat_id = ?', (chat_id,))
        conn.commit()
        conn.close()

    def prune_old_messages(self, days=30):
        """Удаляет сообщения старше указанного количества дней"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
        cursor.execute('DELETE FROM message_history WHERE created_at < ?', (cutoff_date,))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count


# Настройка логирования с ротацией файлов
log_handler = RotatingFileHandler(
    'bot.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5,  # Хранить 5 архивных файлов
    encoding='utf-8'
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log_handler,
        logging.StreamHandler()  # Вывод в консоль
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных из .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Проверка, что ключи загружены
if not GROQ_API_KEY or not TELEGRAM_TOKEN:
    logger.error("Не удалось загрузить GROQ_API_KEY или TELEGRAM_TOKEN из .env")
    raise ValueError("Необходимо указать GROQ_API_KEY и TELEGRAM_TOKEN в файле .env")

# Инициализация клиента Groq
client = Groq(api_key=GROQ_API_KEY)

# Инициализация базы данных
db = DatabaseManager()

MAX_HISTORY = 50  # Максимальное количество сообщений в истории

# Системный промпт по умолчанию
DEFAULT_SYSTEM_PROMPT = """Ты - полезный и дружелюбный ассистент в групповом чате Telegram.
Твоя задача - помогать участникам чата, отвечать на их вопросы и поддерживать беседу.
Старайся давать краткие, но информативные ответы.
Помни, что тебя упоминают по имени, поэтому в ответах не надо обращаться к конкретному пользователю.
Отвечай на русском языке, если не просят иное.
Не используй эмодзи слишком часто.
"""


async def cleanup_old_data(context: ContextTypes.DEFAULT_TYPE):
    """Периодическая очистка старых данных"""
    # Очистка старых сообщений (старше 30 дней)
    deleted_count = db.prune_old_messages(days=30)
    logger.info(f"Очищено {deleted_count} старых сообщений из базы данных")


def process_model_response(text):
    """Обрабатывает ответ модели, удаляя теги <think> и их содержимое"""
    # Удаляем всё содержимое между тегами <think> и </think>, включая сами теги
    import re
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return clean_text.strip()


async def send_long_message(update, text, parse_mode=None):
    """Разбивает длинное сообщение на части с учетом целостности параграфов"""
    MAX_LENGTH = 4000  # Чуть меньше лимита Telegram для безопасности

    if len(text) <= MAX_LENGTH:
        try:
            return await update.message.reply_text(text, parse_mode=parse_mode)
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения с форматированием: {e}")
            # Пробуем отправить без форматирования
            return await update.message.reply_text(text)

    # Разбиваем по абзацам для более естественного деления
    paragraphs = text.split('\n\n')
    current_part = ""

    for paragraph in paragraphs:
        # Если добавление абзаца превысит лимит, отправляем текущую часть
        if len(current_part + paragraph + '\n\n') > MAX_LENGTH:
            if current_part:
                try:
                    await update.message.reply_text(current_part, parse_mode=parse_mode)
                except Exception as e:
                    logger.error(f"Ошибка при отправке части сообщения: {e}")
                    # Пробуем отправить без форматирования
                    await update.message.reply_text(current_part)
                current_part = paragraph + '\n\n'
            else:
                # Если один абзац слишком большой, разбиваем его на части
                for i in range(0, len(paragraph), MAX_LENGTH):
                    chunk = paragraph[i:i + MAX_LENGTH]
                    try:
                        await update.message.reply_text(chunk, parse_mode=parse_mode)
                    except Exception as e:
                        logger.error(f"Ошибка при отправке фрагмента абзаца: {e}")
                        await update.message.reply_text(chunk)
        else:
            current_part += paragraph + '\n\n'

    # Отправляем оставшуюся часть, если она есть
    if current_part:
        try:
            await update.message.reply_text(current_part, parse_mode=parse_mode)
        except Exception as e:
            logger.error(f"Ошибка при отправке последней части: {e}")
            await update.message.reply_text(current_part)


async def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Проверяет, является ли пользователь администратором группы"""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    try:
        # Получаем информацию о пользователях с правами администратора в чате
        chat_admins = await context.bot.get_chat_administrators(chat_id)
        admin_ids = [admin.user.id for admin in chat_admins]

        # Проверяем, есть ли ID пользователя в списке ID администраторов
        is_user_admin = user_id in admin_ids

        logger.info(f"Проверка прав администратора для пользователя {user_id} в чате {chat_id}: {is_user_admin}")
        return is_user_admin
    except Exception as e:
        logger.error(f"Ошибка при проверке прав администратора: {str(e)}")
        # В случае ошибки считаем, что пользователь не администратор
        return False


async def admin_required(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет сообщение о необходимости прав администратора"""
    await update.message.reply_text(
        "⚠️ Для использования этой команды необходимы права администратора группы."
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(
        f"Привет! Я бот на базе Groq API. Я буду отвечать на сообщения, в которых меня упоминают через @{context.bot.username}.\n"
        "Используйте /help для получения списка команд и /models для информации о доступных моделях."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    is_user_admin = await is_admin(update, context)

    basic_commands = (
        "Доступные команды:\n"
        "/start - Запустить бота\n"
        "/help - Показать это сообщение\n"
        "/explain - Руководство по использованию бота\n"
        "/models - Информация о доступных моделях\n\n"
        f"Вы можете обращаться к боту, упоминая его через @{context.bot.username} в сообщении"
    )

    admin_commands = (
        "\n\n<b>Команды только для администраторов:</b>\n"
        "/settings - Показать текущие настройки\n"
        "/set_model [модель] - Установить модель (llama3-70b-8192, llama3-8b-8192 и др.)\n"
        "/set_temp [0.0-1.0] - Установить температуру генерации\n"
        "/set_max_tokens [число] - Установить максимальное количество токенов ответа\n"
        "/toggle - Включить/выключить бота в этом чате\n"
        "/clear_history - Очистить историю чата\n"
    )

    # Если пользователь админ, показываем все команды
    if is_user_admin:
        help_text = basic_commands + admin_commands
        await update.message.reply_text(help_text, parse_mode="HTML")
    else:
        # Обычным пользователям показываем только базовые команды
        await update.message.reply_text(basic_commands)


async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать текущие настройки чата"""
    # Проверяем, является ли пользователь администратором
    if not await is_admin(update, context):
        await admin_required(update, context)
        return

    chat_id = update.effective_chat.id
    settings = db.get_chat_settings(chat_id)

    # Получаем информацию о модели
    model_info = get_model_info(settings['model'])

    history_count = len(db.get_message_history(chat_id))

    settings_text = (
        f"<b>Текущие настройки:</b>\n"
        f"• Модель: <b>{settings['model']}</b>\n"
        f"• Температура: <b>{settings['temperature']}</b>\n"
        f"• Максимальная длина ответа: <b>{settings['max_tokens']} токенов</b>\n"
        f"• Бот: <b>{'активен' if settings['active'] else 'неактивен'}</b>\n"
        f"• Количество сообщений в истории: <b>{history_count}/{MAX_HISTORY}</b>\n\n"
        f"<b>Информация о текущей модели:</b>\n"
        f"• {model_info['description']}\n"
        f"• Рекомендуется для: {model_info['use_case']}\n"
        f"• Лимиты: {model_info['limits']}\n"
    )

    await update.message.reply_text(settings_text, parse_mode="HTML")


async def set_max_tokens(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Установить максимальное количество токенов ответа"""
    # Проверяем, является ли пользователь администратором
    if not await is_admin(update, context):
        await admin_required(update, context)
        return

    chat_id = update.effective_chat.id

    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите максимальное количество токенов (например, 2000)")
        return

    try:
        max_tokens = int(context.args[0])
        if max_tokens > 0:
            settings = db.get_chat_settings(chat_id)
            settings["max_tokens"] = max_tokens
            db.save_chat_settings(chat_id, settings)
            await update.message.reply_text(
                f"Максимальное количество токенов установлено: <b>{max_tokens}</b>\n\n"
                f"Это влияет на максимальную длину ответа бота. Чем больше значение, тем длиннее ответы может давать бот.",
                parse_mode="HTML"
            )
        else:
            await update.message.reply_text("Количество токенов должно быть положительным числом")
    except ValueError:
        await update.message.reply_text("Пожалуйста, укажите корректное целое число")


async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Установить модель для использования"""
    # Проверяем, является ли пользователь администратором
    if not await is_admin(update, context):
        await admin_required(update, context)
        return

    chat_id = update.effective_chat.id

    if not context.args:
        await update.message.reply_text(
            f"Пожалуйста, укажите модель. Например: /set_model llama3-70b-8192\n"
            f"Для просмотра доступных моделей используйте /models"
        )
        return

    model = context.args[0]

    if model not in AVAILABLE_MODELS:
        await update.message.reply_text(
            f"Недопустимая модель. Доступные модели: {', '.join(AVAILABLE_MODELS)}\n"
            f"Для подробной информации используйте /models"
        )
        return

    # Получаем информацию о модели
    model_info = get_model_info(model)

    # Устанавливаем модель
    settings = db.get_chat_settings(chat_id)
    settings["model"] = model
    db.save_chat_settings(chat_id, settings)

    await update.message.reply_text(
        f"✅ Модель установлена: <b>{model}</b>\n\n"
        f"<b>Информация о модели:</b>\n"
        f"• {model_info['description']}\n"
        f"• Рекомендуется для: {model_info['use_case']}\n"
        f"• Лимиты: {model_info['limits']}\n"
        f"• Особенности: {model_info['features']}",
        parse_mode="HTML"
    )


async def set_temperature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Установить температуру генерации"""
    # Проверяем, является ли пользователь администратором
    if not await is_admin(update, context):
        await admin_required(update, context)
        return

    chat_id = update.effective_chat.id

    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите температуру (от 0.0 до 1.0)")
        return

    try:
        temp = float(context.args[0])
        if 0.0 <= temp <= 1.0:
            settings = db.get_chat_settings(chat_id)
            settings["temperature"] = temp
            db.save_chat_settings(chat_id, settings)
            await update.message.reply_text(
                f"Температура установлена: <b>{temp}</b>\n\n"
                f"<b>Что это значит:</b>\n"
                f"• <b>Низкая (0.1-0.3)</b>: более предсказуемые, точные ответы. Хорошо для фактических вопросов и кодирования.\n"
                f"• <b>Средняя (0.4-0.7)</b>: баланс между точностью и разнообразием. Подходит для большинства задач.\n"
                f"• <b>Высокая (0.8-1.0)</b>: более креативные, разнообразные ответы. Подходит для творческих задач.",
                parse_mode="HTML"
            )
        else:
            await update.message.reply_text("Температура должна быть от 0.0 до 1.0")
    except ValueError:
        await update.message.reply_text("Пожалуйста, укажите корректное число")


async def toggle_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Включить/выключить бота в чате"""
    # Проверяем, является ли пользователь администратором
    if not await is_admin(update, context):
        await admin_required(update, context)
        return

    chat_id = update.effective_chat.id
    settings = db.get_chat_settings(chat_id)
    settings["active"] = not settings["active"]
    db.save_chat_settings(chat_id, settings)
    status = "активен" if settings["active"] else "неактивен"
    await update.message.reply_text(f"Бот теперь {status} в этом чате")


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очистить историю чата"""
    # Проверяем, является ли пользователь администратором
    if not await is_admin(update, context):
        await admin_required(update, context)
        return

    chat_id = update.effective_chat.id
    db.clear_chat_history(chat_id)
    await update.message.reply_text("История чата очищена")


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать информацию о доступных моделях"""
    models_info = get_all_models_info()

    report = "<b>Доступные модели:</b>\n\n"

    for model_info in models_info:
        model_name = model_info['name']

        report += (
            f"<b>{model_name}</b>\n"
            f"• {model_info['description']}\n"
            f"• <b>Когда использовать:</b> {model_info['use_case']}\n"
            f"• <b>Особенности:</b> {model_info['features']}\n"
            f"• <b>Лимиты:</b> {model_info['limits']}\n\n"
        )

    # Добавляем информацию о модели для распознавания голосовых сообщений
    report += (
        f"<b>Модель для голосовых сообщений: {WHISPER_MODEL}</b>\n"
        f"• {WHISPER_MODEL_INFO['description']}\n"
        f"• <b>Лимиты:</b> {WHISPER_MODEL_INFO['limits']}\n"
        f"• <b>Особенности:</b> {WHISPER_MODEL_INFO['features']}\n\n"
    )

    report += "Для установки модели используйте команду /set_model [название_модели]"

    try:
        await update.message.reply_text(report, parse_mode="HTML")
    except Exception as e:
        # В случае ошибки форматирования отправляем без HTML
        logger.error(f"Ошибка при отправке информации о моделях: {str(e)}")
        await update.message.reply_text(report.replace("<b>", "").replace("</b>", ""))


async def explain_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):

    """Объяснить все параметры настройки бота"""
    chat_id = update.effective_chat.id
    settings = db.get_chat_settings(chat_id)
    is_user_admin = await is_admin(update, context)
    history_count = len(db.get_message_history(chat_id))

    # Базовая информация для всех пользователей
    basic_explanation = (
        "📋 <b>Руководство по использованию бота</b>\n\n"

        "<b>Как взаимодействовать с ботом:</b>\n"
        f"• Упомянуть бота: @{context.bot.username} [ваш вопрос]\n"
        "• Отправить голосовое сообщение, упомянув бота в подписи или ответив на его сообщение\n\n"
    )

    basic_explanation += (
        "<b>Примеры использования:</b>\n"
        f"• @{context.bot.username} расскажи о квантовой физике\n"
        f"• @{context.bot.username} реши задачу: 2x + 5 = 15\n"
        f"• @{context.bot.username} напиши пример кода на Python для парсинга JSON\n\n"

        "<b>Типы моделей:</b>\n"
        "• Большие модели (llama3-70b-8192) - для сложных задач\n"
        "• Средние модели (mistral-saba-24b) - универсальные\n"
        "• Компактные модели (llama3-8b-8192, gemma-7b-it) - для простых запросов\n\n"

        "Используйте команду /models для подробной информации о каждой модели"
    )

    # Информация по настройкам только для администраторов
    admin_explanation = ""
    if is_user_admin:
        admin_explanation = (
            "\n\n<b>Настройка бота (только для администраторов):</b>\n"

            "<b>Модель ИИ</b>\n"
            "Команда: /set_model [модель]\n"
            f"Текущая модель: <b>{settings['model']}</b>\n\n"

            "<b>Температура</b>\n"
            "Команда: /set_temp [значение от 0.0 до 1.0]\n"
            "• <b>Низкая (0.1-0.3)</b>: более предсказуемые, точные ответы. Хорошо для фактических вопросов и кодирования.\n"
            "• <b>Средняя (0.4-0.7)</b>: баланс между точностью и разнообразием. Подходит для большинства задач.\n"
            "• <b>Высокая (0.8-1.0)</b>: более случайные, творческие ответы. Подходит для креативных задач, генерации идей.\n"
            f"Текущая температура: <b>{settings['temperature']}</b>\n\n"
            
            "<b>Максимальное количество токенов</b>\n"
            "Команда: /set_max_tokens [число]\n"
            "• Определяет максимальную длину ответа бота\n"
            "• Чем выше значение, тем длиннее может быть ответ\n"
            "• Рекомендуемые значения: 1000-4000\n"
            f"Текущее значение: <b>{settings['max_tokens']}</b>\n\n"
            
            "<b>История сообщений</b>\n"
            "Команда для очистки: /clear_history\n"
            "• Бот запоминает историю диалога, что позволяет давать более контекстные ответы.\n"
            "• Очистка истории полезна при смене темы или для экономии токенов.\n"
            f"Текущее количество сообщений в истории: <b>{history_count}/{MAX_HISTORY}</b>\n\n"

            "<b>Активация/деактивация</b>\n"
            "Команда: /toggle\n"
            "• Включает или выключает бота в данном чате.\n"
            f"Текущий статус: <b>{'Активен' if settings['active'] else 'Неактивен'}</b>\n\n"

            "<b>🔍 Рекомендуемые настройки:</b>\n"
            "• Для общего использования: llama3-70b-8192, температура 0.7\n"
            "• Для точных ответов и фактов: llama3-70b-8192, температура 0.2\n"
            "• Для креативных задач: llama3-70b-8192, температура 0.9\n"
            "• Для быстрых ответов на простые вопросы: llama3-8b-8192, температура 0.5\n"
        )

    explanation = basic_explanation + admin_explanation

    try:
        await update.message.reply_text(explanation, parse_mode="HTML")
        logger.info(f"Отправлено руководство по настройкам в чат {chat_id}")
    except Exception as e:
        logger.error(f"Ошибка при отправке руководства: {str(e)}")
        # Запасной вариант без HTML-форматирования
        plain_explanation = explanation.replace("<b>", "").replace("</b>", "")
        await update.message.reply_text(plain_explanation)
        logger.info(f"Отправлена версия руководства без форматирования в чат {chat_id}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка входящих сообщений"""
    if update.message.chat.type in ["group", "supergroup"]:
        chat_id = update.message.chat.id
        username = update.message.from_user.username or update.message.from_user.first_name or "Unknown"

        # Проверяем, активен ли бот в этом чате
        settings = db.get_chat_settings(chat_id)
        if not settings["active"]:
            logger.info(f"Бот неактивен в чате {chat_id}, игнорирую сообщение")
            return

        # Обработка голосовых сообщений
        if update.message.voice:
            logger.info(f"Получено голосовое сообщение от @{username} в чате {chat_id}")

            # Получаем транскрипцию голосового сообщения
            transcribed_text = await process_voice_message(update, context, client)

            # Если не удалось получить текст, прекращаем обработку
            if not transcribed_text:
                logger.warning("Не удалось получить текст голосового сообщения")
                return

            # Устанавливаем транскрибированный текст как сообщение для обработки
            user_message = transcribed_text
            logger.info(f"Транскрипция голосового сообщения: {user_message}")
        else:
            # Обработка текстовых сообщений
            user_message = update.message.text

            # Получаем имя пользователя бота
            bot_username = context.bot.username
            logger.info(f"Проверка упоминания бота @{bot_username} в сообщении: {user_message}")

            # Проверяем, есть ли упоминание бота в сообщении
            is_bot_mentioned = False
            clean_message = user_message

            # Проверяем по entities
            if update.message.entities:
                for entity in update.message.entities:
                    if entity.type == 'mention':
                        mention = user_message[entity.offset:entity.offset + entity.length]
                        logger.info(f"Найдено упоминание: {mention}")
                        if mention.lower() == f"@{bot_username.lower()}":
                            is_bot_mentioned = True
                            # Удаляем упоминание из сообщения
                            clean_message = user_message.replace(mention, "").strip()
                            logger.info(f"Найдено упоминание бота: {mention}, сообщение после очистки: {clean_message}")
                            break

            # Также проверяем простым текстовым поиском (как запасной вариант)
            if not is_bot_mentioned and f"@{bot_username}" in user_message:
                is_bot_mentioned = True
                clean_message = user_message.replace(f"@{bot_username}", "").strip()
                logger.info(f"Найдено текстовое упоминание бота, сообщение после очистки: {clean_message}")

            # Также проверяем, является ли сообщение ответом на сообщение бота
            if not is_bot_mentioned and update.message.reply_to_message:
                if update.message.reply_to_message.from_user.id == context.bot.id:
                    is_bot_mentioned = True
                    clean_message = user_message
                    logger.info(f"Сообщение является ответом на сообщение бота")

            # Если бот не упомянут, игнорируем сообщение
            if not is_bot_mentioned:
                logger.info(f"Бот не упомянут в сообщении, игнорирую")
                return

            # Если после удаления упоминания сообщение пустое
            if not clean_message:
                logger.info("Сообщение содержит только упоминание бота, отправляю приветствие")
                await update.message.reply_text("Здравствуйте! Чем я могу вам помочь?")
                return

            user_message = clean_message

        logger.info(f"Обрабатываю сообщение от @{username} в чате {chat_id}: {user_message}")

        try:
            settings = db.get_chat_settings(chat_id)
            messages = [{"role": "system", "content": settings["system_prompt"]}]
            messages.extend(db.get_message_history(chat_id))

            # Добавляем текущее сообщение
            messages.append({"role": "user", "content": f"{username}: {user_message}"})

            # Получаем модель для использования
            model = settings["model"]

            # Логируем детали запроса для отладки
            logger.info(f"Модель: {model}")
            logger.info(f"Температура: {settings['temperature']}")
            logger.info(f"Макс. токенов: {settings['max_tokens']}")
            messages_history = db.get_message_history(chat_id)
            logger.info(f"Кол-во сообщений в истории: {len(messages_history)}")

            # Отправляем индикатор набора текста
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")

            # Запрос к Groq
            logger.info("Отправка запроса к API Groq...")
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=settings["max_tokens"],
                temperature=settings["temperature"]
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Ответ от Groq получен за {elapsed_time:.2f} секунд")

            reply_text = response.choices[0].message.content

            # Добавляем сообщение пользователя и ответ бота в историю
            db.add_message(chat_id, "user", f"{username}: {user_message}")
            db.add_message(chat_id, "assistant", reply_text)

            # Логируем успешный ответ
            logger.info(f"Отправлен ответ в чат {chat_id}: {reply_text[:50]}...")

            # Очищаем ответ от тегов <think>
            cleaned_reply = process_model_response(reply_text)

            # Отправляем ответ, разбивая на части при необходимости
            await send_long_message(update, cleaned_reply, parse_mode="Markdown")

        except Exception as e:
            # Расширенное логирование ошибки
            logger.error(f"Ошибка при обработке запроса в чате {chat_id}: {str(e)}")
            logger.error(f"Тип ошибки: {type(e).__name__}")

            # Проверяем, связана ли ошибка с моделью
            if "model" in str(e).lower() and "decommissioned" in str(e).lower():
                logger.error("Обнаружена ошибка с моделью! Возможно, модель устарела или не поддерживается.")
                await update.message.reply_text(
                    f"Модель {settings['model']} недоступна или устарела. "
                    f"Переключаюсь на llama3-70b-8192."
                )
                settings['model'] = "llama3-70b-8192"
            elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
                logger.error("Достигнут лимит запросов API")

                # Формируем сообщение с рекомендациями по модели
                error_msg = (
                    "⚠️ <b>Достигнут лимит запросов для модели</b> "
                    f"<b>{settings['model']}</b>\n\n"
                    "Рекомендации:\n"
                    "1. Попробуйте использовать другую модель, например:\n"
                    "• /set_model deepseek-r1-distill-llama-70b (модель без дневного лимита токенов)\n"
                    "• /set_model llama3-8b-8192 (более легкая модель)\n\n"
                    "2. Подождите некоторое время - лимиты обновляются ежедневно\n\n"
                    "Используйте команду /models для просмотра всех доступных моделей и их лимитов."
                )

                try:
                    await update.message.reply_text(error_msg, parse_mode="HTML")
                except:
                    # Если с HTML возникла проблема, отправляем без форматирования
                    await update.message.reply_text(error_msg.replace("<b>", "").replace("</b>", ""))
            else:
                # Общее сообщение об ошибке
                await update.message.reply_text(
                    "Произошла ошибка, попробуйте позже или используйте другую модель (/models для просмотра доступных моделей).")


# Функция для обработки голосовых сообщений
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка голосовых сообщений"""
    # Вызываем общий обработчик сообщений, который определит голосовое сообщение
    await handle_message(update, context)


# Функция для мониторинга состояния
async def log_status(context: ContextTypes.DEFAULT_TYPE):
    """Периодическое логирование состояния бота"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(db.db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT chat_id) FROM chat_settings")
    total_chats = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM message_history")
    total_messages = cursor.fetchone()[0]
    cursor.execute("SELECT chat_id, model FROM chat_settings")
    chat_models = cursor.fetchall()
    conn.close()

    logger.info(f"=== СТАТУС БОТА [{now}] ===")
    logger.info(f"Активных чатов: {total_chats}")
    logger.info(f"Всего сообщений в БД: {total_messages}")
    logger.info(f"Настройки чатов: {', '.join([f'Чат {cid}: {model}' for cid, model in chat_models])}")
    logger.info("=== КОНЕЦ СТАТУСА ===")


# Запуск бота
if __name__ == "__main__":
    logger.info("=========================================")
    logger.info("Запуск бота...")
    logger.info(f"Время запуска: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Используемый API: Groq")
    logger.info(f"Настройки логирования: уровень {logging.getLevelName(logger.level)}")
    logger.info("=========================================")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Добавляем обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("settings", settings))
    app.add_handler(CommandHandler("set_model", set_model))
    app.add_handler(CommandHandler("set_temp", set_temperature))
    app.add_handler(CommandHandler("set_max_tokens", set_max_tokens))
    app.add_handler(CommandHandler("toggle", toggle_bot))
    app.add_handler(CommandHandler("clear_history", clear_history))
    app.add_handler(CommandHandler("explain", explain_settings))
    app.add_handler(CommandHandler("models", models_command))

    # Обработчик для текстовых сообщений
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    ))

    # Обработчик для голосовых сообщений
    app.add_handler(MessageHandler(
        filters.VOICE,
        handle_voice
    ))

    # Добавляем периодическую задачу для логирования состояния
    job_queue = app.job_queue
    job_queue.run_repeating(log_status, interval=3600, first=10)  # Логирование каждый час
    #Очистка базы данных
    job_queue.run_daily(cleanup_old_data, time=datetime.time(hour=3, minute=0))  # Запуск ежедневно в 3:00

    logger.info("Бот готов к работе, начинаю прослушивание...")

    # Запуск бота
    app.run_polling()

    logger.info("=========================================")
    logger.info("Бот остановлен.")
    logger.info(f"Время остановки: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=========================================")