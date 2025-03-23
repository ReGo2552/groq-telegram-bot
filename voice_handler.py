import os
import tempfile
import asyncio
import logging
from pathlib import Path
from telegram import Update
from telegram.ext import ContextTypes

# Импортируем информацию о модели Whisper
from model_info import WHISPER_MODEL_INFO, MAX_VOICE_DURATION

logger = logging.getLogger(__name__)

# Название модели для распознавания голосовых сообщений
WHISPER_MODEL = "whisper-large-v3"


async def process_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE, client=None):
    """Обработка голосовых сообщений и их преобразование в текст"""
    chat_id = update.effective_chat.id
    user = update.message.from_user
    username = user.username or user.first_name or "Пользователь"

    voice = update.message.voice
    if not voice:
        logger.warning(f"Получено голосовое сообщение без данных")
        return None

    # Проверка на превышение максимальной длительности
    if voice.duration > MAX_VOICE_DURATION:
        await update.message.reply_text(
            f"⚠️ Ваше голосовое сообщение слишком длинное (более {MAX_VOICE_DURATION} секунд). "
            f"Пожалуйста, отправьте более короткое сообщение."
        )
        return None

    # Проверка, упомянут ли бот в подписи к голосовому сообщению (если есть)
    bot_username = context.bot.username
    is_bot_mentioned = False

    if update.message.caption:
        caption = update.message.caption
        if f"@{bot_username}" in caption:
            is_bot_mentioned = True
            logger.info(f"Бот упомянут в подписи к голосовому сообщению")

    # Если бот не упомянут, проверяем, является ли сообщение ответом на сообщение бота
    if not is_bot_mentioned and update.message.reply_to_message:
        if update.message.reply_to_message.from_user.id == context.bot.id:
            is_bot_mentioned = True
            logger.info(f"Голосовое сообщение является ответом на сообщение бота")

    # Если бот не упомянут, игнорируем сообщение
    if not is_bot_mentioned:
        logger.info(f"Бот не упомянут в голосовом сообщении, игнорируем")
        return None

    # Получаем файл голосового сообщения
    voice_file = await update.message.voice.get_file()

    # Создаем временный файл для скачивания
    with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
        temp_path = temp_file.name

    # Скачиваем голосовое сообщение
    await voice_file.download_to_drive(custom_path=temp_path)
    logger.info(f"Голосовое сообщение сохранено в {temp_path}")

    # Отправляем статус обработки
    status_msg = await update.message.reply_text("🔄 Обрабатываю голосовое сообщение...")

    try:
        transcribed_text = await transcribe_with_whisper(temp_path, client)
        if not transcribed_text:
            raise Exception("Не удалось получить транскрипцию")

        # Обновляем статусное сообщение
        await status_msg.edit_text(f"🔤 Текст голосового сообщения:\n\n{transcribed_text}")

        # Удаляем временный файл
        os.unlink(temp_path)

        return transcribed_text

    except Exception as e:
        logger.error(f"Ошибка при обработке голосового сообщения: {str(e)}")
        await status_msg.edit_text(
            "❌ Не удалось обработать голосовое сообщение. Пожалуйста, попробуйте еще раз или отправьте текстовое сообщение.")

        # Удаляем временный файл в случае ошибки
        if os.path.exists(temp_path):
            os.unlink(temp_path)

        return None


async def transcribe_with_whisper(audio_path, client):
    """Транскрибирует аудиофайл с помощью Whisper API через Groq"""
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                language="ru"
            )
        return transcription.text
    except Exception as e:
        logger.error(f"Ошибка Whisper API: {str(e)}")
        return None

# Функция для конвертации аудио, если необходимо
'''
def convert_audio_for_api(input_path, output_path):
    """Конвертирует аудиофайл в формат, подходящий для API"""
    try:
        from pydub import AudioSegment

        # Определяем формат входного файла
        audio = AudioSegment.from_file(input_path)

        # Конвертируем в WAV для API (16 кГц, моно)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)

        # Экспортируем в новый файл
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        logger.error(f"Ошибка при конвертации аудио: {str(e)}")
        return False
'''