# Groq Telegram Bot

## Описание
Этот Telegram бот предоставляет доступ к различным моделям искусственного интеллекта через Groq API в групповых чатах. Бот может отвечать на текстовые запросы и обрабатывать голосовые сообщения, превращая их в текст с последующей генерацией ответа.

## Основные возможности
- Взаимодействие с крупными языковыми моделями (LLM) через Groq API
- Обработка текстовых запросов с упоминанием бота
- Распознавание голосовых сообщений с помощью модели Whisper
- Настройка параметров моделей для каждого чата
- Сохранение истории сообщений для контекстного понимания
- Администрирование бота через специальные команды

## Требования
- Python 3.8+
- Telegram Bot Token
- Groq API Key
- Установленные зависимости из `requirements.txt`

## Установка и настройка

### 1. Клонирование репозитория
```
git clone <url-репозитория>
cd <папка-репозитория>
```

### 2. Установка зависимостей
```
pip install -r requirements.txt
```

### 3. Создание .env файла
Создайте файл `.env` в корневой директории проекта со следующим содержимым:
```
TELEGRAM_TOKEN=ваш_токен_телеграм_бота
GROQ_API_KEY=ваш_ключ_groq_api
```

### 4. Запуск бота
```
python bot.py
```

## Структура проекта
- `bot.py` - Основной файл бота с логикой работы
- `model_info.py` - Информация о доступных моделях и их характеристиках
- `voice_handler.py` - Обработчик голосовых сообщений
- `bot_data.db` - SQLite база данных для хранения настроек и истории сообщений

## Команды бота

### Общие команды
- `/start` - Запуск бота и приветствие
- `/help` - Показать список доступных команд
- `/explain` - Подробное руководство по использованию бота
- `/models` - Информация о доступных моделях

### Команды для администраторов
- `/settings` - Показать текущие настройки бота в чате
- `/set_model [модель]` - Установить модель для использования
- `/set_temp [0.0-1.0]` - Установить температуру генерации (креативность)
- `/set_max_tokens [число]` - Установить максимальное количество токенов ответа
- `/toggle` - Включить/выключить бота в текущем чате
- `/clear_history` - Очистить историю сообщений для текущего чата

## Доступные модели
- **llama3-70b-8192** - Мощная модель для сложных задач 
- **llama3-8b-8192** - Быстрая и лёгкая модель для простых вопросов
- **deepseek-r1-distill-llama-70b** - Резервная модель без дневного лимита токенов
- **gemma2-9b-it** - Компактная модель от Google для задач средней сложности
- **mistral-saba-24b** - Универсальная модель среднего размера

## Обработка голосовых сообщений
Бот поддерживает обработку голосовых сообщений с помощью модели `whisper-large-v3`. Чтобы бот обработал голосовое сообщение, необходимо:
- Упомянуть бота в подписи к голосовому сообщению
- Или ответить голосовым сообщением на сообщение бота

## Особенности работы с базой данных
- Бот сохраняет настройки для каждого чата
- История сообщений хранится для контекстного понимания
- Старые сообщения автоматически удаляются через 30 дней
- База данных регулярно очищается для оптимизации производительности

## Логирование
Бот ведет подробное логирование своей работы:
- Все логи сохраняются в файл `bot.log`
- Используется система ротации логов (максимальный размер файла 10 МБ)
- Каждый час происходит запись статистики бота в лог

## Рекомендации по настройке
- Для общего использования: llama3-70b-8192, температура 0.7
- Для точных ответов и фактов: llama3-70b-8192, температура 0.2
- Для креативных задач: llama3-70b-8192, температура 0.9
- Для быстрых ответов на простые вопросы: llama3-8b-8192, температура 0.5

## Ограничения
- Максимальная длительность голосового сообщения: 5 минут
- Максимальное количество сообщений в истории: 50
- Ограничения по количеству запросов и токенов в зависимости от выбранной модели

## Авторы
Telegram: @bob_1985
