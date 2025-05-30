# Code Quality Bot 🤖

Telegram-бот для автоматического анализа качества кода в GitHub-репозиториях с использованием моделей Hugging Face.

## Основные возможности

✅ Автоматический анализ кода на ошибки и антипаттерны  
✅ Генерация отчетов в формате Markdown  
✅ Интеграция с GitHub через API  
✅ Поддержка RAG-контекста для анализа  
✅ Работа как на CPU, так и на GPU (CUDA)

## Установка

### Требования
- Python 3.10+
- Git
- Telegram аккаунт
- [Hugging Face Token](https://huggingface.co/settings/tokens)

### 1. Клонировать репозиторий

```bash
git clone https://github.com/your-username/code-quality-bot.git
cd code-quality-bot
```


2. Установить зависимости

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Настройка


Создать файл .env в корне проекта:

.env
TELEGRAM_TOKEN=your_telegram_bot_token
GITHUB_TOKEN=your_github_personal_token
MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
RAG_DB_PATH=data/rag_db
REPOS_ROOT=data/repos


Инициализировать RAG-базу:

```bash
python bot/code_analysis.py
```

## Использование

Запуск бота
```bash
python -m bot.main
```

## Команды в Telegram:

/start - Приветственное сообщение

/connect <repo-url> - Подключить репозиторий

/analyze - Запустить анализ

/stop - Остановить бота


## Особенности архитектуры

Модели: Использует Phi-3-mini от Microsoft

Анализ кода: Комбинация LLM и RAG-контекста

Безопасность: Изоляция репозиториев в отдельных директориях

Производительность: Оптимизация для работы с большими кодовыми базами

## Docker-развертывание
```bash
docker build -t code-quality-bot .
docker run -d \
  -e TELEGRAM_TOKEN=your_token \
  -e GITHUB_TOKEN=your_github_token \
  -v ./data:/app/data \
  code-quality-bot
```