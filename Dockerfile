FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Сначала установите базовые зависимости без torch
COPY requirements.txt .
RUN pip install --no-cache-dir \
    python-telegram-bot \
    langchain-community \
    sentence-transformers \
    gitpython \
    python-dotenv \
    chardet \
    python-slugify

# Затем установите torch с правильным индексом
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Установите оставшиеся зависимости
RUN pip install --no-cache-dir \
    transformers==4.41.2 \
    faiss-cpu==1.7.4 \
    accelerate==0.30.1 \
    bitsandbytes==0.43.0 \
    huggingface-hub==0.19.4

# Копируем весь проект
COPY . .

# Создаем необходимые директории
RUN mkdir -p /app/data/repos && \
    mkdir -p /app/data/reports && \
    mkdir -p /app/data/rag_db

# Запускаем бота
CMD ["python", "-m", "bot.main"]