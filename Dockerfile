FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV TZ=Europe/Moscow

RUN apt-get update && \
    apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

COPY . .

CMD ["python3", "-m", "bot.main"]