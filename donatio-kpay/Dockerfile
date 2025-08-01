FROM python:3.12-slim

WORKDIR /app

# Replace sources.list with Aliyun mirror for Debian 12 (Bookworm)
RUN echo "deb http://mirrors.aliyun.com/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv sync

ENV PATH="/app/.venv/bin:$PATH"

COPY . .

EXPOSE 8002

CMD ["uvicorn", "main:app", "--reload", "--port", "8002", "--host", "0.0.0.0"]