FROM python:3.12-slim

WORKDIR /app

# Replace sources.list with Aliyun mirror for Debian 12 (Bookworm)
RUN echo "deb http://mirrors.aliyun.com/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends libzbar0 && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy and install dependencies
COPY pyproject.toml .
RUN uv sync

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uv run manage.py boom"]
