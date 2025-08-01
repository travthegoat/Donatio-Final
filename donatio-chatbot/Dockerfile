FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--reload", "--port", "8001", "--host", "0.0.0.0"]
