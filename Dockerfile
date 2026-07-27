FROM python:3.14-slim AS base

# Define o diretório de trabalho do container.
WORKDIR /app

# Evita geração de arquivos pyc e permite saída imediata no terminal.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Instala apenas dependências de produção.
COPY requirements requirements
RUN pip install --no-cache-dir -r requirements/prod.txt

# Copia apenas o código-fonte e arquivos necessários para execução.
COPY src ./src
COPY README.md ./

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
