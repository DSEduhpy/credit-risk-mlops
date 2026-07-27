# 📦 Implantação com Docker e Docker Compose

Este documento descreve como preparar o ambiente de produção e desenvolvimento usando Docker, Docker Compose e o pipeline CI.

## Estrutura de containers

- `api`: container da API FastAPI.
- `mlflow`: container do servidor MLflow.

## Comando de inicialização

Execute no diretório raiz do repositório:

```bash
docker compose up --build
```

O serviço da API ficará disponível em `http://localhost:8000`.
O servidor MLflow ficará disponível em `http://localhost:5000`.

## Variáveis de ambiente

- `CREDIT_RISK_ENV=prod`: define o modo de execução em produção.
- `MLFLOW_TRACKING_URI`: URL do servidor MLflow usado pela API.
- `PORT`: porta de escuta da API.
- `API_WORKERS`: número de workers Uvicorn.

## Volumes persistentes

- `mlflow-artifacts`: persistência de artefatos e experimentos do MLflow.

## Rede

O compose cria uma rede dedicada `credit-risk-net`, garantindo isolamento e comunicação interna entre `api` e `mlflow`.

## Recomendações

- Use `docker compose down` para parar e remover a infraestrutura.
- O container da API é construído a partir do `Dockerfile` do projeto.
- Em produção, prefira `requirements/prod.txt` para reduzir a superfície de dependências.
