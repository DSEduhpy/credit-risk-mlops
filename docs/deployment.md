# 📦 Implantação com Docker, Compose e infraestrutura de referência

Este documento descreve a forma atual de preparar o ambiente local e de referência para produção usando Docker, Docker Compose, observabilidade, automação de CI/CD e uma arquitetura simples para deploy em AWS.

## Estrutura atual de containers

- `api`: container da API FastAPI.
- `prometheus`: coleta métricas da API.
- `grafana`: visualiza métricas e status.

## Comando de inicialização

Execute no diretório raiz do repositório:

```bash
docker compose up --build
```

Os serviços ficam disponíveis em:

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Variáveis de ambiente

- `CREDIT_RISK_ENV=prod`: define o modo de execução em produção.
- `PORT`: porta de escuta da API.
- `API_WORKERS`: número de workers Uvicorn.

## Volumes persistentes

- `api-models`: modelos treinados e persistidos.
- `api-mlruns`: artefatos e experimentos locais.
- `api-logs`: logs da aplicação.

## Rede

O compose cria uma rede dedicada `credit-risk-net`, garantindo isolamento e comunicação interna entre os serviços.

## Health checks

- A API expõe `/health` e `/ready`.
- O container da API realiza um health check simples no endpoint `/health`.

## Recomendações

- Use `docker compose down` para parar e remover a infraestrutura.
- O container da API é construído a partir do `Dockerfile` do projeto.
- Para produção real, substitua as credenciais padrão do Grafana e configure secrets externos.
- A arquitetura de AWS é deixada pronta para extensão com ECR, EC2, IAM, Security Groups e Terraform, mas a validação real depende de credenciais externas.

## Deploy AWS de referência

### 1. EC2
- Criar uma instância EC2 com Ubuntu.
- Conectar via SSH e instalar Docker e Docker Compose.

### 2. ECR
- Criar um repositório ECR para a imagem da API.
- Autenticar o cliente Docker localmente e publicar a imagem.

### 3. Variáveis de ambiente
- `CREDIT_RISK_ENV=prod`
- `PORT=8000`
- `API_WORKERS=2`

### 4. Secrets
- Armazenar credenciais sensíveis em secrets do ambiente ou em um gerenciador externo.

### 5. Deploy
- Clonar o repositório na EC2.
- Executar `docker compose up --build -d`.
- Validar `http://<ip-da-ec2>:8000/health`.
