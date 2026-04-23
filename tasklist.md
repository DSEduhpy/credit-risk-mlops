# Tasklist

## ✅ Concluído

- [x] Implementar pipeline DVC (ingest → processing → features → train)
- [x] Versionamento de artefatos e dados com DVC
- [x] Configurar MLflow como rastreamento de experimentos (SQLite backend)
- [x] Implementar pipeline de modelagem com logging de parâmetros, métricas e artefatos
- [x] Criar API de inferência com FastAPI
- [x] Construir imagem Docker para portabilidade
- [x] Definir arquitetura modular por domínio (ingestion, processing, modeling, api, monitoring)
- [x] Otimização de threshold orientada por função de custo de negócio
- [x] Simulação de impacto financeiro com custo de inadimplência e lucro por cliente

## 🔄 Em andamento

- [ ] Avaliar métricas de estabilidade de modelo em ambientes de produção
- [ ] Consolidar métricas de drift para detecção de alterações de distribuição
- [ ] Refinamento de pipeline para compatibilidade com GitHub Actions e CI/CD
- [ ] Ajuste fino de captura de artefatos e versões de modelo no MLflow

## ⏳ Próximos passos

- [ ] Implementar explicabilidade com SHAP e análise local/global de features
- [ ] Desenvolver monitoramento de drift contínuo em produção
- [ ] Adicionar testes automatizados de integração do pipeline e regressão de dados
- [ ] Incluir validação de schema de dados com Great Expectations ou ferramenta equivalente
- [ ] Projetar fluxo de deploy em cloud (AWS / GCP) com infraestrutura imutável
- [ ] Avaliar arquitetura de Feature Store para ingestão e reuso de features
- [ ] Implantar CI/CD para DVC, MLflow e deploy do serviço de inferência
- [ ] Documentar plano de rollback e validação pós-deploy
