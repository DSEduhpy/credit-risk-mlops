# Roadmap do Projeto

## ✅ Concluído

- [x] Pipeline DVC implementado (ingest → processing → features → train)
- [x] Versionamento de dados e artefatos com DVC
- [x] MLflow configurado para rastreamento de experimentos
- [x] Pipeline de modelagem com logging de parâmetros, métricas e artefatos
- [x] API de inferência com FastAPI criada
- [x] Imagem Docker construída para portabilidade
- [x] Arquitetura modular por domínio definida
- [x] Threshold otimizado por custo de negócio
- [x] Simulação de impacto financeiro implementada
- [x] Explainability SHAP integrado

## 🚧 Em andamento

- [ ] Métricas de estabilidade de modelo em ambiente de produção
- [ ] Métricas de drift consolidadas para detecção de distribuição
- [ ] Pipeline compatível com GitHub Actions / CI-CD
- [ ] Registro de artefatos e versionamento de modelo no MLflow refinado
- [ ] Documentação técnica dos fluxos de arquitetura e projeto

## 📅 Próximas versões

- [ ] Monitoramento contínuo de drift em produção
- [ ] Automação de deploy e rollback em nuvem (AWS / GCP)
- [ ] Validação de schema de dados automatizada
- [ ] Testes de integração de pipeline e regressão de dados
- [ ] Avaliação de Feature Store para reuso de features

## 🎯 Melhorias futuras

- [ ] Observabilidade avançada (alertas, métricas SLA de inferência)
- [ ] Estrutura de governança de dados e métricas de viés
- [ ] Arquitetura de A/B testing de modelos
- [ ] Orquestração de inferência em lote e em tempo real
- [ ] Suporte a modelo em produção com rollback automático
