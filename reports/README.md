# 📊 Explainabilidade do Modelo de Risco de Crédito

## 📖 Índice

1. [O que é SHAP?](#o-que-é-shap)
2. [Importância da Explainabilidade](#importância-da-explainabilidade)
3. [Interpretação dos Gráficos](#interpretação-dos-gráficos)
4. [Impacto no Risco de Crédito](#impacto-no-risco-de-crédito)
5. [Como Usar Este Relatório](#como-usar-este-relatório)

---

## O que é SHAP?

**SHAP** (SHapley Additive exPlanations) é uma metodologia baseada em teoria dos jogos que explica a saída de qualquer modelo de aprendizado de máquina. Ele usa o conceito de valores Shapley da teoria cooperativa dos jogos para atribuir a cada feature uma contribuição para a previsão.

### Características Principais:

- **Fundamentado em Teoria**: Baseado em valores Shapley, com garantias teóricas de equidade
- **Local e Global**: Explica previsões individuais e padrões globais do modelo
- **Agnóstico**: Funciona com qualquer tipo de modelo (linear, árvores, redes neurais, etc.)
- **Interpretável**: Valores são facilmente compreendidos por stakeholders não-técnicos

### Formula Básica:

```
Previsão = Valor Base + Σ(Contribuição de cada Feature)
```

Cada feature recebe um valor SHAP que indica sua contribuição positiva ou negativa para afastar-se do valor base do modelo.

---

## Importância da Explainabilidade

### Por que Explainabilidade em Risco de Crédito?

#### 1. **Conformidade Regulatória**
- Lei Geral de Proteção de Dados (LGPD)
- Resolução 4.945/2021 do Banco Central
- Clientes têm direito de saber por que foram rejeitados

#### 2. **Confiança e Aceitação**
- Modelos inexplicáveis causam desconfiança
- Stakeholders internos precisam entender decisões
- Reduz fricção com áreas de compliance e auditoria

#### 3. **Detecção de Bias**
- Identifica se model aprende discriminações injustas
- Revela dependências inesperadas em dados
- Permite correção antes de impacto financeiro

#### 4. **Melhoria Contínua**
- Identifica features realmente relevantes
- Justifica decisões de feature engineering
- Facilita debugging de modelos com performance baixa

#### 5. **Internabilidade**
- Profissionais de crédito entendem a lógica
- Facilita integração com processos existentes
- Reduz tempo de treinamento de novos users

---

## Interpretação dos Gráficos

### 📌 SHAP Summary Plot (shap_summary.png)

**O que mostra:**
Visualização que combina importância de features e direção do impacto. Cada ponto representa uma amostra individual.

**Como interpretar:**

- **Eixo Horizontal (Feature Value Impact)**: Magnitude da contribuição SHAP
  - À esquerda (negativo): Feature empurra previsão para "Bom Pagador"
  - À direita (positivo): Feature empurra previsão para "Inadimplente"

- **Cores**:
  - 🔴 **Vermelho**: Valor alto da feature
  - 🔵 **Azul**: Valor baixo da feature

- **Altura (Ordem de Features)**: Features ordenadas por importância média

**Exemplo de Interpretação:**

```
Feature: Idade
├─ Pontos Vermelhos à direita → Clientes jovens têm risco maior
├─ Pontos Azuis à esquerda → Clientes idosos têm risco menor
└─ Conclusão: Idade tem forte relação com risco
```

### 📌 SHAP Bar Plot (shap_bar.png)

**O que mostra:**
Importância média absoluta de cada feature. Quanto mais longa a barra, mais impactante a feature nas decisões do modelo.

**Como interpretar:**

- **Comprimento da Barra**: Magnitude média da contribuição |SHAP|
- **Ordem**: Features mais importantes acima
- **Sem noção de direção**: Só mostra magnitude, não se é positivo/negativo

**Exemplo de Leitura:**

```
Feature 1: ████████████ (0.85) - MUITO importante
Feature 2: ████████     (0.52) - Importante
Feature 3: ████         (0.28) - Relevante
Feature 4: █            (0.05) - Pouco relevante
```

---

## Impacto no Risco de Crédito

### 🎯 Features Críticas de Risco Esperadas

Baseado no domínio de risco de crédito, features que costumam ser **muito importantes**:

| Feature | Impacto Esperado | Interpretação |
|---------|-----------------|----------------|
| **Idade** | Alto | Clientes jovens: maior risco; Clientes sênior: menor risco |
| **Renda Anual** | Alto | Renda baixa: maior risco; Renda alta: menor risco |
| **Histórico de Pagamento** | Crítico | Atrasos passados forte preditor de inadimplência |
| **Utilização de Crédito** | Alto | Utilização alta: maior risco (stress financeiro) |
| **Número de Contas Abertas** | Médio | Muitas contas: maior risco (comportamento de risco) |
| **Dias em Atraso** | Crítico | Histórico de atraso: risco muito alto |

### ⚠️ Sinais de Alerta

Se observar estes padrões, pode indicar problema no modelo:

**Sinais Negativos:**
- ❌ Features que não fazem sentido financeiro no top-5
- ❌ Feature completamente irrelevante tem alta importância
- ❌ Padrões muito diferentes entre grupos demográficos (possível bias)

**Sinais Positivos:**
- ✅ Histórico de pagamento entre top-3 features
- ✅ Renda e utilização de crédito têm alta importância
- ✅ Padrões consistentes e interpretáveis

---

## Como Usar Este Relatório

### 📋 Para Analistas de Crédito

1. Visualize os gráficos SHAP
2. Identifique as features mais importantes
3. Valide: "Faz sentido com minha experiência?"
4. Use para apoiar discussões sobre decisões borderline

**Pergunta-chave**: "Para rejeitar este cliente, quais fatores foram decisivos?"

### 📋 Para Cientistas de Dados

1. Analise importância de features vs. correlação
2. Detecte features que o modelo usa de forma inesperada
3. Otimize feature engineering com base em insights SHAP
4. Monitore se SHAP values mudam ao longo do tempo (drift)

**Pergunta-chave**: "O modelo aprendeu relações causais ou correlações espúrias?"

### 📋 Para Compliance / Auditoria

1. Documente como o modelo toma decisões
2. Valide ausência de discriminação ilegal
3. Justifique rejeições para clientes questionadores
4. Prepare argumentação para reguladores

**Pergunta-chave**: "Posso explicar e justificar cada decisão?"

---

## 🔄 Workflow Recomendado

### 1️⃣ **Análise Inicial**
```bash
python -m src.modeling.explain
# Gera shap_summary.png e shap_bar.png
```

### 2️⃣ **Inspeção Visual**
- Abra `/reports/figures/shap_bar.png`
- Identifique top-5 features mais importantes

### 3️⃣ **Validação de Negócio**
- Discuta com especialistas em crédito
- Valide se padrões fazem sentido
- Documente conclusões

### 4️⃣ **Ação**
- Se válido: Mantenha modelo e utilize para explicações
- Se inválido: Revise dados, features ou modelo
- Se parcialmente válido: Ajuste interpretação ou retreine

---

## 📚 Referências

### Literatura Científica
- **Lundberg & Lee (2017)**: "A Unified Approach to Interpreting Model Predictions" ([Paper](https://arxiv.org/abs/1705.07874))
- **Covert et al. (2020)**: "Understanding Global Feature Contributions Through Additive Importance Measures"

### Documentação Online
- [SHAP Documentation](https://shap.readthedocs.io/)
- [SHAP GitHub](https://github.com/slundberg/shap)

### Regulações Brasileiras
- [LGPD - Lei Geral de Proteção de Dados](http://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
- [Resolução 4.945/2021 - Banco Central](https://www.bcb.gov.br/estabilidadefinanceira/exigencias_regulamentares)

---

## 📝 Notas Técnicas

### Configurações Utilizadas

```yaml
max_samples: 2000          # Limite para performance
random_state: 42           # Reprodutibilidade
sample_strategy: stratified # Preserva distribuição de classe
figure_dpi: 300           # Alta resolução
explainer_type: TreeExplainer # Se disponível; fallback para KernelExplainer
```

### Gerar Explicações

```bash
# Básico
python -m src.modeling.explain

# Com parametrização (futuro)
python -m src.modeling.explain --max-samples 5000 --output /custom/path
```

### Resultados Esperados

Arquivos gerados em `/reports/figures/`:

```
📁 reports/figures/
├── shap_summary.png    # Summary plot (força + direção)
└── shap_bar.png        # Bar plot (importância média)
```

---

## 🎯 Conclusão

SHAP fornece uma explicação teoricamente sólida, localmente acurada e globalmente coerente das decisões do modelo. Em risco de crédito, onde decisões impactam vidas e requerem conformidade regulatória, explainabilidade não é apenas desejável — é **essencial**.

Utilize estes gráficos para construir confiança, detectar problemas e melhorar continuamente seu modelo de risco.

---

**Última atualização**: 2026-05-06  
**Versão**: 1.0  
**Autor**: MLOps Engineering Team
