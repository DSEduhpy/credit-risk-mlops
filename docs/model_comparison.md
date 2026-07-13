# Comparação de Modelos

Este documento compara seis algoritmos relevantes para classificação de risco de crédito. O foco é identificar o melhor trade-off entre performance, interpretabilidade, custo de produção e facilidade de operação.

## Modelos comparados

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- MLP (Rede Neural)

## 1. Logistic Regression

### Vantagens
- Modelo linear simples e interpretável.
- Rápido de treinar e inferir.
- Funciona bem como baseline.
- Fácil auditoria e explicação.

### Desvantagens
- Capacidade limitada para relações não lineares.
- Sensível a features não escaladas.
- Menos robusto em dados com interações complexas.

### Tempo de treino
- Muito rápido em datasets médios.

### Interpretabilidade
- Alta. Coeficientes e odds ratios são facilmente compreensíveis.

### SHAP
- Explicações simples e estáveis.

### Produção
- Baixo custo computacional.
- Fácil de servir em API.

### Uso recomendado
- Quando a governança exige explicabilidade rígida.
- Para baseline e validação de conceitos.

## 2. Random Forest

### Vantagens
- Boas métricas sem muita engenharia de features.
- Funciona bem com dados mistos e ruído.
- Não exige escala de features.

### Desvantagens
- Interpretabilidade menor que regressão linear.
- Modelo grande e mais lento para inferência.
- Uso intensivo de memória.

### Tempo de treino
- Moderado a alto, dependendo de número de árvores.

### Interpretabilidade
- Média. Permite feature importance, mas não tão transparente.

### SHAP
- Viável, mas mais caro em tempo de inferência e memória.

### Produção
- Bom para batch scoring, menos ideal para latência baixa.

### Uso recomendado
- Quando a prioridade é precisão sobre interpretabilidade.

## 3. XGBoost

### Vantagens
- Excelente performance em muitos problemas tabulares.
- Balanceia precisão e controle de regularização.
- Ampla adoção e suporte.

### Desvantagens
- Mais complexo de calibrar.
- Tempo de treino e inferência maior que modelos lineares.
- Requer ajuste de hiperparâmetros para melhor performance.

### Tempo de treino
- Moderado a alto.

### Interpretabilidade
- Média. Shapley values ajudam, mas o modelo não é intrinsecamente transparente.

### SHAP
- Funciona bem com TreeExplainer e é relativamente eficiente.

### Produção
- Suporta inferência rápida em CPU com parâmetros otimizados.

### Uso recomendado
- Quando se busca alta acurácia em problemas de crédito sem sacrificar muito a velocidade.

## 4. LightGBM

### Vantagens
- Treina muito rápido em grandes datasets.
- Memória eficiente.
- Bom desempenho em dados tabulares.

### Desvantagens
- Pode exigir cuidados com instabilidades em dados altamente desbalanceados.
- Configuração de bins e regularização influencia resultados.

### Tempo de treino
- Rápido.

### Interpretabilidade
- Média, similar ao XGBoost.

### SHAP
- Compatível com TreeExplainer e rápido.

### Produção
- Um dos mais eficientes para inferência em CPU.

### Uso recomendado
- Quando desempenho e custo de inferência são críticos.

## 5. CatBoost

### Vantagens
- Forte handling de features categóricas.
- Menos necessidade de codificação manual.
- Forte performance out-of-the-box.

### Desvantagens
- Pode ser mais lento para treinar que LightGBM.
- Modelo maior e mais pesado em memória.

### Tempo de treino
- Moderado.

### Interpretabilidade
- Média. Supporte SHAP, mas o modelo é complexo.

### SHAP
- Funciona bem, especialmente em dados categóricos.

### Produção
- Boa opção quando há muitas variáveis categóricas e pouca engenharia de features.

### Uso recomendado
- Quando a preparação de features deve ser minimizada e há forte dependência de categorias.

## 6. MLP (Rede Neural)

### Vantagens
- Capaz de capturar não linearidades complexas.
- Flexível para arquiteturas profundas.

### Desvantagens
- Menor interpretabilidade.
- Requer mais dados e tuning.
- Inferência pode ser mais lenta.
- Menor transparência para compliance financeira.

### Tempo de treino
- Alto dependendo da arquitetura.

### Interpretabilidade
- Baixa a média. Requer técnicas adicionais como SHAP ou LIME.

### SHAP
- Pode ser aplicado, mas é mais custoso e menos estável.

### Produção
- Pode exigir GPU ou otimizações especiais.

### Uso recomendado
- Apenas se houver volume de dados muito grande e padrões complexos que modelos tree-based não capturam.

## Tabela comparativa

| Modelo | Interpretabilidade | Tempo de treino | Inferência | Consumo de memória | SHAP | Recomendado para este projeto |
|--------|--------------------|-----------------|------------|--------------------|------|------------------------------|
| Logistic Regression | Alta | Muito baixo | Muito rápido | Muito baixo | Excelente | Sim, como baseline |
| Random Forest | Média | Médio | Médio | Alto | Possível | Sim, se batch scoring |
| XGBoost | Média | Médio | Rápido | Médio | Bom | Sim, forte candidato |
| LightGBM | Média | Baixo | Muito rápido | Baixo | Bom | Sim, forte candidato |
| CatBoost | Média | Médio | Rápido | Médio | Bom | Sim, bom para dados categóricos |
| MLP | Baixa | Alto | Médio a alto | Alto | Possível, mas caro | Não recomendado |

## Quando vale a pena usar cada modelo

- **Logistic Regression**: primeiro modelo de produção; excelente explicabilidade.
- **Random Forest**: quando há necessidade de robustez sem pré-processamento intenso.
- **XGBoost**: quando a precisão é prioritária e a latência pode ser gerenciada.
- **LightGBM**: quando a velocidade de treino e inferência é crítica.
- **CatBoost**: quando há muitas variáveis categóricas e menos engenharia de features.
- **MLP**: apenas em casos de volume e complexidade extrema, com investimento significativo em explicabilidade.

## Recomendação de implementação

Para este projeto de risco de crédito, **não é recomendado** implementar uma Rede Neural no momento. A razão principal é que o projeto já tem um pipeline robusto para modelos tree-based, que oferecem ótimo equilíbrio entre performance e interpretabilidade.

### Por que não implementar

- A interpretabilidade é crucial em aplicações financeiras.
- Modelos lineares e tree-based já atendem aos requisitos de negócio.
- O custo de desenvolvimento, validação e auditoria de uma rede neural é alto.
- O ganho incremental dificilmente justifica o esforço.

## Se fosse implementada, a arquitetura seria

- Pré-processamento e normalização em `src/processing`.
- Conversão de categorias em embeddings ou one-hot.
- Rede feed-forward com camadas densas e dropout.
- Saída binária com ativação sigmoid.
- Treino com early stopping e validação estratificada.
- Explicabilidade usando SHAP DeepExplainer ou KernelExplainer.

### Exemplo simplificado

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
```

## Conclusão

Os modelos mais alinhados ao projeto são **LightGBM**, **XGBoost** e **CatBoost**, com **Logistic Regression** como baseline de governança. A implementação de redes neurais não é recomendada neste momento devido ao custo adicional e à perda de transparência.
