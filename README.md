# Diagnóstico Sensível ao Custo da Origem de Falhas — Scania APS Failure (ML)

Projeto da disciplina **Tópicos Especiais em Sistemas Inteligentes: Aprendizado de Máquina** (CEFET-MG — Campus V, Divinópolis).  
**Autor:** João Vitor Lobato Romualdo

---

## Sobre o Projeto

Este repositório implementa e avalia um arcabouço experimental de **Aprendizado de Máquina Sensível ao Custo** voltado ao diagnóstico e à triagem da origem de falhas no Sistema de Pressão de Ar (*Air Pressure System* — APS) de caminhões pesados da Scania, utilizando o benchmark público **[APS Failure at Scania Trucks (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)** (60.000 instâncias de treino com 1,67% de falhas no APS, 16.000 de teste com 2,34%, totalizando $\approx 1{,}81\%$ global, e 170 atributos numéricos anonimizados).

Na base oficial, os veículos já apresentaram falha e a tarefa consiste em identificar se a anomalia provém do APS (classe positiva, `pos`) ou de componentes não relacionados ao APS (classe negativa, `neg`). O estudo contrasta um modelo linear transparente e leve (**Regressão Logística** com ponderação balanceada de classes) com comitês não lineares de árvores de decisão (**Random Forest**) sob a métrica oficial assimétrica do fabricante:

$$C_{total} = 10 \cdot FP + 500 \cdot FN$$

onde $FP$ contabiliza verificações diagnósticas desnecessárias do APS em oficina (US&#36; 10) e $FN$ contabiliza falhas reais do APS não diagnosticadas (US&#36; 500).

Além da avaliação nominal com ajuste de limiar *out-of-fold* ($\theta_{OOF}$) em validação cruzada estratificada e aninhada (*Nested CV*) de 5 dobras sem vazamento de dados, reamostragem *bootstrap* pareada ($10.000$ amostras, reportando intervalos de confiança $IC_{95\%}$, $p$-valores unilaterais para a hipótese direcional motivada pelo benchmark $C_{RF} < C_{LR}$ e bilaterais), avaliação multi-semente (5 sementes na *Random Forest*) e análise de interpretabilidade (Top-10 atributos e correlação de Spearman $\rho_S$), o trabalho investiga três eixos complementares:
1. **Estabilidade frente ao Tratamento de Dados Ausentes (E1 a E5):** Comparação entre imputação por mediana, média, constante zero, adição de indicadores binários de ausência (*Missingness Indicators* — MNAR) e preservação integral das 170 colunas.
2. **Calibração Pós-Treinamento de Probabilidades:** Impacto do *Platt Scaling* (Sigmoidal) e da **Regressão Isotônica (PAVA)** sobre o *Brier Score* e sobre a viabilidade de aplicar diretamente o limiar ótimo teórico de Bayes ($\theta^* = \frac{c_{FP}}{c_{FP}+c_{FN}} \approx 0{,}0196$).
3. **Análise de Sensibilidade à Razão de Custos ($\rho = c_{FN}/c_{FP} \in \lbrace 10, 25, 50, 100, 200 \rbrace$):** Delimitação dos regimes operacionais em que o sobrecusto do modelo linear interpretável permanece moderado frente à *Random Forest*.

---

## Principais Resultados (Conjunto de Teste — 16.000 Instâncias)

| Configuração / Modelo | Limiar ($\theta$) | Recall | Precisão | ROC-AUC | PR-AUC | $FP$ (US&#36; 10) | $FN$ (US&#36; 500) | Custo Teste ($IC_{95\%}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| **Random Forest (Padrão)** | 0,50 | 59,20% | 93,28% | 0,9925 | 0,8868 | 16 | 153 | US&#36; 76.660 |
| **Regressão Logística (Padrão)** | 0,50 | 93,87% | 49,51% | 0,9892 | 0,8053 | 359 | 23 | US&#36; 15.090 |
| **Regressão Logística (Limiar OOF - E1, Ref.)** | 0,39 | 94,67% | 44,37% | 0,9892 | 0,8053 | 445 | 20 | **US&#36; 14.450** ([10.330; 19.050]) |
| **Regressão Logística (Mediana + Ind. MNAR - E4)** | 0,29 | 96,27% | 36,68% | 0,9875 | 0,8044 | 591 | 14 | **US&#36; 12.910** ([9.470; 16.870]) |
| **Random Forest (Limiar OOF - E1, Ref.)** | 0,04 | 97,07% | 43,33% | 0,9925 | 0,8868 | 476 | 11 | **US&#36; 10.260** ([7.250; 13.780]) |
| **Random Forest (Média de 5 Sementes - E1)** | 0,04 | — | — | — | — | — | — | **US&#36; 10.008 ± 1.082** |
| **Random Forest (Mediana + Ind. MNAR - E4, Explor.)** | 0,04 | 97,60% | 42,51% | 0,9924 | 0,8901 | 495 | 9 | **US&#36; 9.450** ([6.730; 12.630]) |
| **Random Forest + Calibração de Platt (Bayes $\theta^*$, Explor.)** | 0,0196 | **98,67%** | 38,50% | 0,9925 | 0,8868 | 591 | **5** | **US&#36; 8.410** |

> **Destaques Estatísticos e Operacionais:**
> - Na estratégia primária de referência **E1**, a diferença pareada no teste é $\Delta C = C_{LR} - C_{RF} = +\text{US&#36; } 4.190$ ($IC_{95\%}: [+\text{US&#36; } 250;\; +\text{US&#36; } 8.320]$, $p_{uni} = 0{,}0185$, $p_{bi} = 0{,}0370$).
> - Em contrapartida, a Regressão Logística é **13,4x mais rápida na inferência** ($3{,}66\text{ ms}$ contra $48{,}89\text{ ms}$) e **2,9x mais rápida no treinamento** ($2{,}45\text{ s}$ contra $7{,}13\text{ s}$), exigindo apenas 163 parâmetros.
> - Todas as tabelas completas (incluindo o Top-10 de interpretabilidade e as 15 colunas das estratégias E1–E5) encontram-se documentadas em [RELATORIO_FINAL.md](RELATORIO_FINAL.md) e exportadas em [Praticas/resultados_etapa3.json](Praticas/resultados_etapa3.json).

---

## Estrutura do Repositório

```text
├── experimento_scania.py                   # Pipeline completo (5-fold CV, Nested CV, E1-E5, bootstrap 10k, calibração, sensibilidade e gráficos)
├── requirements.txt                        # Dependências Python (numpy, pandas, scikit-learn, scipy, matplotlib)
├── README.md                               # Visão geral do projeto e instruções de execução
├── RELATORIO_FINAL.md                      # Relatório técnico completo e apêndice de tabelas detalhadas do artigo
├── Praticas/
│   ├── artigo.tex                          # Artigo científico completo no template IEEEtran
│   ├── resultados_etapa3.json              # Métricas detalhadas exportadas pelo experimento
│   ├── fig_custo_limiar.pdf / .png         # Figura 2 do artigo: Curvas de custo OOF vs Teste e decomposição FP/FN
│   ├── fig_estabilidade_imputacao.pdf/.png # Figura 3 do artigo: Estabilidade e IC95% bootstrap entre estratégias E1-E5
│   ├── fig_calibracao_probabilidades.pdf   # Figura 4 do artigo: Diagrama de confiabilidade e Limiar de Bayes vs OOF
│   └── fig_sensibilidade_custos.pdf / .png # Figura 5 do artigo: Sensibilidade à razão de custos c_FN/c_FP (10:1 a 200:1)
└── dados/
    └── aps+failure+at+scania+trucks/
        └── aps_failure_description.txt     # Documentação oficial da base UCI
```

---

## Como Executar

1. Baixe o arquivo `aps+failure+at+scania+trucks.zip` do [repositório oficial da UCI](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) em `dados/aps+failure+at+scania+trucks.zip`.
2. Instale as dependências e execute o script principal:
   ```bash
   pip install -r requirements.txt
   python experimento_scania.py
   ```
