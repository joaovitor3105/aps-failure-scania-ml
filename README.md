# Manutenção Preditiva Sensível ao Custo — Scania APS Failure (ML)

Projeto da disciplina **Tópicos Especiais em Sistemas Inteligentes: Aprendizado de Máquina** (CEFET-MG — Campus V, Divinópolis).  
**Autor:** João Vitor Lobato Romualdo

---

## Sobre o Projeto

Este repositório implementa e avalia um arcabouço experimental de **Aprendizado de Máquina Sensível ao Custo** voltado ao diagnóstico de falhas no Sistema de Pressão de Ar (*Air Pressure System* — APS) de caminhões pesados da Scania, utilizando o benchmark público **[APS Failure at Scania Trucks (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)** (60.000 instâncias de treino, 16.000 de teste, 170 atributos numéricos e ~1,8% de falhas).

O estudo contrasta um modelo linear transparente e leve (**Regressão Logística** com ponderação balanceada de classes) com comitês não lineares de árvores de decisão (**Random Forest**) sob a métrica oficial assimétrica do fabricante:

$$C_{total} = 10 \cdot FP + 500 \cdot FN$$

Além da avaliação nominal com ajuste de limiar *out-of-fold* ($\theta_{OOF}$) em validação cruzada estratificada de 5 dobras sem vazamento de dados, o trabalho investiga três eixos fundamentais:
1. **Estabilidade frente ao Tratamento de Dados Ausentes (E1 a E5):** Comparação entre imputação por mediana, média, constante zero, adição de indicadores binários de ausência (*Missingness Indicators* — MNAR) e preservação integral das 170 colunas.
2. **Calibração Pós-Treinamento de Probabilidades:** Impacto do *Platt Scaling* (Sigmoidal) e da **Regressão Isotônica (PAVA)** sobre o *Brier Score* e sobre a viabilidade de aplicar diretamente o limiar ótimo teórico de Bayes ($\theta^* = \frac{c_{FP}}{c_{FP}+c_{FN}} \approx 0{,}0196$).
3. **Análise de Sensibilidade à Razão de Custos ($\rho = c_{FN}/c_{FP} \in \lbrace 10, 25, 50, 100, 200 \rbrace$):** Delimitação dos regimes operacionais em que o sobrecusto do modelo linear interpretável permanece competitivo frente à *Random Forest*.

---

## Principais Resultados (Conjunto de Teste — 16.000 Instâncias)

| Configuração / Modelo | Limiar ($\theta$) | Recall | Precisão | ROC-AUC | PR-AUC | $FP$ (US&#36; 10) | $FN$ (US&#36; 500) | Custo Scania (US&#36;) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| **Random Forest (Padrão)** | 0,50 | 59,20% | 93,28% | 0,9925 | 0,8868 | 16 | 153 | US&#36; 76.660 |
| **Regressão Logística (Padrão)** | 0,50 | 93,87% | 49,51% | 0,9892 | 0,8053 | 359 | 23 | US&#36; 15.090 |
| **Regressão Logística (Limiar OOF - E1)** | 0,39 | 94,67% | 44,37% | 0,9892 | 0,8053 | 445 | 20 | **US&#36; 14.450** |
| **Regressão Logística (Mediana + Ind. MNAR - E4)** | 0,29 | 96,27% | 36,68% | 0,9875 | 0,8044 | 591 | 14 | **US&#36; 12.910** |
| **Random Forest (Limiar OOF - E1)** | 0,04 | 97,07% | 43,33% | 0,9925 | 0,8868 | 476 | 11 | **US&#36; 10.260** |
| **Random Forest (Mediana + Ind. MNAR - E4)** | 0,04 | 97,60% | 42,51% | 0,9924 | 0,8901 | 495 | 9 | **US&#36; 9.450** |
| **Random Forest + Calibração de Platt (Bayes $\theta^*$)** | 0,0196 | **98,67%** | 38,50% | 0,9925 | 0,8868 | 591 | **5** | **US&#36; 8.410** |

---

## Estrutura do Repositório

```text
├── experimento_scania.py                   # Pipeline completo (5-fold CV, E1-E5, calibração, sensibilidade e gráficos)
├── requirements.txt                        # Dependências Python
├── Praticas/
│   ├── artigo.tex                          # Artigo científico completo no template IEEEtran (Etapas I, II e III)
│   ├── resultados_etapa3.json              # Métricas detalhadas exportadas pelo experimento
│   ├── fig_custo_limiar.pdf / .png         # Figura 1: Curvas de custo OOF vs Teste e decomposição FP/FN
│   ├── fig_estabilidade_imputacao.pdf/.png # Figura 2: Estabilidade entre estratégias de dados ausentes (E1-E5)
│   ├── fig_calibracao_probabilidades.pdf   # Figura 3: Diagrama de confiabilidade e Limiar de Bayes vs OOF
│   ├── fig_sensibilidade_custos.pdf / .png # Figura 4: Sensibilidade à razão de custos c_FN/c_FP (10:1 a 200:1)
│   └── Etapa_I.pdf, Etapa_II.pdf, Etapa_III_Resultados_Discussao_e_Conclusoes-1.pdf
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
