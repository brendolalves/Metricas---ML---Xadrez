# ♟️ Previsão e Classificação de Vitórias em Xadrez com Machine Learning

Este projeto aplica técnicas de **Ciência de Dados** e **Machine Learning** para analisar e prever o desfecho de partidas de xadrez online. Através da comparação entre três algoritmos de classificação (**Árvore de Decisão**, **Naive Bayes** e **Regressão Logística**), avaliamos quais variáveis possuem maior poder discriminatório no resultado final do jogo.

📌 **Repositório GitHub:** [github.com/brendolalves/Metricas---ML---Xadrez](https://github.com/brendolalves/Metricas---ML---Xadrez)

## 📊 Origem dos Dados

O conjunto de dados utilizado (`chess_games.csv`) foi originalmente coletado da plataforma de xadrez [**Lichess.org**](https://lichess.org/) e disponibilizado publicamente no **Kaggle**.

O *dataset* reúne aproximadamente **20.000 partidas** ranqueadas jogadas por usuários reais do Lichess, contendo dados completos de pontuação, status do desfecho, sequência de lances e aberturas empregadas.

## ❓ Perguntas de Negócio e Análise Exploratória

A análise exploratória foi norteada pelas seguintes questões:

1. **Vantagem das Brancas:** Jogar com as peças brancas oferece uma vantagem estatisticamente significativa no Lichess?

2. **Impacto do Rating (Elo):** O quão determinante é a diferença de rating entre os dois jogadores para prever o vencedor?

3. **Influência das Aberturas:** Determinadas aberturas de xadrez aumentam significativamente as chances de vitória das Brancas?

4. **Modos de Encerrar:** Qual a forma mais frequente de término das partidas (Desistência, Xeque-mate ou Tempo)?

## 📖 Dicionário do Banco de Dados

| Coluna | Tipo | Descrição | 
| ----- | ----- | ----- | 
| `id` | String | Identificador único da partida no Lichess. | 
| `rated` | Booleano | Indica se a partida vale pontos de rating (True/False). | 
| `created_at` | Timestamp | Horário de início da partida. | 
| `last_move_at` | Timestamp | Horário de encerramento da partida. | 
| `turns` | Inteiro | Número total de turnos (lances) efetuados na partida. | 
| `victory_status` | Categórico | Tipo de desfecho (`mate`, `resign`, `outoftime`, `draw`). | 
| `winner` | Categórico | Vencedor da partida (`white`, `black`, `draw`). **(Variável Alvo)** | 
| `increment_code` | String | Controle de tempo do jogo (ex: `10+0`, `5+3`). | 
| `white_id` | String | Nome de usuário do jogador de Brancas. | 
| `white_rating` | Inteiro | Pontuação Elo do jogador de Brancas. | 
| `black_id` | String | Nome de usuário do jogador de Pretas. | 
| `black_rating` | Inteiro | Pontuação Elo do jogador de Pretas. | 
| `moves` | Texto | Lista com a sequência de lances em notação algébrica. | 
| `opening_eco` | Categórico | Código ECO de classificação da abertura (ex: `C50`). | 
| `opening_name` | String | Nome completo da abertura utilizada. | 
| `opening_ply` | Inteiro | Número de lances atribuídos à fase de abertura. | 

## 🛠️ Engenharia de Atributos e Pré-processamento

Para otimizar o desempenho e permitir a leitura matemática pelos modelos de Machine Learning, as seguintes transformações foram realizadas:

1. **Criação de Atributo (`rating_diff`):**
   

   $$
   \text{rating\_diff} = \text{white\_rating} - \text{black\_rating}
   $$

   
   Representa numericamente a vantagem (ou desvantagem) teórica do jogador de peças brancas antes do início da partida.

2. **Criação da Variável Resumida de Abertura (`opening_shortname`):**
   Extração da família principal da abertura a partir do campo `opening_name` para reduzir a alta cardinalidade do atributo.

3. **Codificação Dummies (*One-Hot Encoding*):**
   Variáveis categóricas essenciais como `victory_status` e `opening_shortname` foram transformadas em vetores binários ($0$ ou $1$).

4. **Filtragem e Limpeza:**
   Remoção de empates (`draw`) da variável alvo para focar a classificação binária do vencedor (`white` vs. `black`), tratativa de valores nulos e remoção de atributos irrelevantes/identificadores (`id`, `white_id`, `black_id`).

## 🤖 Modelos Treinados e Métricas de Avaliação

Foram treinados três classificadores supervisionados para prever a vitória das peças Brancas (`winner == white`):

1. **Árvore de Decisão (*Decision Tree Classifier*):** Modelo baseado em regras de decisão em formato de árvore hierárquica.

2. **Naive Bayes (*Gaussian Naive Bayes*):** Modelo probabilístico baseado no Teorema de Bayes sob a suposição de independência entre atributos.

3. **Regressão Logística (*Logistic Regression*):** Modelo linear que estima a probabilidade de ocorrência da classe positiva usando a função sigmoide.

### Métricas Utilizadas

* **Acurácia (Accuracy):** Proporção total de acertos do modelo sobre todas as previsões.

* **Precisão (Precision):** Das partidas previstas como vitória das Brancas, quantas realmente foram vitórias das Brancas.

* **Recall (Sensibilidade):** Das partidas que foram realmente vencidas pelas Brancas, quantas o modelo conseguiu identificar.

* **AUC (Área Sob a Curva ROC):** Medida da capacidade do modelo em discriminar entre as duas classes (variação de 0.5 a 1.0).

* **Matriz de Confusão:** Tabela cruzada contendo os Verdadeiros Positivos (VP), Verdadeiros Negativos (VN), Falsos Positivos (FP) e Falsos Negativos (FN).

## 📈 Comparação de Desempenho

Resultados obtidos no conjunto de teste após o treinamento:

| Modelo | Acurácia | Precisão | Recall | AUC-ROC | 
| ----- | ----- | ----- | ----- | ----- | 
| **Árvore de Decisão** | 64.8% | 65.1% | 66.2% | 0.695 | 
| **Naive Bayes** | 62.1% | 62.4% | 63.8% | 0.672 | 
| **Regressão Logística** ⭐ | **66.3%** | **66.5%** | **67.4%** | **0.721** | 

> 🏆 **Conclusão:** A **Regressão Logística** apresentou o melhor desempenho geral, alcançando a maior área sob a curva ROC (**0.721**) e a melhor acurácia (**66.3%**), demonstrando que a relação linear entre a diferença de rating e as variáveis de abertura/status captura bem a dinâmica de vitória do xadrez.

## 📂 Estrutura de Arquivos

```
Metricas---ML---Xadrez/
├── data/
│   └── chess_games.csv                # Dataset original baixado do Kaggle/Lichess
├── apresentação/
│   └── matrizes_confusao_xadrez.xlsx  # Matrizes de confusão exportadas em Excel por modelo
|    └── Classificação e Previsão em Xadrez.pttx
├── main.ipynb      # Notebook completo com EDA e Machine Learning
├── README.md                          # Documentação do projeto
└── requirements.txt                   # Dependências do projeto

```

## ⚙️ Instruções de Instalação e Execução

### Pré-requisitos

* Python 3.8 ou superior

* Gerenciador de pacotes `pip` e ambiente virtual `venv` (recomendado)

### Passo a Passo

1. **Clonar o repositório:**

   ```
   git clone https://github.com/brendolalves/Metricas---ML---Xadrez.git
   cd Metricas---ML---Xadrez
   
   ```

2. **Criar e ativar o ambiente virtual:**

   ```
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   
   ```

3. **Instalar as dependências:**

   ```
   pip install -r requirements.txt
   
   ```

4. **Executar o Jupyter Notebook:**

   ```
   jupyter notebook
   
   ```

   Abra o arquivo `notebooks/analise_e_modelagem.ipynb` no navegador e execute as células.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** [Python 3](https://www.python.org/)

* **Análise e Manipulação de Dados:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)

* **Machine Learning:** [Scikit-learn](https://scikit-learn.org/)

* **Visualização de Dados:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

* **Exportação de Relatórios:** [OpenPyXL](https://openpyxl.readthedocs.io/)

* **Ambiente de Desenvolvimento:** [Jupyter Notebook](https://jupyter.org/) / [VS Code](https://code.visualstudio.com/)

## 👤 Autor

Desenvolvido por **Brendol Alves**.

🔗 GitHub: [@brendolalves](https://github.com/brendolalves)