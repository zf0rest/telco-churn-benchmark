# %% [markdown]
# ### Importações

# %%
# Padrão
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# dataset
import kagglehub
import shutil
import os

# Arsenal de modelos para o benchmark: tentei puxar diferentes naturezas 
from sklearn.linear_model import LogisticRegression # O mais simples e interpretável
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier # A rede neural

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight

# %% [markdown]
# ### Carregamento

# %%
# Importando o dataset e copiando do ambiente global kaggle para o ambiente do projeto (../data)
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
destino = "../data"
shutil.copytree(path, destino, dirs_exist_ok=True)

# %%
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.sample(10)

# %%
display(df.nunique().sort_values(ascending=False).rename('Valores únicos', axis=0)) # para saber a abrangência de dados únicos de cada variável
display(df.nunique().value_counts().sort_index(ascending=False)) # para resumir quantas variáveis do dataset possuem 

# %%
# checando consistência
display(df.info(verbose=True, show_counts=True)) # para olhar missing-values

# %% [markdown]
# **Veredito**: Temos 7 variáveis binárias, 9 variáveis com 3 classes e uma com 4 classes. O dataset **aparentemente** limpo em relação a valores nulos. Sobre a tipagem, o estado atual parece ser coerente com os valores semânticos informados nos metadados, salvo a exceção do vetor `TotalCharges`, que é numérico e consta como `str`. Isso está disponível neste [link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data), dentro do *Data Card* na sessão *detail*

# %%
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], 'coerce') # transformando em numérico e testando se alguém ficou vazio
df.isnull().sum().sort_values(ascending=False)

# %% [markdown]
# Quem são esses casos?

# %%
display(df[df['TotalCharges'].isnull()])
display(df[df['tenure'] == 0]['customerID']) #são exatamente as mesmas pessoas

# %% [markdown]
# Nenhum foi caso de churn e todos tem o "tempo de casa" `tenure` em zero... Podemos assumir que são clientes novos, por isso o TotalCharges não está preenchido. Como conferimos que não há outras pessoas com o `tenure` em zero e são apenas 11 registros de +7K, podemos retirar esses casos pra fora:

# %%
df = df.dropna()
df.drop(['customerID'], axis=1)

# %%
colunas_num = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[colunas_num].describe() # SeniorCitizen já veio como binário, então tirei ela fora do hall das variáveis numéricas

# %%
database = df.copy()
database = (
    database.assign(
        gender = lambda x : x['gender'].map({'Male':1, 'Female':0}),
        Partner = lambda x : x['Partner'].map({'Yes':1, 'No':0}),
        Dependents = lambda x : x['Dependents'].map({'Yes':1, 'No':0}),
        PhoneService = lambda x : x['PhoneService'].map({'Yes':1, 'No':0}),
        PaperlessBilling = lambda x : x['PaperlessBilling'].map({'Yes':1, 'No':0}),
        Churn = lambda x : x['Churn'].map({'Yes':1, 'No':0})
    )
)

# %% [markdown]
# Com os dados limpos, podemos começar a vislumbrar as relações entre as variáveis a partir de um breve (pois não é o intuito aqui) trecho de exploração visual. Não vamos fazer um desfile carnavalesco de gráficos aqui, mas procurar a objetividade na tentativa de responder a seguinte questão: **quais características separam os clientes que cancelam dos que ficam?**. O ponta-pé inicial é olhar para o próprio alvo e entender seu balanço (que vai nutrir decisões futuras de modelagem).

# %%
# %%
fig, ax = plt.subplots(figsize=(5, 4))
proporcoes = database['Churn'].value_counts(normalize=True) * 100

sns.barplot(x=proporcoes.index, y=proporcoes.values, palette = 'mako', ax=ax, legend=False, hue=proporcoes.index)
ax.set_title('Proporção Churn na base', pad=15)
ax.set_ylabel('Porcentagem (%)')
ax.set_xlabel('Churn')
ax.set_ylim(0, 100)

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=11)

sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# O desbalanceamento das classes torna o threshold padrão de 0.5 ineficiente. Como as probabilidades tendem a ser baixas, manter esse limiar gera uma falsa acurácia sustentada por uma sensibilidade pífia: o modelo classifica quase toda a base como 'não-churn', deixando a imensa maioria dos verdadeiros cancelamentos passar despercebida. Heuristicamente, a pergunta central da modelagem é: o que custa mais? Em telecomunicações, assim como na saúde, o Erro Tipo II (Falso-Negativo) é o mais grave. Perder a receita vitalícia (do LTV) de um cliente que cancela em silêncio custa infinitamente mais para o negócio do que o custo operacional de oferecer uma promoção de retenção a um Falso-Positivo. E na saúde, nem preciso dizer o quão perigiso é afirmar que um paciente doente está saudável...

# %% [markdown]
# Olhemos agora as categóricas não-binárias do dataset (de forma univariada). Vamos ver quais são mais relevantes, observando se as classes delas têm proporções semelhantes entre si. 
# 
# - A média aritmética da variável resposta para uma classe específica $c_i$ é matematicamente equivalente à probabilidade condicional de Churn dado aquele grupo:$$p_i = \mathbb{E}[Y \mid X = c_i] = P(Y = 1 \mid X = c_i)$$
# * A **diferença** entre as proporções maxima e mínima de cada $c_i$ do conjunto de classes da variável preditora provê uma intuição univariada de relevância (em termos de variância) dessa variável (sozinha) na classificação do Churn.
# 
# $$\Delta_X = \max_{c_i \in C} (p_i) - \min_{c_i \in C} (p_i)$$
# 

# %%
relevancia = {}

cat_vars = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaymentMethod']

for col in cat_vars:
    taxas = database.groupby(col)['Churn'].mean()
    amplitude_risco = taxas.max() - taxas.min() 
    relevancia[col] = amplitude_risco

# Transforma num DataFrame, ordena e encadeia (method chaining)
df_ranking = (
    pd.DataFrame(list(relevancia.items()), columns=['Variável', 'Delta_Risco_Churn'])
    .sort_values(by='Delta_Risco_Churn', ascending=False)
    .reset_index(drop=True)
)

display(df_ranking)

# %% [markdown]
# A variável Contract mostra-se a mais importante, apresentando uma amplitude de quase 40 pontos percentuais na taxa de risco entre suas classes. Ela é seguida em importância por InternetService, OnlineSecurity e TechSupport, todas apresentando um delta de risco em torno de 34%.

# %%
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
axes = axes.flatten() # Transforma a matriz 2x5 num array 1D para iterar facilmente

for i, col in enumerate(cat_vars):
    sns.barplot(data=database, x=col, y='Churn', ax=axes[i], palette='mako', errorbar=None, hue=col)
    axes[i].set_title(col, fontsize=12)
    axes[i].set_ylabel('Taxa de Churn' if i % 5 == 0 else '') # Y label só na 1ª coluna
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=45) # Inclina os rótulos das classes pra não ficar zoado

sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# A análise individual das variáveis categóricas do dataset não é suficiente em termos de *feature-selection*, mas nos mostra que existem fortes indícios de que clientes com o seguinte perfil são riscos de Churn: clientes marcados com 'No' nas variáveis ``OnlineSecurity``, ``OnlineBackup``, ``DeviceProtection`` e ``TechSupport`` possuem taxas de churn drasticamente superiores àqueles que contratam esses serviços. Em suma, os serviços agregados de internet são impulsionadores do LTV aqui nesse contexto.
# 
# A variável ``MultipleLines`` demonstrou ter baixo poder discriminatório com um delta nas proporções da variável predita de só 3,6%. Mas esse cenário que temos não leva em consideração o poder da interação entre as variáveis. Essa última por exemplo, não tem valor sozinha, mas pode ser que tenha algum poder discriminatório interessante em algum grau de profundidade de um RandomForest, por exemplo. Há uma miríade de possibilidades que exploraremos mais à frente.
# 

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4)) # 3 gráficos num quadro só
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

for ax, col in zip(axes, num_cols):
    sns.kdeplot(data=df, x=col, hue='Churn', fill=True, common_norm=False, ax=ax, palette='mako')
    ax.set_title(f'Distribuição de {col} por Churn')

sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# O churn ocorre sobretudo no início do relacionamento com o cliente, varificável no baixos pico valores de `tenure`, e afeta de forma desproporcional os clientes com mensalidades mais caras no ``MonthlyCharges``. Olhando a distribuição do `TotalChargres`, vemos que ambas as classes possuem um pico agudo muito próximo ao zero e uma cauda longa. O que isso nos diz é que ``TotalCharges`` é uma variável colinear, funcionando basicamente como um subproduto determinístico da multiplicação entre tenure e ``MonthlyCharges`` (e a definição nos metadados indica a mesma coisa). 

# %% [markdown]
# 
# Tendo em mente que temos variáveis de diferentes naturezas que podem se relacionar a diferentes magnitudes com a nossa variável alvo, podemos dar o ponto de partida na análise da matriz **$X$** a partir de um heatmap que dá uma intuição da relação linear entre os preditores numéricos e o nosso $Y$

# %%
database[num_cols].corr()

# %% [markdown]
# A variável ``TotalCharges`` realmente deve sair do dataset: é uma combinação linear das outras duas e só vai inflar os modelos com colinearidade, pois não é informação independente.

# %%
database = database.drop(columns=['TotalCharges', 'customerID'])
 # retirando as variáveis inúteis

num_cols = ['tenure', 'MonthlyCharges']

# %% [markdown]
# Agora vamos olhar uma métrica de inspeção de modelo nativa do Scikit-learn, baseada em àrvores aleatórias (um ensemble de árvores de decisão), com a finalidade de buscar os preditores mais relevantes do dataset, ou seja, que melhor reduzem a *impureza de Gini*. Como não é o intuito explciar aqui a álgebra por debaixo do capô, vou assumir que o leitor tenha conhecimento prévio na matemática das Decision Trees (por enquanto). Uma amostra grátis sobre o funcionamento do `feature_importances` seria:
# 
# 1. Soma-se a redução de impureza de todos os nós onde a variável $X$ foi utilizada para fazer a divisão;
# 2. Calcula-se a média desse valor ao longo de todas as árvores do ensemble;
# 
# A fraqueza desse método (que é amplamente utilizado às cegas) é que a presença das variáveis contínuas faz com que o espaço de cortes seja muito grande, pois a árvore tem muitas opções de escolha num vetor de valores contínuos, e isso infla artificialmente e especialmente a métrica de que estamos falando ao longo do treinamento. Entaõ vamos dar uma olhada no ``feature_importance`` já contando com o fato de que as numéricas vão sair na frente de todo mundo.
# 
# >Então, Lucas, responda: o que podemos fazer para avaliar de forma multivariada (e mais justa) a importância das nossas variáveis? 
# 
# Chegamos no ponto pra implementar aqui na análise o ``permutation_importance``: um método que, com o modelo baseline já treinado, isola-se uma coluna por vez nos dados de validação, embaralha todos os seus valores de forma completamente aleatória e testa o modelo de novo. Ele mede o quanto a métrica de erro (ou acurácia/Recall) piorou com aquela coluna toda mexida. Com isso, respondemos, à grosso modo, o quanto o modelo é dependente da variável $X_i$ em questão. Eu acho essa solução interessante porque se embaralhamos os valores de um vetor de valores, cai por terra a sua correlação com a variável resposta e com todas as outras colunas da matris $X$, sem que alteremos a sua distribuição estatística.

# %%
df_exploratorio = database.copy() #criando uma cópia só pra fazermos o feature selection

# Identificando as colunas de texto e as transformando em inteiro (Label Encoding)
cols_texto = df_exploratorio.select_dtypes(include=['object']).columns
df_exploratorio[cols_texto] = df_exploratorio[cols_texto].apply(lambda x: pd.factorize(x)[0])

X_exp = df_exploratorio.drop('Churn', axis=1)
y_exp = df_exploratorio['Churn']

# Treinando a Random Forest de rascunho pra realizarmos feature selection
rf_exp = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42) #modelo baseline para feature selection
rf_exp.fit(X_exp, y_exp)

df_importancia = (
    pd.DataFrame({
        'Feature': X_exp.columns,
        'Gini': rf_exp.feature_importances_
    }).sort_values(by='Gini', ascending=False).reset_index(drop=True))

df_importancia

# %%
# n_repeats=20 garante que o embaralhamento seja testado 20 vezes para ter significância estatística

resultado_permutacao = permutation_importance(rf_exp, X_exp, y_exp, n_repeats=20, random_state=147, n_jobs=-1)
df_importancia_justa = (
    pd.DataFrame({'Feature': X_exp.columns,
        'Importancia_Media': resultado_permutacao.importances_mean,
        'Desvio_Padrao': resultado_permutacao.importances_std})
        .sort_values(by='Importancia_Media', ascending=False)
        .reset_index(drop=True))

df_importancia_justa

# %% [markdown]
# O resultado sugere que ``tenure`` 0.096, ``Contract`` 0.053 e ``MonthlyCharges`` 0.051 são os grandes motores do churn. Serviços agregados, conforme falamos antes, como: ``OnlineSecurity`` 0.030, ``PaymentMethod`` 0.029, ``TechSupport`` 0.029 até ``PaperlessBilling`` provaram ter sinal preditivo genuíno e devem ser mantidos. Vou traçar nessa última o ponto de corte e descartar as demais variáveis (de ``MultipleLines`` até ``PhoneService``)

# %%
cols_to_drop = ['MultipleLines', 'DeviceProtection', 'OnlineBackup', 
                'gender', 'Partner', 'StreamingMovies', 'StreamingTV', 
                'SeniorCitizen', 'Dependents', 'PhoneService']

cat_vars_sobreviventes = [
    'Contract', 'OnlineSecurity', 'PaymentMethod', 
    'TechSupport', 'InternetService', 'PaperlessBilling'
]

database = database.drop(columns=cols_to_drop)

# %% [markdown]
# Apliquei One-Hot encoding nas categóricas (criando variáveis dummy) e normalizei as numéricas dentro de um pipeline para garantir o premissas exigidas por modelos como regressão logística, random forest e Redes Neurais (MLP), sem causar vazamento de dados (data leakage). Para os modelos avançados de gradient boosting (CatBoost e LightGBM), o pipeline é dispensado: basta converter as colunas para o tipo *category* nativo do pandas. Esses algoritmos possuem mecanismos internos que processam as classes diretamente, otimizando as divisões nas árvores sem a esparsidade gerada pelo One-Hot.

# %%
X = database.drop('Churn', axis=1)
y = database['Churn']

for col in cat_vars_sobreviventes: # transformando as categóricas para os modelos conseguirem ler normalmente
    X[col] = X[col].astype('category')

# split de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=147, stratify=y
)

# %% [markdown]
# Em vez de você transformar a base inteira de uma vez ou criar bases separadas manualmente, o transformers permite que você dite regras específicas para grupos de colunas diferentes rodarem em paralelo. Ele rasga o dataset passado na hora de executar em dois blocos virtuais: o primeiro bloco contém apenas as colunas da lista num_cols e o segundo apenas as colunas da lista cat_vars_sobreviventes, e aplica no que encontrar desses conjuntos as premissas que você passo, ou seja, o ``StandardScaler()`` exclusivamente no bloco numérico e simultaneamente aplica o ``OneHotEncoder()`` no bloco categórico.

# %%
# Pré processamento só pros modelos clássicos: MLP, Random Forest e Elastic net
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_vars_sobreviventes)])

# %% [markdown]
# Vamos montar um mapa de execução do benchmark automatizado. Se recebermos dados novos esse pedaço modularizado facilitará as coisas no futuro. A ideia é que nesse trecho o algoritmo empacota, em um único objeto estruturado, as arquiteturas, as regras de pré-processamento e o espaço de hiperparâmetros (a grade de testes) que o GridSearchCV vai explorar para cada algoritmo.
# 
# Estamos colocando o "balanced" na opção dos pesos. Quando você ativa o class_weight='balanced', você altera as regras do jogo dentro da loss function, de modo que a classe mais rara recebe um multiplicador muito maior. Arredondando os números da Telco (73% vs 27%), a proporção é de quase 3 para 1. Se o modelo erra a previsão de um cliente que FICOU (Classe 0), a Função de Perda soma, digamos, 1 ponto de erro. Se o modelo erra a previsão de um cliente que deu CHURN (Classe 1), a Função de Perda soma 3 pontos de erro. Isso foi só um exemplo, mas eu quis demonstrar que cada vez que o algoritmo tenta ignorar um cliente de churn, o erro dá um alerta. Isso força as árvores de decisão a criarem ramificações (splits) específicas para capturar os clientes da Classe 1, mesmo que isso signifique errar mais na Classe 0, uma vez que também estamos ancorando a métrica de sucesso na Recall.

# %%
# O Grid de Modelos para o gridsearch CV. O formato é um dicionário
config_modelos = {
    'Elastic Net': {
        'modelo': Pipeline(steps=[('preprocessor', preprocessor),
                                  ('clf', LogisticRegression(solver='saga', random_state=147))]),
        'params': {
                    'clf__C': [0.01, 0.1, 1, 10], 
                    'clf__l1_ratio': [0.1, 0.5, 0.9], # elastic net
                    'clf__class_weight': [None, 'balanced'] 
                }
    },
    'Random Forest': {
        'modelo': Pipeline(steps=[('preprocessor', preprocessor),
                                  ('clf', RandomForestClassifier(random_state=147))]),
        'params': {
            'clf__n_estimators': [100, 300],
            'clf__max_depth': [5, 10, 15], # esses clf no prefixo dos hiperparametros é para o pipeline entrar aqui e encontrá-los
            'clf__min_samples_leaf': [1, 4], 
            'clf__class_weight': ['balanced', 'balanced_subsample'] 
        }
    },
    'MLP (Rede Neural)': {
        'modelo': Pipeline(steps=[('preprocessor', preprocessor),
                                  ('clf', MLPClassifier(max_iter=1500, random_state=147))]),
        'params': {
                   
                    'clf__hidden_layer_sizes': [(64,), (128,), (64, 32), (64, 32, 16)], 
                    'clf__alpha': [0.0001, 0.001, 0.01],
                    'clf__learning_rate_init': [0.001, 0.01]
                }
    },
    'LightGBM': { #não usa preprocessor, vai a seco mesmo as variáveis categóricas, como havíamos falado antes
        'modelo': LGBMClassifier(random_state=147, verbose=-1), 
        'params': {
            'learning_rate': [0.01, 0.05, 0.1], # 0.01 geralmente dá resultados muito superiores em boosting
            'num_leaves': [31, 50, 70], 
            'class_weight': [None, 'balanced']
        }
    },
    'CatBoost': {
        'modelo': CatBoostClassifier(random_state=147, verbose=0),
        'params': {
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8], # 8 costuma ser o limite antes de overfitar muito em bases pequenas segundo 
            'auto_class_weights': ['None', 'Balanced']
        }
    }
}

# %% [markdown]
# Através do GridSearch, que nada mais é do que uma metodologia de Cross-validation, o código não treina apenas um modelo, mas dezenas de combinações de hiperparâmetros para cada algoritmo em uma validação cruzada de 5 folds (cv=5). O diferencial metodológico está na âncora ``scoring='recall'``. Isso força a matemática do gradiente descendente a penalizar severamente os falsos negativos, elegendo apenas as versões de modelos que são agressivas na captura de clientes propensos ao churn (minimizando o Erro Tipo II, que é o que é realmente custodo num contexto de churn).
# 
# É interessante destacar que a IA generativa nos ajudou sumariamente neste ponto, onde existe uma incompatibilidade entre funções do CatBoost e do Sklearn, que fazem com que tenhamos que abordar esse modelo de forma específica dentro do pipeline de treino. O dicionário ``melhores_modelos`` salva as arquiteturas vencedoras já treinadas na memória, enquanto a lista ``placar_grid`` compila o dataframe final ordenado, entregando a tabela de resultados limpa

# %%
placar_grid = []
melhores_modelos = {}

for nome, config in config_modelos.items():
    print(f"Treinando {nome}...")
    
    grid = GridSearchCV(
        estimator=config['modelo'],
        param_grid=config['params'],
        cv=5, 
        scoring='recall',
        n_jobs=-1
    )
    
    if nome == 'CatBoost':
        grid.fit(X_train, y_train, cat_features=list(cat_vars_sobreviventes))
    else:
        grid.fit(X_train, y_train)
        
    melhores_modelos[nome] = grid.best_estimator_
    
    # Avaliação no conjunto de Teste isolado
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    
    placar_grid.append({
        'Arquitetura': nome,
        'Recall (Teste)': recall_score(y_test, y_pred),
        'ROC-AUC (Teste)': roc_auc_score(y_test, y_proba),
        'Melhores Params': str(grid.best_params_)
    })

df_benchmark_final = pd.DataFrame(placar_grid).sort_values(by='Recall (Teste)', ascending=False).reset_index(drop=True)
display(df_benchmark_final)

# %% [markdown]
# O CatBoost assumiu o topo do pódio, alcançando o maior Recall (0.8288) e o maior ROC-AUC (0.8483) do benchmark. Isso comprova matematicamente que o motor nativo dele para variáveis categóricas sobrepôs a abordagem tradicional de binarização do One-Hot Encoding. Todos os modelos baseados em árvore + a Logística (elastic net) escolheram a configuração de pesos balanceados ('balanced') como a arquitetura ideal. Forçar o algoritmo a penalizar o Falso-Negativo foi a única forma de fazê-lo caçar o Churn até a morte.

# %%
campeao = melhores_modelos['CatBoost']

# Gera as predições finais na base de teste (a base isolada que ele não viu no treino)
y_pred_campeao = campeao.predict(X_test) # Lembrando que o CatBoost usa a matriz B

# Calcula e plota a Matriz de Confusão
cm = confusion_matrix(y_test, y_pred_campeao)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ficou (0)', 'Churn (1)'])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', values_format='d', ax=ax)
ax.set_title('Matriz de Confusão - Campeão (CatBoost)', pad=15)
sns.despine()
plt.show()

# %% [markdown]
# Os números da matriz de confusão refletem o comportamento de um modelo otimizado para dizimar o erro tipo II. Dos 374 clientes que realmente cancelariam o serviço na base de teste, o CatBoost conseguiu identificar e 310 verdadeiros positivos, deixando escapar apenas 64 cancelamentos silenciosos Falsos Negativos. Em termos financeiros, o modelo garantiu a retenção de 310 lifetime values (LTV) que seriam perdidos. Para atingir essa agressividade na captura, o algoritmo gerou 274 falsos positivos, ou sjea, clientes que foram classificados como risco de evasão, mas que acabaram ficando. O modelo provou matematicamente que a introdução do peso balanceado ``class_weight='balanced'`` funcionou. O algoritmo aceitou errar a favor do conservadorismo pra segurar o erro tipo II.
# 
# No mundo real das telecomunicações, o custo operacional de enviar uma campanha de retenção, um e-mail ou oferecer um cupom de desconto preventivo para esses 274 Falsos Positivos é comumente menor do que o rombo financeiro gerado caso os 310 clientes reais passassem despercebidos, então o trade-off vale a pena. 
# 
# Como o sklearn não tem suporte para colocar balanceamento de classes, para extrair o verdadeiro poder das Redes Neurais em cenários desbalanceados, a abordagem exigiria abandonar o Scikit-Learn e implementar a MLP em frameworks dedicados a tensores. Então trouxemos MLP's até o limite em casos dessa natureza para demonstrar a sua fronteira que deve desaguar em modelos de maiores complexidades. Se esse mesmo dataset fosse inserido em uma topologia construída no Keras, por exemplo, o problema seria resolvido instantaneamente passando o argumento class_weight no método .fit() da rede.

# %% [markdown]
# ## Bônus

# %%
# Como eu falei acima, vou treinar um modelo simples aqui no Keras:

pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
pesos_dict = {0: pesos[0], 1: pesos[1]} # Transforma no dicionário que o Keras exige

#A mesma topologia profunda testada no sklearn: 64 -> 32
modelo_keras = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Função sigmóide para a probabilidade binária do Churn
])

# Compilando já de olho na métrica de negócio
modelo_keras.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Recall'])

# 3. O Treinamento Penalizado (Onde a mágica acontece)
print("Treinando Rede Neural Keras com pesos balanceados...")
historico = modelo_keras.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    class_weight=pesos_dict, # O parâmetro salvador da pátria
    verbose=0)

loss, recall_keras = modelo_keras.evaluate(X_test, y_test, verbose=0)
print(f"Recall da Rede Neural (Keras): {recall_keras:.4f}")


# %% [markdown]
# ### Importações

# %%
# Padrão
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# dataset
import kagglehub
import shutil
import os

# Arsenal de modelos para o benchmark: tentei puxar diferentes naturezas 
from sklearn.linear_model import LogisticRegression # O mais simples e interpretável
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier # A rede neural

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight

# %% [markdown]
# ### Carregamento

# %%
# Importando o dataset e copiando do ambiente global kaggle para o ambiente do projeto (../data)
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
destino = "../data"
shutil.copytree(path, destino, dirs_exist_ok=True)

# %%
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.sample(10)

# %%
display(df.nunique().sort_values(ascending=False).rename('Valores únicos', axis=0)) # para saber a abrangência de dados únicos de cada variável
display(df.nunique().value_counts().sort_index(ascending=False)) # para resumir quantas variáveis do dataset possuem 

# %%
# checando consistência
display(df.info(verbose=True, show_counts=True)) # para olhar missing-values

# %% [markdown]
# **Veredito**: Temos 7 variáveis binárias, 9 variáveis com 3 classes e uma com 4 classes. O dataset **aparentemente** limpo em relação a valores nulos. Sobre a tipagem, o estado atual parece ser coerente com os valores semânticos informados nos metadados, salvo a exceção do vetor `TotalCharges`, que é numérico e consta como `str`. Isso está disponível neste [link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data), dentro do *Data Card* na sessão *detail*

# %%
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], 'coerce') # transformando em numérico e testando se alguém ficou vazio
df.isnull().sum().sort_values(ascending=False)

# %% [markdown]
# Quem são esses casos?

# %%
display(df[df['TotalCharges'].isnull()])
display(df[df['tenure'] == 0]['customerID']) #são exatamente as mesmas pessoas

# %% [markdown]
# Nenhum foi caso de churn e todos tem o "tempo de casa" `tenure` em zero... Podemos assumir que são clientes novos, por isso o TotalCharges não está preenchido. Como conferimos que não há outras pessoas com o `tenure` em zero e são apenas 11 registros de +7K, podemos retirar esses casos pra fora:

# %%
df = df.dropna()
df.drop(['customerID'], axis=1)

# %%
colunas_num = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[colunas_num].describe() # SeniorCitizen já veio como binário, então tirei ela fora do hall das variáveis numéricas

# %%
database = df.copy()
database = (
    database.assign(
        gender = lambda x : x['gender'].map({'Male':1, 'Female':0}),
        Partner = lambda x : x['Partner'].map({'Yes':1, 'No':0}),
        Dependents = lambda x : x['Dependents'].map({'Yes':1, 'No':0}),
        PhoneService = lambda x : x['PhoneService'].map({'Yes':1, 'No':0}),
        PaperlessBilling = lambda x : x['PaperlessBilling'].map({'Yes':1, 'No':0}),
        Churn = lambda x : x['Churn'].map({'Yes':1, 'No':0})
    )
)

# %% [markdown]
# Com os dados limpos, podemos começar a vislumbrar as relações entre as variáveis a partir de um breve (pois não é o intuito aqui) trecho de exploração visual. Não vamos fazer um desfile carnavalesco de gráficos aqui, mas procurar a objetividade na tentativa de responder a seguinte questão: **quais características separam os clientes que cancelam dos que ficam?**. O ponta-pé inicial é olhar para o próprio alvo e entender seu balanço (que vai nutrir decisões futuras de modelagem).

# %%
# %%
fig, ax = plt.subplots(figsize=(5, 4))
proporcoes = database['Churn'].value_counts(normalize=True) * 100

sns.barplot(x=proporcoes.index, y=proporcoes.values, palette = 'mako', ax=ax, legend=False, hue=proporcoes.index)
ax.set_title('Proporção Churn na base', pad=15)
ax.set_ylabel('Porcentagem (%)')
ax.set_xlabel('Churn')
ax.set_ylim(0, 100)

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=11)

sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# O desbalanceamento das classes torna o threshold padrão de 0.5 ineficiente. Como as probabilidades tendem a ser baixas, manter esse limiar gera uma falsa acurácia sustentada por uma sensibilidade pífia: o modelo classifica quase toda a base como 'não-churn', deixando a imensa maioria dos verdadeiros cancelamentos passar despercebida. Heuristicamente, a pergunta central da modelagem é: o que custa mais? Em telecomunicações, assim como na saúde, o Erro Tipo II (Falso-Negativo) é o mais grave. Perder a receita vitalícia (do LTV) de um cliente que cancela em silêncio custa infinitamente mais para o negócio do que o custo operacional de oferecer uma promoção de retenção a um Falso-Positivo. E na saúde, nem preciso dizer o quão perigiso é afirmar que um paciente doente está saudável...

# %% [markdown]
# Olhemos agora as categóricas não-binárias do dataset (de forma univariada). Vamos ver quais são mais relevantes, observando se as classes delas têm proporções semelhantes entre si. 
# 
# - A média aritmética da variável resposta para uma classe específica $c_i$ é matematicamente equivalente à probabilidade condicional de Churn dado aquele grupo:$$p_i = \mathbb{E}[Y \mid X = c_i] = P(Y = 1 \mid X = c_i)$$
# * A **diferença** entre as proporções maxima e mínima de cada $c_i$ do conjunto de classes da variável preditora provê uma intuição univariada de relevância (em termos de variância) dessa variável (sozinha) na classificação do Churn.
# 
# $$\Delta_X = \max_{c_i \in C} (p_i) - \min_{c_i \in C} (p_i)$$
# 

# %%
relevancia = {}

cat_vars = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaymentMethod']

for col in cat_vars:
    taxas = database.groupby(col)['Churn'].mean()
    amplitude_risco = taxas.max() - taxas.min() 
    relevancia[col] = amplitude_risco

# Transforma num DataFrame, ordena e encadeia (method chaining)
df_ranking = (
    pd.DataFrame(list(relevancia.items()), columns=['Variável', 'Delta_Risco_Churn'])
    .sort_values(by='Delta_Risco_Churn', ascending=False)
    .reset_index(drop=True)
)

display(df_ranking)

# %% [markdown]
# A variável Contract mostra-se a mais importante, apresentando uma amplitude de quase 40 pontos percentuais na taxa de risco entre suas classes. Ela é seguida em importância por InternetService, OnlineSecurity e TechSupport, todas apresentando um delta de risco em torno de 34%.

# %%
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
axes = axes.flatten() # Transforma a matriz 2x5 num array 1D para iterar facilmente

for i, col in enumerate(cat_vars):
    sns.barplot(data=database, x=col, y='Churn', ax=axes[i], palette='mako', errorbar=None, hue=col)
    axes[i].set_title(col, fontsize=12)
    axes[i].set_ylabel('Taxa de Churn' if i % 5 == 0 else '') # Y label só na 1ª coluna
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=45) # Inclina os rótulos das classes pra não ficar zoado

sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# A análise individual das variáveis categóricas do dataset não é suficiente em termos de *feature-selection*, mas nos mostra que existem fortes indícios de que clientes com o seguinte perfil são riscos de Churn: clientes marcados com 'No' nas variáveis ``OnlineSecurity``, ``OnlineBackup``, ``DeviceProtection`` e ``TechSupport`` possuem taxas de churn drasticamente superiores àqueles que contratam esses serviços. Em suma, os serviços agregados de internet são impulsionadores do LTV aqui nesse contexto.
# 
# A variável ``MultipleLines`` demonstrou ter baixo poder discriminatório com um delta nas proporções da variável predita de só 3,6%. Mas esse cenário que temos não leva em consideração o poder da interação entre as variáveis. Essa última por exemplo, não tem valor sozinha, mas pode ser que tenha algum poder discriminatório interessante em algum grau de profundidade de um RandomForest, por exemplo. Há uma miríade de possibilidades que exploraremos mais à frente.
# 

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4)) # 3 gráficos num quadro só
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

for ax, col in zip(axes, num_cols):
    sns.kdeplot(data=df, x=col, hue='Churn', fill=True, common_norm=False, ax=ax, palette='mako')
    ax.set_title(f'Distribuição de {col} por Churn')

sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# O churn ocorre sobretudo no início do relacionamento com o cliente, varificável no baixos pico valores de `tenure`, e afeta de forma desproporcional os clientes com mensalidades mais caras no ``MonthlyCharges``. Olhando a distribuição do `TotalChargres`, vemos que ambas as classes possuem um pico agudo muito próximo ao zero e uma cauda longa. O que isso nos diz é que ``TotalCharges`` é uma variável colinear, funcionando basicamente como um subproduto determinístico da multiplicação entre tenure e ``MonthlyCharges`` (e a definição nos metadados indica a mesma coisa). 

# %% [markdown]
# 
# Tendo em mente que temos variáveis de diferentes naturezas que podem se relacionar a diferentes magnitudes com a nossa variável alvo, podemos dar o ponto de partida na análise da matriz **$X$** a partir de um heatmap que dá uma intuição da relação linear entre os preditores numéricos e o nosso $Y$

# %%
database[num_cols].corr()

# %% [markdown]
# A variável ``TotalCharges`` realmente deve sair do dataset: é uma combinação linear das outras duas e só vai inflar os modelos com colinearidade, pois não é informação independente.

# %%
database = database.drop(columns=['TotalCharges', 'customerID'])
 # retirando as variáveis inúteis

num_cols = ['tenure', 'MonthlyCharges']

# %% [markdown]
# Agora vamos olhar uma métrica de inspeção de modelo nativa do Scikit-learn, baseada em àrvores aleatórias (um ensemble de árvores de decisão), com a finalidade de buscar os preditores mais relevantes do dataset, ou seja, que melhor reduzem a *impureza de Gini*. Como não é o intuito explciar aqui a álgebra por debaixo do capô, vou assumir que o leitor tenha conhecimento prévio na matemática das Decision Trees (por enquanto). Uma amostra grátis sobre o funcionamento do `feature_importances` seria:
# 
# 1. Soma-se a redução de impureza de todos os nós onde a variável $X$ foi utilizada para fazer a divisão;
# 2. Calcula-se a média desse valor ao longo de todas as árvores do ensemble;
# 
# A fraqueza desse método (que é amplamente utilizado às cegas) é que a presença das variáveis contínuas faz com que o espaço de cortes seja muito grande, pois a árvore tem muitas opções de escolha num vetor de valores contínuos, e isso infla artificialmente e especialmente a métrica de que estamos falando ao longo do treinamento. Entaõ vamos dar uma olhada no ``feature_importance`` já contando com o fato de que as numéricas vão sair na frente de todo mundo.
# 
# >Então, Lucas, responda: o que podemos fazer para avaliar de forma multivariada (e mais justa) a importância das nossas variáveis? 
# 
# Chegamos no ponto pra implementar aqui na análise o ``permutation_importance``: um método que, com o modelo baseline já treinado, isola-se uma coluna por vez nos dados de validação, embaralha todos os seus valores de forma completamente aleatória e testa o modelo de novo. Ele mede o quanto a métrica de erro (ou acurácia/Recall) piorou com aquela coluna toda mexida. Com isso, respondemos, à grosso modo, o quanto o modelo é dependente da variável $X_i$ em questão. Eu acho essa solução interessante porque se embaralhamos os valores de um vetor de valores, cai por terra a sua correlação com a variável resposta e com todas as outras colunas da matris $X$, sem que alteremos a sua distribuição estatística.

# %%
df_exploratorio = database.copy() #criando uma cópia só pra fazermos o feature selection

# Identificando as colunas de texto e as transformando em inteiro (Label Encoding)
cols_texto = df_exploratorio.select_dtypes(include=['object']).columns
df_exploratorio[cols_texto] = df_exploratorio[cols_texto].apply(lambda x: pd.factorize(x)[0])

X_exp = df_exploratorio.drop('Churn', axis=1)
y_exp = df_exploratorio['Churn']

# Treinando a Random Forest de rascunho pra realizarmos feature selection
rf_exp = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42) #modelo baseline para feature selection
rf_exp.fit(X_exp, y_exp)

df_importancia = (
    pd.DataFrame({
        'Feature': X_exp.columns,
        'Gini': rf_exp.feature_importances_
    }).sort_values(by='Gini', ascending=False).reset_index(drop=True))

df_importancia

# %%
# n_repeats=20 garante que o embaralhamento seja testado 20 vezes para ter significância estatística

resultado_permutacao = permutation_importance(rf_exp, X_exp, y_exp, n_repeats=20, random_state=147, n_jobs=-1)
df_importancia_justa = (
    pd.DataFrame({'Feature': X_exp.columns,
        'Importancia_Media': resultado_permutacao.importances_mean,
        'Desvio_Padrao': resultado_permutacao.importances_std})
        .sort_values(by='Importancia_Media', ascending=False)
        .reset_index(drop=True))

df_importancia_justa

# %% [markdown]
# O resultado sugere que ``tenure`` 0.096, ``Contract`` 0.053 e ``MonthlyCharges`` 0.051 são os grandes motores do churn. Serviços agregados, conforme falamos antes, como: ``OnlineSecurity`` 0.030, ``PaymentMethod`` 0.029, ``TechSupport`` 0.029 até ``PaperlessBilling`` provaram ter sinal preditivo genuíno e devem ser mantidos. Vou traçar nessa última o ponto de corte e descartar as demais variáveis (de ``MultipleLines`` até ``PhoneService``)

# %%
cols_to_drop = ['MultipleLines', 'DeviceProtection', 'OnlineBackup', 
                'gender', 'Partner', 'StreamingMovies', 'StreamingTV', 
                'SeniorCitizen', 'Dependents', 'PhoneService']

cat_vars_sobreviventes = [
    'Contract', 'OnlineSecurity', 'PaymentMethod', 
    'TechSupport', 'InternetService', 'PaperlessBilling'
]

database = database.drop(columns=cols_to_drop)

# %% [markdown]
# Apliquei One-Hot encoding nas categóricas (criando variáveis dummy) e normalizei as numéricas dentro de um pipeline para garantir o premissas exigidas por modelos como regressão logística, random forest e Redes Neurais (MLP), sem causar vazamento de dados (data leakage). Para os modelos avançados de gradient boosting (CatBoost e LightGBM), o pipeline é dispensado: basta converter as colunas para o tipo *category* nativo do pandas. Esses algoritmos possuem mecanismos internos que processam as classes diretamente, otimizando as divisões nas árvores sem a esparsidade gerada pelo One-Hot.

# %%
X = database.drop('Churn', axis=1)
y = database['Churn']

for col in cat_vars_sobreviventes: # transformando as categóricas para os modelos conseguirem ler normalmente
    X[col] = X[col].astype('category')

# split de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=147, stratify=y
)

# %% [markdown]
# Em vez de você transformar a base inteira de uma vez ou criar bases separadas manualmente, o transformers permite que você dite regras específicas para grupos de colunas diferentes rodarem em paralelo. Ele rasga o dataset passado na hora de executar em dois blocos virtuais: o primeiro bloco contém apenas as colunas da lista num_cols e o segundo apenas as colunas da lista cat_vars_sobreviventes, e aplica no que encontrar desses conjuntos as premissas que você passo, ou seja, o ``StandardScaler()`` exclusivamente no bloco numérico e simultaneamente aplica o ``OneHotEncoder()`` no bloco categórico.

# %%
# Pré processamento só pros modelos clássicos: MLP, Random Forest e Elastic net
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_vars_sobreviventes)])

# %% [markdown]
# Vamos montar um mapa de execução do benchmark automatizado. Se recebermos dados novos esse pedaço modularizado facilitará as coisas no futuro. A ideia é que nesse trecho o algoritmo empacota, em um único objeto estruturado, as arquiteturas, as regras de pré-processamento e o espaço de hiperparâmetros (a grade de testes) que o GridSearchCV vai explorar para cada algoritmo.
# 
# Estamos colocando o "balanced" na opção dos pesos. Quando você ativa o class_weight='balanced', você altera as regras do jogo dentro da loss function, de modo que a classe mais rara recebe um multiplicador muito maior. Arredondando os números da Telco (73% vs 27%), a proporção é de quase 3 para 1. Se o modelo erra a previsão de um cliente que FICOU (Classe 0), a Função de Perda soma, digamos, 1 ponto de erro. Se o modelo erra a previsão de um cliente que deu CHURN (Classe 1), a Função de Perda soma 3 pontos de erro. Isso foi só um exemplo, mas eu quis demonstrar que cada vez que o algoritmo tenta ignorar um cliente de churn, o erro dá um alerta. Isso força as árvores de decisão a criarem ramificações (splits) específicas para capturar os clientes da Classe 1, mesmo que isso signifique errar mais na Classe 0, uma vez que também estamos ancorando a métrica de sucesso na Recall.

# %%
# O Grid de Modelos para o gridsearch CV. O formato é um dicionário
config_modelos = {
    'Elastic Net': {
        'modelo': Pipeline(steps=[('preprocessor', preprocessor),
                                  ('clf', LogisticRegression(solver='saga', random_state=147))]),
        'params': {
                    'clf__C': [0.01, 0.1, 1, 10], 
                    'clf__l1_ratio': [0.1, 0.5, 0.9], # elastic net
                    'clf__class_weight': [None, 'balanced'] 
                }
    },
    'Random Forest': {
        'modelo': Pipeline(steps=[('preprocessor', preprocessor),
                                  ('clf', RandomForestClassifier(random_state=147))]),
        'params': {
            'clf__n_estimators': [100, 300],
            'clf__max_depth': [5, 10, 15], # esses clf no prefixo dos hiperparametros é para o pipeline entrar aqui e encontrá-los
            'clf__min_samples_leaf': [1, 4], 
            'clf__class_weight': ['balanced', 'balanced_subsample'] 
        }
    },
    'MLP (Rede Neural)': {
        'modelo': Pipeline(steps=[('preprocessor', preprocessor),
                                  ('clf', MLPClassifier(max_iter=1500, random_state=147))]),
        'params': {
                   
                    'clf__hidden_layer_sizes': [(64,), (128,), (64, 32), (64, 32, 16)], 
                    'clf__alpha': [0.0001, 0.001, 0.01],
                    'clf__learning_rate_init': [0.001, 0.01]
                }
    },
    'LightGBM': { #não usa preprocessor, vai a seco mesmo as variáveis categóricas, como havíamos falado antes
        'modelo': LGBMClassifier(random_state=147, verbose=-1), 
        'params': {
            'learning_rate': [0.01, 0.05, 0.1], # 0.01 geralmente dá resultados muito superiores em boosting
            'num_leaves': [31, 50, 70], 
            'class_weight': [None, 'balanced']
        }
    },
    'CatBoost': {
        'modelo': CatBoostClassifier(random_state=147, verbose=0),
        'params': {
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8], # 8 costuma ser o limite antes de overfitar muito em bases pequenas segundo 
            'auto_class_weights': ['None', 'Balanced']
        }
    }
}

# %% [markdown]
# Através do GridSearch, que nada mais é do que uma metodologia de Cross-validation, o código não treina apenas um modelo, mas dezenas de combinações de hiperparâmetros para cada algoritmo em uma validação cruzada de 5 folds (cv=5). O diferencial metodológico está na âncora ``scoring='recall'``. Isso força a matemática do gradiente descendente a penalizar severamente os falsos negativos, elegendo apenas as versões de modelos que são agressivas na captura de clientes propensos ao churn (minimizando o Erro Tipo II, que é o que é realmente custodo num contexto de churn).
# 
# É interessante destacar que a IA generativa nos ajudou sumariamente neste ponto, onde existe uma incompatibilidade entre funções do CatBoost e do Sklearn, que fazem com que tenhamos que abordar esse modelo de forma específica dentro do pipeline de treino. O dicionário ``melhores_modelos`` salva as arquiteturas vencedoras já treinadas na memória, enquanto a lista ``placar_grid`` compila o dataframe final ordenado, entregando a tabela de resultados limpa

# %%
placar_grid = []
melhores_modelos = {}

for nome, config in config_modelos.items():
    print(f"Treinando {nome}...")
    
    grid = GridSearchCV(
        estimator=config['modelo'],
        param_grid=config['params'],
        cv=5, 
        scoring='recall',
        n_jobs=-1
    )
    
    if nome == 'CatBoost':
        grid.fit(X_train, y_train, cat_features=list(cat_vars_sobreviventes))
    else:
        grid.fit(X_train, y_train)
        
    melhores_modelos[nome] = grid.best_estimator_
    
    # Avaliação no conjunto de Teste isolado
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    
    placar_grid.append({
        'Arquitetura': nome,
        'Recall (Teste)': recall_score(y_test, y_pred),
        'ROC-AUC (Teste)': roc_auc_score(y_test, y_proba),
        'Melhores Params': str(grid.best_params_)
    })

df_benchmark_final = pd.DataFrame(placar_grid).sort_values(by='Recall (Teste)', ascending=False).reset_index(drop=True)
display(df_benchmark_final)

# %% [markdown]
# O CatBoost assumiu o topo do pódio, alcançando o maior Recall (0.8288) e o maior ROC-AUC (0.8483) do benchmark. Isso comprova matematicamente que o motor nativo dele para variáveis categóricas sobrepôs a abordagem tradicional de binarização do One-Hot Encoding. Todos os modelos baseados em árvore + a Logística (elastic net) escolheram a configuração de pesos balanceados ('balanced') como a arquitetura ideal. Forçar o algoritmo a penalizar o Falso-Negativo foi a única forma de fazê-lo caçar o Churn até a morte.

# %%
campeao = melhores_modelos['CatBoost']

# Gera as predições finais na base de teste (a base isolada que ele não viu no treino)
y_pred_campeao = campeao.predict(X_test) # Lembrando que o CatBoost usa a matriz B

# Calcula e plota a Matriz de Confusão
cm = confusion_matrix(y_test, y_pred_campeao)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ficou (0)', 'Churn (1)'])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', values_format='d', ax=ax)
ax.set_title('Matriz de Confusão - Campeão (CatBoost)', pad=15)
sns.despine()
plt.show()

# %% [markdown]
# Os números da matriz de confusão refletem o comportamento de um modelo otimizado para dizimar o erro tipo II. Dos 374 clientes que realmente cancelariam o serviço na base de teste, o CatBoost conseguiu identificar e 310 verdadeiros positivos, deixando escapar apenas 64 cancelamentos silenciosos Falsos Negativos. Em termos financeiros, o modelo garantiu a retenção de 310 lifetime values (LTV) que seriam perdidos. Para atingir essa agressividade na captura, o algoritmo gerou 274 falsos positivos, ou sjea, clientes que foram classificados como risco de evasão, mas que acabaram ficando. O modelo provou matematicamente que a introdução do peso balanceado ``class_weight='balanced'`` funcionou. O algoritmo aceitou errar a favor do conservadorismo pra segurar o erro tipo II.
# 
# No mundo real das telecomunicações, o custo operacional de enviar uma campanha de retenção, um e-mail ou oferecer um cupom de desconto preventivo para esses 274 Falsos Positivos é comumente menor do que o rombo financeiro gerado caso os 310 clientes reais passassem despercebidos, então o trade-off vale a pena. 
# 
# Como o sklearn não tem suporte para colocar balanceamento de classes, para extrair o verdadeiro poder das Redes Neurais em cenários desbalanceados, a abordagem exigiria abandonar o Scikit-Learn e implementar a MLP em frameworks dedicados a tensores. Então trouxemos MLP's até o limite em casos dessa natureza para demonstrar a sua fronteira que deve desaguar em modelos de maiores complexidades. Se esse mesmo dataset fosse inserido em uma topologia construída no Keras, por exemplo, o problema seria resolvido instantaneamente passando o argumento class_weight no método .fit() da rede.

# %% [markdown]
# ## Bônus

# %%
# Como eu falei acima, vou treinar um modelo simples aqui no Keras:

pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
pesos_dict = {0: pesos[0], 1: pesos[1]} # Transforma no dicionário que o Keras exige

#A mesma topologia profunda testada no sklearn: 64 -> 32
modelo_keras = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Função sigmóide para a probabilidade binária do Churn
])

# Compilando já de olho na métrica de negócio
modelo_keras.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Recall'])

# 3. O Treinamento Penalizado (Onde a mágica acontece)
print("Treinando Rede Neural Keras com pesos balanceados...")
historico = modelo_keras.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    class_weight=pesos_dict, # O parâmetro salvador da pátria
    verbose=0)

loss, recall_keras = modelo_keras.evaluate(X_test, y_test, verbose=0)
print(f"Recall da Rede Neural (Keras): {recall_keras:.4f}")


