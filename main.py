import pandas as pd
import seaborn as sns
import matplotlib as mlp

passageiros = pd.read_csv("/Passageiros.csv")
print(passageiros.head())

mlp.rcParams['figure.figsize'] = (10, 6)
mlp.rcParams['font.size'] = 22

sns.lineplot(x='tempo', y='passageiros', data = passageiros, label = "Gráfico")

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(passageiros)
dados_escalado = sc.transform(passageiros)

x = dados_escalado[:,0] #todas as linhas da coluna 1
y = dados_escalado[:,1] #todas as linhas da colunas 2

import matplotlib.pyplot as plt

sns.lineplot(x=x, y=y,label = "Gráfico Escalado")
plt.ylabel("Passageiros")
plt.xlabel("Data")

tamanho_treino = int(len(passageiros) * 0.9)
tamanho_teste = len(passageiros) - tamanho_treino

x_treino = x[0:tamanho_treino] #tamanho treino é exclusivo
y_treino = y[0:tamanho_treino]

x_teste = x[(tamanho_treino):len(passageiros)]
y_teste = y[(tamanho_treino): len(passageiros)]

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size = 0.1,
    shuffle = False
    ) #split treino/teste automático

sns.lineplot(x = x_treino, y = y_treino, label = "Treino")
sns.lineplot(x = x_teste, y = y_teste, label = "Teste")
plt.ylabel('Data')
plt.xlabel('Passageiros')

from tensorflow.keras.models import Sequential #importação da classe Sequencial
from tensorflow.keras.layers import Dense #importação da classe Dense

regressor = Sequential() #criando um modelo sequencial vazio
regressor.add(Dense(1,input_dim = 1, kernel_initializer = 'ones', activation = 'linear', use_bias = True)) #adicionando uma camada com 1 neuronio
#regressor.add(Dense(4,input_dim = 1, kernel_initializer = 'ones', activation = 'linear', use_bias = True)) #adicionando uma camada densa com 4 neuronios, cada um com seu peso e vies
#1 representa o número de neurônios
#inpu_dim representa o numero de features
#kernel_initializer = 'ones'significa que todos os pesos(w) comecam com valor 1
#activation = 'linear' função de ativacao de regressao linear
#use_bias = True adiciona ou nao um vies
regressor.compile(loss='mean_squared_error', optimizer = 'adam') #compile define como o modelo vai aprender
#loss='mean_squared_error' e a funcao de erro
#optimizer = 'adam' e a funcao de otimizacao

regressor.summary() #resumo

regressor.fit(x_treino, y_treino)

y_predict = regressor.predict(x_treino)
y_predict.shape #para 129 entradas, o modelo retornou 1 valor previsto por entrada

sns.lineplot(x = x_treino, y = y_treino, label = 'Treino')
sns.lineplot(x = x_treino, y = y_predict[:,0], label = 'Ajuste_Treino')
#[:,0] tem que usar pois o predict retorna uma matriz 2D

"""Voltando para a Escala"""

dados = {'tempo': x_treino, 'passageiros':y_predict[:,0]}

resultados = pd.DataFrame(data = dados)
resultados #exibe o dataframe com as colunas normalizadas

resultados_transformados = sc.inverse_transform(resultados)
resultados_transformados = pd.DataFrame(data = resultados_transformados)
resultados_transformados.columns = ['tempo', 'passageiros']
resultados_transformados

sns.lineplot(x = 'tempo', y = 'passageiros', data = passageiros, label = 'Transformados')
sns.lineplot(x = 'tempo', y = 'passageiros', data = resultados_transformados, label = 'Previsao_treino')

predict_teste = regressor.predict(x_teste)
predict_teste.shape

dados = {'tempo': x_teste, 'passageiros':predict_teste[:,0]}
dados = pd.DataFrame(data = dados)
resultados_predicao_transformados = sc.inverse_transform(dados)
resultados_predicao_transformados = pd.DataFrame(data = resultados_predicao_transformados)
resultados_predicao_transformados.columns = ['tempo', 'passageiros']

sns.lineplot(x = 'tempo', y = 'passageiros', data = passageiros, label = 'Transformados')
sns.lineplot(x = 'tempo', y = 'passageiros', data = resultados_transformados, label = 'Previsao_treino')
sns.lineplot(x = 'tempo', y = 'passageiros', data = resultados_predicao_transformados, label = 'Previsao_treino')

"""Mult Layer Perceptron"""

regressor2 = Sequential()
regressor2.add(Dense(8, input_dim = 1, kernel_initializer = 'random_uniform', activation = 'sigmoid', use_bias = False))
regressor2.add(Dense(8, kernel_initializer = 'random_uniform', activation = 'sigmoid', use_bias = False))
regressor2.add(Dense(1, kernel_initializer = 'random_uniform', activation = 'linear', use_bias = False))

regressor2.compile(loss = 'mean_squared_error', optimizer = 'adam')
regressor2.summary()

regressor2.fit(x_treino, y_treino, epochs = 200)

y_predict_treino = regressor2.predict(x_treino)
y_predict_teste = regressor2.predict(x_teste)

sns.lineplot(x = x_treino, y = y_treino, label = 'Treino')
sns.lineplot(x = x_teste, y = y_teste, label = 'Teste')
sns.lineplot(x = x_treino, y = y_predict_treino[:,0] , label = 'Previsao_treino')
sns.lineplot(x = x_teste, y = y_predict_teste[:,0], label = 'Previsao_teste')
