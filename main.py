import pandas as pd
import seaborn as sns
import matplotlib as mlp

passageiros = pd.read_csv("/content/Passageiros.csv")
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
#optimizer = 'adam' e a funcao de otimizacao, atualizando os valores dos pesos e bias

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

import numpy as np

vetor = pd.DataFrame(y_treino)[0] #[0] significa que é uma Series que sera entrada para a funcao separa_dados
vetor

def separa_dados(vetor, n_passos):

  x_novo, y_novo = [], []

  for i in range(n_passos, vetor.shape[0]): #range(inicio, fim), vetor.shape[0] retorna o numero de linhas de um vetor ou matriz
    x_novo.append(list(vetor.loc[i-n_passos:i-1])) #.loc[inicio:fim] retorna todos os elementos cujo indice esteja entre inicio e fim, list transforma em uma lista
    y_novo.append(vetor.loc[i])
  x_novo, y_novo = np.array(x_novo), np.array(y_novo) #transforma em arrays, oq antes era lista, ideal para ML, pois as libs esperam arrays numpy
  return x_novo, y_novo

x_treino_novo, y_treino_novo = separa_dados(vetor, 1)

x_treino_novo[0:5]

y_treino_novo[0:5]

vetor2 = pd.DataFrame(y_teste)[0]
x_teste_novo, y_teste_novo = separa_dados(vetor2, 1)

x_teste_novo

y_teste_novo

"""Previsão com Janelas Temporais"""

regressor3 = Sequential()

regressor3.add(Dense(8, input_dim = 1, kernel_initializer = 'ones', activation = 'linear',use_bias = False))
#kernel_initializer = 'ones' define que todos os pesos de todos os neuronios sao 1
regressor3.add(Dense(64, kernel_initializer = 'random_uniform', activation = 'sigmoid', use_bias = False))
#kernel_initializer = 'random_uniform' inicializa pesos com valores aleatorios
regressor3.add(Dense(1, kernel_initializer = 'random_uniform', activation = 'linear', use_bias = False)) #apenas um neuronio na ultima camada pois a saida sera apenas um valor

regressor3.compile(loss = 'mean_squared_error', optimizer = 'adam')
#compile define as regras de aprendizado
#loss = 'mean_squared_error' função de perda, boa para analisar se esta diminuindo perda de acordo com as epocas
#optimizer = 'adam' ajusta os pesos para diminuir a função de perda

regressor3.fit(x_treino_novo, y_treino_novo, epochs = 100)

y_predict_novo = regressor3.predict(x_treino_novo)

y_predict_novo

sns.lineplot(x = 'tempo', y = y_treino_novo, data = passageiros[1:129], label = 'Treino')
sns.lineplot(x = 'tempo', y = pd.DataFrame(y_predict_novo)[0], data = passageiros[1:129], label = 'Previsao_treino')
#pd.DataFrame(y_predict_novo)[0] tem que transformar em DF pois o keras retorna uma lista de listas array([[-1.3029567 ],[-1.2666299 ], ...]]

resultado_teste = regressor3.predict(x_teste_novo)

resultado = pd.DataFrame(resultado_teste)[0]

resultado

sns.lineplot(x = 'tempo', y = y_treino_novo, data = passageiros[1:129], label = 'Treino')
sns.lineplot(x = 'tempo', y = pd.DataFrame(y_predict_novo)[0], data = passageiros[1:129], label = 'Previsao_treino')
#pd.DataFrame(y_predict_novo)[0] tem que transformar em DF pois o keras retorna uma lista de listas array([[-1.3029567 ],[-1.2666299 ], ...]]
sns.lineplot(x = 'tempo', y = y_teste_novo, data = passageiros[130:144], label = 'Teste')
sns.lineplot(x = 'tempo', y = resultado.values, data = passageiros[130:144], label = 'Previsao_teste')

xtreino_novo, ytreino_novo = separa_dados(vetor, 4)

xtreino_novo[0:5]

ytreino_novo[0:5]

xteste_novo, yteste_novo = separa_dados(vetor2, 4)

regressor4 = Sequential()

regressor4.add(Dense(8, input_dim = 4, kernel_initializer='ones', activation = 'linear', use_bias = 'False'))
regressor4.add(Dense(64, kernel_initializer='random_uniform', activation = 'sigmoid', use_bias = 'False'))
regressor4.add(Dense(1, kernel_initializer='random_uniform', activation = 'linear', use_bias = 'False'))

regressor4.compile(loss = 'mean_squared_error', optimizer = 'adam')

regressor4.fit(xtreino_novo, ytreino_novo, epochs = 200)

y_predict_novo = regressor4.predict(xtreino_novo)

y_predict_novo = pd.DataFrame(y_predict_novo)[0]

resultado_novo = regressor4.predict(xteste_novo)

resultado_novo = pd.DataFrame(resultado_novo)[0]

sns.lineplot(x = 'tempo', y = ytreino_novo, data = passageiros[4:129], label = 'treino')
sns.lineplot(x = 'tempo', y = y_predict_novo.values, data = passageiros[4:129], label = 'Previsao_treino')
sns.lineplot(x = 'tempo', y = yteste_novo, data = passageiros[133:144], label = 'Teste')
sns.lineplot(x = 'tempo', y = resultado_novo.values, data = passageiros[133:144], label = 'Presvisao_teste')
