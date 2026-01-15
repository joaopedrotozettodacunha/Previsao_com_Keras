bike = pd.read_csv('/content/bicicletas.csv')

bike.head()

bike['datas'] = pd.to_datetime(bike['datas'])

bike.head()

sns.lineplot(x = 'datas', y ='contagem', data = bike)
plt.xticks(rotation = 70) #serve para inclinar os textos do eixo x

sc2 = StandardScaler()

sc2.fit(bike['contagem'].values.reshape(-1,1))
#values converte uma Series para um array, to_numpy() é mais moderno
#depois do values vc tem isso array([120, 135, 150, 160, ...])
#scikit-learn exige entradas 2D, (n_amostras, n_features)
#reshape(-1, 1) -1 calcula o numero de linhas, 1 forca exatamente a uma coluna
'''array([
 [120],
 [135],
 [150],
 [160],
 ...
])'''

y = sc2.transform(bike['contagem'].values.reshape(-1, 1))
y

tamanho_treino = int(len(bike) * 0.9)
tamanho_treino

tamanho_teste = int(len(bike) - tamanho_treino)
tamanho_teste

ytreino = y[0:tamanho_treino]
yteste = y[tamanho_treino:len(y)]

sns.lineplot(x = 'datas', y = ytreino[:,0], data = bike[0:tamanho_treino], label = 'treino')
sns.lineplot(x = 'datas', y = yteste[:,0], data = bike[tamanho_treino:len(y)], label = 'Teste')
plt.xticks(rotation = 70)

vetor = pd.DataFrame(ytreino)[0]

vetor



xtreino_novo, ytreino_novo = separa_dados(vetor, 10)

xtreino_novo

xtreino_novo.shape

ytreino_novo

vetor2 = pd.DataFrame(yteste)[0]

xteste_novo, yteste_novo = separa_dados(vetor2, 10)

xteste_novo.shape

"""Redes Neurais Recorrentes"""

xtreino_novo = xtreino_novo.reshape((xtreino_novo.shape[0], xtreino_novo.shape[1], 1))

xtreino_novo.shape

xteste_novo = xteste_novo.reshape((xteste_novo.shape[0], xteste_novo.shape[1], 1))

recorrente =  Sequential()

from tensorflow.keras.layers import LSTM

recorrente.add(LSTM(128, input_shape = (xtreino_novo.shape[1], xtreino_novo.shape[2])))
#128 neuronios recorrentes
#input_shape = (passos, num_features) passos significa quantidade de valores anteriores usados para prever o próximo e num_features representa o numero de features
#no caso 1 pois é apenas a feature de contagem
recorrente.add(Dense(units = 1))
#units = 1 produz um valor de saida

recorrente.compile(loss = 'mean_squared_error', optimizer = 'RMSProp')
#optimizer = 'RMSProp' muitas vezes otimizador padrao de LSTM
#LSTM mitiga o vanish gradient

recorrente.summary()
