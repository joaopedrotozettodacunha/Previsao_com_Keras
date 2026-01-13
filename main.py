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
