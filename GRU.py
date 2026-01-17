from tensorflow.keras.layers import GRU

xtreino_novo.shape

recorrente_g = Sequential()

recorrente_g.add(GRU(128, input_shape = (xtreino_novo.shape[1], xtreino_novo.shape[2])))
recorrente_g.add(Dense(units = 1))

recorrente_g.compile(loss = 'mean_squared_error', optimizer = 'RMSProp')

recorrente.summary()

recorrente_g.summary()

resultado_treino = recorrente_g.fit(xtreino_novo, ytreino_novo, validation_data = (xteste_novo, yteste_novo), epochs = 10)
#validation_data = (xteste_novo, yteste_novo) dados de validação

resultado_teste = recorrente_g.predict(xteste_novo)

xteste_novo.shape

yteste_novo.shape

resultado_teste

sns.lineplot(x = 'datas', y = yteste[:,0], data = bike[tamanho_treino:len(bike)], label = 'Teste')
sns.lineplot(x = 'datas', y = resultado_teste[:,0], data = bike[tamanho_treino + 10: len(bike)], label = 'Previsao_Teste')
plt.xticks(rotation = 70)

print(resultado.history.keys())

plt.plot(resultado.history['loss'])
plt.plot(resultado.history['val_loss'])
plt.legend('Treino', 'Teste')
plt.xlabel('Epocas')
plt.ylabel('Custo')

plt.plot(resultado_treino.history['loss'])
plt.plot(resultado_treino.history['val_loss'])
plt.legend('Treino', 'Teste')
plt.xlabel('Epocas')
plt.ylabel('Custo')
