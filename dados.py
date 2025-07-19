    
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



dados = pd.read_csv('dados.csv')
df = pd.DataFrame(dados)
print(df.columns)

x = np.array(df.index).reshape(-1,1)
y = np.array(df['Tempo_espera'])

modelo = LinearRegression()
modelo.fit(x, y)

previsao = modelo.predict([[len(df)+1]])

print('PREVISAO DE TEMPO DE ESPERA PARA O PROXIMO PACIENTE', previsao)