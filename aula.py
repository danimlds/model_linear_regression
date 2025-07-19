import numpy as np
from sklearn.linear_model import LinearRegression

vendas = {
    'jan':2000,
    'fev':3000,
    'mar':4000
}

#Transforma em dados numericos
meses = np.array([1,2,3]).reshape(-1,1)
valores = np.array([2000,3000,4000])

#Criando modelo

modelo = LinearRegression()
modelo.fit(meses, valores)

#previsão
proximo_mes = 4
venda_prevista = modelo.predict([[proximo_mes]])[0]
print(f'Previsao de venda para o proximo mes - {proximo_mes} -> {venda_prevista}')

