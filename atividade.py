import numpy as np
from sklearn.linear_model import LinearRegression

vendas = {
    'jul': 500000,
    'ago': 700000,
    'set': 900000,
    'out': 90000,
    'nov': 1000000,
    'dez': 200000
}

meses = np.array([1,2,3,4,5,6]).reshape(-1,1)
valores = np.array([500000, 700000, 900000, 90000, 1000000, 200000])
modelo = LinearRegression()
modelo.fit(meses, valores)
proximo_mes = 7
venda_prevista = modelo.predict([[proximo_mes]])[0]
print(f'Previsao de venda para o proximo mes - {proximo_mes} -> {venda_prevista}')